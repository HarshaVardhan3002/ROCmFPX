#include "models.h"

#include <algorithm>
#include <cmath>
#include <string>

// audio8 (arktts) generation stages
//
// GEN_CODE: slow AR hidden state + code 0 -> codes 1..9 (fast AR) + the next slow AR input
// GEN_WAV : a window of 10-code frames -> 44.1 kHz PCM, through the arktts codec decoder
//
// the fast AR predicts code L from the prefix [h, code 0 .. code L-1] at positions 0..L.
// attention is causal, so a level is one pass over L+1 rows, no kv cache needed

static constexpr float audio8_fast_rope_theta = 1000000.0f;

// fast AR: llama style layers, rope on interleaved pairs, GQA, no bias
// x: [dim, T]. returns [dim, T]
ggml_tensor * clip_graph_audio8_gen::fast_forward(ggml_tensor * x) const {
    const int     n_head    = hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;
    const int64_t T         = x->ne[1];

    ggml_tensor * pos = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) T, 1.0f), GGML_TYPE_I32);

    for (size_t il = 0; il < model.layers.size(); il++) {
        const auto & layer = model.layers[il];
        const int64_t d_head = layer.q_w->ne[1] / n_head;
        ggml_tensor * inp = x;

        ggml_tensor * cur = build_norm(x, layer.ln_1_w, nullptr, NORM_TYPE_RMS, hparams.eps, (int) il);

        ggml_tensor * q = ggml_reshape_3d(ctx0, build_mm(layer.q_w, cur), d_head, n_head,    T);
        ggml_tensor * k = ggml_reshape_3d(ctx0, build_mm(layer.k_w, cur), d_head, n_head_kv, T);
        ggml_tensor * v = ggml_reshape_3d(ctx0, build_mm(layer.v_w, cur), d_head, n_head_kv, T);

        q = ggml_rope_ext(ctx0, q, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NORMAL, 0,
                          audio8_fast_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NORMAL, 0,
                          audio8_fast_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [d_head, T, n_head]
        k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [d_head, T, n_head_kv]

        // mul_mat broadcasts kv head h / (n_head / n_head_kv), same as repeat_interleave
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q); // [T_k, T_q, n_head]
        kq = ggml_diag_mask_inf(ctx0, kq, 0);
        kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / sqrtf((float) d_head), 0.0f);

        ggml_tensor * vt  = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3)); // [T_k, d_head, n_head_kv]
        ggml_tensor * kqv = ggml_mul_mat(ctx0, vt, kq);                         // [d_head, T_q, n_head]
        cur = ggml_cont_2d(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3), d_head * n_head, T);

        cur = build_mm(layer.o_w, cur);
        x   = ggml_add(ctx0, cur, inp);

        inp = x;
        cur = build_norm(x, layer.ln_2_w, nullptr, NORM_TYPE_RMS, hparams.eps, (int) il);
        ggml_tensor * gate = build_mm(layer.ff_gate_w, cur);
        ggml_tensor * up   = build_mm(layer.ff_up_w, cur);
        cur = build_mm(layer.ff_down_w, ggml_swiglu_split(ctx0, gate, up));
        x   = ggml_add(ctx0, cur, inp);
    }
    return x;
}

// the reference sampler (legacy top-k/top-p), on device:
// sort desc, drop where cumsum(softmax(all logits)) > top_p or rank >= top_k (never rank 0),
// then divide by the temperature and draw where the cdf crosses inp_rand
// logits: [vocab]. returns the vocab id [1] I32
ggml_tensor * clip_graph_audio8_gen::sample(ggml_tensor * logits, ggml_tensor * inp_rand) const {
    const int64_t n_vocab = logits->ne[0];
    const int64_t k       = top_k > 0 && top_k < n_vocab ? top_k : n_vocab;

    auto sort_by = [this](ggml_tensor * a, ggml_tensor * idx) {
        ggml_tensor * a2d = ggml_reshape_2d(ctx0, a, 1, a->ne[0]);
        return ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, a2d, idx), idx->ne[0]);
    };

    ggml_tensor * p_full = ggml_soft_max(ctx0, logits);

    // top_k gives no order, sort the k candidates by logit
    ggml_tensor * idx = k < n_vocab ? ggml_top_k(ctx0, logits, (int) k) : ggml_argsort(ctx0, logits, GGML_SORT_ORDER_DESC);
    if (k < n_vocab) {
        idx = sort_by(idx, ggml_argsort(ctx0, sort_by(logits, idx), GGML_SORT_ORDER_DESC));
    }
    ggml_tensor * vals = sort_by(logits, idx);

    if (top_p < 1.0f) {
        // keep[i] = cdf[i] < top_p, the crossing element is dropped like in the reference
        ggml_tensor * cdf  = ggml_cumsum(ctx0, sort_by(p_full, idx));
        ggml_tensor * keep = ggml_step(ctx0, ggml_scale_bias(ctx0, cdf, -1.0f, top_p));
        // rank 0 is always kept
        ggml_tensor * first = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_arange(ctx0, 0.0f, (float) k, 1.0f), -1.0f, 0.5f));
        keep = ggml_clamp(ctx0, ggml_add(ctx0, keep, first), 0.0f, 1.0f);
        // log(1) = 0 (keep), log(0) = -inf (drop)
        vals = ggml_add(ctx0, vals, ggml_log(ctx0, keep));
    }
    vals = ggml_scale(ctx0, vals, 1.0f / std::max(temp, 1e-5f));
    cb(vals, "sample_logits", -1);

    // draw one candidate: count the cdf entries at or past inp_rand
    ggml_tensor * cumsum = ggml_cumsum(ctx0, ggml_soft_max(ctx0, vals));
    ggml_tensor * cross  = ggml_step(ctx0, ggml_sub(ctx0, cumsum, inp_rand));
    ggml_tensor * rank   = ggml_scale_bias(ctx0, ggml_sum(ctx0, cross), -1.0f, (float) k);
    // a rounding short cdf can leave no crossing, take the last candidate then
    rank = ggml_clamp(ctx0, rank, 0.0f, (float) (k - 1));

    ggml_tensor * id = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, idx, 1, k), ggml_cast(ctx0, rank, GGML_TYPE_I32));
    id = ggml_reshape_1d(ctx0, id, 1);
    cb(id, "sample_token_id", -1);
    return id;
}

// arktts codec decoder: RVQ decode -> window transformer -> 2x (ConvTranspose1d + ConvNeXt) -> DAC -> tanh
// inp_codes: [n_frames, 10] I32, already clamped to each codebook. returns [n_frames * 2048] F32
ggml_tensor * clip_graph_audio8_gen::wav_decode(clip_graph_qwen3tts_gen::code2wav & c2w, ggml_tensor * inp_codes) const {
    const auto &  w = model.c2w;
    const int64_t T = inp_codes->ne[0];

    auto ids = [&](int g) {
        return ggml_view_1d(ctx0, inp_codes, T, (size_t) g * inp_codes->nb[1]);
    };

    // 1. codebooks already hold out_proj(code) + bias, see conversion/audio8.py
    ggml_tensor * cur = ggml_get_rows(ctx0, w.quant_first_cb_w, ids(0)); // [1024, T]
    const int64_t n_rest = w.quant_rest_cb_w->ne[2];
    for (int g = 1; g <= n_rest; g++) {
        ggml_tensor * cb_g = ggml_view_2d(ctx0, w.quant_rest_cb_w, w.quant_rest_cb_w->ne[0], w.quant_rest_cb_w->ne[1],
                                          w.quant_rest_cb_w->nb[1], (size_t) (g - 1) * w.quant_rest_cb_w->nb[2]);
        cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, cb_g, ids(g)));
    }
    cb(cur, "wav_quant_hidden", -1);

    // 2. post_module: causal transformer with a 128 frame window, q/k permuted for NEOX rope at conversion
    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        cur = c2w.tfm_layer_forward(cur, w.tfm_layers[il], il);
    }
    cur = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    cur = ggml_mul(ctx0, cur, w.tfm_output_norm_w);
    cb(cur, "wav_tfm_out", -1);

    // 3. upsample, kernel == stride so no overlap tail
    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [T, 1024]
    for (size_t il = 0; il < w.upsample.size(); il++) {
        const auto & up = w.upsample[il];
        x = c2w.causal_conv_transpose1d(x, up.conv_w, up.conv_b, 2, "");
        x = c2w.convnext_block(x, up, "up" + std::to_string(il));
        cb(x, "wav_upsample_out", (int) il);
    }

    // 4. DAC decoder, kernel == 2 * stride in every block
    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };

    x = c2w.causal_conv1d(x, w.dac_entry_w, w.dac_entry_b, 1, "dac_entry");
    for (size_t il = 0; il < w.dac.size(); il++) {
        const auto & blk = w.dac[il];
        const int stride = (int) (blk.conv_w->ne[0] / 2);
        const std::string blk_name = "dac" + std::to_string(il);
        x = c2w.snake(x, blk.snake_alpha, blk.snake_beta);
        x = c2w.causal_conv_transpose1d(x, blk.conv_w, blk.conv_b, stride, blk_name + "_tail");
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            x = c2w.dac_res_unit(x, blk.res[ir], DAC_DILATIONS[ir], blk_name + "_res" + std::to_string(ir));
        }
        cb(x, "wav_dac_block_out", (int) il);
    }

    x = c2w.snake(x, w.dac_post_snake_alpha, w.dac_post_snake_beta);
    x = c2w.causal_conv1d(x, w.dac_post_conv_w, w.dac_post_conv_b, 1, "dac_post_conv"); // [n_samples, 1]

    // the reference ends in tanh, not a clamp
    x = ggml_tanh(ctx0, x);
    x = ggml_reshape_1d(ctx0, x, x->ne[0]);
    cb(x, "wav_audio_out", -1);
    return x;
}

// persisted state of the codec decoder: rope position, K/V window per transformer layer,
// left context / tail per stateful conv. same slots as code2wav minus pre_conv
std::vector<c2w_state_slot> list_audio8_state_slots(const clip_hparams & hparams, const clip_model & model) {
    const auto & w = model.c2w;
    std::vector<c2w_state_slot> slots;

    if (w.tfm_layers.empty()) {
        return slots;
    }

    slots.push_back({"tfm_pos", 1, 1});

    const int64_t d_head = w.tfm_layers[0].q_w->ne[1] / hparams.wav_tfm_n_head;
    const int64_t kv_ch  = d_head * hparams.wav_tfm_n_head_kv;
    const int64_t prefix = hparams.wav_tfm_swa - 1;
    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        slots.push_back({"tfm_k_" + std::to_string(il), kv_ch, prefix});
        slots.push_back({"tfm_v_" + std::to_string(il), kv_ch, prefix});
    }

    for (size_t il = 0; il < w.upsample.size(); il++) {
        const auto & up = w.upsample[il];
        slots.push_back({"up" + std::to_string(il) + "_dwconv", up.dwconv_w->ne[0] - 1, up.dwconv_w->ne[2]});
    }

    slots.push_back({"dac_entry", w.dac_entry_w->ne[0] - 1, w.dac_entry_w->ne[1]});

    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };
    for (size_t il = 0; il < w.dac.size(); il++) {
        const auto & blk = w.dac[il];
        const int64_t stride = blk.conv_w->ne[0] / 2;
        const std::string blk_name = "dac" + std::to_string(il);
        slots.push_back({blk_name + "_tail", stride, blk.conv_w->ne[1]});
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & res = blk.res[ir];
            slots.push_back({blk_name + "_res" + std::to_string(ir),
                              (res.conv1_w->ne[0] - 1) * DAC_DILATIONS[ir], res.conv1_w->ne[1]});
        }
    }

    slots.push_back({"dac_post_conv", w.dac_post_conv_w->ne[0] - 1, w.dac_post_conv_w->ne[1]});

    return slots;
}

ggml_cgraph * clip_graph_audio8_gen::build() {
    if (gen_process == CLIP_GEN_PROCESS_GEN_WAV) {
        ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_q);
        ggml_set_name(inp_codes, "inp_codes");
        ggml_set_input(inp_codes);

        clip_graph_qwen3tts_gen::code2wav c2w(*this);
        for (const auto & slot : list_audio8_state_slots(hparams, model)) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
            ggml_set_name(t, ("state_in_" + slot.name).c_str());
            ggml_set_input(t);
            c2w.state_in[slot.name] = t;
        }

        ggml_tensor * out_audio = wav_decode(c2w, inp_codes);
        ggml_set_name(out_audio, "out_audio");
        ggml_set_output(out_audio);
        ggml_build_forward_expand(gf, out_audio);

        for (auto & slot : c2w.state_out) {
            ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
            ggml_set_output(slot.second);
            ggml_build_forward_expand(gf, slot.second);
        }
        return gf;
    }

    const int64_t dim   = model.gen_code_embd_w->ne[0];
    const int64_t vocab = model.gen_code_head_w->ne[1];
    const int     n_gen = n_q - 1;

    // the slow AR hidden state (after its output norm) arrives as the single batch entry
    ggml_tensor * h_state = build_inp_raw(1);
    h_state = ggml_reshape_2d(ctx0, h_state, n_mmproj_embd, 1);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    // per-level logit bias, -inf everywhere but the code for teacher forcing, else 0
    ggml_tensor * inp_bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vocab, n_gen);
    ggml_set_name(inp_bias, "inp_bias");
    ggml_set_input(inp_bias);

    ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_gen);
    ggml_set_name(inp_rand, "inp_rand");
    ggml_set_input(inp_rand);

    ggml_tensor * x = h_state;
    std::vector<ggml_tensor *> codes = { code0 };
    ggml_tensor * all_logits = nullptr;
    for (int level = 1; level <= n_gen; level++) {
        x = ggml_concat(ctx0, x, ggml_get_rows(ctx0, model.gen_code_embd_w, codes.back()), 1); // [dim, level + 1]

        ggml_tensor * y = fast_forward(x);
        y = ggml_view_2d(ctx0, y, dim, 1, y->nb[1], (size_t) level * y->nb[1]);
        y = build_norm(y, model.gen_code_norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);

        ggml_tensor * logits = build_mm(model.gen_code_head_w, y); // [vocab, 1]
        cb(logits, "fast_logits", level);
        all_logits = all_logits ? ggml_concat(ctx0, all_logits, logits, 1) : logits;

        ggml_tensor * z = ggml_add(ctx0, ggml_reshape_1d(ctx0, logits, vocab),
                                   ggml_view_1d(ctx0, inp_bias, vocab, (size_t) (level - 1) * inp_bias->nb[1]));
        ggml_tensor * r = ggml_view_1d(ctx0, inp_rand, 1, (size_t) (level - 1) * inp_rand->nb[0]);
        codes.push_back(sample(z, r));
    }

    // output 1: the 10 codes of this frame, gathered as floats from an iota then cast back
    ggml_tensor * iota = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) vocab, 1.0f), 1, vocab);
    ggml_tensor * out_codes = nullptr;
    for (ggml_tensor * c : codes) {
        ggml_tensor * f = ggml_get_rows(ctx0, iota, c); // [1, 1]
        out_codes = out_codes ? ggml_concat(ctx0, out_codes, f, 1) : f;
    }
    out_codes = ggml_cast(ctx0, ggml_reshape_1d(ctx0, out_codes, n_q), GGML_TYPE_I32);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    // output 2: the fast AR logits [vocab, 9], for checks against the reference
    ggml_set_name(all_logits, "out_feats");
    ggml_set_output(all_logits);
    ggml_build_forward_expand(gf, all_logits);

    // output 3: the slow AR input of the next frame, sum_i table[i * vocab + code_i]
    // the semantic token embedding is folded into the code 0 rows at conversion
    ggml_tensor * table = model.gen_audio_embd_w;
    ggml_tensor * embd  = nullptr;
    for (size_t l = 0; l < codes.size(); l++) {
        ggml_tensor * t = ggml_view_2d(ctx0, table, table->ne[0], vocab, table->nb[1], l * vocab * table->nb[1]);
        ggml_tensor * e = ggml_get_rows(ctx0, t, codes[l]);
        embd = embd ? ggml_add(ctx0, embd, e) : e;
    }
    cb(embd, "gen_embd", -1);

    // out embd goes last, clip_encode() reads it back via ggml_graph_node(gf, -1)
    ggml_build_forward_expand(gf, embd);

    return gf;
}

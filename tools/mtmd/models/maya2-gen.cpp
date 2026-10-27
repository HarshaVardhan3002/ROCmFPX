#include "models.h"

#include <cmath>

// maya2 generation stages
//
// GEN_CODE: backbone hidden state + code 0 -> codes 1..31 (depth decoder) + the next backbone input
// GEN_WAV : a window of 32-code frames -> PCM, through the kyutai mimi decoder
//
// the depth decoder predicts level L from the prefix [h, code 0 .. code L-1]. each prefix row is embedded
// from its slot alone and the self-attention is causal, so a level only needs one pass over L+1 rows
// through the 4 layers of its weight group

// one weight group: causal self-attention + GELU FFN (pre-norm), then the cross-attention constant
ggml_tensor * clip_graph_maya2_gen::depth_trunk(ggml_tensor * x, int group) const {
    const auto & depth = model.depth;
    const int64_t dim  = x->ne[0];
    const int64_t T    = x->ne[1];
    const int64_t nh   = dim / depth_d_head;
    const size_t  es   = ggml_element_size(x);

    for (int il = 0; il < depth.n_layer; il++) {
        const auto & layer = depth.blocks[group * depth.n_layer + il];
        ggml_tensor * inp = x;

        ggml_tensor * cur = build_norm(x, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, 1e-5f, il);
        ggml_tensor * qkv = build_mm(layer.qkv_w, cur); // [3*dim, T]

        ggml_tensor * q = ggml_view_3d(ctx0, qkv, depth_d_head, nh, T, depth_d_head * es, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_3d(ctx0, qkv, depth_d_head, nh, T, depth_d_head * es, qkv->nb[1], dim * es);
        ggml_tensor * v = ggml_view_3d(ctx0, qkv, depth_d_head, nh, T, depth_d_head * es, qkv->nb[1], 2 * dim * es);

        q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [d_head, T, nh]
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);

        // no positional encoding, the slot is in the input embedding
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q); // [T_k, T_q, nh]
        kq = ggml_diag_mask_inf(ctx0, kq, 0);
        kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / sqrtf((float) depth_d_head), 0.0f);

        ggml_tensor * vt  = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3)); // [T_k, d_head, nh]
        ggml_tensor * kqv = ggml_mul_mat(ctx0, vt, kq);                         // [d_head, T_q, nh]
        cur = ggml_cont_2d(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3), dim, T);

        cur = build_mm(layer.o_w, cur);
        x   = ggml_add(ctx0, cur, inp);

        inp = x;
        cur = build_norm(x, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, 1e-5f, il);
        cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_GELU_ERF, il);
        x   = ggml_add(ctx0, cur, inp);

        // the speaker memory is one token, so the cross-attention adds o(v(mem)), folded at conversion
        x = ggml_add(ctx0, x, ggml_view_1d(ctx0, depth.cross_w, dim, (size_t) il * depth.cross_w->nb[1]));
    }
    return x;
}

ggml_cgraph * clip_graph_maya2_gen::build() {
    const auto & depth = model.depth;
    const int64_t dim   = depth.in_proj_w->ne[1];
    const int64_t vocab = depth.head_w->ne[1];
    const int64_t rank  = depth.code_lo_w->ne[0];

    if (gen_process == CLIP_GEN_PROCESS_GEN_WAV) {
        // [n_frames, 32] group-major codes -> PCM
        ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_level + 1);
        ggml_set_name(inp_codes, "inp_codes");
        ggml_set_input(inp_codes);

        // split RVQ: semantic codebook + 31 acoustic ones, each with its output projection
        clip_graph_qwen3tts_gen::code2wav c2w(*this);
        ggml_tensor * cur = c2w.quant_decode(inp_codes);

        // hf mimi keeps q/k in the rotate_half layout and uses the exact gelu
        clip_graph_pockettts_seanet seanet(*this);
        cur = seanet.mimi_decode(cur, GGML_ROPE_TYPE_NEOX, FFN_GELU_ERF);
        ggml_set_name(cur, "out_audio");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // the backbone hidden state arrives as the single batch entry
    ggml_tensor * h_state = build_inp_raw(1);
    h_state = ggml_reshape_2d(ctx0, h_state, n_mmproj_embd, 1);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    // per-level logit bias, -inf everywhere but the code for teacher forcing, else 0
    ggml_tensor * inp_bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vocab, n_level);
    ggml_set_name(inp_bias, "inp_bias");
    ggml_set_input(inp_bias);

    ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_level);
    ggml_set_name(inp_rand, "inp_rand");
    ggml_set_input(inp_rand);

    auto level_embd = [&](int slot) {
        return ggml_view_1d(ctx0, depth.level_embd_w, dim, (size_t) slot * depth.level_embd_w->nb[1]);
    };

    // prefix row for slot k: code_up(code_lo[k * vocab + code]) + level_embd[k + 1]
    auto code_row = [&](int slot, ggml_tensor * code) {
        ggml_tensor * lo = ggml_view_2d(ctx0, depth.code_lo_w, rank, vocab, depth.code_lo_w->nb[1],
                                        (size_t) slot * vocab * depth.code_lo_w->nb[1]);
        ggml_tensor * e = build_mm(depth.code_up_w, ggml_get_rows(ctx0, lo, code)); // [dim, 1]
        return ggml_add(ctx0, e, level_embd(slot + 1));
    };

    ggml_tensor * x = build_mm(depth.in_proj_w, h_state);
    x = ggml_add(ctx0, x, level_embd(0));

    clip_graph_qwen3tts_gen::code_gen sampler(*this, top_k, 1.0f);

    std::vector<ggml_tensor *> codes = { code0 };
    ggml_tensor * all_logits = nullptr;
    for (int level = 1; level <= n_level; level++) {
        x = ggml_concat(ctx0, x, code_row(level - 1, codes.back()), 1); // [dim, level + 1]

        ggml_tensor * y = depth_trunk(x, group_of_level(level));
        y = ggml_view_2d(ctx0, y, dim, 1, y->nb[1], (size_t) level * y->nb[1]);
        y = build_norm(y, depth.norm_w, depth.norm_b, NORM_TYPE_NORMAL, 1e-5f, -1);

        ggml_tensor * head = ggml_view_2d(ctx0, depth.head_w, dim, vocab, depth.head_w->nb[1],
                                          (size_t) (level - 1) * depth.head_w->nb[2]);
        ggml_tensor * logits = ggml_mul_mat(ctx0, head, y); // [vocab, 1]
        cb(logits, "depth_logits", level);
        all_logits = all_logits ? ggml_concat(ctx0, all_logits, logits, 1) : logits;

        ggml_tensor * z = ggml_scale(ctx0, logits, 1.0f / temp);
        z = ggml_add(ctx0, ggml_reshape_1d(ctx0, z, vocab),
                     ggml_view_1d(ctx0, inp_bias, vocab, (size_t) (level - 1) * inp_bias->nb[1]));
        ggml_tensor * r = ggml_view_1d(ctx0, inp_rand, 1, (size_t) (level - 1) * inp_rand->nb[0]);
        codes.push_back(sampler.do_sampling(z, r));
    }

    // output 1: the 32 codes of this frame, gathered as floats from an iota then cast back
    ggml_tensor * iota = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) vocab, 1.0f), 1, vocab);
    ggml_tensor * out_codes = nullptr;
    for (ggml_tensor * c : codes) {
        ggml_tensor * f = ggml_get_rows(ctx0, iota, c); // [1, 1]
        out_codes = out_codes ? ggml_concat(ctx0, out_codes, f, 1) : f;
    }
    out_codes = ggml_cast(ctx0, ggml_reshape_1d(ctx0, out_codes, n_level + 1), GGML_TYPE_I32);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    // output 2: the depth logits [vocab, 31], for checks against the reference
    ggml_set_name(all_logits, "out_feats");
    ggml_set_output(all_logits);
    ggml_build_forward_expand(gf, all_logits);

    // output 3: the backbone input of the next frame, RMSNorm(sum_l table[l * vocab + code_l] / sqrt(32))
    // eps is the nn.RMSNorm default, the float32 machine epsilon
    ggml_tensor * table = model.gen_audio_embd_w;
    ggml_tensor * embd  = nullptr;
    for (size_t l = 0; l < codes.size(); l++) {
        ggml_tensor * t = ggml_view_2d(ctx0, table, table->ne[0], vocab, table->nb[1], l * vocab * table->nb[1]);
        ggml_tensor * e = ggml_get_rows(ctx0, t, codes[l]);
        embd = embd ? ggml_add(ctx0, embd, e) : e;
    }
    embd = ggml_scale(ctx0, embd, 1.0f / sqrtf((float) codes.size()));
    embd = ggml_rms_norm(ctx0, embd, 1.1920929e-07f);
    embd = ggml_mul(ctx0, embd, model.gen_audio_embd_norm_w);
    cb(embd, "gen_embd", -1);

    // out embd goes last, clip_encode() reads it back via ggml_graph_node(gf, -1)
    ggml_build_forward_expand(gf, embd);

    return gf;
}

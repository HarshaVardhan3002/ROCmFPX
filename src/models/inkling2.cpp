#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-impl.h"

#include <cfloat>

//
// Inkling 2 (thinkingmachines/Inkling-2-Small), text model.
// What it does that other archs do not:
//   1. no RoPE: positions come from a per-token relative bias over backward distance (0 <= d < rel_extent)
//      and from short convolutions on k, v, the attention output and the MoE output
//   2. per-token attention sinks (a projection of the hidden state), sigmoid output gate
//   3. L2 clipping of q, k and the relative states; unweighted RMS norm on v
//   4. full-attention layers: dynamic k/v convolutions (static taps plus per-token taps) and log-length scaling
//   5. block attention residuals: each layer reads a softmax mix of the block states through learned queries
// Every layer has both an attention cache and a short-conv state (hybrid iswa memory, filters all true).
// The conv state row holds four convolutions: k | v | attention output | MoE output.
//

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping);

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    load_swa_pattern(ml, 6);

    ml.get_key(LLM_KV_INKLING_REL_EXTENT,        hparams.inkling_rel_extent);
    ml.get_key(LLM_KV_INKLING_REL_DIM,           hparams.inkling_rel_dim);
    ml.get_key(LLM_KV_INKLING_QK_CLIP,           hparams.inkling_qk_clip);
    ml.get_key(LLM_KV_INKLING_REL_CLIP,          hparams.inkling_rel_clip);
    ml.get_key(LLM_KV_INKLING_LOG_SCALING_ALPHA, hparams.inkling_log_alpha);
    ml.get_key(LLM_KV_INKLING_LOG_SCALING_FLOOR, hparams.inkling_log_floor);
    ml.get_key(LLM_KV_INKLING_ATTN_RES_BLOCK,    hparams.inkling_attn_res_block);
    ml.get_key(LLM_KV_INKLING_CONV_KERNEL,       hparams.inkling_conv_kernel);

    GGML_ASSERT(hparams.inkling_conv_kernel > 1);
    GGML_ASSERT(hparams.inkling_attn_res_block > 0);
    GGML_ASSERT(hparams.inkling_rel_extent > 0);

    // the conv state row must fit the widest layer
    uint32_t kv_max = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        kv_max = std::max(kv_max, std::max(hparams.n_embd_k_gqa(il), hparams.n_embd_v_gqa(il)));
    }
    hparams.inkling_conv_channels = 2*kv_max + 2*hparams.n_embd;

    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), true);

    type = LLM_TYPE_UNKNOWN; // Inkling-2-Small: 48 layers, 594B total / 17B active
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp();
    const int64_t d_rel    = hparams.inkling_rel_dim;
    const int64_t n_k      = hparams.inkling_conv_kernel;

    tok_embd      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, 0);
    tok_norm      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), {n_embd}, 0);
    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), {n_embd, n_vocab}, 0);
    output_res_q  = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_Q,    "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool    is_swa  = hparams.is_swa(i);
        const int64_t n_rel   = is_swa ? hparams.n_swa : hparams.inkling_rel_extent;
        const int64_t kv_dim  = hparams.n_embd_k_gqa(i);
        const int64_t hd      = hparams.n_embd_head_k(i);
        const int64_t nh      = hparams.n_head(i);
        const int64_t nh_kv   = hparams.n_head_kv(i);

        if (is_swa) {
            layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        } else {
            layer.attn_q_in_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_IN_NORM, "weight", i), {n_embd}, 0);
            layer.attn_k_in_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_IN_NORM, "weight", i), {n_embd}, 0);
            layer.attn_v_in_norm = create_tensor(tn(LLM_TENSOR_ATTN_V_IN_NORM, "weight", i), {n_embd}, 0);
        }

        if (i > 0) {
            if (is_swa) {
                layer.attn_res_q = create_tensor(tn(LLM_TENSOR_ATTN_RES_Q, "weight", i), {n_embd}, 0);
            } else {
                layer.attn_res_q = create_tensor(tn(LLM_TENSOR_ATTN_RES_Q, "weight", i), {n_embd, 3}, 0);
            }
        }
        layer.ffn_res_q = create_tensor(tn(LLM_TENSOR_FFN_RES_Q, "weight", i), {n_embd}, 0);

        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, nh*hd}, 0);
        layer.wk             = create_tensor(tn(LLM_TENSOR_ATTN_K,         "weight", i), {n_embd, kv_dim}, 0);
        layer.wv             = create_tensor(tn(LLM_TENSOR_ATTN_V,         "weight", i), {n_embd, kv_dim}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {nh*hd, n_embd}, 0);
        layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), {n_embd, nh*hd}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, nh}, 0);
        layer.attn_rel       = create_tensor(tn(LLM_TENSOR_ATTN_REL,       "weight", i), {n_embd, nh*d_rel}, 0);
        layer.attn_rel_proj  = create_tensor(tn(LLM_TENSOR_ATTN_REL_PROJ,  "weight", i), {d_rel, n_rel}, 0);
        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {hd, nh}, 0);
        layer.attn_rel_clip  = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP,  "weight", i), {d_rel, nh}, 0);

        // static taps in ggml_ssm_conv order: index n_k-1 weights the current token
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_k, kv_dim}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_k, kv_dim}, 0);
        if (!is_swa) {
            // per-token taps, row h*n_k + lag (lag 0 = current token)
            layer.attn_k_conv_dyn = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_DYN, "weight", i), {n_embd, nh_kv*n_k}, 0);
            layer.attn_v_conv_dyn = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_DYN, "weight", i), {n_embd, nh_kv*n_k}, 0);
        }
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_k, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_k, n_embd}, 0);

        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        // NVFP4 experts carry a per-expert scale (ModelOpt amax / 2688)
        layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps_s   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// relative-distance indices for both caches, and the per-token log scaling of the full-attention layers
class llm_graph_input_inkling2 : public llm_graph_input_i {
public:
    llm_graph_input_inkling2(const llama_kv_cache_iswa_context * mctx, int32_t rel_base, int32_t rel_swa,
                             float alpha, float floor) :
        mctx(mctx), rel_base(rel_base), rel_swa(rel_swa), alpha(alpha), floor(floor) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (dist_base) {
            mctx->get_base()->set_input_rel_dist(dist_base, ubatch, rel_base);
        }
        if (dist_swa) {
            mctx->get_swa()->set_input_rel_dist(dist_swa, ubatch, rel_swa);
        }
        if (tau) {
            GGML_ASSERT(ggml_backend_buffer_is_host(tau->buffer));
            float * data = (float *) tau->data;
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                const double r = ((double) ubatch->pos[i] + 1.0) / (double) floor;
                data[i] = (float) (1.0 + alpha * std::log(std::max(r, 1.0)));
            }
        }
    }

    ggml_tensor * dist_base = nullptr; // I32 [n_kv_base, n_tokens]
    ggml_tensor * dist_swa  = nullptr; // I32 [n_kv_swa,  n_tokens]
    ggml_tensor * tau       = nullptr; // F32 [1, 1, n_tokens]

    const llama_kv_cache_iswa_context * mctx;
    const int32_t rel_base;
    const int32_t rel_swa;
    const float   alpha;
    const float   floor;
};

// stacked = [blocks..., cur], RMS-normalized without weight; out_q = softmax_j(q . s_j / sqrt(n_embd)) s_j
// returns [n_embd, n_q, n_tokens]
ggml_tensor * llama_model_inkling2::graph::block_mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur,
                                                     ggml_tensor * q, int il, const char * name) {
    const int64_t n_src = (int64_t) blocks.size() + 1;
    const int64_t n_tok = cur->ne[1];

    ggml_tensor * stack = nullptr;
    for (ggml_tensor * b : blocks) {
        ggml_tensor * b3 = ggml_reshape_3d(ctx0, b, n_embd, 1, n_tok);
        stack = stack ? ggml_concat(ctx0, stack, b3, 1) : b3;
    }
    ggml_tensor * c3 = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tok);
    stack = stack ? ggml_concat(ctx0, stack, c3, 1) : c3;                   // [n_embd, n_src, n_tok]

    stack = ggml_rms_norm(ctx0, stack, 1e-6f);

    ggml_tensor * scores = ggml_mul_mat(ctx0, q, stack);                    // [n_q, n_src, n_tok]
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));        // [n_src, n_q, n_tok]
    ggml_tensor * probs = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf((float) n_embd), 0.0f);
    cb(probs, name, il);

    ggml_tensor * stack_t = ggml_cont(ctx0, ggml_transpose(ctx0, stack));   // [n_src, n_embd, n_tok]
    GGML_ASSERT(probs->ne[0] == n_src);
    return ggml_mul_mat(ctx0, stack_t, probs);                              // [n_embd, n_q, n_tok]
}

// x * min(1, threshold / (|x| + 1e-6)) per head vector, after an optional learned scale
ggml_tensor * llama_model_inkling2::graph::l2_clip(ggml_tensor * x, ggml_tensor * scale, float threshold) {
    if (scale) {
        x = ggml_mul(ctx0, x, scale);
    }
    ggml_tensor * n = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));
    // max(1, (|x| + eps)/threshold) == 1/min(1, threshold/(|x| + eps))
    n = ggml_scale_bias(ctx0, n, 1.0f/threshold, 1e-6f/threshold);
    n = ggml_clamp(ctx0, n, 1.0f, FLT_MAX);
    return ggml_div(ctx0, x, n);
}

// causal depthwise conv over [state | x], with the state row shared by the layer's four convs
//   out = (conv(x) + x) / sqrt(K + 1)                         static
//   out = (sum_lag x[t-lag] (w[lag] + dyn[t, lag]) + x) / sqrt(2K + 1)   dynamic (dyn_in != nullptr)
ggml_tensor * llama_model_inkling2::graph::short_conv(ggml_tensor * x, ggml_tensor * conv_rs, ggml_tensor * conv_state,
                                                      int64_t ch_off, ggml_tensor * kernel,
                                                      ggml_tensor * dyn_w, int64_t n_dyn_heads, int il) {
    const auto *  mctx_recr    = static_cast<const llama_memory_hybrid_iswa_context *>(mctx)->get_recr();
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_k          = hparams.inkling_conv_kernel;
    const int64_t d_conv       = n_k - 1;
    const int64_t n_ch_all     = hparams.inkling_conv_channels;
    const int64_t C            = x->ne[0];

    GGML_ASSERT(ch_off + C <= n_ch_all);

    ggml_tensor * xs = ggml_reshape_3d(ctx0, x, C, n_seq_tokens, n_seqs);

    // this conv's channels of the state: [d_conv, C, n_seqs]
    ggml_tensor * st = ggml_view_3d(ctx0, conv_rs, d_conv, C, n_seqs, conv_rs->nb[1], conv_rs->nb[2],
                                    ch_off*conv_rs->nb[1]);
    ggml_tensor * bx = ggml_concat(ctx0, st, ggml_transpose(ctx0, xs), 0);  // [d_conv + T, C, n_seqs]

    // write the new state: slot 0 = final, slot s = s tokens back (partial rollback)
    {
        const int64_t K         = cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
        const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
        const auto    mem_size  = mctx_recr->get_size();
        const auto    kv_head   = mctx_recr->get_head();
        const size_t  row_size  = ggml_row_size(conv_state->type, d_conv*n_ch_all);
        const size_t  ch_bytes  = ggml_row_size(conv_state->type, d_conv*ch_off);

        for (int64_t slot = 0; slot < n_written; ++slot) {
            ggml_tensor * snap = ggml_view_3d(ctx0, bx, d_conv, C, n_seqs, bx->nb[1], bx->nb[2],
                                              (bx->ne[0] - d_conv - slot)*ggml_element_size(bx));
            ggml_tensor * dst  = ggml_view_2d(ctx0, conv_state, d_conv*C, n_seqs, conv_state->nb[1],
                                              ((size_t) slot*mem_size + kv_head)*row_size + ch_bytes);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, snap, dst));
        }
    }

    ggml_tensor * out = ggml_ssm_conv(ctx0, bx, kernel);                     // [C, T, n_seqs]

    float norm = 1.0f/sqrtf((float) (n_k + 1));

    if (dyn_w) {
        const int64_t hd = C / n_dyn_heads;

        ggml_tensor * dw = ggml_reshape_4d(ctx0, dyn_w, n_k, n_dyn_heads, n_seq_tokens, n_seqs);
        dw = ggml_cont(ctx0, ggml_permute(ctx0, dw, 1, 2, 0, 3));          // [T, n_k, heads, n_seqs]

        ggml_tensor * acc = nullptr;
        for (int64_t lag = 0; lag < n_k; ++lag) {
            ggml_tensor * xl = ggml_view_4d(ctx0, bx, n_seq_tokens, hd, n_dyn_heads, n_seqs,
                                            bx->nb[1], bx->nb[1]*hd, bx->nb[2],
                                            (d_conv - lag)*ggml_element_size(bx));
            ggml_tensor * wl = ggml_cont(ctx0, ggml_view_4d(ctx0, dw, n_seq_tokens, 1, n_dyn_heads, n_seqs,
                                                            dw->nb[1], dw->nb[2], dw->nb[3], lag*dw->nb[1]));
            ggml_tensor * t = ggml_mul(ctx0, xl, wl);                       // [T, hd, heads, n_seqs]
            acc = acc ? ggml_add(ctx0, acc, t) : t;
        }
        acc = ggml_reshape_3d(ctx0, acc, n_seq_tokens, C, n_seqs);
        acc = ggml_cont(ctx0, ggml_transpose(ctx0, acc));                   // [C, T, n_seqs]
        out = ggml_add(ctx0, out, acc);

        norm = 1.0f/sqrtf((float) (2*n_k + 1));
    }

    out = ggml_scale(ctx0, ggml_add(ctx0, out, xs), norm);
    cb(out, "short_conv", il);

    return ggml_reshape_2d(ctx0, out, C, n_seq_tokens*n_seqs);
}

llama_model_inkling2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params), model(model) {
    GGML_ASSERT(ubatch.equal_seqs());

    const int64_t d_rel = hparams.inkling_rel_dim;
    const uint32_t blk  = hparams.inkling_attn_res_block;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL = build_norm(inpL, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    cb(inpL, "inp_embd_norm", -1);

    auto * inp_mem  = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp_mem->get_attn();
    auto * inp_rs   = inp_mem->get_recr();

    const auto * mctx_hyb  = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
    const auto * mctx_attn = mctx_hyb->get_attn();
    const auto * mctx_recr = mctx_hyb->get_recr();

    // relative distances and log scaling
    llm_graph_input_inkling2 * inp_ink;
    {
        auto inp = std::make_unique<llm_graph_input_inkling2>(mctx_attn,
                (int32_t) hparams.inkling_rel_extent, (int32_t) hparams.n_swa,
                hparams.inkling_log_alpha, hparams.inkling_log_floor);

        inp->dist_base = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_attn->get_base()->get_n_kv(), n_tokens);
        ggml_set_input(inp->dist_base);
        inp->dist_swa  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_attn->get_swa()->get_n_kv(), n_tokens);
        ggml_set_input(inp->dist_swa);
        inp->tau       = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(inp->tau);

        inp_ink = (llm_graph_input_inkling2 *) res->add_input(std::move(inp));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    std::vector<ggml_tensor *> blocks;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const bool    is_swa   = hparams.is_swa(il);
        const bool    boundary = (uint32_t) il % blk == 0;
        const int64_t hd       = hparams.n_embd_head_k(il);
        const int64_t nh       = hparams.n_head(il);
        const int64_t nh_kv    = hparams.n_head_kv(il);
        const int64_t kv_dim   = hparams.n_embd_k_gqa(il);

        ggml_tensor * partial = inpL;

        // conv state of this layer: [d_conv, channels, n_seqs]
        ggml_tensor * conv_state = mctx_recr->get_r_l(il);
        ggml_tensor * conv_rs    = build_rs(inp_rs, conv_state, hparams.n_embd_r(), ubatch.n_seqs);
        conv_rs = ggml_reshape_3d(ctx0, conv_rs, hparams.inkling_conv_kernel - 1, hparams.inkling_conv_channels, ubatch.n_seqs);

        // attention inputs
        ggml_tensor * xq;
        ggml_tensor * xk;
        ggml_tensor * xv;
        if (il == 0) {
            xq = xk = xv = partial;
        } else {
            ggml_tensor * mix = block_mix(blocks, partial, layer.attn_res_q, il, "attn_res_probs");
            const int64_t n_q = mix->ne[1];
            auto pick = [&](int64_t iq) {
                return ggml_cont(ctx0, ggml_view_2d(ctx0, mix, n_embd, n_tokens, mix->nb[2], iq*mix->nb[1]));
            };
            xq = pick(0);
            xk = n_q > 1 ? pick(1) : xq;
            xv = n_q > 1 ? pick(2) : xq;
        }

        if (is_swa) {
            xq = build_norm(xq, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            xk = xv = xq;
        } else {
            xq = build_norm(xq, layer.attn_q_in_norm, nullptr, LLM_NORM_RMS, il);
            xk = build_norm(xk, layer.attn_k_in_norm, nullptr, LLM_NORM_RMS, il);
            xv = build_norm(xv, layer.attn_v_in_norm, nullptr, LLM_NORM_RMS, il);
        }
        cb(xq, "attn_norm", il);

        ggml_tensor * Qcur  = build_lora_mm(layer.wq, xq);
        ggml_tensor * Kcur  = build_lora_mm(layer.wk, xk);
        ggml_tensor * Vcur  = build_lora_mm(layer.wv, xv);
        ggml_tensor * Rcur  = build_lora_mm(layer.attn_rel, xq);
        ggml_tensor * gate  = build_lora_mm(layer.wqkv_gate, xq);
        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, xq);      // [nh, n_tokens]

        ggml_tensor * k_dyn = is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_dyn, xk);
        ggml_tensor * v_dyn = is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_dyn, xv);

        Kcur = short_conv(Kcur, conv_rs, conv_state, 0,      layer.attn_k_conv, k_dyn, nh_kv, il);
        Vcur = short_conv(Vcur, conv_rs, conv_state, kv_dim, layer.attn_v_conv, v_dyn, nh_kv, il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, hd, nh,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, hd, nh_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, hd, nh_kv, n_tokens);
        Rcur = ggml_reshape_3d(ctx0, Rcur, d_rel, nh, n_tokens);

        Qcur = l2_clip(Qcur, layer.attn_q_clip,   hparams.inkling_qk_clip);
        Kcur = l2_clip(Kcur, nullptr,             hparams.inkling_qk_clip);
        Rcur = l2_clip(Rcur, layer.attn_rel_clip, hparams.inkling_rel_clip);
        Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, inp_ink->tau);
            Rcur = ggml_mul(ctx0, Rcur, inp_ink->tau);
        }
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // relative bias: bias[kv, t, h] = (r[t, h] . proj[:, d]) with d = dist(t, kv), zero outside [0, n_rel)
        ggml_tensor * rel = ggml_mul_mat(ctx0, layer.attn_rel_proj, Rcur);   // [n_rel, nh, n_tokens]
        rel = ggml_pad(ctx0, rel, 1, 0, 0, 0);                               // [n_rel + 1, nh, n_tokens]
        rel = ggml_cont(ctx0, ggml_transpose(ctx0, rel));                    // [nh, n_rel + 1, n_tokens]
        ggml_tensor * bias = ggml_get_rows(ctx0, rel, is_swa ? inp_ink->dist_swa : inp_ink->dist_base); // [nh, n_kv, n_tokens]
        bias = ggml_cont(ctx0, ggml_permute(ctx0, bias, 2, 0, 1, 3));        // [n_kv, n_tokens, nh]
        cb(bias, "rel_bias", il);

        // attention with per-token sinks (dense, F32)
        ggml_tensor * cur;
        {
            auto * k_rot = is_swa ? inp_attn->self_k_rot_swa : inp_attn->self_k_rot;
            auto * v_rot = is_swa ? inp_attn->self_v_rot_swa : inp_attn->self_v_rot;
            if (k_rot) {
                Qcur = llama_mul_mat_hadamard(ctx0, Qcur, k_rot);
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }

            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            const auto * mctx_cur = is_swa ? mctx_attn->get_swa() : mctx_attn->get_base();

            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs(), il));

            ggml_tensor * kq_mask = is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask();
            GGML_ASSERT(kq_mask->ne[3] == 1 && "inkling2: multiple KV streams are not supported yet");
            if (kq_mask->type != GGML_TYPE_F32) {
                kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
            }

            ggml_tensor * k = mctx_cur->get_k(ctx0, il);
            ggml_tensor * v = mctx_cur->get_v(ctx0, il);
            const bool v_trans = v->nb[1] > v->nb[2];

            ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);          // [hd, n_tokens, nh]
            k = ggml_permute(ctx0, k, 0, 2, 1, 3);                           // [hd, n_kv, nh_kv]
            v = ggml_permute(ctx0, v, 0, 2, 1, 3);

            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);                     // [n_kv, n_tokens, nh]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_scale(ctx0, kq, 1.0f/sqrtf((float) hd));
            kq = ggml_add(ctx0, kq, bias);
            kq = ggml_add(ctx0, kq, kq_mask);

            // one extra logit per (token, head) with a zero value
            ggml_tensor * s = ggml_cont(ctx0, ggml_transpose(ctx0, sinks)); // [n_tokens, nh]
            s = ggml_reshape_3d(ctx0, s, 1, n_tokens, nh);
            kq = ggml_concat(ctx0, kq, s, 0);                                // [n_kv + 1, n_tokens, nh]
            kq = ggml_soft_max(ctx0, kq);
            kq = ggml_cont(ctx0, ggml_view_3d(ctx0, kq, kq->ne[0] - 1, n_tokens, nh, kq->nb[1], kq->nb[2], 0));
            cb(kq, "kq_soft_max", il);

            if (!v_trans) {
                v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            }
            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);                   // [hd, n_tokens, nh]
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, hd*nh, n_tokens);

            if (v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
            }
        }
        cb(cur, "kqv_out", il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cur = build_lora_mm(layer.wo, cur);
        cb(cur, "attn_out", il);

        cur = short_conv(cur, conv_rs, conv_state, 2*kv_dim, layer.attn_post_conv, nullptr, 0, il);

        // block residual
        ggml_tensor * residual;
        if (boundary) {
            blocks.push_back(partial);
            residual = cur;
        } else {
            residual = ggml_add(ctx0, partial, cur);
        }
        cb(residual, "attn_residual", il);

        // MoE
        cur = block_mix(blocks, residual, layer.ffn_res_q, il, "ffn_res_probs");
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SITU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr, nullptr,
                layer.ffn_up_exps_s, layer.ffn_gate_exps_s, layer.ffn_down_exps_s);
        cb(cur, "ffn_moe_out", il);

        cur = short_conv(cur, conv_rs, conv_state, 2*kv_dim + n_embd, layer.ffn_post_conv, nullptr, 0, il);

        cur = ggml_add(ctx0, residual, cur);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = block_mix(blocks, inpL, model.output_res_q, -1, "output_res_probs");
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    const float cap = hparams.f_final_logit_softcapping;
    cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

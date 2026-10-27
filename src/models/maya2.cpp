#include "models.h"

#include <cmath>

void llama_model_maya2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // the cb0 head has one row per codebook 0 code plus AEOS, much less than the text vocab
    const auto * w_out = ml.get_weight(tn(LLM_TENSOR_OUTPUT, "weight").str().c_str());
    if (w_out == nullptr) {
        throw std::runtime_error("the cb0 head (output.weight) is missing");
    }
    const int64_t n_cb0 = w_out->tensor->ne[1];
    if (n_cb0 <= 0 || n_cb0 > n_vocab) {
        throw std::runtime_error("the cb0 head must have between 1 and n_vocab rows");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_cb0}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_maya2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_maya2::graph::graph(const llama_model & model, const llm_graph_params & params)
        : llama_model_qwen3::graph(params, no_build{}) {
    ggml_tensor * cur = build_backbone(model);

    // cb0 head, logit i is codebook 0 code i (the last row is AEOS)
    cur = build_lora_mm(model.output, cur, model.output_s);

    // the logits must have n_vocab entries, pad the text part with -inf
    const int64_t n_vocab = model.tok_embd->ne[1];
    const int64_t pad     = n_vocab - model.output->ne[1];
    if (pad > 0) {
        // a scalar -inf from a weight, as in qwen3vl.cpp
        ggml_tensor * neg_inf = ggml_scale_bias(ctx0,
                ggml_view_1d(ctx0, model.output_norm, 1, 0),
                0.0f, -INFINITY);
        neg_inf = ggml_repeat_4d(ctx0, neg_inf, pad, cur->ne[1], 1, 1);
        cur = ggml_concat(ctx0, cur, neg_inf, 0);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

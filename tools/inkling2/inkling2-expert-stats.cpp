// Per-expert routing statistics for MoE pruning (REAP-style saliency).
//   inkling2-expert-stats model.gguf text.txt out.tsv [n_ctx=512] [n_par=8] [max_chunks=0]
// The text is cut into chunks of n_ctx tokens; n_par chunks are decoded together as separate sequences
// (unified KV cache). For every layer and expert it accumulates over the tokens routed to it:
//   count, sum of router weights w, sum of w * ||expert output||, sum of ||expert output||
// REAP saliency = sum_w_norm / count. out.tsv is rewritten after every decode call.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct stats {
    int n_layer = 0, n_expert = 0;
    std::vector<double> cnt, sw, swn, sn; // [n_layer * n_expert]
    std::vector<std::vector<int32_t>> ids; // per layer: last selected experts [n_used * T]
    std::vector<std::vector<float>> w;     // per layer: last router weights   [n_used * T]
    long long n_tok = 0;
};
static stats g;

static int layer_of(const char * n) {
    const char * d = strrchr(n, '-');
    return d ? atoi(d + 1) : -1;
}

static bool starts(const char * n, const char * p) { return strncmp(n, p, strlen(p)) == 0; }

// read a (possibly strided) 4-byte tensor into a dense array, ne0 fastest
template <typename T>
static void read_dense(const ggml_tensor * t, std::vector<T> & out) {
    static_assert(sizeof(T) == 4, "4-byte elements");
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    out.resize(ggml_nelements(t));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    memcpy(&out[k++], &raw[i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]], 4);
}

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    const bool topk = starts(n, "ffn_moe_topk-");
    const bool wts  = starts(n, "ffn_moe_weights") && !strstr(n, "_sum") && t->type == GGML_TYPE_F32;
    const bool down = starts(n, "ffn_moe_down-") || starts(n, "ffn_moe_down_scaled-");
    if (ask) {
        return topk || wts || down;
    }
    const int il = layer_of(n);
    if (il < 0 || il >= g.n_layer) {
        return true;
    }
    const size_t ne = ggml_nelements(t);
    if (topk && t->type == GGML_TYPE_I32) {
        read_dense(t, g.ids[il]);
    } else if (wts && ne == g.ids[il].size()) {
        read_dense(t, g.w[il]);
    } else if (down && t->type == GGML_TYPE_F32) {
        const int64_t n_embd = t->ne[0];
        const int64_t n_slot = t->ne[1] * t->ne[2];
        if ((size_t) n_slot != g.ids[il].size() || g.w[il].size() != g.ids[il].size()) {
            fprintf(stderr, "layer %d: shape mismatch (%lld slots, %zu ids, %zu w)\n", il, (long long) n_slot,
                    g.ids[il].size(), g.w[il].size());
            return true;
        }
        std::vector<float> buf;
        read_dense(t, buf);
        double * c = &g.cnt[(size_t) il * g.n_expert];
        double * a = &g.sw[(size_t) il * g.n_expert];
        double * b = &g.swn[(size_t) il * g.n_expert];
        double * s = &g.sn[(size_t) il * g.n_expert];
        for (int64_t k = 0; k < n_slot; ++k) {
            const int e = g.ids[il][k];
            if (e < 0 || e >= g.n_expert) continue;
            const float * x = &buf[k * n_embd];
            double ss = 0;
            for (int64_t i = 0; i < n_embd; ++i) ss += (double) x[i] * x[i];
            const double nr = sqrt(ss), wk = g.w[il][k];
            c[e] += 1; a[e] += wk; b[e] += wk * nr; s[e] += nr;
        }
        g.ids[il].clear();
        g.w[il].clear();
    }
    return true;
}

static void save(const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) { perror("out"); return; }
    fprintf(f, "# tokens %lld\nlayer\texpert\tcount\tsum_w\tsum_w_norm\tsum_norm\n", g.n_tok);
    for (int l = 0; l < g.n_layer; ++l)
        for (int e = 0; e < g.n_expert; ++e) {
            const size_t i = (size_t) l * g.n_expert + e;
            fprintf(f, "%d\t%d\t%.0f\t%.9g\t%.9g\t%.9g\n", l, e, g.cnt[i], g.sw[i], g.swn[i], g.sn[i]);
        }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf text.txt out.tsv [n_ctx=512] [n_par=8] [max_chunks=0]\n", argv[0]);
        return 1;
    }
    const int n_ctx = argc > 4 ? atoi(argv[4]) : 512;
    const int n_par = argc > 5 ? atoi(argv[5]) : 8;
    const int max_chunks = argc > 6 ? atoi(argv[6]) : 0;

    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss; ss << in.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, true);
    if (nt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(nt);
    int n_chunks = nt / n_ctx;
    if (max_chunks > 0 && n_chunks > max_chunks) n_chunks = max_chunks;
    fprintf(stderr, "%d tokens -> %d chunks of %d, %d per call\n", nt, n_chunks, n_ctx, n_par);

    g.n_layer = llama_model_n_layer(model);
    char buf[64];
    if (llama_model_meta_val_str(model, "inkling2.expert_count", buf, sizeof(buf)) < 0) {
        fprintf(stderr, "no inkling2.expert_count\n"); return 1;
    }
    g.n_expert = atoi(buf);
    const size_t N = (size_t) g.n_layer * g.n_expert;
    g.cnt.assign(N, 0); g.sw.assign(N, 0); g.swn.assign(N, 0); g.sn.assign(N, 0);
    g.ids.resize(g.n_layer); g.w.resize(g.n_layer);

    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx * n_par;
    cp.n_batch = n_ctx * n_par;
    cp.n_ubatch = n_ctx * n_par;
    cp.n_seq_max = n_par;
    cp.kv_unified = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    llama_batch b = llama_batch_init(n_ctx * n_par, 0, 1);
    for (int c0 = 0; c0 < n_chunks; c0 += n_par) {
        const int np = std::min(n_par, n_chunks - c0);
        llama_memory_clear(llama_get_memory(ctx), true);
        b.n_tokens = 0;
        for (int s = 0; s < np; ++s) {
            for (int i = 0; i < n_ctx; ++i) {
                const int k = b.n_tokens++;
                b.token[k] = toks[(size_t) (c0 + s) * n_ctx + i];
                b.pos[k] = i;
                b.n_seq_id[k] = 1;
                b.seq_id[k][0] = s;
                b.logits[k] = i == n_ctx - 1;
            }
        }
        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at chunk %d\n", c0); return 1; }
        g.n_tok += (long long) np * n_ctx;
        save(argv[3]);
        fprintf(stderr, "chunks %d/%d  %.1f s\n", c0 + np, n_chunks, (ggml_time_us() - t0) / 1e6);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

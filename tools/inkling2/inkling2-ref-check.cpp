// Run token ids through a model and write the logits of every token, for checks against a PyTorch reference.
//   inkling2-ref-check model.gguf ids.bin out.bin [chunk] [dump_dir]
// ids.bin: int32 token ids. out.bin: float32 [n_tokens, n_vocab].
// chunk: tokens per llama_decode call (default: all at once); 1 = token-by-token decode.
// dump_dir: if set, writes the F32 tensors named l_out-N / inp_embd_norm of the first call there.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_dump_dir;

static bool dump_cb(struct ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    static int counter = 0;
    const size_t ln = strlen(n);
    const bool layer0 = getenv("DUMP_L0") && ln > 2 && strcmp(n + ln - 2, "-0") == 0;
    const bool want = layer0 || strncmp(n, "l_out-", 6) == 0 || strcmp(n, "inp_embd_norm") == 0 || strcmp(n, "result_norm") == 0;
    if (ask) {
        return want;
    }
    if (want && t->type == GGML_TYPE_F32 && !g_dump_dir.empty()) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        char pre[16];
        snprintf(pre, sizeof(pre), "%04d_", counter++);
        std::string p = g_dump_dir + "/" + (layer0 ? std::string(pre) : std::string()) + n + ".bin";
        FILE * f = fopen(p.c_str(), "wb");
        if (f) {
            fwrite(buf.data(), sizeof(float), buf.size(), f);
            fclose(f);
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf ids.bin out.bin [chunk] [dump_dir]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    FILE * fi = fopen(argv[2], "rb");
    if (!fi) { perror("ids"); return 1; }
    std::vector<llama_token> ids;
    int32_t v;
    while (fread(&v, 4, 1, fi) == 1) ids.push_back(v);
    fclose(fi);
    const int n = (int) ids.size();
    int chunk = argc > 4 ? atoi(argv[4]) : n;
    if (chunk <= 0) chunk = n;
    if (argc > 5) g_dump_dir = argv[5];

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = n;
    cp.n_ubatch = n;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (getenv("KV_F16") == nullptr) {
        cp.type_k = GGML_TYPE_F32; // exact checks need an F32 cache
        cp.type_v = GGML_TYPE_F32;
    }
    cp.cb_eval = dump_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    std::vector<float> out((size_t) n * n_vocab);
    for (int p0 = 0; p0 < n; p0 += chunk) {
        const int m = std::min(chunk, n - p0);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; ++i) {
            b.token[i] = ids[p0 + i];
            b.pos[i] = p0 + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = m;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", p0); return 1; }
        for (int i = 0; i < m; ++i) {
            memcpy(&out[(size_t) (p0 + i) * n_vocab], llama_get_logits_ith(ctx, i), sizeof(float) * n_vocab);
        }
        llama_batch_free(b);
        g_dump_dir.clear(); // dump the first call only
    }

    FILE * fo = fopen(argv[3], "wb");
    fwrite(out.data(), sizeof(float), out.size(), fo);
    fclose(fo);
    fprintf(stderr, "wrote %d x %d logits\n", n, n_vocab);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

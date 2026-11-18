// llama-haru: Haru-4B full-duplex speech model (higurashi-ai/Haru-4B).
//
// One language-model position is one 133 ms frame (3200 samples at 24 kHz). Its input is the sum of the model's
// text token embedding, the model's previous audio latent and the user's audio latent (both through a small
// projector). The hidden state predicts the next text token (tied output embedding) and, through a MeanFlow MLP
// head, the next 64-dim audio latent, which the causal VibeVoice codec decodes to audio.
//
// The text backbone is a plain qwen35 GGUF run by libllama (embedding input, hidden state + logits output). The
// projectors, head and codec come from the haru-audio GGUF (tools/haru/convert_haru_audio.py) and run as ggml
// graphs here.
//
//   offline duplex:  llama-haru -m lm.gguf --audio audio.gguf --voice voice.wav --user user.wav -o out.wav
//   verification:    llama-haru -m lm-F32.gguf --audio audio-F32.gguf --check REF_DIR --voice v.wav --user u.wav
//                    (REF_DIR from haru_ref.py: every part is compared against the official PyTorch code)

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#define HARU_ARCH "haru-audio"

static void die(const char * msg, const std::string & arg = "") {
    fprintf(stderr, "llama-haru: %s%s\n", msg, arg.c_str());
    exit(1);
}

// ------------------------------------------------------------------------------------------------ files

struct npy {
    std::vector<int64_t> shape;
    std::vector<float>   f;  // float32 or int32 (converted)
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

static npy npy_load(const std::string & path) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) die("cannot open ", path);
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) die("not a npy file: ", path);
    uint32_t hlen = 0;
    if (magic[6] == 1) { uint16_t h; if (fread(&h, 2, 1, fp) != 1) die("bad npy ", path); hlen = h; }
    else if (fread(&hlen, 4, 1, fp) != 1) die("bad npy ", path);
    std::string hdr(hlen, ' ');
    if (fread(hdr.data(), 1, hlen, fp) != hlen) die("bad npy ", path);
    const bool is_f4 = hdr.find("'<f4'") != std::string::npos, is_i4 = hdr.find("'<i4'") != std::string::npos;
    if (!is_f4 && !is_i4) die("npy dtype must be <f4 or <i4: ", path);
    const bool fortran = hdr.find("'fortran_order': True") != std::string::npos;
    npy r;
    const size_t a = hdr.find('('), b = hdr.find(')');
    std::string dims = hdr.substr(a + 1, b - a - 1);
    for (size_t p = 0; p < dims.size();) {
        while (p < dims.size() && (dims[p] == ' ' || dims[p] == ',')) p++;
        if (p >= dims.size()) break;
        r.shape.push_back(strtoll(dims.c_str() + p, nullptr, 10));
        while (p < dims.size() && dims[p] != ',') p++;
    }
    r.f.resize(r.numel());
    if (is_f4) {
        if (fread(r.f.data(), 4, r.f.size(), fp) != r.f.size()) die("short npy ", path);
    } else {
        std::vector<int32_t> v(r.f.size());
        if (fread(v.data(), 4, v.size(), fp) != v.size()) die("short npy ", path);
        for (size_t i = 0; i < v.size(); i++) r.f[i] = (float) v[i];
    }
    fclose(fp);
    if (fortran && r.shape.size() > 1) {  // column-major on disk (a transposed torch view): reorder to row-major
        const size_t nd = r.shape.size();
        std::vector<float> c(r.f.size());
        std::vector<int64_t> idx(nd, 0);
        for (size_t fi = 0; fi < r.f.size(); fi++) {  // fi walks column-major: first index fastest
            size_t ci = 0;
            for (size_t k = 0; k < nd; k++) ci = ci * r.shape[k] + idx[k];
            c[ci] = r.f[fi];
            for (size_t k = 0; k < nd && ++idx[k] == r.shape[k]; k++) idx[k] = 0;
        }
        r.f.swap(c);
    }
    return r;
}

// mono WAV, PCM16 or float32, must be 24 kHz
static std::vector<float> wav_read(const std::string & path, int want_rate) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) die("cannot open ", path);
    char riff[12];
    if (fread(riff, 1, 12, fp) != 12 || memcmp(riff, "RIFF", 4) || memcmp(riff + 8, "WAVE", 4)) die("not a WAV: ", path);
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t rate = 0;
    std::vector<float> out;
    char id[4]; uint32_t len;
    while (fread(id, 1, 4, fp) == 4 && fread(&len, 4, 1, fp) == 1) {
        if (!memcmp(id, "fmt ", 4)) {
            std::vector<uint8_t> b(len);
            if (fread(b.data(), 1, len, fp) != len) die("bad WAV ", path);
            memcpy(&fmt, &b[0], 2); memcpy(&ch, &b[2], 2); memcpy(&rate, &b[4], 4); memcpy(&bits, &b[14], 2);
            if (fmt == 0xFFFE && len >= 26) memcpy(&fmt, &b[24], 2);
        } else if (!memcmp(id, "data", 4)) {
            std::vector<uint8_t> b(len);
            if (fread(b.data(), 1, len, fp) != len) die("short WAV ", path);
            const int bps = bits / 8, n = (int) (len / (bps * ch));
            out.resize(n);
            for (int i = 0; i < n; i++) {
                float s = 0;
                for (int c = 0; c < ch; c++) {  // downmix
                    const uint8_t * p = &b[(size_t) (i * ch + c) * bps];
                    if (fmt == 3 && bits == 32) { float v; memcpy(&v, p, 4); s += v; }
                    else if (fmt == 1 && bits == 16) { int16_t v; memcpy(&v, p, 2); s += v / 32768.0f; }
                    else die("WAV must be PCM16 or float32: ", path);
                }
                out[i] = s / ch;
            }
        } else {
            fseek(fp, len + (len & 1), SEEK_CUR);
        }
    }
    fclose(fp);
    if ((int) rate != want_rate) die("WAV must be 24000 Hz (resample first): ", path);
    return out;
}

static void wav_write(const std::string & path, const std::vector<float> & a, int rate, int channels = 1) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) die("cannot write ", path);
    const uint32_t data = (uint32_t) a.size() * 2, riff = 36 + data, fmtlen = 16, br = rate * channels * 2;
    const uint16_t pcm = 1, ch = channels, ba = channels * 2, bits = 16;
    fwrite("RIFF", 1, 4, fp); fwrite(&riff, 4, 1, fp); fwrite("WAVEfmt ", 1, 8, fp); fwrite(&fmtlen, 4, 1, fp);
    fwrite(&pcm, 2, 1, fp); fwrite(&ch, 2, 1, fp); fwrite(&rate, 4, 1, fp); fwrite(&br, 4, 1, fp);
    fwrite(&ba, 2, 1, fp); fwrite(&bits, 2, 1, fp); fwrite("data", 1, 4, fp); fwrite(&data, 4, 1, fp);
    for (float v : a) { const int16_t s = (int16_t) lrintf(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f); fwrite(&s, 2, 1, fp); }
    fclose(fp);
}

// ------------------------------------------------------------------------------------------------ audio model

struct haru_audio {
    // hparams
    int sample_rate = 24000, frame = 3200, latent = 64, hidden = 2560, pad_id = 0, eop_id = 0, eos_id = -1;
    int base_ch = 32, head_layers = 4, head_steps = 1;
    std::vector<int> ratios, enc_depths, dec_depths;
    float codec_eps = 1e-5f, latent_std = 0.5f, head_eps = 1e-5f, interval_embed = 7.0f, guidance = 1.5f;
    float latent_scale = 1.0f, latent_bias = 0.0f;
    std::vector<float> stream_embd;  // [2, hidden]: PAD, EOP rows in float32

    gguf_context * gguf = nullptr;
    ggml_context * ctx_w = nullptr;
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_t> bufs;
    ggml_backend_sched_t sched = nullptr;
    std::unordered_map<std::string, ggml_tensor *> w;

    ggml_tensor * t(const std::string & name) const {
        auto it = w.find(name);
        if (it == w.end()) die("missing tensor ", name);
        return it->second;
    }
};

static std::vector<int> gguf_ints(gguf_context * g, const char * key) {
    const int64_t k = gguf_find_key(g, key);
    if (k < 0) die("missing key ", key);
    const int n = (int) gguf_get_arr_n(g, k);
    std::vector<int> r(n);
    const enum gguf_type ty = gguf_get_arr_type(g, k);
    const void * d = gguf_get_arr_data(g, k);
    for (int i = 0; i < n; i++) {
        if (ty == GGUF_TYPE_INT32) r[i] = ((const int32_t *) d)[i];
        else if (ty == GGUF_TYPE_UINT32) r[i] = (int) ((const uint32_t *) d)[i];
        else if (ty == GGUF_TYPE_INT64) r[i] = (int) ((const int64_t *) d)[i];
        else die("bad int array ", key);
    }
    return r;
}

static uint32_t gguf_u32(gguf_context * g, const std::string & key) {
    const int64_t k = gguf_find_key(g, (HARU_ARCH "." + key).c_str());
    if (k < 0) die("missing key ", key);
    return gguf_get_val_u32(g, k);
}

static float gguf_f32(gguf_context * g, const std::string & key) {
    const int64_t k = gguf_find_key(g, (HARU_ARCH "." + key).c_str());
    if (k < 0) die("missing key ", key);
    return gguf_get_val_f32(g, k);
}

static void audio_load(haru_audio & m, const std::string & path, bool cpu_only, int n_threads) {
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &m.ctx_w };
    m.gguf = gguf_init_from_file(path.c_str(), ip);
    if (!m.gguf) die("cannot load ", path);
    gguf_context * g = m.gguf;
    m.sample_rate = gguf_u32(g, "sample_rate"); m.frame = gguf_u32(g, "frame_samples"); m.latent = gguf_u32(g, "latent_dim");
    m.hidden = gguf_u32(g, "hidden_size"); m.pad_id = gguf_u32(g, "pad_token_id"); m.eop_id = gguf_u32(g, "eop_token_id");
    m.eos_id = gguf_u32(g, "eos_token_id");
    m.base_ch = gguf_u32(g, "codec.base_channels");
    m.ratios = gguf_ints(g, HARU_ARCH ".codec.ratios");
    m.enc_depths = gguf_ints(g, HARU_ARCH ".codec.encoder_depths");
    m.dec_depths = gguf_ints(g, HARU_ARCH ".codec.decoder_depths");
    m.codec_eps = gguf_f32(g, "codec.norm_eps"); m.latent_std = gguf_f32(g, "codec.latent_std");
    m.head_layers = gguf_u32(g, "head.num_layers"); m.head_eps = gguf_f32(g, "head.norm_eps");
    m.interval_embed = gguf_f32(g, "head.interval_embed"); m.head_steps = gguf_u32(g, "head.num_steps");
    m.guidance = gguf_f32(g, "head.guidance_scale");

    // backends: the best GPU (if any) then the CPU; the scheduler moves unsupported ops to the CPU
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) die("no CPU backend");
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(cpu);
    if (auto * set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(cpu_dev), "ggml_backend_set_n_threads")) {
        set_threads(cpu, n_threads);
    }
    if (!cpu_only) {
        if (ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)) m.backends.push_back(gpu);
    }
    m.backends.push_back(cpu);

    // weights on the first backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backends[0]);
    if (!buf) die("cannot allocate the audio weights");
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    m.bufs.push_back(buf);
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) die("cannot open ", path);
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * tt = ggml_get_tensor(m.ctx_w, name);
        const size_t n = ggml_nbytes(tt);
        tmp.resize(n);
#ifdef _WIN32
        _fseeki64(fp, (long long) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET);
#else
        fseeko(fp, (off_t) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET);
#endif
        if (fread(tmp.data(), 1, n, fp) != n) die("short read of ", name);
        ggml_backend_tensor_set(tt, tmp.data(), 0, n);
        m.w[name] = tt;
    }
    fclose(fp);
    auto scalar = [&](const char * name) { float v; ggml_backend_tensor_get(m.t(name), &v, 0, 4); return v; };
    m.latent_scale = scalar("latent_scale");
    m.latent_bias = scalar("latent_bias");
    ggml_tensor * se = m.t("stream_embd");
    m.stream_embd.resize(ggml_nelements(se));
    ggml_backend_tensor_get(se, m.stream_embd.data(), 0, ggml_nbytes(se));

    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto * b : m.backends) bufts.push_back(ggml_backend_get_default_buffer_type(b));
    m.sched = ggml_backend_sched_new(m.backends.data(), bufts.data(), (int) m.backends.size(), 65536, false, true);
    fprintf(stderr, "llama-haru: audio %s on %s, latent scale %.6f bias %.6f\n", path.c_str(),
            ggml_backend_name(m.backends[0]), m.latent_scale, m.latent_bias);
}

// one graph evaluation: build -> alloc -> set inputs -> compute -> read outputs
struct graph_call {
    haru_audio & m;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<std::pair<ggml_tensor *, const float *>> inputs;  // data must outlive run()
    std::list<std::vector<float>> owned;

    explicit graph_call(haru_audio & m) : m(m) {
        const size_t n_nodes = 65536;
        ggml_init_params p = { ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        ctx = ggml_init(p);
        gf = ggml_new_graph_custom(ctx, n_nodes, false);
    }
    ~graph_call() { ggml_free(ctx); }

    ggml_tensor * input(const std::vector<float> & data, int64_t ne0, int64_t ne1 = 1) {
        GGML_ASSERT((int64_t) data.size() == ne0 * ne1);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(x);
        inputs.emplace_back(x, data.data());
        return x;
    }
    ggml_tensor * input_copy(std::vector<float> data, int64_t ne0, int64_t ne1 = 1) {
        owned.push_back(std::move(data));
        return input(owned.back(), ne0, ne1);
    }
    void output(ggml_tensor * x) { ggml_set_output(x); ggml_build_forward_expand(gf, x); }

    void run() {
        ggml_backend_sched_reset(m.sched);
        if (!ggml_backend_sched_alloc_graph(m.sched, gf)) die("graph allocation failed");
        for (auto & [x, d] : inputs) ggml_backend_tensor_set(x, d, 0, ggml_nbytes(x));
        if (ggml_backend_sched_graph_compute(m.sched, gf) != GGML_STATUS_SUCCESS) die("graph compute failed");
    }
    static std::vector<float> get(ggml_tensor * x) {
        std::vector<float> r(ggml_nelements(x));
        ggml_backend_tensor_get(x, r.data(), 0, ggml_nbytes(x));
        return r;
    }
};

// ---- codec layers. Activations are [C, T] (channels contiguous); every layer state is [C, context].

struct codec_state { std::vector<std::vector<float>> s; };

struct layer_io {
    graph_call & g;
    const codec_state * in;
    std::vector<ggml_tensor *> new_states;
    size_t idx = 0;
    ggml_tensor * state(int64_t c, int64_t n) {  // the next layer's input state
        const std::vector<float> & v = in->s[idx++];
        return g.input(v, c, n);
    }
};

static ggml_tensor * rms(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    x = ggml_rms_norm(c, x, eps);
    return w ? ggml_mul(c, x, w) : x;
}

static ggml_tensor * linear(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, x, b) : x;
}

// causal dense conv, kernel K, stride s; weight [K * I, O] tap-major
static ggml_tensor * conv_dense(layer_io & io, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int K, int s) {
    ggml_context * c = io.g.ctx;
    const int64_t I = x->ne[0], ctxn = K - s;
    if (x->ne[1] % s) x = ggml_pad(c, x, 0, s - x->ne[1] % s, 0, 0);  // final partial hop of a whole-clip encode
    ggml_tensor * full = ggml_concat(c, io.state(I, ctxn), x, 1);  // [I, ctx + T]
    const int64_t L = full->ne[1], OL = (L - K) / s + 1;
    io.new_states.push_back(ggml_cont(c, ggml_view_2d(c, full, I, ctxn, full->nb[1], (L - ctxn) * full->nb[1])));
    ggml_tensor * cols;
    if (K == 2 * s) {  // windows = two consecutive hops
        ggml_tensor * hops = ggml_reshape_2d(c, ggml_cont(c, full), I * s, L / s);
        cols = ggml_concat(c, ggml_view_2d(c, hops, I * s, OL, hops->nb[1], 0),
                              ggml_view_2d(c, hops, I * s, OL, hops->nb[1], hops->nb[1]), 0);
    } else {
        GGML_ASSERT(s == 1);
        cols = nullptr;
        for (int k = 0; k < K; k++) {
            ggml_tensor * v = ggml_view_2d(c, full, I, OL, full->nb[1], k * full->nb[1]);
            cols = cols ? ggml_concat(c, cols, v, 0) : ggml_cont(c, v);
        }
    }
    return linear(c, cols, w, b);  // [O, OL]
}

// causal depthwise conv; weight [C, K] (ne0 = C: one row of channel weights per tap). Built from K shifted
// views times a broadcast row (plain mul/add, so every backend runs it; ggml_ssm_conv on CUDA needs C % 128 == 0)
static ggml_tensor * conv_dw(layer_io & io, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_context * c = io.g.ctx;
    const int64_t C = x->ne[0], T = x->ne[1], K = w->ne[1];
    GGML_ASSERT(w->ne[0] == C);
    ggml_tensor * full = ggml_cont(c, ggml_concat(c, io.state(C, K - 1), x, 1));  // [C, K - 1 + T]
    io.new_states.push_back(ggml_cont(c, ggml_view_2d(c, full, C, K - 1, full->nb[1], T * full->nb[1])));
    ggml_tensor * y = nullptr;
    for (int64_t k = 0; k < K; k++) {
        ggml_tensor * v = ggml_view_2d(c, full, C, T, full->nb[1], k * full->nb[1]);
        ggml_tensor * wk = ggml_view_1d(c, w, C, k * w->nb[1]);
        ggml_tensor * p = ggml_mul(c, v, wk);
        y = y ? ggml_add(c, y, p) : p;
    }
    return ggml_add(c, y, b);
}

// causal transposed conv, kernel 2s, stride s; weight [I, 2s * O] tap-major; state = overlap tail [O, s]
static ggml_tensor * conv_up(layer_io & io, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int s) {
    ggml_context * c = io.g.ctx;
    const int64_t T = x->ne[1], O = w->ne[1] / (2 * s);
    ggml_tensor * z = ggml_mul_mat(c, w, x);  // [2s * O, T]
    const size_t es = ggml_element_size(z);
    ggml_tensor * A = ggml_reshape_2d(c, ggml_cont(c, ggml_view_3d(c, z, O, s, T, O * es, z->nb[1], 0)), O, s * T);
    ggml_tensor * B = ggml_reshape_2d(c, ggml_cont(c, ggml_view_3d(c, z, O, s, T, O * es, z->nb[1], s * O * es)), O, s * T);
    ggml_tensor * st = io.state(O, s);
    ggml_tensor * Bs = ggml_concat(c, st, B, 1);  // [O, s * (T + 1)]
    io.new_states.push_back(ggml_cont(c, ggml_view_2d(c, B, O, s, B->nb[1], s * (T - 1) * B->nb[1])));
    ggml_tensor * y = ggml_add(c, A, ggml_view_2d(c, Bs, O, s * T, Bs->nb[1], 0));
    return ggml_add(c, y, b);
}

static ggml_tensor * codec_block(layer_io & io, ggml_tensor * x, const std::string & p, float eps) {
    ggml_context * c = io.g.ctx;
    haru_audio & m = io.g.m;
    ggml_tensor * y = conv_dw(io, rms(c, x, m.t(p + "norm.weight"), eps), m.t(p + "conv.weight"), m.t(p + "conv.bias"));
    x = ggml_add(c, x, ggml_mul(c, y, m.t(p + "gamma")));
    ggml_tensor * h = rms(c, x, m.t(p + "ffn_norm.weight"), eps);
    h = linear(c, ggml_gelu_erf(c, linear(c, h, m.t(p + "fc1.weight"), m.t(p + "fc1.bias"))), m.t(p + "fc2.weight"), m.t(p + "fc2.bias"));
    return ggml_add(c, x, ggml_mul(c, h, m.t(p + "ffn_gamma")));
}

// layer list of the encoder / decoder in state order, with their state shapes
struct codec_layout { std::vector<std::pair<int64_t, int64_t>> shapes; };

static codec_layout encoder_layout(const haru_audio & m) {
    codec_layout L;
    const int n = (int) m.ratios.size();
    std::vector<int> strides(m.ratios.rbegin(), m.ratios.rend());
    for (int i = 0; i <= n; i++) {
        const int ch = m.base_ch << i;
        if (i == 0) L.shapes.push_back({1, 7 - 1});
        else L.shapes.push_back({m.base_ch << (i - 1), strides[i - 1]});  // kernel 2s, stride s: context s
        for (int j = 0; j < m.enc_depths[i]; j++) L.shapes.push_back({ch, 6});
    }
    L.shapes.push_back({m.base_ch << n, 6});
    return L;
}

static codec_layout decoder_layout(const haru_audio & m) {
    codec_layout L;
    const int n = (int) m.ratios.size();
    for (int i = 0; i <= n; i++) {
        const int ch = m.base_ch << (n - i);
        if (i == 0) L.shapes.push_back({m.latent, 6});
        else L.shapes.push_back({ch, m.ratios[i - 1]});  // overlap tail of the transposed conv: [O, s]
        for (int j = 0; j < m.dec_depths[i]; j++) L.shapes.push_back({ch, 6});
    }
    L.shapes.push_back({m.base_ch, 6});
    return L;
}

static codec_state zero_state(const codec_layout & L) {
    codec_state s;
    for (auto & [c, n] : L.shapes) s.s.emplace_back((size_t) (c * n), 0.0f);
    return s;
}

// audio [n] -> latent means [T, latent] (whole clip from a zero state, or whole frames from a running state)
static std::vector<float> codec_encode(haru_audio & m, const std::vector<float> & audio, codec_state & st) {
    graph_call g(m);
    layer_io io{g, &st, {}, 0};
    const int n = (int) m.ratios.size();
    std::vector<int> strides(m.ratios.rbegin(), m.ratios.rend());
    ggml_tensor * x = g.input(audio, 1, (int64_t) audio.size());
    const std::string P = "codec.encoder.";
    for (int i = 0; i <= n; i++) {
        if (i == 0) x = conv_dense(io, x, m.t(P + "stem.weight"), m.t(P + "stem.bias"), 7, 1);
        else {
            const std::string d = P + "down." + std::to_string(i - 1) + ".";
            x = conv_dense(io, x, m.t(d + "weight"), m.t(d + "bias"), 2 * strides[i - 1], strides[i - 1]);
        }
        for (int j = 0; j < m.enc_depths[i]; j++) {
            x = codec_block(io, x, P + "stages." + std::to_string(i) + "." + std::to_string(j) + ".", m.codec_eps);
        }
    }
    x = conv_dense(io, x, m.t(P + "head.weight"), m.t(P + "head.bias"), 7, 1);
    g.output(x);
    for (auto * s : io.new_states) g.output(s);
    g.run();
    for (size_t i = 0; i < io.new_states.size(); i++) st.s[i] = graph_call::get(io.new_states[i]);
    return graph_call::get(x);
}

// latents [T, latent] -> audio [T * frame]
static std::vector<float> codec_decode(haru_audio & m, const std::vector<float> & lat, codec_state & st) {
    graph_call g(m);
    layer_io io{g, &st, {}, 0};
    const int n = (int) m.ratios.size();
    ggml_tensor * x = g.input(lat, m.latent, (int64_t) lat.size() / m.latent);
    const std::string P = "codec.decoder.";
    for (int i = 0; i <= n; i++) {
        if (i == 0) x = conv_dense(io, x, m.t(P + "stem.weight"), m.t(P + "stem.bias"), 7, 1);
        else {
            const std::string u = P + "up." + std::to_string(i - 1) + ".";
            x = conv_up(io, x, m.t(u + "weight"), m.t(u + "bias"), m.ratios[i - 1]);
        }
        for (int j = 0; j < m.dec_depths[i]; j++) {
            x = codec_block(io, x, P + "stages." + std::to_string(i) + "." + std::to_string(j) + ".", m.codec_eps);
        }
    }
    x = conv_dense(io, x, m.t(P + "head.weight"), m.t(P + "head.bias"), 7, 1);  // [1, T * frame]
    g.output(x);
    for (auto * s : io.new_states) g.output(s);
    g.run();
    for (size_t i = 0; i < io.new_states.size(); i++) st.s[i] = graph_call::get(io.new_states[i]);
    return graph_call::get(x);
}

// raw codec latents [T, latent] (+ masks [T]) -> projected embeddings [T, hidden] of the model / user projector
static std::vector<float> project(haru_audio & m, const std::string & which, const std::vector<float> & raw) {
    graph_call g(m);
    ggml_context * c = g.ctx;
    const int64_t T = (int64_t) raw.size() / m.latent;
    std::vector<float> scaled(raw.size());
    for (size_t i = 0; i < raw.size(); i++) scaled[i] = (raw[i] + m.latent_bias) * m.latent_scale;
    ggml_tensor * x = g.input(scaled, m.latent, T);
    const std::string p = which + "_audio_proj.";
    x = linear(c, x, m.t(p + "fc1.weight"), m.t(p + "fc1.bias"));
    x = rms(c, x, m.t(p + "norm.weight"), 1e-6f);
    x = linear(c, x, m.t(p + "fc2.weight"), m.t(p + "fc2.bias"));
    g.output(x);
    g.run();
    return graph_call::get(x);
}

static std::vector<float> scalar_features(float v) {  // sinusoidal features of a scalar: [cos, sin] of 128 freqs
    std::vector<float> f(256);
    for (int i = 0; i < 128; i++) {
        const float freq = expf(-logf(10000.0f) * (float) i / 128.0f);
        const float a = v * freq;
        f[i] = cosf(a);
        f[128 + i] = sinf(a);
    }
    return f;
}

// MeanFlow head: LM hidden [hidden] + noise [latent] -> normalized latent [latent]
static std::vector<float> head_sample(haru_audio & m, const float * hid, const std::vector<float> & noise, float guidance) {
    std::vector<float> z = noise;
    const float h = 1.0f / (float) m.head_steps;
    for (int step = 0; step < m.head_steps; step++) {
        graph_call g(m);
        ggml_context * c = g.ctx;
        ggml_tensor * hx = g.input_copy(std::vector<float>(hid, hid + m.hidden), m.hidden);
        ggml_tensor * zx = g.input(z, m.latent);
        ggml_tensor * cond = ggml_mul_mat(c, m.t("head.cond_proj.weight"), hx);
        auto embed = [&](const char * name, float v) {
            ggml_tensor * f = g.input_copy(scalar_features(v), 256);
            const std::string p = std::string("head.") + name + ".";
            return ggml_mul_mat(c, m.t(p + "fc2.weight"), ggml_silu(c, ggml_mul_mat(c, m.t(p + "fc1.weight"), f)));
        };
        ggml_tensor * cc = ggml_add(c, cond, embed("t_embed", step * h));
        cc = ggml_add(c, cc, embed("dt_embed", m.interval_embed));
        cc = ggml_add(c, cc, embed("h_embed", h));
        cc = ggml_add(c, cc, embed("omega_embed", 1.0f - 1.0f / guidance));
        cc = ggml_silu(c, cc);
        const int64_t H = m.hidden;
        ggml_tensor * x = ggml_mul_mat(c, m.t("head.latent_proj.weight"), zx);
        for (int l = 0; l < m.head_layers; l++) {
            const std::string p = "head.layers." + std::to_string(l) + ".";
            ggml_tensor * ada = ggml_mul_mat(c, m.t(p + "ada.weight"), cc);  // [3H]: shift, scale, gate
            ggml_tensor * shift = ggml_view_1d(c, ada, H, 0);
            ggml_tensor * scale = ggml_view_1d(c, ada, H, H * sizeof(float));
            ggml_tensor * gate = ggml_view_1d(c, ada, H, 2 * H * sizeof(float));
            ggml_tensor * hh = rms(c, x, m.t(p + "norm.weight"), m.head_eps);
            hh = ggml_add(c, ggml_add(c, hh, ggml_mul(c, hh, scale)), shift);
            ggml_tensor * f = ggml_mul_mat(c, m.t(p + "ffn.down_proj.weight"),
                ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, m.t(p + "ffn.gate_proj.weight"), hh)),
                            ggml_mul_mat(c, m.t(p + "ffn.up_proj.weight"), hh)));
            x = ggml_add(c, x, ggml_mul(c, gate, f));
        }
        ggml_tensor * ada = ggml_mul_mat(c, m.t("head.final.ada.weight"), cc);  // [2H]: shift, scale
        ggml_tensor * shift = ggml_view_1d(c, ada, H, 0);
        ggml_tensor * scale = ggml_view_1d(c, ada, H, H * sizeof(float));
        ggml_tensor * hh = rms(c, x, nullptr, m.head_eps);
        hh = ggml_add(c, ggml_add(c, hh, ggml_mul(c, hh, scale)), shift);
        ggml_tensor * u = ggml_mul_mat(c, m.t("head.final.proj.weight"), hh);
        ggml_tensor * out = ggml_add(c, zx, ggml_scale(c, u, h));
        g.output(out);
        g.run();
        z = graph_call::get(out);
    }
    return z;
}

// ------------------------------------------------------------------------------------------------ backbone

struct backbone {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_embd = 0, n_vocab = 0, n_pos_per_embd = 1;
    llama_pos n_past = 0;
    // token embedding rows, read from the GGUF and converted with ggml's own row conversion
    FILE * fp = nullptr;
    size_t row_off = 0, row_bytes = 0;
    ggml_type row_type = GGML_TYPE_F32;
    std::unordered_map<int, std::vector<float>> rows;

    void open(const std::string & path, int n_ctx, int n_threads, int n_gpu_layers) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = n_gpu_layers;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) die("cannot load ", path);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.n_threads = cp.n_threads_batch = n_threads;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) die("cannot create the context");
        vocab = llama_model_get_vocab(model);
        n_embd = llama_model_n_embd(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        n_pos_per_embd = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE || llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE ? 4 : 1;

        ggml_context * meta = nullptr;
        gguf_init_params ip = { true, &meta };
        gguf_context * g = gguf_init_from_file(path.c_str(), ip);
        if (!g) die("cannot read ", path);
        const int64_t ti = gguf_find_tensor(g, "token_embd.weight");
        if (ti < 0) die("no token_embd.weight in ", path);
        ggml_tensor * te = ggml_get_tensor(meta, "token_embd.weight");
        row_type = te->type;
        row_bytes = ggml_row_size(row_type, te->ne[0]);
        row_off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
        GGML_ASSERT(te->ne[0] == n_embd);
        gguf_free(g);
        ggml_free(meta);
        fp = fopen(path.c_str(), "rb");
    }

    const std::vector<float> & row(int id) {
        auto it = rows.find(id);
        if (it != rows.end()) return it->second;
        std::vector<uint8_t> raw(row_bytes);
#ifdef _WIN32
        _fseeki64(fp, (long long) (row_off + (size_t) id * row_bytes), SEEK_SET);
#else
        fseeko(fp, (off_t) (row_off + (size_t) id * row_bytes), SEEK_SET);
#endif
        if (fread(raw.data(), 1, row_bytes, fp) != row_bytes) die("short read of a token embedding row");
        std::vector<float> r(n_embd);
        if (row_type == GGML_TYPE_F32) memcpy(r.data(), raw.data(), n_embd * sizeof(float));
        else ggml_get_type_traits(row_type)->to_float(raw.data(), r.data(), n_embd);
        return rows.emplace(id, std::move(r)).first->second;
    }

    // run embeddings [n, n_embd]; hidden [n, n_embd] (all positions) and logits of the last position
    void forward(const std::vector<float> & embd, std::vector<float> * hidden, std::vector<float> * last_logits) {
        const int n = (int) (embd.size() / n_embd);
        llama_batch_ext * b = llama_batch_ext_init(ctx);
        for (int s = 0; s < n; s += 512) {
            const int e = std::min(n, s + 512);
            llama_batch_ext_clear(b);
            for (int i = s; i < e; i++) {
                const llama_embd le = { embd.data() + (size_t) i * n_embd, 1, (size_t) n_embd };
                const int32_t idx = llama_batch_ext_add_embd(b, 0, le);
                GGML_ASSERT(idx >= 0);
                llama_pos p[4] = { n_past, n_past, n_past, 0 };
                if (n_pos_per_embd == 1) p[1] = p[2] = 0;
                llama_batch_ext_set_pos(b, idx, p);
                llama_batch_ext_set_output_embd(b, idx, hidden != nullptr || i == n - 1);
                n_past++;
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) die("llama_process failed");
            if (hidden) {
                for (int i = s; i < e; i++) {
                    const float * h = llama_get_embeddings_ith(ctx, i - s);
                    hidden->insert(hidden->end(), h, h + n_embd);
                }
            } else if (e == n) {
                // keep only the last hidden
            }
        }
        if (last_logits) {
            const float * l = llama_get_logits_ith(ctx, -1);
            last_logits->assign(l, l + n_vocab);
        }
        llama_batch_ext_free(b);
    }

    std::vector<float> last_hidden() {
        const float * h = llama_get_embeddings_ith(ctx, -1);
        return std::vector<float>(h, h + n_embd);
    }
};

// ------------------------------------------------------------------------------------------------ model

struct haru {
    haru_audio a;
    backbone lm;
    std::mt19937_64 rng;
    std::normal_distribution<float> normal{0.0f, 1.0f};

    std::vector<float> randn(size_t n) { std::vector<float> r(n); for (auto & v : r) v = normal(rng); return r; }

    // codec.sample: latent means [T, D] -> means + s * (latent_std / 0.8) * noise, one s per call
    std::vector<float> codec_sample(const std::vector<float> & mean) {
        const float s = normal(rng) * (a.latent_std / 0.8f);
        std::vector<float> r(mean.size());
        for (size_t i = 0; i < r.size(); i++) r[i] = mean[i] + s * normal(rng);
        return r;
    }

    // sum of the three streams for T positions; latents raw [T, D]; masks per position (nullptr: all on)
    std::vector<float> embed(const std::vector<int> & tokens, const std::vector<float> & model_lat, const std::vector<float> & user_lat,
                             const std::vector<float> * mmask, const std::vector<float> * umask) {
        const int T = (int) tokens.size(), H = lm.n_embd;
        std::vector<float> e((size_t) T * H);
        for (int t = 0; t < T; t++) memcpy(&e[(size_t) t * H], lm.row(tokens[t]).data(), H * sizeof(float));
        const std::vector<float> pm = project(a, "model", model_lat), pu = project(a, "user", user_lat);
        for (int t = 0; t < T; t++) {
            const float mm = mmask ? (*mmask)[t] : 1.0f, um = umask ? (*umask)[t] : 1.0f;
            for (int i = 0; i < H; i++) e[(size_t) t * H + i] += mm * pm[(size_t) t * H + i] + um * pu[(size_t) t * H + i];
        }
        return e;
    }
};

// ------------------------------------------------------------------------------------------------ check

static void compare(const char * what, const std::vector<float> & got, const npy & ref, size_t off = 0, size_t n = 0) {
    if (!n) n = ref.f.size() - off;
    if (got.size() < n) { printf("%-28s SIZE got %zu want %zu\n", what, got.size(), n); return; }
    double dot = 0, ng = 0, nr = 0, maxd = 0, sumd = 0, maxr = 0;
    for (size_t i = 0; i < n; i++) {
        const double g = got[i], r = ref.f[off + i];
        dot += g * r; ng += g * g; nr += r * r;
        maxd = std::max(maxd, std::fabs(g - r)); sumd += std::fabs(g - r); maxr = std::max(maxr, std::fabs(r));
    }
    printf("%-28s n %8zu  max|d| %.3e  mean|d| %.3e  max|ref| %.3e  cos %.9f\n", what, n, maxd, sumd / n, maxr,
           dot / (std::sqrt(ng * nr) + 1e-30));
}

static int run_check(haru & h, const std::string & dir, const std::vector<float> & voice, const std::vector<float> & user) {
    haru_audio & a = h.a;
    auto R = [&](const char * k) { return npy_load(dir + "/" + k + ".npy"); };
    // codec
    {
        codec_state st = zero_state(encoder_layout(a));
        compare("codec encode (whole clip)", codec_encode(a, voice, st), R("enc_voice"));
        st = zero_state(encoder_layout(a));
        const npy ref = R("enc_user");
        std::vector<float> all;
        for (int64_t i = 0; i < ref.shape[0]; i++) {
            std::vector<float> fr(user.begin() + i * a.frame, user.begin() + (i + 1) * a.frame);
            auto l = codec_encode(a, fr, st);
            all.insert(all.end(), l.begin(), l.end());
        }
        compare("codec encode (streaming)", all, ref);
        const npy ev = R("enc_voice");
        codec_state ds = zero_state(decoder_layout(a));
        std::vector<float> aud;
        for (int64_t i = 0; i < ev.shape[0]; i++) {
            std::vector<float> l(ev.f.begin() + i * a.latent, ev.f.begin() + (i + 1) * a.latent);
            auto o = codec_decode(a, l, ds);
            aud.insert(aud.end(), o.begin(), o.end());
        }
        compare("codec decode (streaming)", aud, R("dec_audio"));
        codec_state zs = zero_state(encoder_layout(a));
        auto sl = codec_encode(a, std::vector<float>(16 * a.frame, 0.0f), zs);
        compare("silence latent", std::vector<float>(sl.end() - a.latent, sl.end()), R("silence"));
    }
    // prompt
    const npy pt = R("p_tokens"), pmod = R("p_model"), pusr = R("p_user"), pmm = R("p_mmask"), pum = R("p_umask");
    std::vector<int> toks(pt.f.begin(), pt.f.end());
    std::vector<float> e = h.embed(toks, pmod.f, pusr.f, &pmm.f, &pum.f);
    compare("prompt embeddings", e, R("p_embd"));
    std::vector<float> hid, logits;
    h.lm.forward(e, &hid, &logits);
    compare("prompt hidden (all)", hid, R("p_hidden"));
    compare("prompt logits (last)", logits, R("p_logits_last"));
    // frames, teacher forced
    const npy fu = R("f_user"), fti = R("f_tok_in"), fli = R("f_lat_in"), fh = R("f_hidden"), fl = R("f_logits");
    const npy fn = R("f_noise"), flo = R("f_lat_out"), fa = R("f_audio");
    const int nf = (int) fh.shape[0], H = h.lm.n_embd, D = a.latent;
    std::vector<float> all_e, all_h, all_l, all_z;
    for (int t = 0; t < nf; t++) {
        std::vector<int> tk = { (int) fti.f[t] };
        std::vector<float> ml(fli.f.begin() + t * D, fli.f.begin() + (t + 1) * D), ul(fu.f.begin() + t * D, fu.f.begin() + (t + 1) * D);
        std::vector<float> ee = h.embed(tk, ml, ul, nullptr, nullptr);
        all_e.insert(all_e.end(), ee.begin(), ee.end());
        std::vector<float> hh, ll;
        h.lm.forward(ee, &hh, &ll);
        all_h.insert(all_h.end(), hh.begin(), hh.end());
        all_l.insert(all_l.end(), ll.begin(), ll.end());
        // head on the reference hidden, so its check is independent of the backbone
        std::vector<float> noise(fn.f.begin() + t * D, fn.f.begin() + (t + 1) * D);
        std::vector<float> z = head_sample(a, fh.f.data() + (size_t) t * H, noise, a.guidance);
        for (auto & v : z) v = v / a.latent_scale - a.latent_bias;
        all_z.insert(all_z.end(), z.begin(), z.end());
    }
    compare("frame embeddings", all_e, R("f_embd"));
    compare("frame hidden", all_h, fh);
    compare("frame logits", all_l, fl);
    compare("frame head latents", all_z, flo);
    {  // the same frames in one batch (teacher forced): separates the one-row decode path from the batched one
        llama_memory_clear(llama_get_memory(h.lm.ctx), true);
        h.lm.n_past = 0;
        h.lm.forward(e, nullptr, nullptr);
        std::vector<float> bh;
        h.lm.forward(all_e, &bh, nullptr);
        compare("frame hidden (one batch)", bh, fh);
    }
    // decoded frame audio: decoder stream over lat_in[0] (frame 0), then lat_out[0..]
    codec_state ds = zero_state(decoder_layout(a));
    codec_decode(a, std::vector<float>(fli.f.begin(), fli.f.begin() + D), ds);
    std::vector<float> aud;
    for (int t = 0; t < nf; t++) {
        auto o = codec_decode(a, std::vector<float>(flo.f.begin() + t * D, flo.f.begin() + (t + 1) * D), ds);
        for (auto & v : o) v = std::max(-1.0f, std::min(1.0f, v));
        aud.insert(aud.end(), o.begin(), o.end());
    }
    if (fa.f.size() == (size_t) nf) {  // haru_ref.py keeps the first sample of each decoded frame
        std::vector<float> first(nf);
        for (int t = 0; t < nf; t++) first[t] = aud[(size_t) t * a.frame];
        compare("frame audio (first samples)", first, fa);
    } else {
        compare("frame audio", aud, fa);
    }
    // top-1 agreement of the text logits
    int same = 0;
    for (int t = 0; t < nf; t++) {
        const float * g = &all_l[(size_t) t * h.lm.n_vocab], * r = &fl.f[(size_t) t * h.lm.n_vocab];
        same += std::max_element(g, g + h.lm.n_vocab) - g == std::max_element(r, r + h.lm.n_vocab) - r;
    }
    printf("frame logits top-1 same       %d / %d\n", same, nf);
    return 0;
}

// ------------------------------------------------------------------------------------------------ duplex

struct dry_state {
    float mult = 0.8f, base = 1.75f, max_pen = 32.0f;
    int allowed = 1, lookback = 1024, max_match = 32;
    std::vector<int> hist;
    std::map<int, float> pen;
    std::vector<int> exempt;

    void update(int tok) {
        if (std::find(exempt.begin(), exempt.end(), tok) != exempt.end()) return;
        hist.push_back(tok);
        const int keep = lookback + max_match + 1;
        if ((int) hist.size() > keep) hist.erase(hist.begin(), hist.end() - keep);
        pen.clear();
        const int n = (int) hist.size();
        if (n < std::max(1, allowed)) return;
        std::map<int, int> longest;
        const int last = hist[n - 1];
        for (int j = n - 2; j >= std::max(0, n - 1 - lookback); j--) {
            if (hist[j] != last) continue;
            int len = 1;
            while (len < max_match && j - len >= 0 && hist[j - len] == hist[n - 1 - len]) len++;
            if (len >= allowed) {
                const int cand = hist[j + 1];
                if (len > longest[cand]) longest[cand] = len;
            }
        }
        for (auto & [t, l] : longest) {
            if (std::find(exempt.begin(), exempt.end(), t) == exempt.end()) pen[t] = std::min(max_pen, mult * powf(base, (float) (l - allowed)));
        }
    }
};

struct duplex_params {
    float temp = 0.7f, guidance = 1.5f, eop_bias = 0.0f;
    int top_k = 25;
    bool speak_first = false, dry = true;
    float tail_seconds = 4.0f, ref_seconds = 180.0f;
};

static int sample_token(std::vector<float> & logits, const duplex_params & dp, std::mt19937_64 & rng) {
    const int n = (int) logits.size();
    if (dp.temp <= 0) return (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    const int k = dp.top_k > 0 && dp.top_k < n ? dp.top_k : n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    std::vector<double> p(k);
    const double mx = logits[idx[0]] / dp.temp;
    double sum = 0;
    for (int i = 0; i < k; i++) { p[i] = std::exp(logits[idx[i]] / dp.temp - mx); sum += p[i]; }
    std::uniform_real_distribution<double> u(0.0, sum);
    double r = u(rng);
    for (int i = 0; i < k; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
    return idx[k - 1];
}

static std::string piece(const llama_vocab * v, const std::vector<int> & toks) {
    std::string s(256, '\0');
    int n = llama_detokenize(v, toks.data(), (int) toks.size(), s.data(), (int) s.size(), false, false);
    if (n < 0) { s.resize(-n); n = llama_detokenize(v, toks.data(), (int) toks.size(), s.data(), (int) s.size(), false, false); }
    s.resize(std::max(0, n));
    return s;
}

static int run_duplex(haru & h, const std::string & card, const std::vector<float> & voice_in, const std::vector<float> & user,
                      const duplex_params & dp, const std::string & out_path) {
    haru_audio & a = h.a;
    const int F = a.frame, D = a.latent;
    std::vector<float> voice = voice_in;
    const size_t lim = (size_t) (dp.ref_seconds * a.sample_rate);
    if (voice.size() > lim) voice.erase(voice.begin(), voice.end() - lim);

    // prompt: card, then the voice reference on the model channel (PAD ... PAD EOP), then the opening grid
    std::vector<int> toks;
    if (!card.empty()) {
        const std::string txt = "<|im_start|>system\n" + card + "<|im_end|>\n";
        std::vector<llama_token> t(txt.size() + 16);
        const int n = llama_tokenize(h.lm.vocab, txt.c_str(), (int) txt.size(), t.data(), (int) t.size(), false, true);
        if (n < 0) die("tokenize failed");
        toks.assign(t.begin(), t.begin() + n);
    }
    const int n_card = (int) toks.size();
    std::vector<float> ref_lat;
    if (!voice.empty()) {
        codec_state st = zero_state(encoder_layout(a));
        ref_lat = h.codec_sample(codec_encode(a, voice, st));
    }
    const int n_ref = (int) ref_lat.size() / D;
    for (int i = 0; i < n_ref; i++) toks.push_back(i == n_ref - 1 ? a.eop_id : a.pad_id);
    std::vector<int> grid;
    if (dp.speak_first) grid = { a.pad_id, a.eop_id };
    if (toks.empty() && grid.empty()) grid = { a.pad_id };
    toks.insert(toks.end(), grid.begin(), grid.end());
    const int n = (int) toks.size(), n_grid = (int) grid.size();

    codec_state zs = zero_state(encoder_layout(a));
    const std::vector<float> enc_sil = codec_encode(a, std::vector<float>(16 * F, 0.0f), zs);
    const std::vector<float> silence(enc_sil.end() - D, enc_sil.end());
    std::vector<float> room = silence;
    if (n_ref && voice.size() >= (size_t) F) {
        const int whole = (int) (voice.size() / F);
        int best = 0; double be = 1e300;
        for (int f = 0; f < whole; f++) {
            double e = 0;
            for (int i = 0; i < F; i++) e += (double) voice[(size_t) f * F + i] * voice[(size_t) f * F + i];
            if (e < be) { be = e; best = f; }
        }
        room.assign(ref_lat.begin() + best * D, ref_lat.begin() + (best + 1) * D);
    }
    std::vector<float> mlat((size_t) n * D, 0.0f), ulat((size_t) n * D, 0.0f), mm(n), um(n);
    for (int i = 0; i < n_ref; i++) std::copy(ref_lat.begin() + i * D, ref_lat.begin() + (i + 1) * D, mlat.begin() + (size_t) (n_card + i) * D);
    for (int i = n - n_grid; i < n; i++) {
        std::copy(room.begin(), room.end(), mlat.begin() + (size_t) i * D);
        std::copy(silence.begin(), silence.end(), ulat.begin() + (size_t) i * D);
    }
    for (int i = 0; i < n; i++) { mm[i] = i >= n_card ? 1.0f : 0.0f; um[i] = i >= n - n_grid ? 1.0f : 0.0f; }
    std::vector<float> logits;
    h.lm.forward(h.embed(toks, mlat, ulat, &mm, &um), nullptr, &logits);
    std::vector<float> hid = h.lm.last_hidden();
    fprintf(stderr, "llama-haru: prompt %d positions (card %d, voice %d frames)\n", n, n_card, n_ref);

    dry_state dry;
    dry.exempt = { a.pad_id, a.eop_id };
    if (a.eos_id >= 0) dry.exempt.push_back(a.eos_id);
    int force_text = dp.speak_first ? 1 : 0;
    codec_state enc = zero_state(encoder_layout(a)), dec = zero_state(decoder_layout(a));
    std::vector<float> out_model, out_user;
    std::vector<int> pending;
    bool in_gap = false;
    const int n_frames = (int) (user.size() / F) + (int) (dp.tail_seconds * a.sample_rate / F);
    for (int t = 0;; t++) {
        // emit model frame t from hid / logits
        if (dp.dry) for (auto & [tk, p] : dry.pen) logits[tk] -= p;
        for (int k = 0; k < 2; k++) {  // PAD / EOP logits in float32
            const float * r = &a.stream_embd[(size_t) k * a.hidden];
            double s = 0;
            for (int i = 0; i < a.hidden; i++) s += (double) hid[i] * r[i];
            logits[k == 0 ? a.pad_id : a.eop_id] = (float) s;
        }
        logits[a.eop_id] += dp.eop_bias;
        if (force_text > 0) {
            for (int b : dry.exempt) logits[b] = -INFINITY;
            force_text--;
        }
        const int tok = sample_token(logits, dp, h.rng);
        const std::vector<float> noise = h.randn(D);
        std::vector<float> lat = head_sample(a, hid.data(), noise, dp.guidance);
        for (auto & v : lat) v = v / a.latent_scale - a.latent_bias;
        std::vector<float> aud = codec_decode(a, lat, dec);
        for (auto & v : aud) v = std::max(-1.0f, std::min(1.0f, v));
        out_model.insert(out_model.end(), aud.begin(), aud.end());
        if (dp.dry) dry.update(tok);
        if (tok == a.pad_id || tok == a.eop_id) {
            if (!in_gap) { printf(" "); in_gap = true; }
        } else {
            in_gap = false;
            pending.push_back(tok);
            const std::string s = piece(h.lm.vocab, pending);
            if (s.size() < 3 || s.compare(s.size() - 3, 3, "\xEF\xBF\xBD") != 0) { printf("%s", s.c_str()); pending.clear(); }
        }
        fflush(stdout);
        if (t >= n_frames) break;
        // consume user frame t (silence after the clip ends)
        std::vector<float> uf(F, 0.0f);
        if ((size_t) (t + 1) * F <= user.size()) std::copy(user.begin() + (size_t) t * F, user.begin() + (size_t) (t + 1) * F, uf.begin());
        out_user.insert(out_user.end(), uf.begin(), uf.end());
        const std::vector<float> ul = h.codec_sample(codec_encode(a, uf, enc));
        h.lm.forward(h.embed({ tok }, lat, ul, nullptr, nullptr), nullptr, &logits);
        hid = h.lm.last_hidden();
    }
    printf("\n");
    // stereo: user left, model right (the model's frame t plays while user frame t arrives)
    std::vector<float> st;
    const size_t L = std::max(out_model.size(), out_user.size());
    for (size_t i = 0; i < L; i++) {
        st.push_back(i < out_user.size() ? out_user[i] : 0.0f);
        st.push_back(i < out_model.size() ? out_model[i] : 0.0f);
    }
    wav_write(out_path, out_model, a.sample_rate, 1);
    const std::string sp = out_path.size() > 4 ? out_path.substr(0, out_path.size() - 4) + "-stereo.wav" : out_path + "-stereo.wav";
    wav_write(sp, st, a.sample_rate, 2);
    fprintf(stderr, "llama-haru: wrote %s and %s (%.1f s)\n", out_path.c_str(), sp.c_str(), (double) out_model.size() / a.sample_rate);
    return 0;
}

// ------------------------------------------------------------------------------------------------ main

static void usage() {
    fprintf(stderr,
        "usage: llama-haru -m LM.gguf --audio AUDIO.gguf [options]\n"
        "  --voice WAV        voice reference (24 kHz mono, 3 s to 3 min)\n"
        "  --user WAV         user speech for an offline conversation (24 kHz mono)\n"
        "  --card TEXT        character card (system prompt); --card-file FILE\n"
        "  -o OUT.wav         model audio (also writes OUT-stereo.wav: user left, model right)\n"
        "  --seed N --temp T --top-k K --guidance G (1-2) --eop-bias B --speak-first --no-dry --tail SECONDS\n"
        "  -t THREADS -ngl N --cpu (audio graphs on the CPU) -c CTX\n"
        "  --check REF_DIR    compare every part with haru_ref.py dumps (use F32 files)\n");
    exit(1);
}

int main(int argc, char ** argv) {
    std::string lm_path, audio_path, voice_path, user_path, card, out = "haru-out.wav", check;
    int n_threads = 8, ngl = 99, n_ctx = 16384;
    uint64_t seed = 0;
    bool cpu = false, has_seed = false;
    duplex_params dp;
    for (int i = 1; i < argc; i++) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (k == "-m") lm_path = next();
        else if (k == "--audio") audio_path = next();
        else if (k == "--voice") voice_path = next();
        else if (k == "--user") user_path = next();
        else if (k == "--card") card = next();
        else if (k == "--card-file") {
            FILE * fp = fopen(next().c_str(), "rb");
            if (!fp) die("cannot open the card file");
            char b[4096]; size_t r;
            while ((r = fread(b, 1, sizeof b, fp)) > 0) card.append(b, r);
            fclose(fp);
        }
        else if (k == "-o") out = next();
        else if (k == "--check") check = next();
        else if (k == "--seed") { seed = std::stoull(next()); has_seed = true; }
        else if (k == "--temp") dp.temp = std::stof(next());
        else if (k == "--top-k") dp.top_k = std::stoi(next());
        else if (k == "--guidance") dp.guidance = std::stof(next());
        else if (k == "--eop-bias") dp.eop_bias = std::stof(next());
        else if (k == "--speak-first") dp.speak_first = true;
        else if (k == "--no-dry") dp.dry = false;
        else if (k == "--tail") dp.tail_seconds = std::stof(next());
        else if (k == "-t") n_threads = std::stoi(next());
        else if (k == "-ngl") ngl = std::stoi(next());
        else if (k == "-c") n_ctx = std::stoi(next());
        else if (k == "--cpu") cpu = true;
        else usage();
    }
    if (lm_path.empty() || audio_path.empty()) usage();
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);
    haru h;
    h.rng.seed(has_seed ? seed : std::random_device{}());
    audio_load(h.a, audio_path, cpu, n_threads);
    h.lm.open(lm_path, n_ctx, n_threads, ngl);
    if (h.lm.n_embd != h.a.hidden) die("the backbone and the audio file do not match");
    const std::vector<float> voice = voice_path.empty() ? std::vector<float>() : wav_read(voice_path, h.a.sample_rate);
    const std::vector<float> user = user_path.empty() ? std::vector<float>() : wav_read(user_path, h.a.sample_rate);
    const int rc = check.empty() ? run_duplex(h, card, voice, user, dp, out) : run_check(h, check, voice, user);
    llama_free(h.lm.ctx);
    llama_model_free(h.lm.model);
    return rc;
}

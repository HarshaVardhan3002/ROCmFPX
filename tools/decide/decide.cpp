// llama-decide: calibrated decisions from decision models (baselabs elephant-1 / mammoth-1 style).
//
// A request is a state plus questions (noul, choice, score), as in OneJev's qev runtime. Every question is
// rendered into the same chat prompt: a shared prefix (system turn and state block) and a question suffix
// (question, options, answer line, generation prompt). The answer is a softmax over the LM-head logits of the
// option letters at the last suffix token (readout "slot_lm").
//
// The GGUF has <arch>.decision.type = "baselabs-qev" (convert_hf_to_gguf.py). Models trained with a
// block-bidirectional mask on the full-attention layers (<arch>.decision.attention = "block_bidirectional") run non-causally: the prefix is decoded as one block,
// then each suffix as one block on a copy of the prefix sequence, so attention is bidirectional within a block
// and causal across blocks. Recurrent layers are causal by construction.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h" // llama_model_decision_label

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// qev prompt "qev-labels-v2"
static const char * SYSTEM_PROMPT =
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning.";
static const char * PREFIX_MARK = "QEV_PREFIX_MARK";
static const char * NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?";
static const char * NOUL_DEFAULT_TRUE  = "the statement is true / the answer is yes";
static const char * NOUL_DEFAULT_FALSE = "the statement is false / the answer is no";

// qev SLOT_LABELS: A..Z, then two-letter labels that are single tokens in the Qwen tokenizer, 256 in total
static std::vector<std::string> default_slot_labels() {
    static const char * not_one_token[] = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV",
        "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"};
    std::vector<std::string> out;
    for (char c = 'A'; c <= 'Z'; ++c) {
        out.emplace_back(1, c);
    }
    for (char a = 'A'; a <= 'Z' && out.size() < 256; ++a) {
        for (char b = 'A'; b <= 'Z' && out.size() < 256; ++b) {
            const std::string s = {a, b};
            if (std::find_if(std::begin(not_one_token), std::end(not_one_token),
                    [&](const char * x) { return s == x; }) == std::end(not_one_token)) {
                out.push_back(s);
            }
        }
    }
    return out;
}

static std::string strip(const std::string & s) {
    const char * ws = " \t\n\r\f\v";
    const size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) {
        return "";
    }
    return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

// json.dumps(x, ensure_ascii=False, indent=2)
static std::string dump(const json & x) {
    return x.dump(2, ' ', false, json::error_handler_t::replace);
}

static std::string render_entry(const json & e) {
    if (e.is_null()) {
        return "";
    }
    if (e.is_string()) {
        return strip(e.get<std::string>());
    }
    return dump(e);
}

static std::string render_state(const json & s) {
    if (s.is_null()) {
        return "";
    }
    if (s.is_string()) {
        return s.get<std::string>();
    }
    return dump(s);
}

struct question {
    std::string qid;
    std::string kind;
    std::string instructions;
    std::vector<std::string> labels;
    std::vector<std::string> names;
    std::vector<json>        descs;
};

struct branch {
    size_t q;                 // index into questions
    std::vector<int> order;   // slot i shows option order[i]
    std::string suffix;
    std::vector<llama_token> tokens; // suffix tokens
    std::vector<float> logits;       // slot logits
    float mass = 0.0f;               // probability mass of the slot tokens over the vocabulary
};

static question parse_question(const std::string & qid, const json & q) {
    question r;
    r.qid  = qid;
    r.kind = q.value("type", "");
    const json instr = q.contains("instructions") ? q["instructions"] : json();
    if (r.kind == "noul") {
        r.instructions = render_entry(instr);
        if (r.instructions.empty()) {
            r.instructions = NOUL_DEFAULT_INSTRUCTIONS;
        }
        const json crit = q.contains("criteria") ? q["criteria"] : json();
        const bool has_t = crit.is_object() && crit.contains("true")  && !crit["true"].is_null();
        const bool has_f = crit.is_object() && crit.contains("false") && !crit["false"].is_null();
        r.labels = {"yes", "no"};
        r.names  = {"yes", "no"};
        r.descs  = {has_t ? json(render_entry(crit["true"]))  : json(NOUL_DEFAULT_TRUE),
                    has_f ? json(render_entry(crit["false"])) : json(NOUL_DEFAULT_FALSE)};
    } else if (r.kind == "choice") {
        r.instructions = render_entry(instr);
        if (r.instructions.empty()) {
            r.instructions = "Which option applies to the state?";
        }
        if (!q.contains("criteria") || !q["criteria"].is_object() || q["criteria"].empty()) {
            throw std::runtime_error("choice question " + qid + " needs criteria {name: description}");
        }
        for (const auto & [name, desc] : q["criteria"].items()) {
            r.labels.push_back(name);
            r.names.push_back(name);
            r.descs.push_back(desc);
        }
    } else if (r.kind == "score") {
        r.instructions = render_entry(instr);
        if (r.instructions.empty()) {
            r.instructions = "Which level describes the state?";
        }
        if (!q.contains("criteria") || !q["criteria"].is_array() || q["criteria"].empty()) {
            throw std::runtime_error("score question " + qid + " needs criteria [levels]");
        }
        for (size_t i = 0; i < q["criteria"].size(); ++i) {
            r.labels.push_back(std::to_string(i));
            r.names.push_back("level " + std::to_string(i));
            r.descs.push_back(q["criteria"][i]);
        }
    } else {
        throw std::runtime_error("question " + qid + " has unsupported type '" + r.kind + "'");
    }
    return r;
}

static std::string render_suffix(const question & q, const std::vector<int> & order,
                                 const std::vector<std::string> & slots) {
    const size_t k = q.labels.size();
    std::string lines;
    for (size_t i = 0; i < k; ++i) {
        const std::string & name = q.names[order[i]];
        const std::string text = render_entry(q.descs[order[i]]);
        lines += (i ? "\n" : "") + slots[i] + ". " + (text.empty() ? name : name + ": " + text);
    }
    std::string letters;
    for (size_t i = 0; i < k; ++i) {
        letters += (i ? ", " : "") + slots[i];
    }
    const std::string header = q.kind == "score" ? "Rate the state:" : "Question:";
    const std::string answer = (k <= 26 ? "Answer with one letter: " : "Answer with one label: ") + letters + ".";
    return header + " " + q.instructions + "\n\nOptions:\n" + lines + "\n\n" + answer;
}

static std::string render_prompt(const common_chat_templates * tmpls, const std::string & state, const std::string & suffix) {
    common_chat_templates_inputs in;
    common_chat_msg sys;
    sys.role    = "system";
    sys.content = SYSTEM_PROMPT;
    common_chat_msg usr;
    usr.role    = "user";
    usr.content = "<state>\n" + state + "\n</state>\n\n" + suffix;
    in.messages              = {sys, usr};
    in.add_generation_prompt = true;
    in.use_jinja             = true;
    in.enable_thinking       = false;
    in.chat_template_kwargs["enable_thinking"] = "false";
    return common_chat_templates_apply(tmpls, in).prompt.str();
}

static std::string meta_str(const llama_model * model, const char * key) {
    char buf[512];
    const int n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n < 0 ? "" : std::string(buf);
}

static std::vector<float> softmax(const std::vector<float> & z, float t) {
    float m = -INFINITY;
    for (float x : z) {
        m = std::max(m, x / t);
    }
    std::vector<float> p(z.size());
    double s = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        p[i] = std::exp(z[i] / t - m);
        s += p[i];
    }
    for (float & x : p) {
        x = (float) (x / s);
    }
    return p;
}

static double round6(double x) {
    return std::round(x * 1e6) / 1e6;
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m elephant-1-Q8_0_ROCMFPX.gguf -f request.json [--debias N] [--branches N] [--causal] [--debug]\n", argv[0]);
    LOG("\n    request.json: {\"state\": ..., \"questions\": {\"id\": {\"type\": \"noul|choice|score\", "
        "\"instructions\": ..., \"criteria\": ...}}}\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // tool options, removed before the common parser sees argv
    int  debias   = 1;
    int  n_group  = 8;
    bool force_causal = false;
    bool debug    = false;
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--debias") && i + 1 < argc) {
            debias = std::max(1, atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--branches") && i + 1 < argc) {
            n_group = std::max(1, atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--causal")) {
            force_causal = true;
        } else if (!strcmp(argv[i], "--debug")) {
            debug = true;
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }
    common_init();

    json request;
    try {
        request = json::parse(params.prompt);
    } catch (const std::exception & e) {
        LOG_ERR("%s: cannot parse the request (-f): %s\n", __func__, e.what());
        return 1;
    }
    if (!request.contains("questions") || !request["questions"].is_object() || request["questions"].empty()) {
        LOG_ERR("%s: the request needs at least one question\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // how the checkpoint was trained (written by convert_hf_to_gguf.py from config.json and calibration.json)
    const std::string kd        = meta_str(model, "general.architecture") + ".decision.";
    const std::string dtype     = meta_str(model, (kd + "type").c_str());
    const std::string full_attn = meta_str(model, (kd + "attention").c_str());
    if (dtype.empty()) {
        LOG_WRN("%s: no decision metadata in the GGUF, running causal with the qev prompt\n", __func__);
    } else if (dtype != "baselabs-qev") {
        LOG_ERR("%s: decision type '%s' is not a qev model, serve it with llama-server /v1/systemone\n", __func__, dtype.c_str());
        return 1;
    }
    const bool noncausal = full_attn == "block_bidirectional" && !force_causal;

    // calibration temperatures "<arch>.decision.temperature.<kind>[.<K>]"
    std::map<std::string, float> temps;
    {
        const std::string pt = kd + "temperature.";
        for (int32_t i = 0; i < llama_model_meta_count(model); ++i) {
            char key[256];
            char val[64];
            if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || strncmp(key, pt.c_str(), pt.size()) != 0 ||
                llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
                continue;
            }
            temps[key + pt.size()] = std::strtof(val, nullptr);
        }
    }
    auto temperature = [&](const std::string & kind, size_t k) {
        auto it = temps.find(kind + "." + std::to_string(k));
        if (it != temps.end()) {
            return it->second;
        }
        it = temps.find(kind);
        return it != temps.end() ? it->second : 1.0f;
    };

    // answer slots: every label must be one token that decodes back to itself
    std::vector<std::string> labels;
    for (uint32_t i = 0; llama_model_decision_label(model, i) != nullptr; ++i) {
        labels.emplace_back(llama_model_decision_label(model, i));
    }
    if (labels.empty()) {
        labels = default_slot_labels();
    }
    std::vector<llama_token> slot_ids;
    for (const auto & l : labels) {
        const auto t = common_tokenize(vocab, l, false, false);
        if (t.size() != 1 || common_token_to_piece(vocab, t[0], false) != l) {
            break;
        }
        slot_ids.push_back(t[0]);
    }
    if (slot_ids.size() < 26) {
        LOG_ERR("%s: answer letters are not single tokens in this vocabulary\n", __func__);
        return 1;
    }
    if (slot_ids.size() < labels.size()) {
        slot_ids.resize(26);
    }

    // render every branch (one per question and debias shift)
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    const std::string state = render_state(request.contains("state") ? request["state"] : json());

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<question> questions;
    std::vector<branch>   branches;
    try {
        for (const auto & [qid, q] : request["questions"].items()) {
            questions.push_back(parse_question(qid, q));
            const question & qq = questions.back();
            const size_t k = qq.labels.size();
            if (k > slot_ids.size()) {
                throw std::runtime_error("question " + qid + " has " + std::to_string(k) + " options, the model has " +
                                         std::to_string(slot_ids.size()) + " answer slots");
            }
            const int shifts = k > 1 ? std::min<int>(debias, (int) k) : 1;
            for (int s = 0; s < shifts; ++s) {
                branch b;
                b.q = questions.size() - 1;
                for (size_t i = 0; i < k; ++i) {
                    b.order.push_back((int) ((i + s) % k));
                }
                b.suffix = render_suffix(qq, b.order, labels);
                branches.push_back(std::move(b));
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }

    // shared prefix: tokens of the prompt up to the end of the state block, cut to the common token prefix
    std::vector<std::vector<llama_token>> full;
    for (const auto & b : branches) {
        full.push_back(common_tokenize(vocab, render_prompt(tmpls.get(), state, b.suffix), false, true));
    }
    const std::string pt = render_prompt(tmpls.get(), state, PREFIX_MARK);
    std::vector<llama_token> prefix = common_tokenize(vocab, pt.substr(0, pt.find(PREFIX_MARK)), false, true);
    size_t P = prefix.size();
    while (P > 0 && !std::all_of(full.begin(), full.end(), [&](const std::vector<llama_token> & ids) {
               return ids.size() > P && std::equal(prefix.begin(), prefix.begin() + P, ids.begin()); })) {
        --P;
    }
    prefix.resize(P);
    size_t longest = 0;
    for (size_t i = 0; i < branches.size(); ++i) {
        branches[i].tokens.assign(full[i].begin() + P, full[i].end());
        longest = std::max(longest, branches[i].tokens.size());
    }
    if (P == 0) {
        LOG_ERR("%s: empty shared prefix\n", __func__);
        return 1;
    }

    // one context sized for the prefix plus a group of suffixes; non-causal blocks must fit one ubatch each.
    // The hybrid memory splits multi-sequence batches with split_equal, which would cut suffixes of different
    // lengths across ubatches, so non-causal suffixes are decoded one per call
    const int group = noncausal ? 1 : std::min<int>(n_group, (int) branches.size());
    llama_context_params cparams = common_context_params_to_llama(params);
    const uint32_t need = (uint32_t) (P + group * longest);
    cparams.n_ctx      = std::max<uint32_t>(cparams.n_ctx, GGML_PAD(need, 256));
    cparams.n_batch    = std::max<uint32_t>(cparams.n_batch, cparams.n_ctx);
    cparams.n_ubatch   = (uint32_t) GGML_PAD(std::max(P, longest), 64);
    cparams.n_seq_max  = group + 1;
    cparams.kv_unified = true;
    cparams.attention_type = noncausal ? LLAMA_ATTENTION_TYPE_NON_CAUSAL : LLAMA_ATTENTION_TYPE_CAUSAL;
    cparams.embeddings = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create the context\n", __func__);
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const auto t1 = std::chrono::steady_clock::now();

    // prefix block on sequence 0
    {
        common_batch batch(ctx);
        for (size_t i = 0; i < P; ++i) {
            batch.add(prefix[i], (llama_pos) i, 0, i + 1 == P);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: prefix decode failed\n", __func__);
            return 1;
        }
    }

    // suffix blocks, `group` at a time, each on its own copy of sequence 0
    for (size_t g0 = 0; g0 < branches.size(); g0 += group) {
        const size_t g1 = std::min(branches.size(), g0 + group);
        common_batch batch(ctx);
        std::vector<int> out_idx;
        for (size_t i = g0; i < g1; ++i) {
            const llama_seq_id s = (llama_seq_id) (1 + i - g0);
            llama_memory_seq_rm(mem, s, -1, -1);
            llama_memory_seq_cp(mem, 0, s, -1, -1);
            const auto & t = branches[i].tokens;
            for (size_t j = 0; j < t.size(); ++j) {
                batch.add(t[j], (llama_pos) (P + j), s, j + 1 == t.size());
            }
            out_idx.push_back(batch.size() - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: suffix decode failed\n", __func__);
            return 1;
        }
        for (size_t i = g0; i < g1; ++i) {
            branch & b = branches[i];
            const float * v = llama_get_logits_ith(ctx, out_idx[i - g0]);
            const size_t k = b.order.size();
            double m = -INFINITY;
            for (int j = 0; j < n_vocab; ++j) {
                m = std::max<double>(m, v[j]);
            }
            double all = 0.0, sel = 0.0;
            for (int j = 0; j < n_vocab; ++j) {
                all += std::exp(v[j] - m);
            }
            for (size_t j = 0; j < k; ++j) {
                b.logits.push_back(v[slot_ids[j]]);
                sel += std::exp(v[slot_ids[j]] - m);
            }
            b.mass = (float) (sel / all);
        }
    }
    const auto t2 = std::chrono::steady_clock::now();

    // probabilities per question, averaged over the debias shifts in the original option order
    json answers = json::object();
    json diag    = json::object();
    size_t total_tokens = P;
    for (const auto & b : branches) {
        total_tokens += b.tokens.size();
    }
    for (size_t qi = 0; qi < questions.size(); ++qi) {
        const question & q = questions[qi];
        const size_t k = q.labels.size();
        std::vector<double> probs(k, 0.0);
        int n = 0;
        double mass = 0.0;
        json slot_logits = json::array();
        for (const auto & b : branches) {
            if (b.q != qi) {
                continue;
            }
            const auto p = softmax(b.logits, temperature(q.kind, k));
            for (size_t i = 0; i < k; ++i) {
                probs[b.order[i]] += p[i];
            }
            mass += b.mass;
            slot_logits.push_back(b.logits);
            ++n;
        }
        double sum = 0.0;
        for (double & p : probs) {
            p /= n;
            sum += p;
        }
        for (double & p : probs) {
            p = sum > 0 ? p / sum : 1.0 / k;
        }
        const size_t best = std::max_element(probs.begin(), probs.end()) - probs.begin();
        json a;
        a["type"] = q.kind;
        if (q.kind == "noul") {
            a["noul"] = round6(std::min(1.0, std::max(0.0, probs[0])));
        } else if (q.kind == "choice") {
            a["choice"] = q.labels[best];
            json pr = json::object();
            for (size_t i = 0; i < k; ++i) {
                pr[q.labels[i]] = round6(probs[i]);
            }
            a["probabilities"] = pr;
            a["confidence"] = round6(k == 1 ? 1.0 : (probs[best] - 1.0 / k) / (1.0 - 1.0 / k));
        } else {
            double es = 0.0, dist = 0.0, umad = 0.0;
            for (size_t i = 0; i < k; ++i) {
                es   += i * probs[i];
                dist += probs[i] * std::fabs((double) i - (double) best);
                umad += std::fabs((double) i - (k - 1) / 2.0) / k;
            }
            a["score"] = round6(es);
            json lg = json::object();
            json pr = json::object();
            for (size_t i = 0; i < k; ++i) {
                lg[std::to_string(i)] = q.descs[i];
                pr[std::to_string(i)] = round6(probs[i]);
            }
            a["legend"] = lg;
            a["probabilities"] = pr;
            a["confidence"] = round6(k == 1 ? 1.0 : std::max(0.0, 1.0 - dist / umad));
        }
        answers[q.qid] = a;
        diag[q.qid] = {{"allowed_token_mass", mass / n}, {"slot_logits", slot_logits},
                       {"temperature", temperature(q.kind, k)}};
    }

    json resp;
    resp["answers"] = answers;
    resp["usage"]   = {{"input_tokens", total_tokens}, {"output_tokens", questions.size()}};
    resp["meta"]    = {
        {"attention", noncausal ? "block_bidirectional" : "causal"},
        {"prefix_tokens", P},
        {"branches", branches.size()},
        {"timing", {{"render_s", std::chrono::duration<double>(t1 - t0).count()},
                    {"forward_s", std::chrono::duration<double>(t2 - t1).count()}}},
    };
    if (debug) {
        resp["meta"]["diagnostics"] = diag;
    }
    printf("%s\n", resp.dump(2).c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

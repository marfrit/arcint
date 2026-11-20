// A router-input capture for campaign expert-foresight's layer-ahead predictor
// (docs/campaigns/expert-foresight.md, "Queued (operator, 2026-11-19): a
// layer-ahead predictor", plan step 1: "a trace with the hidden state at each
// MoE input"). It records, for every token of a prompt and of a greedy decode,
// each MoE layer's router input -- the hidden state that feeds ffn_gate_inp,
// after the layer's ffn norm -- and the routed expert ids the router selected,
// in rank order. The replay (tools/expert_cache_replay.py's sibling) then asks
// how well layer l+d's router, applied to layer l's input, predicts layer
// l+d's real routing. Off the serving path: no server change, no patch, no
// kernel; the graph evaluation callback reads tensors the graph computes.
//
// WHERE THE TENSORS ARE (read, not guessed):
//   * build_moe_ffn calls build_lora_mm(gate_inp, cur) for the router
//     (src/llama-graph.cpp:2154) and names the product "ffn_moe_logits"
//     (src/llama-graph.cpp:2157); build_lora_mm is ggml_mul_mat(w, cur)
//     (src/llama-graph.cpp:1618), so the mul_mat's activation operand t->src[1]
//     IS the router input; build_hadamard_input returns cur untouched for
//     ffn_gate_inp (no rotation is registered for a router).
//   * the selected ids are the "ffn_moe_topk" tensor, I32 [n_used, n_tokens],
//     the rows of ggml_argsort_top_k over the probs, descending
//     (src/llama-graph.cpp:2238-2240).
//   * for qwen4exp (Flash-Next) that cur is the hyper-connection FFN mix
//     build_hc_mix(res_hc, hc_ffn_norm, ...) (src/models/qwen4exp.cpp:484-494,
//     built at src/models/qwen4exp.cpp:313, output named "hc_mixed"), passed
//     straight into build_moe_ffn (src/models/qwen4exp.cpp:1176). It is NOT
//     matched by name: "hc_mixed" is emitted twice a layer (the attention mix
//     at src/models/qwen4exp.cpp:457 and the ffn mix at :484) with the same
//     name and il.
//
// The eval callback (llama_context_params.cb_eval) is asked for each node
// (ggml-backend.cpp:1910); we return true for the two names above, the
// scheduler computes up to that node and synchronises (ggml-backend.cpp:1922,
// 1928), then calls us with ask=false (ggml-backend.cpp:1930) and we read the
// tensor. Reading goes through ggml_backend_tensor_get and honours t->nb[1], so
// a view row is read correctly; CPU only (-ngl 0) means host memory anyway.
//
// Second cell in the tool: after the first decode token is captured, the tool
// recomputes the top-k from the CAPTURED input times the layer's
// ffn_gate_inp weight, on the CPU, and compares the first two ranks with the
// captured ids. It aborts (exit 3, SELF-CHECK FAIL) on any mismatch of those
// two ranks. The
// weight is read from the run's own GGUF with the public gguf API (no internal
// llama symbols) and dequantised with ggml_get_type_traits(type)->to_float; the
// Flash-Next GGUF is split, so every shard named by the file's
// "-NNNNN-of-MMMMM.gguf" is opened and the tensor looked up in each.
//
// Output: one binary file (magic "LAC1") and one text sidecar `<out>.info`
// (the prompt file's name, the decode count, the model files' names -- names
// only, no paths). File format, documented by tools/expert_router_capture_format.h:
//
//   header   b"LAC1", u32 version (1), u32 n_layer, u32 n_embd, u32 n_expert,
//            u32 n_used, u8 dtype (1: f16) -- 25 bytes, little endian
//   token    i32 position, i32 token id, u8 phase (0 prompt, 1 decode), then
//            per MoE layer: n_embd x u16 f16 router input (one contiguous row,
//            token-major), then n_used x i16 ids in rank order
//
// A token record is 8 + 1 + n_layer*(2*n_embd + 2*n_used) bytes: Flash-Next
// (48 layers, n_embd 2560, top-10 of 512 experts) 246,729 bytes, ~241 KiB,
// prompt and decode alike. A 100k-token prompt is about 24 GB; keep --out on a
// dataset with room. The file has no trailer; tools/expert_router_capture_read.py
// reads it (and drops a torn last record).
//
// Usage (CPU only; needs the GGUF and the prompt file, no card):
//   arcint-expert-router-capture --model MODEL.gguf --prompt PROMPT.txt
//       --decode 64 --out capture.lac1
//
// Red cases (the round trip, tests/python/test_expert_router_capture.py; the
// mutants are named there and in tools/expert_router_capture_read.py): a writer
// that drops the phase byte and a writer that writes n_used+1 ids shift every
// later field, and the reader's strict torn-record cell fails when the reader
// accepts a short record.
//
// The reference idea (a router's own scores plus a small learned correction
// for the next layers' routers) is Strata-HotRod's, relayed in the campaign
// section above; this tool only records what is needed to measure it, nothing
// of its code is used.

#include "llama.h"

#include "llama-ext.h"   // llama_model_n_expert (staging header of the pin)

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "expert_router_capture_format.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// args

struct args {
    std::string model;
    std::string prompt;
    std::string out;
    int32_t decode = 0;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model MODEL.gguf --prompt PROMPT.txt --decode N --out FILE\n"
        "  CPU only (-ngl 0, no expert cache, no MTP). Writes FILE (LAC1) and FILE.info.\n",
        argv0);
}

bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--model") {
            const char * v = value("--model"); if (!v) return false; a.model = v;
        } else if (key == "--prompt") {
            const char * v = value("--prompt"); if (!v) return false; a.prompt = v;
        } else if (key == "--decode") {
            const char * v = value("--decode"); if (!v) return false;
            char * end = nullptr;
            const long n = std::strtol(v, &end, 10);
            if (*end != 0 || n < 0) {
                std::fprintf(stderr, "--decode %s is not a non-negative count\n", v);
                return false;
            }
            a.decode = (int32_t) n;
        } else if (key == "--out") {
            const char * v = value("--out"); if (!v) return false; a.out = v;
        } else if (key == "-h" || key == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", key.c_str());
            usage(argv[0]);
            return false;
        }
    }
    if (a.model.empty() || a.prompt.empty() || a.out.empty() || a.decode < 1) {
        std::fprintf(stderr, "--model, --prompt, --out and a --decode of at least 1 are required\n");
        usage(argv[0]);
        return false;
    }
    return true;
}

std::string basename_of(const std::string & path) {
    return std::filesystem::path(path).filename().string();
}

// The router's top-k from the GGUF metadata: llama.h has no n_expert_used
// getter, and the LAC1 header must carry it before the first graph runs.
bool model_n_expert_used(const llama_model * model, uint32_t & n_used) {
    char arch[64] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        return false;
    }
    const std::string key = std::string(arch) + ".expert_used_count";
    char buf[64] = {0};
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return false;
    }
    char * end = nullptr;
    const unsigned long v = std::strtoul(buf, &end, 10);
    if (*end != '\0' || v == 0 || v > 32767) return false;
    n_used = (uint32_t) v;
    return true;
}

// ---------------------------------------------------------------------------
// the run's own GGUF: the ffn_gate_inp weight the self-check needs

struct shard {
    std::string name;
    gguf_context * ctx = nullptr;
    int fd = -1;
    void * base = nullptr;
    size_t size = 0;
};

struct weights {
    std::vector<shard> shards;
    std::string why;

    ~weights() {
        for (auto & s : shards) {
            if (s.ctx) gguf_free(s.ctx);
            if (s.base) munmap(s.base, s.size);
            if (s.fd >= 0) ::close(s.fd);
        }
    }

    bool open_shard(const std::string & name) {
        shard s;
        s.name = name;
        s.fd = ::open(name.c_str(), O_RDONLY);
        if (s.fd < 0) { why = "cannot open " + basename_of(name); return false; }
        struct stat st{};
        if (::fstat(s.fd, &st) != 0 || st.st_size <= 0) {
            ::close(s.fd);
            why = "cannot stat " + basename_of(name);
            return false;
        }
        s.size = (size_t) st.st_size;
        s.base = mmap(nullptr, s.size, PROT_READ, MAP_PRIVATE, s.fd, 0);
        if (s.base == MAP_FAILED) {
            s.base = nullptr;
            ::close(s.fd);
            why = "cannot mmap " + basename_of(name);
            return false;
        }
        struct gguf_init_params p{};
        p.no_alloc = true;
        p.ctx = nullptr;
        s.ctx = gguf_init_from_file(name.c_str(), p);
        if (!s.ctx) {
            munmap(s.base, s.size);
            ::close(s.fd);
            why = "not a GGUF: " + basename_of(name);
            return false;
        }
        shards.push_back(std::move(s));
        return true;
    }

    // every shard of the split GGUF named by the file's "-NNNNN-of-MMMMM.gguf"
    bool open(const std::string & model_path) {
        const std::string base = basename_of(model_path);
        const std::filesystem::path dir = std::filesystem::path(model_path).parent_path();
        // <stem>-NNNNN-of-MMMMM.gguf
        const size_t of = base.rfind("-of-");
        const size_t dash = of == std::string::npos ? std::string::npos : base.rfind('-', of - 1);
        const bool split = of != std::string::npos && dash != std::string::npos
                && of - dash == 6 && base.size() >= of + 4 + 5 + 5
                && base.compare(base.size() - 5, 5, ".gguf") == 0;
        if (!split) {
            return open_shard(model_path);
        }
        const std::string stem = base.substr(0, dash);
        const std::string total = base.substr(of + 4, 5);
        std::vector<std::string> names;
        for (const auto & e : std::filesystem::directory_iterator(dir)) {
            const std::string n = e.path().filename().string();
            if (n.size() != stem.size() + 1 + 5 + 4 + 5 + 5) continue;
            if (n.compare(0, stem.size(), stem) != 0) continue;
            if (n.compare(stem.size(), 1, "-") != 0) continue;
            if (n.compare(stem.size() + 1 + 5, 4, "-of-") != 0) continue;
            if (n.compare(stem.size() + 1 + 5 + 4, 5, total) != 0) continue;
            if (n.compare(n.size() - 5, 5, ".gguf") != 0) continue;
            names.push_back(e.path().string());
        }
        std::sort(names.begin(), names.end());
        if (names.empty()) { why = "no shards of " + base + " beside it"; return false; }
        for (const auto & n : names) {
            if (!open_shard(n)) return false;
        }
        return true;
    }

    // the tensor's bytes in whichever shard holds them, or nullptr
    const shard * find(const std::string & tensor_name, int64_t & tensor_id) const {
        for (const auto & s : shards) {
            const int64_t id = gguf_find_tensor(s.ctx, tensor_name.c_str());
            if (id >= 0) { tensor_id = id; return &s; }
        }
        return nullptr;
    }

    bool has_bias(int il) const {
        int64_t id = -1;
        const std::string name = "blk." + std::to_string(il) + ".ffn_gate_inp.bias";
        return find(name, id) != nullptr;
    }

    // the layer's ffn_gate_inp weight as f32 [n_expert][n_embd], row-major
    bool load_gate_inp(int il, uint32_t n_embd, uint32_t n_expert,
                       std::vector<float> & w, std::string & detail) const {
        const std::string name = "blk." + std::to_string(il) + ".ffn_gate_inp.weight";
        int64_t id = -1;
        const shard * s = find(name, id);
        if (s == nullptr) { detail = name + " is in no shard"; return false; }
        const enum ggml_type type = gguf_get_tensor_type(s->ctx, id);
        const int64_t * ne = gguf_get_tensor_ne(s->ctx, id);
        if (ne[0] != (int64_t) n_embd || ne[1] != (int64_t) n_expert) {
            detail = name + " is " + std::to_string(ne[0]) + "x" + std::to_string(ne[1])
                   + ", not n_embd x n_expert";
            return false;
        }
        const ggml_type_traits * tr = ggml_get_type_traits(type);
        if (tr == nullptr || tr->to_float == nullptr) {
            detail = name + " has type " + ggml_type_name(type) + " with no to_float";
            return false;
        }
        const size_t row_bytes = ggml_row_size(type, n_embd);
        const uint8_t * data = (const uint8_t *) s->base
                + gguf_get_data_offset(s->ctx) + gguf_get_tensor_offset(s->ctx, id);
        w.assign((size_t) n_embd * n_expert, 0.0f);
        std::vector<uint8_t> row(row_bytes);
        for (uint32_t j = 0; j < n_expert; ++j) {
            std::memcpy(row.data(), data + row_bytes * j, row_bytes);
            tr->to_float(row.data(), w.data() + (size_t) j * n_embd, n_embd);
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// the callback state

struct layer_capture {
    std::vector<uint16_t> in;     // graph_tokens rows x n_embd f16 bits
    std::vector<int32_t>  ids;    // graph_tokens rows x n_used ids
    bool has_in  = false;
    bool has_ids = false;
};

struct capture_state {
    lac1::geometry g;
    lac1::writer * out = nullptr;
    std::vector<layer_capture> layers;

    int32_t graph_tokens = -1;     // rows of the pending graph, -1 when none
    int32_t pos_cursor   = 0;      // next position to write
    uint8_t phase        = lac1::PHASE_PROMPT;
    std::vector<int32_t> queued_tokens;   // the current llama_decode call's token ids
    size_t queued_index  = 0;             // how many of them are written

    bool first_decode_check = true;   // the next decode graph is the self-checked one
    bool failed = false;
    std::string failure;

    const weights * w = nullptr;
};

bool widen_to_f32(const void * raw, enum ggml_type type, uint32_t n, std::vector<float> & f) {
    f.resize(n);
    if (type == GGML_TYPE_F32) {
        std::memcpy(f.data(), raw, sizeof(float) * n);
        return true;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, f.data(), n);
        return true;
    }
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (tr == nullptr || tr->to_float == nullptr) return false;
    tr->to_float(raw, f.data(), n);
    return true;
}

bool read_rows_f16(const ggml_tensor * t, uint32_t n_embd, uint32_t n_tokens,
                   std::vector<uint16_t> & bits) {
    if (t->ne[0] != (int64_t) n_embd || t->ne[1] != (int64_t) n_tokens) return false;
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    bits.assign((size_t) n_embd * n_tokens, 0);
    std::vector<uint8_t> raw(row_bytes);
    std::vector<float>   f;
    for (uint32_t k = 0; k < n_tokens; ++k) {
        ggml_backend_tensor_get(t, raw.data(), (size_t) k * t->nb[1], row_bytes);
        if (!widen_to_f32(raw.data(), t->type, n_embd, f)) return false;
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) (bits.data() + (size_t) k * n_embd),
                              (int64_t) n_embd);
    }
    return true;
}

bool read_ids(const ggml_tensor * t, uint32_t n_used, uint32_t n_tokens,
              std::vector<int32_t> & ids) {
    if (t->ne[0] != (int64_t) n_used || t->ne[1] != (int64_t) n_tokens) return false;
    if (t->ne[2] != 1 || t->ne[3] != 1) return false;
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    ids.assign((size_t) n_used * n_tokens, -1);
    std::vector<uint8_t> raw(row_bytes);
    for (uint32_t k = 0; k < n_tokens; ++k) {
        ggml_backend_tensor_get(t, raw.data(), (size_t) k * t->nb[1], row_bytes);
        for (uint32_t j = 0; j < n_used; ++j) {
            int64_t v = -1;
            switch (t->type) {
                case GGML_TYPE_I32: {
                    int32_t x; std::memcpy(&x, raw.data() + 4 * j, 4); v = x; break;
                }
                case GGML_TYPE_I64: {
                    int64_t x; std::memcpy(&x, raw.data() + 8 * j, 8); v = x; break;
                }
                default: return false;
            }
            if (v < 0 || v > 32767) return false;
            ids[(size_t) k * n_used + j] = (int32_t) v;
        }
    }
    return true;
}

// the first decode token's cell: captured input x the layer's gate weight
int self_check(capture_state & s, int il, const std::vector<float> & w,
               std::string & why) {
    const layer_capture & c = s.layers[il];
    std::vector<float> x(s.g.n_embd);
    ggml_fp16_to_fp32_row((const ggml_fp16_t *) c.in.data(), x.data(), (int64_t) s.g.n_embd);
    // the top two of the logits, then of the softmax: the same order
    int32_t best = -1, second = -1;
    float best_v = 0.0f, second_v = 0.0f;
    for (uint32_t j = 0; j < s.g.n_expert; ++j) {
        const float * row = w.data() + (size_t) j * s.g.n_embd;
        float dot = 0.0f;
        for (uint32_t k = 0; k < s.g.n_embd; ++k) dot += x[k] * row[k];
        if (best == -1 || dot > best_v) {
            second = best; second_v = best_v;
            best = (int32_t) j; best_v = dot;
        } else if (second == -1 || dot > second_v) {
            second = (int32_t) j; second_v = dot;
        }
    }
    if (c.ids.empty() || c.ids.size() < 2) { why = "fewer than two captured ids"; return 1; }
    if (best != c.ids[0] || second != c.ids[1]) {
        why = "captured ids " + std::to_string(c.ids[0]) + "," + std::to_string(c.ids[1])
            + ", recomputed " + std::to_string(best) + "," + std::to_string(second)
            + " (logits " + std::to_string(best_v) + "," + std::to_string(second_v) + ")";
        return 1;
    }
    return 0;
}

// Write the pending graph's rows and reset; positions come from pos_cursor,
// token ids from the current call's queue. Returns 0, or an error code.
int flush_graph(capture_state & s) {
    if (s.graph_tokens < 0) return 0;
    const uint32_t n_layer = s.g.n_layer;
    std::string err;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const layer_capture & c = s.layers[il];
        if (!c.has_in || !c.has_ids) {
            err = "layer " + std::to_string(il) + " has no captured input or ids";
            break;
        }
        if ((int32_t) (c.in.size() / s.g.n_embd) != s.graph_tokens ||
            (int32_t) (c.ids.size() / s.g.n_used) != s.graph_tokens) {
            err = "layer " + std::to_string(il) + " has a row count other than the graph's";
            break;
        }
    }
    if (!err.empty()) {
        s.failure = err;
        s.failed = true;
        return 2;
    }
    if (s.queued_index + (size_t) s.graph_tokens > s.queued_tokens.size()) {
        s.failure = "the graph's token count runs past the call's token queue";
        s.failed = true;
        return 2;
    }

    if (s.phase == lac1::PHASE_DECODE && s.first_decode_check && s.graph_tokens == 1) {
        int bad = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (s.w->has_bias((int) il)) {
                std::fprintf(stderr,
                    "SELF-CHECK: blk.%u.ffn_gate_inp.bias is present; this tool's check has no bias term\n",
                    il);
                s.failure = "a router bias is not handled by the self-check";
                s.failed = true;
                return 4;
            }
            std::vector<float> w;
            std::string detail;
            if (!s.w->load_gate_inp((int) il, s.g.n_embd, s.g.n_expert, w, detail)) {
                std::fprintf(stderr, "SELF-CHECK: layer %u: %s\n", il, detail.c_str());
                s.failure = "the gate weight could not be read: " + detail;
                s.failed = true;
                return 4;
            }
            std::string why;
            if (self_check(s, (int) il, w, why) != 0) {
                std::fprintf(stderr, "SELF-CHECK FAIL layer %u: %s\n", il, why.c_str());
                ++bad;
            }
        }
        if (bad > 0) {
            std::fprintf(stderr, "SELF-CHECK: %d of %u layers' top-2 ranks differ from the captured ids\n",
                         bad, n_layer);
            s.failure = "the captured ids do not match the input x ffn_gate_inp";
            s.failed = true;
            return 4;
        }
        std::fprintf(stderr, "SELF-CHECK: first decode token, %u layers, top-2 ranks exact\n", n_layer);
        s.first_decode_check = false;
    }

    std::vector<uint16_t> inputs((size_t) n_layer * s.g.n_embd);
    std::vector<int16_t>  ids((size_t) n_layer * s.g.n_used);
    for (int32_t k = 0; k < s.graph_tokens; ++k) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            std::memcpy(inputs.data() + (size_t) il * s.g.n_embd,
                        s.layers[il].in.data() + (size_t) k * s.g.n_embd,
                        sizeof(uint16_t) * s.g.n_embd);
            for (uint32_t j = 0; j < s.g.n_used; ++j) {
                ids[(size_t) il * s.g.n_used + j] =
                    (int16_t) s.layers[il].ids[(size_t) k * s.g.n_used + j];
            }
        }
        const int32_t pos = s.pos_cursor + k;
        const int32_t token = s.queued_tokens[s.queued_index + (size_t) k];
        if (!s.out->write_token(pos, token, s.phase, inputs.data(), ids.data())) {
            s.failure = s.out->error() ? s.out->error() : "the write failed";
            s.failed = true;
            return 2;
        }
    }
    s.pos_cursor += s.graph_tokens;
    s.queued_index += (size_t) s.graph_tokens;
    s.graph_tokens = -1;
    for (auto & c : s.layers) {
        c.in.clear(); c.ids.clear();
        c.has_in = false; c.has_ids = false;
    }
    return 0;
}

bool parse_layer_name(const char * name, const char * prefix, int & il) {
    const size_t pl = std::strlen(prefix);
    if (std::strncmp(name, prefix, pl) != 0) return false;
    const char * rest = name + pl;
    if (*rest == '\0') return false;
    char * end = nullptr;
    const long v = std::strtol(rest, &end, 10);
    if (*end != '\0') return false;
    il = (int) v;
    return true;
}

bool capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * s = (capture_state *) user_data;
    int il = -1;
    const bool is_logits = parse_layer_name(t->name, "ffn_moe_logits-", il);
    const bool is_topk   = !is_logits && parse_layer_name(t->name, "ffn_moe_topk-", il);
    if (!is_logits && !is_topk) return false;

    if (ask) return true;   // compute this node, synchronise, then ask=false

    if (s->failed) return true;
    if (il < 0 || (uint32_t) il >= s->g.n_layer) {
        s->failure = std::string("a tensor named for layer ") + t->name + " outside 0..n_layer-1";
        s->failed = true;
        return true;
    }

    if (is_logits) {
        const ggml_tensor * in = t->src[1];
        if (in == nullptr || t->src[0] == in) {
            s->failure = std::string("ffn_moe_logits-") + std::to_string(il) + " has no distinct input operand";
            s->failed = true;
            return true;
        }
        if (in->ne[0] != (int64_t) s->g.n_embd || in->ne[2] != 1 || in->ne[3] != 1) {
            s->failure = std::string("the router input at layer ") + std::to_string(il)
                       + " is not [n_embd, n_tokens]";
            s->failed = true;
            return true;
        }
        if (il == 0) {
            // a new graph: flush the pending one first
            if (s->graph_tokens >= 0 && flush_graph(*s) != 0) return true;
            s->graph_tokens = (int32_t) in->ne[1];
            for (auto & c : s->layers) {
                c.in.clear(); c.ids.clear();
                c.has_in = false; c.has_ids = false;
            }
        }
        layer_capture & c = s->layers[il];
        if (c.has_in) {
            s->failure = "layer " + std::to_string(il) + " was captured twice in one graph";
            s->failed = true;
            return true;
        }
        std::vector<uint16_t> bits;
        if (!read_rows_f16(in, s->g.n_embd, (uint32_t) s->graph_tokens, bits)) {
            s->failure = std::string("cannot read the router input at layer ") + std::to_string(il);
            s->failed = true;
            return true;
        }
        c.in = std::move(bits);
        c.has_in = true;
    } else {
        if (s->graph_tokens < 0) {
            s->failure = std::string("ffn_moe_topk-") + std::to_string(il) + " arrived before its logits";
            s->failed = true;
            return true;
        }
        layer_capture & c = s->layers[il];
        if (c.has_ids) {
            s->failure = "layer " + std::to_string(il) + " ids were captured twice in one graph";
            s->failed = true;
            return true;
        }
        std::vector<int32_t> ids;
        if (!read_ids(t, s->g.n_used, (uint32_t) s->graph_tokens, ids)) {
            s->failure = std::string("cannot read ffn_moe_topk at layer ") + std::to_string(il);
            s->failed = true;
            return true;
        }
        c.ids = std::move(ids);
        c.has_ids = true;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) return 2;

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;          // CPU only
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", basename_of(a.model).c_str());
        llama_backend_free();
        return 1;
    }

    const uint32_t n_layer  = (uint32_t) llama_model_n_layer(model);
    const uint32_t n_embd   = (uint32_t) llama_model_n_embd(model);
    const uint32_t n_expert = (uint32_t) llama_model_n_expert(model);
    uint32_t n_used = 0;
    if (!model_n_expert_used(model, n_used)) {
        std::fprintf(stderr, "the GGUF has no <arch>.expert_used_count; the LAC1 header needs the top-k\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // the prompt: tokenised the way the server does (special tokens parsed)
    std::FILE * pf = std::fopen(a.prompt.c_str(), "rb");
    if (pf == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", basename_of(a.prompt).c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::string text;
    char buf[1 << 16];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof(buf), pf)) > 0) text.append(buf, got);
    std::fclose(pf);
    if (text.empty()) {
        std::fprintf(stderr, "%s is empty\n", basename_of(a.prompt).c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens(text.size() + 1);
    int n_prompt = llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                                  prompt_tokens.data(), (int32_t) prompt_tokens.size(),
                                  /*add_special*/ true, /*parse_special*/ true);
    if (n_prompt < 0) {
        prompt_tokens.resize((size_t) -n_prompt);
        n_prompt = llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                                  prompt_tokens.data(), (int32_t) prompt_tokens.size(),
                                  true, true);
    }
    if (n_prompt <= 0) {
        std::fprintf(stderr, "the prompt does not tokenise\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    prompt_tokens.resize((size_t) n_prompt);

    const int32_t n_ctx = n_prompt + a.decode + 4;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = (uint32_t) n_ctx;
    cp.n_batch   = (uint32_t) std::min<int32_t>(n_ctx, 2048);
    cp.n_ubatch  = cp.n_batch;
    cp.n_threads = (int32_t) std::max(1u, std::thread::hardware_concurrency());
    cp.n_threads_batch = cp.n_threads;
    cp.embeddings = false;
    cp.no_perf = true;

    // the run's weights for the self-check, before the context exists
    weights w;
    if (!w.open(a.model)) {
        std::fprintf(stderr, "SELF-CHECK: %s\n", w.why.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    capture_state st;
    st.g.n_layer  = n_layer;
    st.g.n_embd   = n_embd;
    st.g.n_expert = n_expert;
    st.g.n_used   = n_used;
    st.layers.resize(n_layer);   // one slot a layer: capture_cb indexes it by the tensor's layer
    st.w = &w;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot create the context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // the output: the capture file and its sidecar, before anything decodes
    lac1::writer out;
    st.out = &out;
    std::string info;
    if (!out.open(a.out.c_str(), st.g)) {
        std::fprintf(stderr, "%s: the capture file cannot be opened\n", basename_of(a.out).c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::FILE * inf = std::fopen((a.out + ".info").c_str(), "wb");
    if (inf != nullptr) {
        std::fprintf(inf, "capture LAC1\nprompt %s\ndecode %d\n",
                     basename_of(a.prompt).c_str(), a.decode);
        for (const auto & s : w.shards) std::fprintf(inf, "model %s\n", basename_of(s.name).c_str());
        std::fclose(inf);
    } else {
        std::fprintf(stderr, "WARN: cannot write %s.info\n", basename_of(a.out).c_str());
    }

    const int32_t prompt_len = (int32_t) prompt_tokens.size();
    llama_batch batch = llama_batch_init(cp.n_batch, 0, 1);
    auto decode_tokens = [&](const std::vector<llama_token> & tokens, int32_t pos0, uint8_t phase) -> bool {
        st.phase = phase;
        st.queued_tokens.assign(tokens.begin(), tokens.end());
        st.queued_index = 0;
        const size_t chunk = (size_t) cp.n_batch;   // llama_decode's graph size
        for (size_t off = 0; off < tokens.size(); off += chunk) {
            const size_t n = std::min(chunk, tokens.size() - off);
            batch.n_tokens = (int32_t) n;
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                batch.token[i]     = tokens[off + (size_t) i];
                batch.pos[i]       = pos0 + (int32_t) off + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                // every row an output: llama.cpp computes the last layer only for the output rows
                // (inp_out_ids), so with one output the last layer's router saw one token
                batch.logits[i]    = true;
            }
            if (llama_decode(ctx, batch) != 0) {
                st.failure = "llama_decode failed";
                st.failed = true;
                return false;
            }
            if (flush_graph(st) != 0 || st.failed) return false;
        }
        return true;
    };

    bool ok = decode_tokens(prompt_tokens, 0, lac1::PHASE_PROMPT);

    // greedy decode: sample, then decode the sampled token
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    for (int32_t i = 0; ok && i < a.decode; ++i) {
        const llama_token t = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, t);
        std::vector<llama_token> one{t};
        ok = decode_tokens(one, prompt_len + i, lac1::PHASE_DECODE);
    }
    llama_sampler_free(smpl);

    if (out.records == 0 && !st.failed) {   // no graph ever wrote a token (an earlier failure keeps its own text)
        st.failure = "no token was captured";
        st.failed = true;
    }

    out.close();
    const uint64_t tokens_written = out.records;
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (!ok || st.failed) {
        std::fprintf(stderr, "FAIL: %s\n", st.failure.c_str());
        std::fprintf(stderr, "%llu tokens were written to %s\n",
                     (unsigned long long) tokens_written, basename_of(a.out).c_str());
        return 3;
    }
    std::fprintf(stderr,
        "captured %llu tokens (%d prompt, %d decode), %llu bytes, %llu bytes a token, %u layers, "
        "n_embd %u, %u of %u experts\n",
        (unsigned long long) tokens_written, n_prompt, a.decode,
        (unsigned long long) (lac1::HEADER_BYTES + tokens_written * lac1::token_bytes(st.g)),
        (unsigned long long) lac1::token_bytes(st.g),
        n_layer, n_embd, st.g.n_used, n_expert);
    return 0;
}

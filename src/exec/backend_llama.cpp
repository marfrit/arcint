// The libllama executor (0.5.3.1459): arcint without OpenVINO. The GGUF is
// the model; ggml's OpenCL backend runs it on the Arc card (each Arc card is
// its own OpenCL platform under the NEO driver). arcint keeps the server,
// the chat template rendering, the sampler, stop handling and the lanes;
// llama.cpp keeps the weights, the tokenizer, the attention KV and the
// recurrent (gated delta-net) state, one sequence per lane.
#include "exec/backend.h"

#ifdef ARCINT_LLAMA

#include <CL/cl.h>
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <minja/chat-template.hpp>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "core/sampler.h"
#include "exec/chat_json.h"
#include "util/log.h"

namespace lgc {
namespace {

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

// The OpenCL platform holding the requested card: "GPU.N" is the N-th GPU in
// OpenCL's enumeration, anything else a substring of the device name ("B60",
// "A770"). -1 when nothing matches.
int opencl_platform_for(const std::string& want, std::string& name_out) {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return -1;
    std::vector<cl_platform_id> ps(np);
    clGetPlatformIDs(np, ps.data(), nullptr);
    int ordinal = -1;
    if (want.rfind("GPU", 0) == 0) {
        ordinal = 0;
        if (want.size() > 4 && want[3] == '.') ordinal = std::atoi(want.c_str() + 4);
    }
    int seen = 0;
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        cl_device_id d = nullptr;
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 1, &d, nullptr);
        char name[256] = {};
        clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
        const bool hit = ordinal >= 0 ? seen == ordinal : std::string(name).find(want) != std::string::npos;
        if (hit) {
            name_out = name;
            return static_cast<int>(i);
        }
        ++seen;
    }
    return -1;
}

void route_llama_log(ggml_log_level level, const char* text, void*) {
    std::string s(text != nullptr ? text : "");
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty()) return;
    if (level == GGML_LOG_LEVEL_ERROR) log::error("llama", "%s", s.c_str());
    else if (level == GGML_LOG_LEVEL_WARN) log::warn("llama", "%s", s.c_str());
    else log::verbose("llama", "%s", s.c_str());
}

class LlamaTokenizer final : public Tokenizer {
public:
    explicit LlamaTokenizer(const llama_vocab* v) : v_(v) {}

    std::vector<int> encode(std::string_view text) override {
        const int need = -llama_tokenize(v_, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
                                         /*add_special=*/false, /*parse_special=*/true);
        std::vector<llama_token> t(static_cast<size_t>(std::max(need, 0)));
        if (need > 0)
            llama_tokenize(v_, text.data(), static_cast<int32_t>(text.size()), t.data(), need, false, true);
        return std::vector<int>(t.begin(), t.end());
    }

    std::string decode_one(int id) override {
        char buf[256];
        const int n = llama_token_to_piece(v_, id, buf, sizeof(buf), 0, /*special=*/true);
        if (n >= 0) return std::string(buf, static_cast<size_t>(n));
        std::string s(static_cast<size_t>(-n), '\0');
        llama_token_to_piece(v_, id, s.data(), -n, 0, true);
        return s;
    }

    std::string decode(const std::vector<int>& ids) override {
        std::string out;
        for (int id : ids) out += decode_one(id);
        return out;
    }

    int eos_id() const override { return llama_vocab_eos(v_); }

private:
    const llama_vocab* v_;
};

class LlamaBackend final : public Backend {
public:
    LlamaBackend(const Config& cfg, int n_ctx) {
        std::string card;
        const int platform = opencl_platform_for(cfg.device, card);
        if (platform < 0)
            throw std::runtime_error(log::format("no OpenCL GPU matches --device %s", cfg.device.c_str()));
        setenv("GGML_OPENCL_PLATFORM", std::to_string(platform).c_str(), 1);
        setenv("GGML_OPENCL_DEVICE", "0", 1);
        llama_log_set(route_llama_log, nullptr);
        llama_backend_init();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers       = 999;
        // --llama-cpu-moe: the first N layers' experts in host memory, as
        // llama.cpp's --n-cpu-moe (common/common.h llm_add_n_cpu_ffn_overrides).
        // They are memory-mapped from the GGUF: the OpenCL device declares no
        // mmap support, and the default load mode would read the whole model
        // into anonymous memory (OOM for Flash-Next's 84 GiB).
        for (int i = 0; i < cfg.llama_cpu_moe; ++i)
            cpu_moe_patterns_.push_back(log::format("blk\\.%d\\.ffn_(up|down|gate|gate_up)_(ch|)exps", i));
        for (const std::string& pat : cpu_moe_patterns_)
            buft_overrides_.push_back({ pat.c_str(), ggml_backend_cpu_buffer_type() });
        if (!buft_overrides_.empty()) {
            buft_overrides_.push_back({ nullptr, nullptr });
            mp.tensor_buft_overrides = buft_overrides_.data();
            mp.load_mode             = LLAMA_LOAD_MODE_MMAP;
        }
        const auto t_load     = clock_type::now();
        model_                = llama_model_load_from_file(cfg.gguf_path.c_str(), mp);
        if (model_ == nullptr)
            throw std::runtime_error(log::format("llama.cpp could not load %s", cfg.gguf_path.c_str()));

        // The models this engine serves, by the GGUF's own architecture; the
        // allowlist rule (DESIGN §3.1) holds here as on the OpenVINO path.
        char arch[64] = {};
        llama_model_meta_val_str(model_, "general.architecture", arch, sizeof(arch));
        const std::string a(arch);
        if (a != "qwen35" && a != "qwen35moe" && a != "qwen4exp")
            throw std::runtime_error(log::format("%s is a '%s' model; --engine llama serves qwen35, qwen35moe and qwen4exp",
                                                 cfg.gguf_path.c_str(), arch));
        vocab_     = llama_model_get_vocab(model_);
        tokenizer_ = std::make_unique<LlamaTokenizer>(vocab_);
        n_vocab_   = static_cast<size_t>(llama_vocab_n_tokens(vocab_));

        lanes_      = std::max(1, cfg.parallel);
        n_ctx_      = n_ctx > 0 ? n_ctx : std::min(llama_model_n_ctx_train(model_), 32768);
        n_batch_    = cfg.prefill_chunk > 0 ? cfg.prefill_chunk : 2048;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = static_cast<uint32_t>(n_ctx_) * static_cast<uint32_t>(lanes_);
        cp.n_seq_max       = static_cast<uint32_t>(lanes_);
        cp.n_batch         = static_cast<uint32_t>(n_batch_);
        cp.n_ubatch        = static_cast<uint32_t>(std::min(n_batch_, 512));
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        // CPU threads: the CPU-side experts' decode is memory-bound, and the
        // physical cores beat the SMT threads (measured-here, Flash-Next on a
        // 8-core/16-thread host: 11.4 t/s at 8 threads, 9.0 at 16)
        const int threads  = cfg.llama_threads > 0 ? cfg.llama_threads
                                                    : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        ctx_               = llama_init_from_model(model_, cp);
        if (ctx_ == nullptr) throw std::runtime_error("llama.cpp could not create a context");
        slot_tokens_.resize(static_cast<size_t>(lanes_));

        const char* tmpl = llama_model_chat_template(model_, nullptr);
        if (tmpl == nullptr) throw std::runtime_error(log::format("%s carries no chat template", cfg.gguf_path.c_str()));
        template_src_ = tmpl;
        const char* bos = llama_vocab_get_text(vocab_, llama_vocab_bos(vocab_));
        const char* eos = llama_vocab_get_text(vocab_, llama_vocab_eos(vocab_));
        template_ = std::make_unique<minja::chat_template>(template_src_, bos ? bos : "", eos ? eos : "");

        char desc[128] = {};
        llama_model_desc(model_, desc, sizeof(desc));
        const std::string file = std::filesystem::path(cfg.gguf_path).filename().string();
        status_.id          = file;
        status_.served_id   = cfg.served_model_name.empty() ? file : cfg.served_model_name;
        status_.loaded      = true;
        status_.n_ctx       = n_ctx_;
        status_.n_ctx_train = llama_model_n_ctx_train(model_);
        status_.n_layer     = llama_model_n_layer(model_);
        status_.weights_bytes = llama_model_size(model_);
        status_.kv_precision  = "f16";
        SamplerDefaults d;
        d.temperature        = 0.7f;
        d.top_p              = 0.8f;
        d.top_k              = 20;
        d.repetition_penalty = 1.05f;
        d.presence_penalty   = 0.0f;
        d.provenance         = "provisional";
        if (auto err = apply_operator_defaults(cfg, d)) throw std::runtime_error(*err);
        status_.sampler_defaults = d;
        log::info("load", "llama.cpp %s (%s, %.2f GiB) on %s, OpenCL platform %d, in %.1f s; %d lane%s x %d ctx",
                  desc, arch, static_cast<double>(llama_model_size(model_)) / (1u << 30), card.c_str(), platform,
                  seconds_since(t_load), lanes_, lanes_ == 1 ? "" : "s", n_ctx_);
    }

    ~LlamaBackend() override {
        if (ctx_ != nullptr) llama_free(ctx_);
        if (model_ != nullptr) llama_model_free(model_);
    }

    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return *tokenizer_; }

    json template_caps() const override {
        const minja::chat_template_caps& c = template_->original_caps();
        return json{{"supports_tools", c.supports_tools},
                    {"supports_tool_calls", c.supports_tool_calls},
                    {"supports_tool_responses", c.supports_tool_responses},
                    {"supports_system_role", c.supports_system_role},
                    {"supports_parallel_tool_calls", c.supports_parallel_tool_calls},
                    {"supports_object_arguments", c.requires_object_arguments},
                    {"supports_string_content", !c.requires_typed_content},
                    {"supports_typed_content", c.requires_typed_content},
                    {"supports_preserve_reasoning", template_src_.find("reasoning_content") != std::string::npos}};
    }

    std::string render_chat(const ChatRequest& req) const override {
        minja::chat_template_inputs inputs;
        inputs.messages = chat_messages_json(req, template_->original_caps().requires_object_arguments);
        inputs.tools    = chat_tools_json(req);
        inputs.add_generation_prompt = true;
        if (req.has_enable_thinking) inputs.extra_context = json{{"enable_thinking", req.enable_thinking}};
        minja::chat_template_options opts;
        opts.apply_polyfills = false;   // the GGUF's template is the contract (§3.7)
        return template_->apply(inputs, opts);
    }

    FinishReason generate(const GenerationInput& in, int slot, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        const int seq = std::min(std::max(slot, 0), lanes_ - 1);
        const std::vector<int> prompt =
            in.prompt_ids.empty() ? tokenizer_->encode(in.prompt) : in.prompt_ids;
        stats.prompt_tokens = static_cast<int>(prompt.size());
        if (prompt.empty()) return FinishReason::Stop;
        if (static_cast<int>(prompt.size()) >= n_ctx_) {
            log::warn("slot", "prompt of %zu tokens does not fit n_ctx %d", prompt.size(), n_ctx_);
            return FinishReason::Length;
        }

        uint64_t seed = in.sampler.seed;
        if (!in.sampler.seeded) {
            std::random_device rd;
            seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        }
        Sampler sampler(in.sampler, seed);
        sampler.set_prompt(prompt);

        std::vector<int>&  have = slot_tokens_[static_cast<size_t>(seq)];
        std::vector<float> logits(n_vocab_);

        // ------------------------------------------------------------ prefill
        const auto t_prefill = clock_type::now();
        {
            std::lock_guard<std::mutex> lk(mu_);
            // The longest prefix this lane still holds is reused; one token is
            // always run, for its logits. A recurrent state cannot step back,
            // so where llama.cpp refuses the partial removal the lane starts over.
            size_t common = 0;
            while (common < have.size() && common < prompt.size() && have[common] == prompt[common]) ++common;
            if (common == prompt.size()) --common;
            llama_memory_t mem = llama_get_memory(ctx_);
            if (common < have.size() && !llama_memory_seq_rm(mem, seq, static_cast<llama_pos>(common), -1)) {
                llama_memory_seq_rm(mem, seq, -1, -1);
                common = 0;
            }
            have.resize(common);
            stats.cache_hit_tokens = static_cast<int>(common);
            for (size_t at = common; at < prompt.size(); at += static_cast<size_t>(n_batch_)) {
                const size_t n    = std::min(prompt.size() - at, static_cast<size_t>(n_batch_));
                const bool   last = at + n == prompt.size();
                if (!decode_locked(seq, prompt.data() + at, n, at, last))
                    throw std::runtime_error("llama_decode failed during prefill");
                have.insert(have.end(), prompt.begin() + static_cast<long>(at),
                            prompt.begin() + static_cast<long>(at + n));
            }
            copy_logits_locked(logits);
        }
        stats.prefill_seconds = seconds_since(t_prefill);

        // ------------------------------------------------------------- decode
        const auto t_decode = clock_type::now();
        FinishReason reason = FinishReason::Stop;
        const auto   t_s0   = clock_type::now();
        int          next   = sampler.sample(logits.data(), n_vocab_);
        stats.decode_sample_seconds += seconds_since(t_s0);
        auto is_stop_token = [&](int tok) {
            if (!in.sampler.ignore_eos && llama_vocab_is_eog(vocab_, tok)) return true;
            return std::find(in.sampler.stop_token_ids.begin(), in.sampler.stop_token_ids.end(), tok) !=
                   in.sampler.stop_token_ids.end();
        };
        while (true) {
            if (is_stop_token(next)) break;
            if (in.sampler.max_tokens >= 0 && stats.completion_tokens >= in.sampler.max_tokens) {
                reason = FinishReason::Length;
                break;
            }
            if (static_cast<int>(have.size()) + 1 >= n_ctx_) {
                reason = FinishReason::Length;
                break;
            }
            ++stats.completion_tokens;
            const auto    t_emit = clock_type::now();
            const Control ctl    = on_piece(tokenizer_->decode_one(next), next);
            stats.decode_emit_seconds += seconds_since(t_emit);
            sampler.observe(next);
            if (ctl != Control::Continue) {
                reason = ctl == Control::Cancel ? FinishReason::Abort : FinishReason::Stop;
                break;
            }
            const auto t_fwd = clock_type::now();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!decode_locked(seq, &next, 1, have.size(), true))
                    throw std::runtime_error("llama_decode failed during decode");
                have.push_back(next);
                copy_logits_locked(logits);
            }
            stats.decode_forward_seconds += seconds_since(t_fwd);
            const auto t_s = clock_type::now();
            next           = sampler.sample(logits.data(), n_vocab_);
            stats.decode_sample_seconds += seconds_since(t_s);
        }
        stats.decode_seconds = seconds_since(t_decode);
        return reason;
    }

private:
    // One llama_decode of n tokens of `seq` at positions [pos, pos + n), with
    // logits for the last one when `want_last`. Caller holds mu_.
    bool decode_locked(int seq, const int* toks, size_t n, size_t pos, bool want_last) {
        llama_batch b = llama_batch_init(static_cast<int32_t>(n), 0, 1);
        for (size_t i = 0; i < n; ++i) {
            b.token[i]     = toks[i];
            b.pos[i]       = static_cast<llama_pos>(pos + i);
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq;
            b.logits[i]    = want_last && i + 1 == n;
        }
        b.n_tokens  = static_cast<int32_t>(n);
        const int r = llama_decode(ctx_, b);
        llama_batch_free(b);
        if (r != 0) log::error("llama", "llama_decode returned %d", r);
        return r == 0;
    }

    void copy_logits_locked(std::vector<float>& out) {
        const float* l = llama_get_logits_ith(ctx_, -1);
        if (l == nullptr) throw std::runtime_error("llama.cpp returned no logits");
        std::copy(l, l + n_vocab_, out.begin());
    }

    // --llama-cpu-moe's tensor patterns, alive for the load
    std::vector<std::string>                       cpu_moe_patterns_;
    std::vector<llama_model_tensor_buft_override> buft_overrides_;
    llama_model*                          model_ = nullptr;
    llama_context*                        ctx_   = nullptr;
    const llama_vocab*                    vocab_ = nullptr;
    std::unique_ptr<LlamaTokenizer>       tokenizer_;
    std::unique_ptr<minja::chat_template> template_;
    std::string                           template_src_;
    ModelStatus                           status_;
    size_t                                n_vocab_ = 0;
    int                                   lanes_   = 1;
    int                                   n_ctx_   = 0;
    int                                   n_batch_ = 2048;
    std::mutex                            mu_;    // llama_context is not thread-safe: one call at a time
    std::vector<std::vector<int>>         slot_tokens_;
};

}  // namespace

std::unique_ptr<Backend> make_llama_backend(const Config& cfg, int n_ctx) {
    return std::make_unique<LlamaBackend>(cfg, n_ctx);
}

}  // namespace lgc

#endif  // ARCINT_LLAMA

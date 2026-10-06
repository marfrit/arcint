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
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "exec/llama_spec.h"
#include "exec/verify_walk.h"
#include "util/log.h"

namespace lgc {
namespace {

using json = nlohmann::json;

// --llama-kv's names (config.cpp admits only these)
ggml_type kv_type(const std::string& t) {
    return t == "q8_0" ? GGML_TYPE_Q8_0 : t == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;   // q4_0: V only
}

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
    // A context checkpoint (--llama-checkpoints): what a lane holds after
    // n_tokens tokens
    struct Checkpoint {
        size_t               n_tokens = 0;
        uint64_t             req      = 0;   // the prefill that made it
        std::vector<uint8_t> state;          // the memory's PARTIAL_ONLY state
        std::vector<float>   dft_row;        // the drafter's carried row, --llama-mtp
    };

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
        mp.load_mtp           = cfg.llama_mtp > 0;   // the GGUF's MTP layer, for --llama-mtp
        // --llama-expert-cache: slots on the card for the experts --llama-cpu-moe
        // keeps in host memory (contrib/llama.cpp patch 0021)
#ifdef ARCINT_LLAMA_EXPERT_CACHE
        expert_profile_          = cfg.llama_expert_profile;
        mp.expert_cache_bytes    = static_cast<size_t>(cfg.llama_expert_cache_mib) << 20;
        mp.expert_cache_profile  = expert_profile_.empty() ? nullptr : expert_profile_.c_str();
        // the experts are computed on the card (slots) and read by it from the
        // bank: the CPU's repacked copies would only cost host memory
        if (cfg.llama_expert_cache_mib > 0) mp.use_extra_bufts = false;
#else
        if (cfg.llama_expert_cache_mib > 0)
            throw std::runtime_error("--llama-expert-cache: this build's llama.cpp lacks contrib/llama.cpp patch 0021");
#endif
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
#ifdef ARCINT_LLAMA_EXPERT_CACHE
        if (cfg.llama_expert_cache_mib > 0) {
            llama_expert_cache_info ec{};
            if (!llama_model_expert_cache_info(model_, &ec))
                throw std::runtime_error("--llama-expert-cache: llama.cpp built no expert cache (no MoE layer in host memory?)");
            log::info("load", "expert cache: %lld slots, %.2f GiB on the card, the other experts %.2f GiB in host memory",
                      static_cast<long long>(ec.slots), static_cast<double>(ec.bytes) / (1u << 30),
                      static_cast<double>(ec.bank_bytes) / (1u << 30));
        }
#endif

        // The models this engine serves, by the GGUF's own architecture; the
        // allowlist rule (DESIGN §3.1) holds here as on the OpenVINO path.
        // 'llama' is admitted at one geometry only, the Mistral Small 24B
        // family's: served for creative writing with Mistral Small 3.2's
        // finetune Cydonia 24B (operator, 2026-10-04). Others of the family
        // load too; their tool calls are not parsed (Qwen formats only,
        // src/core/toolcall.h). Their think blocks are split: <thinking>
        // (Cydonia) and [THINK] (Magistral), see ModelStatus::think_tags_extended
        char arch[64] = {};
        llama_model_meta_val_str(model_, "general.architecture", arch, sizeof(arch));
        const std::string a(arch);
        vocab_     = llama_model_get_vocab(model_);
        char head_dim[16] = {};
        llama_model_meta_val_str(model_, "llama.attention.key_length", head_dim, sizeof(head_dim));
        const bool mistral_small_24b = a == "llama" && llama_model_n_layer(model_) == 40 && llama_model_n_embd(model_) == 5120 &&
                                       llama_model_n_head(model_) == 32 && llama_model_n_head_kv(model_) == 8 &&
                                       std::string(head_dim) == "128" && llama_vocab_n_tokens(vocab_) == 131072;
        mistral_small_24b_ = mistral_small_24b;
        if (a != "qwen35" && a != "qwen35moe" && a != "qwen4exp" && !mistral_small_24b)
            throw std::runtime_error(log::format("%s is a '%s' model; --engine llama serves qwen35, qwen35moe, qwen4exp "
                                                 "and Mistral Small 3.2 24B ('llama', 40 x 5120, 32/8 heads)",
                                                 cfg.gguf_path.c_str(), arch));
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
        cp.type_k          = kv_type(cfg.llama_kv_k);
        cp.type_v          = kv_type(cfg.llama_kv_v);
        // CPU threads: the CPU-side experts' decode is memory-bound, and the
        // physical cores beat the SMT threads (measured-here, Flash-Next on a
        // 8-core/16-thread host: 11.4 t/s at 8 threads, 9.0 at 16)
        const int threads  = cfg.llama_threads > 0 ? cfg.llama_threads
                                                    : std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2);
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        // --llama-mtp: per-token snapshots of the recurrent state, so a
        // rejected draft rolls back on the device (llama_memory_seq_rm)
        if (cfg.llama_mtp > 0) cp.n_rs_seq = static_cast<uint32_t>(cfg.llama_mtp);
        // logits rows: the last token a step, or a verify's 1 + drafts, per
        // lane (as llama.cpp's speculative loop sets them). Unbounded,
        // llama.cpp reserves them for a whole ubatch, vocabulary-wide: the
        // coder with its MTP layer then no longer fit the A770 (engine
        // resets, measured-here)
        cp.n_outputs_max_per_seq = static_cast<uint32_t>(1 + std::max(cfg.llama_mtp, 0));
        cp.n_outputs_max         = cp.n_outputs_max_per_seq * static_cast<uint32_t>(lanes_);
        ctx_               = llama_init_from_model(model_, cp);
        if (ctx_ == nullptr) throw std::runtime_error("llama.cpp could not create a context");
        if (cfg.llama_mtp > 0) {
            std::string err;
            spec_ = make_llama_mtp(model_, ctx_, cfg.llama_mtp, lanes_, n_batch_, static_cast<int>(cp.n_ubatch),
                                   threads, cfg.gguf_path, cfg.llama_mtp_vocab, cp.type_k, cp.type_v, err);
            if (!spec_) throw std::runtime_error(log::format("--llama-mtp %d: %s", cfg.llama_mtp, err.c_str()));
            n_draft_ = cfg.llama_mtp;
        }
        status_.mtp_enabled = n_draft_ > 0;
        slot_tokens_.resize(static_cast<size_t>(lanes_));
        // context checkpoints only where llama.cpp makes them: a memory that
        // cannot remove a partial sequence (recurrent, or the recurrent part
        // of a hybrid). An attention-only cache trims, and its PARTIAL_ONLY
        // state would be the whole KV
        ckpt_max_  = cfg.llama_checkpoints;
        ckpt_step_ = static_cast<size_t>(std::max(cfg.llama_checkpoint_step, 1));
        ckpt_on_   = ckpt_max_ > 0 && (llama_model_is_hybrid(model_) || llama_model_is_recurrent(model_));
        n_ubatch_  = static_cast<int>(cp.n_ubatch);
        ckpts_.resize(static_cast<size_t>(lanes_));

        const char* tmpl = llama_model_chat_template(model_, nullptr);
        if (tmpl == nullptr) throw std::runtime_error(log::format("%s carries no chat template", cfg.gguf_path.c_str()));
        template_src_ = tmpl;
        const char* bos = llama_vocab_get_text(vocab_, llama_vocab_bos(vocab_));
        const char* eos = llama_vocab_get_text(vocab_, llama_vocab_eos(vocab_));
        template_ = std::make_unique<minja::chat_template>(template_src_, bos ? bos : "", eos ? eos : "");
        if (ckpt_on_) {
            user_start_ = tokenizer_->encode(user_start_text());
            log::info("load", "context checkpoints: up to %d a lane, %zu tokens apart, user messages %s", ckpt_max_,
                      ckpt_step_, user_start_.empty() ? "not found in the template" : "found");
        }

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
        status_.think_tags_extended = mistral_small_24b_;
        SamplerDefaults d;
        d.temperature        = 0.7f;
        d.top_p              = 0.8f;
        d.top_k              = 20;
        d.repetition_penalty = 1.05f;
        d.presence_penalty   = 0.0f;
        d.provenance         = "provisional";
        if (auto err = apply_operator_defaults(cfg, d)) throw std::runtime_error(*err);
        status_.sampler_defaults = d;
        log::info("load", "llama.cpp %s (%s, %.2f GiB) on %s, OpenCL platform %d, in %.1f s; %d lane%s x %d ctx%s",
                  desc, arch, static_cast<double>(llama_model_size(model_)) / (1u << 30), card.c_str(), platform,
                  seconds_since(t_load), lanes_, lanes_ == 1 ? "" : "s", n_ctx_,
                  n_draft_ > 0 ? log::format("; MTP drafts up to %d", n_draft_).c_str() : "");
    }

    ~LlamaBackend() override {
        spec_.reset();   // its draft context reads the target's
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
        if (spec_) return generate_spec(in, slot, on_piece, stats);
        const int seq = std::min(std::max(slot, 0), lanes_ - 1);
        const std::vector<int> prompt = prompt_tokens(in);
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
            // always run, for its logits. A recurrent state cannot step back:
            // where llama.cpp refuses the partial removal the lane resumes from
            // its newest checkpoint inside the prefix, else it starts over.
            size_t common = 0;
            while (common < have.size() && common < prompt.size() && have[common] == prompt[common]) ++common;
            if (common == prompt.size()) --common;
            llama_memory_t mem = llama_get_memory(ctx_);
            if (common < have.size() && !llama_memory_seq_rm(mem, seq, static_cast<llama_pos>(common), -1)) {
                const Checkpoint* c = restore_ckpt_locked(seq, common);
                if (c != nullptr && llama_memory_seq_rm(mem, seq, static_cast<llama_pos>(c->n_tokens), -1)) {
                    common = c->n_tokens;
                } else {
                    llama_memory_seq_rm(mem, seq, -1, -1);
                    common = 0;
                }
            }
            drop_ckpts_after_locked(seq, common);
            have.resize(common);
            stats.cache_hit_tokens = static_cast<int>(common);
            // all but the last token, checkpointed along the way; the last
            // runs alone, for its logits
            const size_t n_pre = prompt.size() - 1;
            prefill_locked(seq, prompt, common, n_pre, have, [&](const int* t, size_t n, size_t at) {
                return decode_locked(seq, t, n, at, false);
            });
            if (!decode_locked(seq, &prompt.back(), 1, n_pre, true))
                throw std::runtime_error("llama_decode failed during prefill");
            have.push_back(prompt.back());
            copy_logits_locked(logits);
        }
        stats.prefill_seconds = seconds_since(t_prefill);

        // ------------------------------------------------------------- decode
        const auto t_decode = clock_type::now();
        FinishReason reason = FinishReason::Stop;
        const auto   t_s0   = clock_type::now();
        int          next   = sampler.sample(logits.data(), n_vocab_);
        stats.decode_sample_seconds += seconds_since(t_s0);
        auto is_stop_token = [&](int tok) { return is_stop(in.sampler, tok); };
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
        log_expert_cache(seq);
        return reason;
    }

private:
    // --llama-mtp: draft, verify [last, drafts...] in one target decode, keep
    // what the sampler agrees with. The sampler draws every verified position
    // in order, from the logits of the prefix it has itself accepted, so the
    // output is what the plain loop would sample (examples/speculative-simple
    // of the pinned llama.cpp, with arcint's sampler).
    FinishReason generate_spec(const GenerationInput& in, int slot, const TokenCallback& on_piece,
                               GenerationStats& stats) {
        const int seq = std::min(std::max(slot, 0), lanes_ - 1);
        try {
            return generate_spec_lane(in, seq, on_piece, stats);
        } catch (...) {
            // a failure between the target's decode and the bookkeeping would
            // leave the lane holding tokens `have` does not record: start over
            std::lock_guard<std::mutex> lk(mu_);
            slot_tokens_[static_cast<size_t>(seq)].clear();
            ckpts_[static_cast<size_t>(seq)].clear();
            spec_->seq_rm(seq, 0);
            spec_->reset(seq);
            throw;
        }
    }

    // The text that opens a user message after an assistant reply (ChatML:
    // "<|im_start|>user"), from the template by difference, as llama.cpp's
    // autoparser finds its user_start: what stands between a reply and the
    // next user message, less what ends a reply. Empty when the template
    // renders neither.
    std::string user_start_text() const {
        static const char* u1 = "ARCINTPROBEUSERONE";
        static const char* a1 = "ARCINTPROBEREPLYONE";
        static const char* u2 = "ARCINTPROBEUSERTWO";
        try {
            minja::chat_template_options opts;
            opts.apply_polyfills = false;
            const json m_u1{{"role", "user"}, {"content", u1}}, m_a1{{"role", "assistant"}, {"content", a1}};
            minja::chat_template_inputs two, three;
            two.messages              = json::array({m_u1, m_a1});
            three.messages            = json::array({m_u1, m_a1, json{{"role", "user"}, {"content", u2}}});
            two.add_generation_prompt = three.add_generation_prompt = false;
            const std::string r2 = template_->apply(two, opts), r3 = template_->apply(three, opts);
            const size_t e2 = r2.rfind(a1), e3 = r3.find(a1), s3 = r3.rfind(u2);
            if (e2 == std::string::npos || e3 == std::string::npos || s3 == std::string::npos) return {};
            const std::string end = r2.substr(e2 + std::strlen(a1));
            std::string       mid = r3.substr(e3 + std::strlen(a1), s3 - e3 - std::strlen(a1));
            size_t k = 0;
            while (k < end.size() && k < mid.size() && end[k] == mid[k]) ++k;
            mid.erase(0, k);
            // trimmed, as llama.cpp's detect_user_start_marker: a trailing
            // newline or space would tokenize with the message's first
            // characters and miss it
            const auto ws = " \t\r\n";
            const size_t a = mid.find_first_not_of(ws), b = mid.find_last_not_of(ws);
            return a == std::string::npos ? std::string{} : mid.substr(a, b - a + 1);
        } catch (const std::exception&) {
            return {};
        }
    }

    // The expert cache's counters since the previous request: the share of
    // decode's routed experts served from a slot, and the swaps
    void log_expert_cache(int seq) {
#ifndef ARCINT_LLAMA_EXPERT_CACHE
        (void) seq;
    }
#else
        std::lock_guard<std::mutex> lk(mu_);   // the counters move inside llama_decode
        llama_expert_cache_info ec{};
        if (!llama_model_expert_cache_info(model_, &ec)) return;
        const uint64_t h = ec.decode_hits - ec_prev_.decode_hits, m = ec.decode_misses - ec_prev_.decode_misses;
        if (h + m > 0)
            log::info("slot", "lane %d: expert cache: %.1f %% of decode's routed experts on the card (%llu / %llu), %llu swaps in %llu adapts, %.2f s writing them",
                      seq, 100.0 * static_cast<double>(h) / static_cast<double>(h + m), static_cast<unsigned long long>(h),
                      static_cast<unsigned long long>(h + m), static_cast<unsigned long long>(ec.swaps - ec_prev_.swaps),
                      static_cast<unsigned long long>(ec.adapts - ec_prev_.adapts),
                      static_cast<double>(ec.apply_us - ec_prev_.apply_us) / 1e6);
        ec_prev_ = ec;
    }
#endif

    // The prompt's ids. A vocabulary that wants BOS (Mistral's Tekken) gets it
    // when the rendered template did not write it: encode() adds no special
    // tokens, so a finetune's template without {{ bos_token }} would otherwise
    // run without BOS
    std::vector<int> prompt_tokens(const GenerationInput& in) const {
        std::vector<int> ids = in.prompt_ids.empty() ? tokenizer_->encode(in.prompt) : in.prompt_ids;
        const llama_token bos = llama_vocab_bos(vocab_);
        if (in.prompt_ids.empty() && llama_vocab_get_add_bos(vocab_) && bos != LLAMA_TOKEN_NULL &&
            (ids.empty() || ids.front() != bos))
            ids.insert(ids.begin(), bos);
        return ids;
    }

    FinishReason generate_spec_lane(const GenerationInput& in, int seq, const TokenCallback& on_piece,
                                    GenerationStats& stats) {
        const std::vector<int> prompt = prompt_tokens(in);
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
        std::vector<float> rows;   // the verified rows' logits

        // ------------------------------------------------------------ prefill
        // All but the last prompt token: the loop below starts from it.
        const auto   t_prefill = clock_type::now();
        const size_t n_pre     = prompt.size() - 1;
        {
            std::lock_guard<std::mutex> lk(mu_);
            size_t common = 0;
            while (common < have.size() && common < n_pre && have[common] == prompt[common]) ++common;
            if (common < have.size()) {
                // where the target's recurrent state cannot step back to the
                // cut, it resumes from its newest checkpoint inside the prefix
                // (llama.cpp leaves the memory untouched when it refuses);
                // without one, or if the drafter's cut is refused, the lane
                // clears. The drafter keeps what stands below the cut.
                const Checkpoint* c = nullptr;
                if (!llama_memory_seq_rm(llama_get_memory(ctx_), seq, static_cast<llama_pos>(common), -1)) {
                    c      = restore_ckpt_locked(seq, common);
                    common = c != nullptr ? c->n_tokens : 0;
                }
                if (!spec_->seq_rm(seq, common)) common = 0;
                else if (c != nullptr) spec_->set_carried_row(seq, common, c->dft_row);
            }
            drop_ckpts_after_locked(seq, common);
            have.resize(common);
            stats.cache_hit_tokens = static_cast<int>(common);
            prefill_locked(seq, prompt, common, n_pre, have, [&](const int* t, size_t n, size_t at) {
                return spec_->decode(t, n, at, seq, false) == 0;
            });
        }
        stats.prefill_seconds = seconds_since(t_prefill);

        // ------------------------------------------------------------- decode
        const auto   t_decode = clock_type::now();
        FinishReason reason   = FinishReason::Stop;
        int          id_last  = prompt.back();
        std::vector<int> batch;
        while (true) {
            if (static_cast<int>(have.size()) + 1 >= n_ctx_) {
                reason = FinishReason::Length;
                break;
            }
            // tokens that may still be emitted: the plain loop's max_tokens
            // and context checks (it emits a token while have + 1 < n_ctx)
            int budget = n_ctx_ - static_cast<int>(have.size()) - 2;
            if (in.sampler.max_tokens >= 0) budget = std::min(budget, in.sampler.max_tokens - stats.completion_tokens);
            const int n_max = std::max(0, std::min(n_draft_, budget - 1));
            std::vector<int> draft;
            {
                const auto t_p = clock_type::now();
                std::lock_guard<std::mutex> lk(mu_);
                if (n_max > 0) draft = spec_->draft(seq, id_last, have.size(), n_max);
                stats.draft_propose_seconds += seconds_since(t_p);
            }
            stats.draft_proposed += static_cast<int>(draft.size());
            batch.assign(1, id_last);
            batch.insert(batch.end(), draft.begin(), draft.end());
            {
                // the verified rows' logits leave the context under the lock:
                // another lane's decode would overwrite them
                std::lock_guard<std::mutex> lk(mu_);
                const auto t_v = clock_type::now();
                if (spec_->decode(batch.data(), batch.size(), have.size(), seq, true) != 0)
                    throw std::runtime_error("llama_decode failed during verify");
                rows.resize(batch.size() * n_vocab_);
                for (size_t i = 0; i < batch.size(); ++i) {
                    const float* l = llama_get_logits_ith(ctx_, static_cast<int32_t>(i));
                    if (l == nullptr) throw std::runtime_error("llama.cpp returned no logits");
                    std::copy(l, l + n_vocab_, rows.begin() + static_cast<long>(i * n_vocab_));
                }
                stats.draft_verify_seconds += seconds_since(t_v);
            }
            double     emit_s = 0.0;
            const auto t_walk = clock_type::now();
            const VerifyWalk w = walk_verify(
                rows.data(), batch.size(), n_vocab_, draft, sampler, budget,
                [&](int tok) { return is_stop(in.sampler, tok); },
                [&](int tok) {
                    const auto t_emit = clock_type::now();
                    const Control ctl = on_piece(tokenizer_->decode_one(tok), tok);
                    emit_s += seconds_since(t_emit);
                    return ctl;
                });
            stats.decode_emit_seconds += emit_s;
            stats.decode_sample_seconds += seconds_since(t_walk) - emit_s;
            stats.completion_tokens += w.emitted;
            stats.draft_accepted += w.accepted;
            have.push_back(id_last);
            have.insert(have.end(), draft.begin(), draft.begin() + w.accepted);
            id_last = w.last;
            const auto t_r = clock_type::now();
            {
                std::lock_guard<std::mutex> lk(mu_);
                spec_->accept(seq, w.accepted);
                // at most n_draft positions back: within the target's snapshots
                if (!spec_->seq_rm(seq, have.size()))
                    throw std::runtime_error("llama.cpp refused the draft rollback");
            }
            stats.draft_rollback_seconds += seconds_since(t_r);
            if (w.done) {
                reason = w.reason;
                break;
            }
        }
        stats.decode_seconds = seconds_since(t_decode);
        log_expert_cache(seq);
        return reason;
    }

    // Prompt tokens [from, to) of `seq` in batches, appended to `have`, with
    // checkpoints where llama.cpp's server makes them
    // (tools/server/server-context.cpp at the pin): at the start of a batch
    // that begins the last user message, a user message more than
    // --llama-checkpoint-step tokens past the lane's newest checkpoint, or
    // 4 + n_ubatch and 4 tokens before the prompt's end, the batches broken
    // there. A follow-up whose template re-renders the reply (the generation
    // prompt's think block dropped) shares the prompt up to a few tokens
    // before its end; an edited message, up to that message's start. Caller
    // holds mu_.
    template <class Decode>
    void prefill_locked(int seq, const std::vector<int>& prompt, size_t from, size_t to, std::vector<int>& have,
                        Decode&& decode) {
        const bool   ckpt = ckpt_on_;
        const size_t P    = prompt.size();
        const size_t nub  = static_cast<size_t>(n_ubatch_);
        ++ckpt_req_;
        std::vector<size_t> users;   // the user messages' first tokens
        if (ckpt && !user_start_.empty())
            for (size_t i = 0; i + user_start_.size() <= P; ++i)
                if (std::equal(user_start_.begin(), user_start_.end(), prompt.begin() + static_cast<long>(i)))
                    users.push_back(i);
        const size_t last_user = users.empty() ? SIZE_MAX : users.back();
        std::vector<size_t> ends;    // 4 + n_ubatch and 4 before the end (llama.cpp's checkpoint_offsets)
        for (size_t off : {4 + nub, size_t{4}}) {
            const size_t n_last = std::min(static_cast<size_t>(n_batch_), off);
            if (P > n_last) ends.push_back(P - n_last);
        }
        const auto& list   = ckpts_[static_cast<size_t>(seq)];
        auto newest        = [&] { return list.empty() ? size_t{0} : list.back().n_tokens; };
        auto is_user       = [&](size_t at) { return std::binary_search(users.begin(), users.end(), at); };
        // where a batch stops early: a user start the checkpoint rule takes, or an end offset
        auto breaks_at     = [&](size_t at) {
            if (!ckpt) return false;
            if (is_user(at) && (at == last_user || list.empty() || at > newest() + ckpt_step_)) return true;
            return std::find(ends.begin(), ends.end(), at) != ends.end();
        };
        for (size_t at = from; at < to;) {
            size_t e = std::min(to, at + static_cast<size_t>(n_batch_));
            for (size_t i = at + 1; i < e; ++i)
                if (breaks_at(i)) {
                    e = i;
                    break;
                }
            if (ckpt && at > 0) {
                // llama.cpp: a mid-prompt batch only when it starts a user
                // message or ends within a ubatch of the prompt's end; then
                // not within the step of the newest, unless it is the last
                // user message or near the end
                const bool near_end = P < e + nub;
                const bool last_one = e == to;
                bool       take     = last_one || is_user(at) || near_end;
                take = take && (list.empty() || at == last_user || near_end || at > newest() + ckpt_step_);
                if (take) save_ckpt_locked(seq, at);
            }
            if (!decode(prompt.data() + at, e - at, at))
                throw std::runtime_error("llama_decode failed during prefill");
            have.insert(have.end(), prompt.begin() + static_cast<long>(at), prompt.begin() + static_cast<long>(e));
            at = e;
        }
    }

    // A checkpoint of what `seq` holds after n_tokens tokens: only the part of
    // the memory llama.cpp cannot roll back (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY:
    // a hybrid model's recurrent state), in host memory, with the drafter's
    // carried row. At most --llama-checkpoints; when full, llama.cpp's
    // eviction: first the ones within the step of an earlier one that an
    // earlier request made, then the oldest. One already at n_tokens is kept
    // (the lane's tokens below it are unchanged, so is its state). Caller
    // holds mu_.
    void save_ckpt_locked(int seq, size_t n_tokens) {
        auto& list = ckpts_[static_cast<size_t>(seq)];
        while (!list.empty() && list.back().n_tokens > n_tokens) list.pop_back();
        if (!list.empty() && list.back().n_tokens == n_tokens) {
            list.back().req = ckpt_req_;   // as llama.cpp's supersede: the current request's now
            return;
        }
        Checkpoint c;
        c.n_tokens = n_tokens;
        c.req      = ckpt_req_;
        const size_t size = llama_state_seq_get_size_ext(ctx_, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (size == 0) return;
        c.state.resize(size);
        if (llama_state_seq_get_data_ext(ctx_, c.state.data(), size, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size) {
            log::warn("slot", "lane %d: checkpoint at %zu tokens failed", seq, n_tokens);
            return;
        }
        if (spec_) c.dft_row = spec_->carried_row(seq, n_tokens);
        size_t last = SIZE_MAX;   // the condition re-checked each step: only as many as make room
        for (auto it = list.begin(); list.size() + 1 >= static_cast<size_t>(ckpt_max_) && it != list.end();) {
            if (it->req != ckpt_req_ && last != SIZE_MAX && it->n_tokens <= last + ckpt_step_) {
                it = list.erase(it);
                continue;
            }
            last = it->n_tokens;
            ++it;
        }
        while (!list.empty() && list.size() >= static_cast<size_t>(ckpt_max_)) list.erase(list.begin());
        log::verbose("slot", "lane %d: checkpoint at %zu tokens (%.1f MiB), %zu kept", seq, n_tokens,
                     static_cast<double>(size) / (1u << 20), list.size() + 1);
        list.push_back(std::move(c));
    }

    // Loads the newest checkpoint of `seq` at or below `limit` tokens; null
    // when there is none or loading failed (the state is then undefined and
    // the caller clears the sequence). Caller holds mu_.
    const Checkpoint* restore_ckpt_locked(int seq, size_t limit) {
        const auto& list = ckpts_[static_cast<size_t>(seq)];
        for (auto it = list.rbegin(); it != list.rend(); ++it) {
            if (it->n_tokens > limit) continue;
            if (llama_state_seq_set_data_ext(ctx_, it->state.data(), it->state.size(), seq,
                                             LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != it->state.size()) {
                log::warn("slot", "lane %d: loading the checkpoint at %zu tokens failed", seq, it->n_tokens);
                return nullptr;
            }
            log::info("slot", "lane %d resumes from its checkpoint at %zu tokens (%.1f MiB)", seq, it->n_tokens,
                      static_cast<double>(it->state.size()) / (1u << 20));
            return &*it;
        }
        return nullptr;
    }

    // Checkpoints past `n` tokens no longer describe the lane. Caller holds mu_.
    void drop_ckpts_after_locked(int seq, size_t n) {
        auto& list = ckpts_[static_cast<size_t>(seq)];
        while (!list.empty() && list.back().n_tokens > n) list.pop_back();
    }

    bool is_stop(const SamplerParams& sp, int tok) const {
        if (!sp.ignore_eos && llama_vocab_is_eog(vocab_, tok)) return true;
        return std::find(sp.stop_token_ids.begin(), sp.stop_token_ids.end(), tok) != sp.stop_token_ids.end();
    }

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

    std::string             expert_profile_;   // --llama-expert-profile, alive for the load
#ifdef ARCINT_LLAMA_EXPERT_CACHE
    llama_expert_cache_info ec_prev_{};        // the counters at the previous request's end
#endif
    // --llama-cpu-moe's tensor patterns, alive for the load
    std::vector<std::string>                       cpu_moe_patterns_;
    std::vector<llama_model_tensor_buft_override> buft_overrides_;
    llama_model*                          model_ = nullptr;
    llama_context*                        ctx_   = nullptr;
    const llama_vocab*                    vocab_ = nullptr;
    std::unique_ptr<LlamaTokenizer>       tokenizer_;
    bool mistral_small_24b_ = false;   // the admitted llama geometry (think tags, BOS)
    std::unique_ptr<minja::chat_template> template_;
    std::string                           template_src_;
    ModelStatus                           status_;
    size_t                                n_vocab_ = 0;
    int                                   lanes_   = 1;
    int                                   n_ctx_   = 0;
    int                                   n_batch_ = 2048;
    std::mutex                            mu_;    // llama_context is not thread-safe: one call at a time
    std::vector<std::vector<int>>         slot_tokens_;
    std::unique_ptr<LlamaSpec>            spec_;      // --llama-mtp
    int                                   n_draft_ = 0;
    // --llama-checkpoints: per lane, the recurrent state at token counts the
    // lane's tokens still match, oldest first
    std::vector<std::vector<Checkpoint>>  ckpts_;
    bool                                  ckpt_on_   = false;   // a hybrid or recurrent model, N > 0
    int                                   ckpt_max_  = 32;
    size_t                                ckpt_step_ = 8192;
    uint64_t                              ckpt_req_  = 0;
    std::vector<int>                      user_start_;   // a user message's first tokens
    int                                   n_ubatch_  = 512;
};

}  // namespace

std::unique_ptr<Backend> make_llama_backend(const Config& cfg, int n_ctx) {
    return std::make_unique<LlamaBackend>(cfg, n_ctx);
}

}  // namespace lgc

#endif  // ARCINT_LLAMA

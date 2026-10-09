// The libllama executor (0.5.3.1459): arcint without OpenVINO. The GGUF is
// the model; ggml's OpenCL backend runs it on the Arc card (each Arc card is
// its own OpenCL platform under the NEO driver). arcint keeps the server,
// the chat template rendering, the sampler, stop handling and the lanes;
// llama.cpp keeps the weights, the tokenizer, the attention KV and the
// recurrent (gated delta-net) state, one sequence per lane.
#include "exec/backend.h"

#ifdef ARCINT_LLAMA

#include <CL/cl.h>
#include <gguf.h>
#include <llama.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <optional>
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
#include "exec/llama_layer_split.h"
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

// The OpenCL card --device names: its platform, its index among the
// platform's devices as ggml-opencl numbers them (every device type, as
// GGML_OPENCL_DEVICES takes it), and its name. "GPU.N" is the N-th platform
// with a GPU in OpenCL's enumeration (the NEO driver lists each Arc card as
// its own platform), anything else a substring of the device name ("B60",
// "A770"); the platform's first GPU either way. platform -1 when nothing
// matches.
struct ClCard {
    int         platform = -1;
    int         device   = 0;
    std::string name;
};

ClCard opencl_card_for(const std::string& want) {
    ClCard  c;
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return c;
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
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> ds(nd);
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr);
        int gpu = -1;
        for (cl_uint j = 0; j < nd && gpu < 0; ++j) {
            cl_device_type t = 0;
            clGetDeviceInfo(ds[j], CL_DEVICE_TYPE, sizeof(t), &t, nullptr);
            if (t & CL_DEVICE_TYPE_GPU) gpu = static_cast<int>(j);
        }
        if (gpu < 0) continue;
        char name[256] = {};
        clGetDeviceInfo(ds[static_cast<size_t>(gpu)], CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr);
        const bool hit = ordinal >= 0 ? seen == ordinal : std::string(name).find(want) != std::string::npos;
        if (hit) {
            c.platform = static_cast<int>(i);
            c.device   = gpu;
            c.name     = name;
            return c;
        }
        ++seen;
    }
    return c;
}

// llama.cpp names OpenCL devices GPUOpenCL, GPUOpenCL1, ...; the logs name
// the card (its description) instead
const char* card_desc(const char* dev_name) {
    ggml_backend_dev_t d = dev_name != nullptr ? ggml_backend_dev_by_name(dev_name) : nullptr;
    return d != nullptr ? ggml_backend_dev_description(d) : (dev_name != nullptr ? dev_name : "?");
}

// The GGUF's <arch>.block_count (MTP layers included): llama.cpp's layer
// split places that many layers and the output; -1 when unreadable. `nextn`
// gets <arch>.nextn_predict_layers (0 when absent).
int gguf_block_count(const std::string& path, int* nextn = nullptr) {
    gguf_context* g = gguf_init_from_file(path.c_str(), gguf_init_params{ /*no_alloc=*/true, nullptr });
    if (g == nullptr) return -1;
    int           n  = -1;
    const int64_t ka = gguf_find_key(g, "general.architecture");
    if (ka >= 0 && gguf_get_kv_type(g, ka) == GGUF_TYPE_STRING) {
        const std::string arch = gguf_get_val_str(g, ka);
        auto int_of = [g](int64_t k) {
            switch (gguf_get_kv_type(g, k)) {
                case GGUF_TYPE_UINT32: return static_cast<int>(gguf_get_val_u32(g, k));
                case GGUF_TYPE_INT32:  return static_cast<int>(gguf_get_val_i32(g, k));
                case GGUF_TYPE_UINT64: return static_cast<int>(gguf_get_val_u64(g, k));
                default:               return -1;
            }
        };
        const int64_t kb = gguf_find_key(g, (arch + ".block_count").c_str());
        if (kb >= 0) n = int_of(kb);
        if (nextn != nullptr) {
            const int64_t kn = gguf_find_key(g, (arch + ".nextn_predict_layers").c_str());
            *nextn           = kn >= 0 ? std::max(0, int_of(kn)) : 0;
        }
    }
    gguf_free(g);
    return n;
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
        // --device: one card, or the two cards of --llama-layer-split in layer order
        const std::vector<std::string> wanted = cfg.llama_devices.empty() ? std::vector<std::string>{ cfg.device } : cfg.llama_devices;
        std::vector<ClCard> cards;
        for (const std::string& w : wanted) {
            ClCard c = opencl_card_for(w);
            if (c.platform < 0) throw std::runtime_error(log::format("no OpenCL GPU matches --device %s", w.c_str()));
            for (const ClCard& o : cards)
                if (o.platform == c.platform && o.device == c.device)
                    throw std::runtime_error(log::format("--device %s: %s names one card twice", cfg.device.c_str(), c.name.c_str()));
            cards.push_back(c);
        }
        const bool split = cards.size() > 1;
        if (split && cfg.llama_layer_split <= 0)
            throw std::runtime_error("two cards need --llama-layer-split K");
        std::string card_names, cl_ids;
        for (const ClCard& c : cards) {
            card_names += (card_names.empty() ? "" : " + ") + c.name;
            cl_ids += (cl_ids.empty() ? "" : ",") + log::format("%d:%d", c.platform, c.device);
        }
        if (!split) {
            setenv("GGML_OPENCL_PLATFORM", std::to_string(cards[0].platform).c_str(), 1);
            setenv("GGML_OPENCL_DEVICE", "0", 1);
            unsetenv("GGML_OPENCL_DEVICES");   // it would win over the two above (0023)
        } else {
            // one context per card's platform, the cards in this order (0023)
            setenv("GGML_OPENCL_DEVICES", cl_ids.c_str(), 1);
        }
        llama_log_set(route_llama_log, nullptr);
        llama_backend_init();

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers       = 999;
        // the GGUF's MTP layer, for --llama-mtp (unless --llama-mtp-gguf brings it)
        mp.load_mtp           = cfg.llama_mtp > 0 && cfg.llama_mtp_gguf.empty();
        int n_layer_gguf      = 0;
        if (split) {
            // the GPU devices ggml lists are the cards GGML_OPENCL_DEVICES
            // selected, in its order (the build's one GPU backend is OpenCL);
            // a llama.cpp without 0023 lists one
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) devs_.push_back(d);
            }
            if (devs_.size() != cards.size())
                throw std::runtime_error(log::format("--device %s: llama.cpp lists %zu GPU devices for GGML_OPENCL_DEVICES=%s, not %zu "
                                                     "(contrib/llama.cpp patch 0023)", cfg.device.c_str(), devs_.size(), cl_ids.c_str(),
                                                     cards.size()));
            for (size_t i = 0; i < cards.size(); ++i)
                if (cards[i].name != ggml_backend_dev_description(devs_[i]))
                    log::warn("load", "--device %s: llama.cpp's device %zu is %s (%s), OpenCL's %s", cfg.device.c_str(), i,
                              ggml_backend_dev_name(devs_[i]), ggml_backend_dev_description(devs_[i]), cards[i].name.c_str());
            devs_.push_back(nullptr);
            // --llama-layer-split K: layers [0, K) on the first card, the rest
            // and the output on the second, exactly (exec/llama_layer_split.h)
            int nextn    = 0;
            n_layer_gguf = gguf_block_count(cfg.gguf_path, &nextn);
            if (n_layer_gguf < 0)
                throw std::runtime_error(log::format("--llama-layer-split: no <arch>.block_count in %s", cfg.gguf_path.c_str()));
            // the placement counts every block (llama.cpp's n_layer_all), but
            // an MTP block not loaded from this file (--llama-mtp-gguf, or no
            // --llama-mtp) runs nothing: K must leave the second card a block
            // that does
            const int n_run = mp.load_mtp ? n_layer_gguf : n_layer_gguf - nextn;
            if (!llama_layer_split_valid(n_run, cfg.llama_layer_split))
                throw std::runtime_error(log::format("--llama-layer-split %d: %s runs %d layers; K from 1 to %d puts layers on both cards",
                                                     cfg.llama_layer_split, cfg.gguf_path.c_str(), n_run, n_run - 1));
            const auto fr = llama_layer_split_fractions(n_layer_gguf, cfg.llama_layer_split);
            tensor_split_.assign(std::max<size_t>(llama_max_devices(), 2), 0.0f);
            tensor_split_[0] = fr[0];
            tensor_split_[1] = fr[1];
            mp.devices       = devs_.data();
            mp.split_mode    = LLAMA_SPLIT_MODE_LAYER;
            mp.tensor_split  = tensor_split_.data();
        }
        // --llama-expert-cache: slots on the card for the experts --llama-cpu-moe
        // keeps in host memory (contrib/llama.cpp patch 0021); under the layer
        // split a cache per card for its own layers (0024)
        const bool ec_on = cfg.llama_expert_cache_on();
#ifdef ARCINT_LLAMA_EXPERT_CACHE
        expert_profile_          = cfg.llama_expert_profile;
        mp.expert_cache_bytes    = static_cast<size_t>(cfg.llama_expert_cache_mib) << 20;
        mp.expert_cache_profile  = expert_profile_.empty() ? nullptr : expert_profile_.c_str();
        if (split && ec_on) {
#ifdef ARCINT_LLAMA_EXPERT_CACHE_DEV
            for (int mib : cfg.llama_expert_cache_mib_dev) expert_cache_bytes_dev_.push_back(static_cast<size_t>(mib) << 20);
            mp.expert_cache_bytes          = 0;
            mp.expert_cache_bytes_dev      = expert_cache_bytes_dev_.data();
            mp.n_expert_cache_bytes_dev    = expert_cache_bytes_dev_.size();
#else
            throw std::runtime_error("--llama-expert-cache on two cards: this build's llama.cpp lacks contrib/llama.cpp patch 0024");
#endif
        }
        // the experts are computed on the card (slots) and read by it from the
        // bank: the CPU's repacked copies would only cost host memory
        if (ec_on) mp.use_extra_bufts = false;
#else
        if (ec_on)
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
        if (ec_on) {
            llama_expert_cache_info ec{};
            if (!llama_model_expert_cache_info(model_, &ec))
                throw std::runtime_error("--llama-expert-cache: llama.cpp built no expert cache (no MoE layer in host memory?)");
#ifdef ARCINT_LLAMA_EXPERT_CACHE_DEV
            // one line per card's cache (the device names are llama.cpp's,
            // in --device order)
            const int n_ec = llama_model_expert_cache_count(model_);
            for (int i = 0; i < n_ec; ++i) {
                llama_expert_cache_info ei{};
                if (!llama_model_expert_cache_info_dev(model_, i, &ei)) continue;
                const char* dn = card_desc(llama_model_expert_cache_device(model_, i));
                log::info("load", "expert cache: %lld slots, %.2f GiB on the card %s, the other experts of its layers %.2f GiB in host memory",
                          static_cast<long long>(ei.slots), static_cast<double>(ei.bytes) / (1u << 30), dn ? dn : "?",
                          static_cast<double>(ei.bank_bytes) / (1u << 30));
            }
            if (split) {
                // a card given a budget that built no cache (no MoE layer of
                // its range in host memory, or no memory) would run its
                // experts from host buffers: refused, as one card is
                int want = 0;
                for (int mib : cfg.llama_expert_cache_mib_dev) want += mib > 0;
                if (n_ec != want)
                    throw std::runtime_error(log::format("--llama-expert-cache: %d of the %d cards given a budget built a cache (llama.cpp's log says why)", n_ec, want));
            }
            ec_prev_dev_.assign(static_cast<size_t>(n_ec), llama_expert_cache_info{});
#else
            log::info("load", "expert cache: %lld slots, %.2f GiB on the card, the other experts %.2f GiB in host memory",
                      static_cast<long long>(ec.slots), static_cast<double>(ec.bytes) / (1u << 30),
                      static_cast<double>(ec.bank_bytes) / (1u << 30));
#endif
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

        // Named lanes (--served-model-name A,B --lane-ctx CA,CB; DESIGN 4.2,
        // docs/campaigns/lanes-agent-subagent.md): a cap per lane, one KV
        // pool of their sum that every lane's cells live in (llama.cpp's
        // kv_unified, its server's --kv-unified; a sequence can use the whole
        // pool, llama-context.cpp:293-294, so the caps below are what keeps
        // each lane in its share). Equal lanes (--parallel N alone) keep a
        // stream of n_ctx each, as before.
        named_      = cfg.named_lanes();
        lanes_      = named_ ? static_cast<int>(cfg.lane_ctx.size()) : std::max(1, cfg.parallel);
        const int n_ctx_train = llama_model_n_ctx_train(model_);
        int64_t   pool        = 0;
        if (named_) {
            for (size_t i = 0; i < cfg.lane_ctx.size(); ++i) {
                if (n_ctx_train > 0 && cfg.lane_ctx[i] > n_ctx_train)
                    throw std::runtime_error(log::format("--lane-ctx %d (lane '%s'): %s was trained for %d", cfg.lane_ctx[i],
                                                         cfg.lane_names[i].c_str(), cfg.gguf_path.c_str(), n_ctx_train));
                pool += cfg.lane_ctx[i];
            }
            cap_   = cfg.lane_ctx;
            n_ctx_ = cap_[0];   // the first lane's: what an unnamed request gets
        } else {
            n_ctx_ = n_ctx > 0 ? n_ctx : std::min(n_ctx_train, 32768);
            cap_.assign(static_cast<size_t>(lanes_), n_ctx_);
            pool   = static_cast<int64_t>(n_ctx_) * lanes_;
        }
        n_batch_    = cfg.prefill_chunk > 0 ? cfg.prefill_chunk : 2048;
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = static_cast<uint32_t>(pool);
        cp.n_seq_max       = static_cast<uint32_t>(lanes_);
        cp.kv_unified      = named_;
        cp.n_batch         = static_cast<uint32_t>(n_batch_);
        // with the expert cache a prefill ubatch gathers the bank's routed experts
        // over the link once: the whole batch in one ubatch reads them once per
        // n_batch tokens instead of per 512
        cp.n_ubatch        = static_cast<uint32_t>(ec_on ? n_batch_ : std::min(n_batch_, 512));
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
        // llama.cpp pads the context to 256 (llama-context.cpp:291); the caps
        // are multiples of 256 (config.cpp), so a pool of any other size is a
        // context this engine did not ask for
        if (named_ && static_cast<int64_t>(llama_n_ctx(ctx_)) != pool)
            throw std::runtime_error(log::format("named lanes: llama.cpp made a KV pool of %u tokens, not the lanes' %lld",
                                                 llama_n_ctx(ctx_), static_cast<long long>(pool)));
        // a cell window per lane in the pool (contrib/llama.cpp 0026): lane i
        // finds its cells in its own range and attends over that range only.
        // Without it a lane's attention spans the pool up to its highest used
        // cell, the other lane's included: the subagent decoded 14.2 t/s with
        // the agent lane 130k deep, 38.4 with the pool empty (measured-here,
        // docs/campaigns/lanes-agent-subagent.md)
        std::vector<uint32_t> windows;
        if (named_) {
            windows.assign(cap_.begin(), cap_.end());
            windows_ = llama_memory_seq_windows(llama_get_memory(ctx_), windows.data(), lanes_);
            if (!windows_)
                log::warn("load", "named lanes: this model's KV cache takes no cell windows; a lane's attention "
                                  "spans the pool's used cells");
        }
        if (cfg.llama_mtp > 0) {
            // --llama-mtp-gguf: the MTP block (with its own embeddings and LM
            // head) as a second model, as llama.cpp loads an MTP-only file
            // for draft-mtp (common/arg.cpp, speculative.draft.mparams);
            // every tensor on the card. Under the layer split, on
            // --llama-mtp-card (default the first): it reads only the last
            // layer's rows, through host memory, so any card can run it.
            // Strata keeps its drafter on the last stage
            // (generate.cpp:2498-2500); here the A770, the last card, drafted
            // in 1.95 s a request against the B60's 0.94 (measured-here). The
            // GGUF's own MTP layer is its last layer, on the last card
            llama_model* mtp_model = model_;
            if (!cfg.llama_mtp_gguf.empty()) {
                llama_model_params dp = llama_model_default_params();
                dp.n_gpu_layers       = 999;
                dp.load_mtp           = true;
                if (split) {
                    mtp_devs_     = { devs_[static_cast<size_t>(std::max(0, cfg.llama_mtp_card))], nullptr };
                    dp.devices    = mtp_devs_.data();
                }
                mtp_model_            = llama_model_load_from_file(cfg.llama_mtp_gguf.c_str(), dp);
                if (mtp_model_ == nullptr)
                    throw std::runtime_error(log::format("llama.cpp could not load %s", cfg.llama_mtp_gguf.c_str()));
                mtp_model = mtp_model_;
            }
            std::string err;
            spec_ = make_llama_mtp(mtp_model, ctx_, cfg.llama_mtp, lanes_, n_batch_, static_cast<int>(cp.n_ubatch),
                                   threads, cfg.llama_mtp_gguf.empty() ? cfg.gguf_path : cfg.llama_mtp_gguf,
                                   cfg.llama_mtp_vocab, cfg.llama_mtp_min_p, cp.type_k, cp.type_v, cp.kv_unified, err);
            if (!spec_) throw std::runtime_error(log::format("--llama-mtp %d: %s", cfg.llama_mtp, err.c_str()));
            n_draft_ = cfg.llama_mtp;
            if (windows_ && !spec_->set_kv_windows(windows.data(), lanes_))
                log::warn("load", "named lanes: the MTP draft context takes no cell windows; its attention spans "
                                  "the pool's used cells");
        }
        status_.mtp_enabled = n_draft_ > 0;
        slot_tokens_.resize(static_cast<size_t>(lanes_));
        comb_decoding_.assign(static_cast<size_t>(lanes_), false);
        comb_stalled_.assign(static_cast<size_t>(lanes_), false);
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
        std::string lane_desc = log::format("%d lane%s x %d ctx", lanes_, lanes_ == 1 ? "" : "s", n_ctx_);
        if (named_) {
            lane_desc = log::format("%d named lanes,", lanes_);
            for (int i = 0; i < lanes_; ++i) lane_desc += log::format(" %s%d", i == 0 ? "" : "+ ", cap_[static_cast<size_t>(i)]);
            lane_desc += log::format(" = %lld ctx in one pool", static_cast<long long>(pool));
        }
        log::info("load", "llama.cpp %s (%s, %.2f GiB) on %s, OpenCL %s %s, in %.1f s; %s%s",
                  desc, arch, static_cast<double>(llama_model_size(model_)) / (1u << 30), card_names.c_str(),
                  split ? "devices" : "platform", split ? cl_ids.c_str() : std::to_string(cards[0].platform).c_str(),
                  seconds_since(t_load), lane_desc.c_str(),
                  n_draft_ > 0 ? log::format("; MTP drafts up to %d", n_draft_).c_str() : "");
        if (named_) {
            // one line per lane, and the pool they share
            for (int i = 0; i < lanes_; ++i)
                log::info("load", "lane %d '%s': cap %d tokens", i, cfg.lane_names[static_cast<size_t>(i)].c_str(),
                          cap_[static_cast<size_t>(i)]);
            log::info("load", "KV pool: %u tokens shared by %d lanes (kv_unified; %s)%s",
                      llama_n_ctx(ctx_), lanes_,
                      windows_ ? "a cell window per lane: a lane attends over its own cells"
                               : "a lane's attention runs over the pool's used cells, masked",
                      spec_ ? "; the MTP draft context likewise" : "");
        }
        if (split)
            log::info("load", "layer split: layers 0-%d on %s, %d-%d and the output on %s (tensor_split %.1f/%.1f)%s",
                      cfg.llama_layer_split - 1, cards[0].name.c_str(), cfg.llama_layer_split, n_layer_gguf - 1,
                      cards[1].name.c_str(), static_cast<double>(tensor_split_[0]), static_cast<double>(tensor_split_[1]),
                      mtp_model_ != nullptr ? log::format("; the MTP model on %s", cards[static_cast<size_t>(std::max(0, cfg.llama_mtp_card))].name.c_str()).c_str() : "");
    }

    ~LlamaBackend() override {
        spec_.reset();   // its draft context reads the target's
        if (ctx_ != nullptr) llama_free(ctx_);
        if (mtp_model_ != nullptr) llama_model_free(mtp_model_);
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
        const int cap = cap_[static_cast<size_t>(seq)];   // the lane's context
        const std::vector<int> prompt = prompt_tokens(in);
        stats.prompt_tokens = static_cast<int>(prompt.size());
        if (prompt.empty()) return FinishReason::Stop;
        if (static_cast<int>(prompt.size()) >= cap) {
            log::warn("slot", "lane %d: prompt of %zu tokens does not fit n_ctx %d", seq, prompt.size(), cap);
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
            if (static_cast<int>(have.size()) + 1 >= cap) {
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
        // The counters are the model's, not a sequence's: with more than one
        // lane they count every lane's decode since the previous line (any
        // lane's), so the line says so rather than passing for this request's
        // (snapshotting around each locked decode would put two counter reads
        // on every token's critical path for a log line)
        const char* scope = lanes_ > 1 ? " (process-wide, all lanes since the previous line)" : "";
        if (h + m > 0)
            log::info("slot", "lane %d: expert cache%s: %.1f %% of decode's routed experts on the card (%llu / %llu), %llu swaps in %llu adapts, %.2f s writing them",
                      seq, scope, 100.0 * static_cast<double>(h) / static_cast<double>(h + m), static_cast<unsigned long long>(h),
                      static_cast<unsigned long long>(h + m), static_cast<unsigned long long>(ec.swaps - ec_prev_.swaps),
                      static_cast<unsigned long long>(ec.adapts - ec_prev_.adapts),
                      static_cast<double>(ec.apply_us - ec_prev_.apply_us) / 1e6);
        ec_prev_ = ec;
#ifdef ARCINT_LLAMA_EXPERT_CACHE_DEV
        // under the layer split, each card's share too (the line above sums them)
        const int n_ec = llama_model_expert_cache_count(model_);
        for (int i = 0; n_ec > 1 && i < n_ec && i < static_cast<int>(ec_prev_dev_.size()); ++i) {
            llama_expert_cache_info ei{};
            if (!llama_model_expert_cache_info_dev(model_, i, &ei)) continue;
            llama_expert_cache_info& p = ec_prev_dev_[static_cast<size_t>(i)];
            const uint64_t hi = ei.decode_hits - p.decode_hits, mi = ei.decode_misses - p.decode_misses;
            const char*    dn = card_desc(llama_model_expert_cache_device(model_, i));
            // every card, also one that counted nothing (its ids not read)
            {
                log::info("slot", "lane %d: expert cache %s%s: %.1f %% of decode's routed experts on the card (%llu / %llu), %llu swaps",
                          seq, dn ? dn : "?", lanes_ > 1 ? " (process-wide)" : "", hi + mi > 0 ? 100.0 * static_cast<double>(hi) / static_cast<double>(hi + mi) : 0.0,
                          static_cast<unsigned long long>(hi), static_cast<unsigned long long>(hi + mi),
                          static_cast<unsigned long long>(ei.swaps - p.swaps));
            }
            p = ei;
        }
#endif
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
        const int              cap    = cap_[static_cast<size_t>(seq)];   // the lane's context
        const std::vector<int> prompt = prompt_tokens(in);
        stats.prompt_tokens = static_cast<int>(prompt.size());
        if (prompt.empty()) return FinishReason::Stop;
        if (static_cast<int>(prompt.size()) >= cap) {
            log::warn("slot", "lane %d: prompt of %zu tokens does not fit n_ctx %d", seq, prompt.size(), cap);
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
        // a lane in this loop: another lane's step waits a moment for its step
        struct Decoding {
            LlamaBackend& b;
            int           seq;
            Decoding(LlamaBackend& be, int s) : b(be), seq(s) { b.set_decoding(seq, true); }
            ~Decoding() { b.set_decoding(seq, false); }
        };
        std::optional<Decoding> decoding(std::in_place, *this, seq);
        while (true) {
            if (static_cast<int>(have.size()) + 1 >= cap) {
                reason = FinishReason::Length;
                break;
            }
            // tokens that may still be emitted: the plain loop's max_tokens
            // and context checks (it emits a token while have + 1 < cap); the
            // verify of 1 + n_max then ends below the lane's cap, so a lane's
            // cells never reach into the other lanes' share of the pool
            int budget = cap - static_cast<int>(have.size()) - 2;
            if (in.sampler.max_tokens >= 0) budget = std::min(budget, in.sampler.max_tokens - stats.completion_tokens);
            const int n_max = std::max(0, std::min(n_draft_, budget - 1));
            std::vector<int> draft;
            {
                // the step, with the other lanes' steps that are ready: one draft
                // step for all, then one target decode for all the verifies
                // (llama.cpp's server drafts and batches its slots the same way);
                // the rows leave the context under the lock, before another
                // decode overwrites them
                StepReq req{ seq, id_last, have.size(), n_max, &draft, &batch, &rows };
                if (step_combined(req) != 0) throw std::runtime_error("llama_decode failed during verify");
                stats.draft_propose_seconds += req.t_propose;
                stats.draft_verify_seconds += req.t_verify;
            }
            stats.draft_proposed += static_cast<int>(draft.size());
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
        decoding.reset();   // this lane posts no more steps
        stats.decode_seconds = seconds_since(t_decode);
        if (lanes_ > 1) {
            std::lock_guard<std::mutex> lk(mu_);
            log::info("slot", "lane %d: %llu steps so far shared with another lane's, %llu of them with verifies of "
                              "different lengths (process-wide)", seq,
                      static_cast<unsigned long long>(n_combined_), static_cast<unsigned long long>(n_uneven_));
        }
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
    // A lane's step, handed to the combiner: its drafts (up to n_max after
    // id_last at pos), then the verify of [id_last, drafts...]
    struct StepReq {
        int                 seq     = 0;
        int                 id_last = 0;
        size_t              pos     = 0;
        int                 n_max   = 0;
        std::vector<int>*   draft   = nullptr;
        std::vector<int>*   batch   = nullptr;
        std::vector<float>* rows    = nullptr;
        bool                done    = false;
        int                 rc      = 0;
        double              t_propose = 0.0, t_verify = 0.0;
    };

    // a lane enters or leaves its decode loop; a leader waiting for it stops
    // waiting when it leaves
    void set_decoding(int seq, bool on) {
        std::lock_guard<std::mutex> lk(comb_mu_);
        comb_decoding_[static_cast<size_t>(seq)] = on;
        comb_stalled_[static_cast<size_t>(seq)]  = false;
        comb_cv_.notify_all();
    }

    // Steps of several lanes at once. The first lane to arrive leads: while
    // other lanes are in their decode loop and have not posted their step yet
    // it waits for them (at most comb_wait_), then, under mu_, drafts for every
    // posted step in shared draft steps, verifies them in one target decode and
    // hands each lane its drafts and rows. A lane that arrives meanwhile posts
    // for the next round. One lane alone runs at once, as before. A lane that
    // misses a round's deadline (its client stalls the emit, a long walk) is
    // not waited for again until it posts: a stalled lane costs the others one
    // wait, not one a step.
    int step_combined(StepReq& r) {
        std::unique_lock<std::mutex> ul(comb_mu_);
        comb_q_.push_back(&r);
        comb_stalled_[static_cast<size_t>(r.seq)] = false;
        comb_cv_.notify_all();
        while (!r.done) {
            if (comb_leader_) {
                comb_cv_.wait(ul);
                continue;
            }
            comb_leader_ = true;
            // the decoding lanes that are not stalled and have not posted
            auto missing = [&](bool mark) {
                bool any = false;
                for (size_t s = 0; s < comb_decoding_.size(); ++s) {
                    if (!comb_decoding_[s] || comb_stalled_[s]) continue;
                    bool posted = false;
                    for (const StepReq* q : comb_q_) posted |= q->seq == static_cast<int>(s);
                    if (posted) continue;
                    any = true;
                    if (mark) comb_stalled_[s] = true;
                }
                return any;
            };
            const auto deadline = clock_type::now() + comb_wait_;
            while (missing(false)) {
                if (comb_cv_.wait_until(ul, deadline) == std::cv_status::timeout) {
                    missing(true);
                    break;
                }
            }
            std::vector<StepReq*> reqs;
            reqs.swap(comb_q_);
            ul.unlock();
            int    rc = 0;
            double t_p = 0.0, t_v = 0.0;
            // a throw here would leave the other lanes waiting: every exception is a
            // failed step for the lanes in it
            try {
                std::lock_guard<std::mutex> lk(mu_);
                const auto             t0 = clock_type::now();
                std::vector<DraftPart> dparts;
                dparts.reserve(reqs.size());
                for (StepReq* q : reqs)
                    if (q->n_max > 0) dparts.push_back(DraftPart{ q->seq, q->id_last, q->pos, q->n_max, q->draft });
                    else q->draft->clear();
                if (!dparts.empty()) spec_->draft_multi(dparts.data(), dparts.size());
                t_p = seconds_since(t0);
                const auto            t1 = clock_type::now();
                std::vector<SpecPart> parts;
                parts.reserve(reqs.size());
                for (StepReq* q : reqs) {
                    q->batch->assign(1, q->id_last);
                    q->batch->insert(q->batch->end(), q->draft->begin(), q->draft->end());
                    parts.push_back(SpecPart{ q->seq, q->batch->data(), q->batch->size(), q->pos, true });
                }
                // the rows of `idx` parts of `reqs`, decoded from batch offset 0
                auto copy_rows = [&](const std::vector<StepReq*>& qs) {
                    size_t off = 0;
                    for (StepReq* q : qs) {
                        const size_t n = q->batch->size();
                        q->rows->resize(n * n_vocab_);
                        for (size_t i = 0; i < n; ++i) {
                            const float* l = llama_get_logits_ith(ctx_, static_cast<int32_t>(off + i));
                            if (l == nullptr) return -1;
                            std::copy(l, l + n_vocab_, q->rows->begin() + static_cast<long>(i * n_vocab_));
                        }
                        off += n;
                    }
                    return 0;
                };
                rc = spec_->decode_multi(parts.data(), parts.size());
                if (rc == 0) rc = copy_rows(reqs);
                for (StepReq* q : reqs) q->rc = rc;
                if (rc != 0 && reqs.size() > 1) {
                    // lane by lane, so only the lane that fails errors. A lane whose
                    // part the target already holds (the MTP context failing after the
                    // target's decode, an earlier ubatch of an uneven split) cannot be
                    // decoded twice: llama.cpp refuses positions that do not continue
                    // the sequence ("inconsistent sequence positions", measured with an
                    // injected failure), that lane errors and generate_spec clears it
                    for (size_t k = 0; k < reqs.size(); ++k) {
                        reqs[k]->rc = spec_->decode_multi(&parts[k], 1);
                        if (reqs[k]->rc == 0) reqs[k]->rc = copy_rows({ reqs[k] });
                    }
                }
                t_v = seconds_since(t1);
                if (reqs.size() > 1) {
                    ++n_combined_;
                    for (const SpecPart& p : parts)
                        if (p.n != parts[0].n) {
                            // a hybrid model's equal split then runs them one by one
                            ++n_uneven_;
                            break;
                        }
                }
            } catch (const std::exception& e) {
                log::error("slot", "a shared step failed: %s", e.what());
                for (StepReq* q : reqs) q->rc = -1;
            } catch (...) {
                for (StepReq* q : reqs) q->rc = -1;
            }
            ul.lock();
            for (StepReq* q : reqs) {
                q->t_propose = t_p;
                q->t_verify  = t_v;
                q->done      = true;
            }
            comb_leader_ = false;
            comb_cv_.notify_all();
        }
        return r.rc;
    }

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
    std::vector<size_t>     expert_cache_bytes_dev_;   // --llama-expert-cache A,B, alive for the load
#endif
#ifdef ARCINT_LLAMA_EXPERT_CACHE_DEV
    std::vector<llama_expert_cache_info> ec_prev_dev_;   // per card's cache, the same
#endif
    // --device GPU.0,GPU.1 --llama-layer-split K: the cards (null-terminated),
    // the MTP model's card, the split; alive for the load
    std::vector<ggml_backend_dev_t> devs_;
    std::vector<ggml_backend_dev_t> mtp_devs_;
    std::vector<float>              tensor_split_;
    // --llama-cpu-moe's tensor patterns, alive for the load
    std::vector<std::string>                       cpu_moe_patterns_;
    std::vector<llama_model_tensor_buft_override> buft_overrides_;
    llama_model*                          model_ = nullptr;
    llama_model*                          mtp_model_ = nullptr;   // --llama-mtp-gguf
    llama_context*                        ctx_   = nullptr;
    const llama_vocab*                    vocab_ = nullptr;
    std::unique_ptr<LlamaTokenizer>       tokenizer_;
    bool mistral_small_24b_ = false;   // the admitted llama geometry (think tags, BOS)
    std::unique_ptr<minja::chat_template> template_;
    std::string                           template_src_;
    ModelStatus                           status_;
    size_t                                n_vocab_ = 0;
    int                                   lanes_   = 1;
    int                                   n_ctx_   = 0;   // the first lane's cap (status, the load line)
    bool                                  named_   = false;   // --lane-ctx: kv_unified, a cap per lane
    bool                                  windows_ = false;   // named lanes: a KV cell window per lane (0026)
    std::vector<int>                      cap_;       // per lane (seq): its context
    int                                   n_batch_ = 2048;
    std::mutex                            mu_;    // llama_context is not thread-safe: one call at a time
    // the step combiner (step_combined)
    std::mutex                            comb_mu_;
    std::condition_variable               comb_cv_;
    std::vector<StepReq*>                 comb_q_;
    bool                                  comb_leader_   = false;
    std::vector<bool>                     comb_decoding_;   // per lane: in its decode loop
    std::vector<bool>                     comb_stalled_;    // per lane: missed a round's deadline
    uint64_t                              n_combined_    = 0;   // steps shared with another lane's
    uint64_t                              n_uneven_      = 0;   // of them, verifies of different lengths
    std::chrono::microseconds             comb_wait_{ 20000 };
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

#pragma once

#include <cstdint>
#include <string>
#include <optional>
#include <vector>

namespace lgc {

struct SamplerDefaults;  // core/sampling.h

struct Config {
    // What gets served: the GGUF (--gguf), or the device-free stub (--stub).
    std::string gguf_path;   // --gguf: the file as published; the first shard of a split file
    // --engine: "llama" (ggml's OpenCL backend through libllama; the model is
    // --gguf) is the only engine. The flag is kept for the units' sake;
    // "ov" is refused with a pointer to the tag openvino-history.
    std::string engine;
    // --llama-cpu-moe N: on the libllama engine, the experts of the first N
    // layers stay in host memory (memory-mapped from the GGUF) and llama.cpp's
    // CPU backend computes them -- a model whose experts exceed VRAM.
    // --llama-threads N: that backend's threads (0: half the hardware threads,
    // the physical cores on SMT hosts).
    // --llama-mtp N: up to N tokens drafted per verify by the GGUF's own MTP
    // head (llama.cpp's draft-mtp); 0 off. A verify of N + 1 rows.
    int llama_cpu_moe = 0;
    int llama_threads = 0;
    int llama_mtp     = 0;
    // --llama-mtp-vocab FILE: the token ids the MTP head may draft (int32,
    // or a JSON list); the draft steps then read those rows of the output
    // head instead of all of them
    std::string llama_mtp_vocab;
    // --llama-mtp-gguf FILE: the MTP layer from a separate MTP-only GGUF
    // (llama.cpp's converter --mtp: the MTP block, the embeddings and the
    // LM head) instead of the model's own file
    std::string llama_mtp_gguf;
    // --llama-mtp-min-p P: a draft step continues while its token's
    // probability under the MTP head is at least P, and a draft below P is
    // not verified (Strata's --spec-min-p); 0 drafts --llama-mtp tokens always
    double llama_mtp_min_p = 0.0;
    // --llama-reserve-cells N: the compute buffers are reserved for a prompt
    // ubatch at N cache cells instead of the whole context, and a prompt chunk
    // deeper in the cache takes smaller ubatches to stay inside them (contrib
    // llama.cpp 0052): the full-context reservation's VRAM goes to the expert
    // slots. 0: reserve for the whole context
    uint32_t llama_reserve_cells = 0;
    // --llama-kv K[:V]: the libllama engine's attention cache types, f16,
    // q8_0 or q8_0:q4_0 (V defaults to K). Quantized K/V run on the Intel
    // attention kernels of contrib/llama.cpp 0015.
    std::string llama_kv_k = "f16";
    std::string llama_kv_v = "f16";
    // --llama-checkpoints N / --llama-checkpoint-step T: per lane, up to N
    // snapshots of the state llama.cpp cannot roll back (a hybrid model's
    // recurrent layers), taken where llama.cpp's server takes its context
    // checkpoints (near the prompt's end, at the last user message, at
    // earlier user messages T apart; 32 a slot, 8192), so a request sharing
    // a prefix with an earlier one resumes from the newest snapshot inside it
    // instead of prefilling from token 0. 0 off.
    int llama_checkpoints     = 32;
    // --llama-expert-cache MIB / --llama-expert-profile FILE: a per-expert GPU
    // cache of this many MiB for the MoE layers whose experts --llama-cpu-moe
    // keeps in host memory, ranked by a Strata STRP profile (empty: round
    // robin); the other experts in a bank in pinned host memory the card reads
    // over the link, hot ones swapped in during decode (contrib/llama.cpp
    // patch 0021). 0 off.
    // Under a layer split, --llama-expert-cache A,B: a cache per card, in
    // --device order (patch 0024); 0 keeps none on that card. One value is
    // the one card's. llama_expert_cache_mib is the first entry.
    int         llama_expert_cache_mib = 0;
    std::vector<int> llama_expert_cache_mib_dev;
    std::string llama_expert_profile;
    // --device GPU.0,GPU.1 with --engine llama: the cards of a layer split,
    // in layer order (filled from --device by parse_args; one entry for one
    // card). --llama-layer-split K: layers [0, K) on the first card, [K,
    // n_layer) and the output head (and the MTP layer) on the second
    // (llama.cpp's LLAMA_SPLIT_MODE_LAYER over two OpenCL platforms, patch
    // 0023). Two cards need K: there is no automatic split.
    std::vector<std::string> llama_devices;
    int llama_layer_split = 0;
    // --llama-mtp-card N: the card (index into llama_devices) that runs the
    // --llama-mtp-gguf model under the split; it reads only the last
    // layer's rows, through host memory. -1: unset, the first card.
    int llama_mtp_card = -1;
    bool llama_expert_cache_on() const {
        for (int m : llama_expert_cache_mib_dev)
            if (m > 0) return true;
        return llama_expert_cache_mib > 0;
    }
    int llama_checkpoint_step = 8192;
    bool        stub = false;

    // The name this endpoint answers to on /v1/models and /props. Empty means
    // the GGUF's file name (the stub: "stub"). DESIGN.md §4.2.
    std::string served_model_name;
    // Named lanes (DESIGN.md §4.2, amended 2026-10-07;
    // docs/campaigns/lanes-agent-subagent.md): --served-model-name A,B with
    // --lane-ctx CA,CB, paired by index. A lane's name picks it, and its cap
    // is its context. lane_names is empty unless the name list has two or
    // more entries; served_model_name is then the first (the agent lane, where
    // a request without a name goes). parallel is set to the lane count.
    // (The stub too, for the device-free suites.)
    std::vector<std::string> lane_names;
    std::vector<int>         lane_ctx;
    bool named_lanes() const { return !lane_ctx.empty(); }

    // --device: the N-th GPU in OpenCL enumeration (GPU.N) or a device-name
    // substring; two cards, GPU.0,GPU.1, for --llama-layer-split.
    std::string device = "GPU.0";

    std::string host = "127.0.0.1";
    int         port = 8090;

    int  n_ctx          = 0;      // 0: min(the model's trained context, 32768)
    // True only when --n-ctx was actually passed (any value, including 0);
    // named lanes refuse it, their contexts are --lane-ctx.
    bool n_ctx_explicit = false;
    int parallel = 1;  // lanes (DESIGN.md §4 /health reports free/total)
    bool parallel_explicit = false;  // --parallel was given (named lanes refuse a disagreeing one)

    // How long a request waits for a lane before it is refused with a 503
    // that carries the lane's free count (DESIGN.md §4.3). Zero refuses
    // immediately, which is the default because a lane count is a *memory*
    // reservation: with N lanes, an N+1st concurrent sequence has nowhere to
    // live, and a client that waits behind a 30k-context session cannot tell a
    // queue from a hang. A non-zero value restores queueing for deployments
    // that prefer it, and /health reports the queue depth either way.
    double queue_timeout_s = 0.0;

    // Prefill chunk in tokens (llama.cpp's n_batch), 0 = the default 2048.
    // Bounding the chunk bounds activation memory, which is how deep context
    // is served. Chunk boundaries are not bit-exact (DESIGN.md §3.2); 2048
    // keeps ordinary prompts in a single chunk.
    int prefill_chunk = 2048;

    // Operator serving defaults — the layer between the artifact and the
    // request (sampler-defaults order, 2026-09-01). Each flag overrides only
    // its own field of the artifact's sampler defaults; a request field still
    // wins over everything. Setting any of them turns the served provenance
    // string to "operator", so /props keeps telling the truth about where the
    // regime came from. Unset means "the artifact decides", exactly as before.
    std::optional<float> temp;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<int>   top_k;
    std::optional<float> repetition_penalty;
    std::optional<float> presence_penalty;

    // --chat-template-kwarg enable_thinking=BOOL: the default a request gets
    // when it sends neither chat_template_kwargs.enable_thinking nor
    // reasoning_effort. Only enable_thinking is accepted, because it is the
    // only kwarg the render path implements — an accepted-but-ignored key
    // would be a lie.
    std::optional<bool> think_default;

    // Stub-only: milliseconds of artificial latency per emitted token. Exists
    // so the cancellation path (§3.7) can be demonstrated against a backend
    // that would otherwise finish before a client could disconnect.
    int stub_delay_ms = 0;

    int http_threads = 0;  // 0: httplib default
    int verbosity    = 0;  // -v, -vv

    bool show_help    = false;
    bool show_version = false;
};

struct ArgParse {
    bool        ok = true;
    std::string error;
};

ArgParse    parse_args(int argc, char** argv, Config& cfg);
std::string usage_text();

// The --llama-kv pair as /props reports it: "K" when both sides are one type,
// "K:V" otherwise ("f16", "q8_0", "q8_0:q4_0"). The llama backend publishes this
// as the KV precision it serves.
std::string llama_kv_dtype(const Config& cfg);

// The names of options the retired engine had and 0.8.0 removed. parse_args
// refuses each by name ("<flag> was retired with the OpenVINO engine in 0.8.0
// (see docs/history.md)") rather than as an unknown option.
bool is_retired_flag(const std::string& flag);

// The operator serving-defaults layer: applies the --temp/--top-p/--top-k/
// --repetition-penalty/--presence-penalty flags (only those that were set) to
// the artifact's sampler defaults and marks their provenance "operator".
// Returns the same refusal a request would get for an out-of-range value.
std::optional<std::string> apply_operator_defaults(const Config& cfg, SamplerDefaults& d);

}  // namespace lgc

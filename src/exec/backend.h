#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/prefix_cache.h"
#include <nlohmann/json.hpp>
#include "core/chat.h"
#include "core/model_registry.h"
#include "core/sampling.h"

// The seam between "arcint owns the state" and "OV owns the math"
// (DESIGN.md §1). Everything above this interface — HTTP, scheduling, stop
// handling, tool-call parsing, usage accounting — is backend-agnostic and
// testable on a machine with no GPU in it.
namespace lgc {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int> encode(std::string_view text)        = 0;
    virtual std::string      decode(const std::vector<int>& ids)  = 0;
    virtual std::string      decode_one(int id)                   = 0;
    virtual int              eos_id() const                       = 0;
};

// The startup reservation (DESIGN.md §7.0.2a), every term measured rather than
// assumed. It is carried on the status so that /props can print it and an
// admission refusal can quote it: a refusal without numbers is the failure mode
// this engine exists to avoid, and with two lanes the arithmetic is the whole
// argument for why there are exactly two.
struct Reservation {
    bool     measured           = false;
    uint64_t device_total_bytes = 0;
    uint64_t weights_bytes      = 0;   // weights + graph, device-resident after compile
    uint64_t activation_bytes   = 0;   // per lane, at the chunk actually served
    uint64_t la_slab_bytes      = 0;   // per lane: its GDN checkpoint rows
    uint64_t kv_bytes_per_token = 0;
    uint64_t margin_bytes       = 0;
    uint64_t pool_blocks        = 0;   // the shared KV page pool
    int      kv_block_tokens    = 0;
    int      lanes              = 1;
    int      prefill_chunk      = 0;
    int      n_ctx              = 0;

    // M7 (the auto-fit design, §2/§4): the terms the pre-M7 budget did not
    // charge for. `drafter_bytes` is the resident delta from compiling
    // embeddings/MTP/DFlash after the main model (previously folded into
    // activation_bytes, attributed to the wrong line).
    //
    // The expert slot pool is two ledgers, not one (measured on the card,
    // see backend_ov.cpp Phase B): `expert_slot_bytes` is the DEVICE (VRAM)
    // charge -- the OTD LRU working set --offload-ratio > 0 actually keeps
    // resident, and the only one counted in the device budget below.
    // `expert_slot_host_bytes` is the HOST (GTT) estimate -- everything the
    // ratio makes eligible to be paged in -- informational only, never
    // subtracted from device_total_bytes. Both are zero when offload is
    // off. `slot_source` says how the DEVICE figure was priced: "forced"
    // (ARCINT_FIT_SLOT_BYTES) or "probe" (the plateau probe); empty when
    // offload is off or the probe failed with an explicit --n-ctx to fall
    // back on.
    uint64_t    expert_slot_bytes      = 0;
    uint64_t    expert_slot_host_bytes = 0;
    uint64_t    drafter_bytes          = 0;
    std::string slot_source;
};

struct ModelStatus {
    // Two names, and the distinction matters at every boundary. `id` is the
    // allowlist's canonical name: it is artifact identity, it is what the model
    // registry is keyed by, and it never moves. `served_id` is what the outside
    // world is told — the same thing unless --served-model-name says otherwise
    // (DESIGN.md §4.2). A roster or proxy that discovers names from /v1/models
    // pins itself to whatever it finds there, so that name has to be settable
    // without touching which artifact is asserted.
    std::string id;
    std::string served_id;
    Quant       quant        = Quant::Q4;
    bool        loaded       = false;
    bool        stub         = false;
    int         n_ctx        = 0;   // what this server is running with
    int         n_ctx_train  = 0;   // what the artifact was trained for
    int         n_layer      = 0;
    int         n_gdn_layer  = 0;
    int         n_attn_layer = 0;
    bool        mtp_enabled  = false;
    uint64_t    weights_bytes = 0;

    Reservation     reservation;
    SamplerDefaults sampler_defaults;
};

// What the caller does with each emitted piece.
enum class Control {
    Continue,
    Stop,    // a stop sequence matched — finish_reason "stop"
    Cancel,  // the client is gone — abort at the next boundary (§3.7)
};

using TokenCallback = std::function<Control(std::string_view piece, int token_id)>;

enum class FinishReason { Stop, Length, Abort };

const char* finish_reason_name(FinishReason r);

struct GenerationInput {
    std::string   prompt;
    SamplerParams sampler;

    // Names of the tools this request declared. A real backend ignores this —
    // the tools already reached the model through the chat template — but the
    // stub backend needs them to synthesise a tool call so the parsing path can
    // be exercised without a GPU.
    std::vector<std::string> tool_names;
};

struct GenerationStats {
    int    prompt_tokens     = 0;
    int    completion_tokens = 0;
    int    cache_hit_tokens  = 0;
    int    draft_proposed    = 0;
    int    draft_accepted    = 0;
    // Wall time spent copying the whole model state so a rejected draft can be
    // rolled back. On the stateful path this is the price of speculation.
    // What the prefix cache costs on the way in. It is the same host round-trip
    // that makes speculation unprofitable, paid once per prompt, and it would
    // otherwise hide inside prefill_seconds and be read as prefill being slow.
    double snapshot_seconds = 0.0;
    // Where a decode step actually goes. The graph is only part of it, and the
    // rest was invisible until these existed.
    double decode_forward_seconds = 0.0;   // the language model
    double decode_embed_seconds   = 0.0;   // the embeddings model, once per token
    double decode_sample_seconds  = 0.0;   // logits copy + penalties + argmax
    double decode_emit_seconds    = 0.0;   // detokenize, stop scan, stream out
    // Waiting for the device while the other lane had it. Broken out because it
    // otherwise hides inside whichever phase happened to block, and a decode
    // step that spent 10 ms queueing did not spend 10 ms gathering embeddings.
    double decode_wait_seconds    = 0.0;
    double draft_rollback_seconds = 0.0;
    double draft_verify_seconds   = 0.0;   // the one pass that checks all drafts
    // The propose phase's own cost -- turnstile wait, embed.infer() for the
    // anchor token, and the drafter itself (MTP layer+head, DFlash, or the
    // n-gram drafter) -- previously invisible (M11, DESIGN §7.0.2aa row):
    // nothing timed it, so it fell into the served decode line's "other" on
    // every drafting request.
    double draft_propose_seconds   = 0.0;
    double draft_reforward_seconds = 0.0;  // re-running the accepted prefix after a reject
    double prefill_seconds   = 0.0;
    // Where prefill goes, mirroring the decode breakdown. It existed for decode
    // and not for prefill, which is why a 2026-08-29 profile could account for
    // ~100 us/token of node time against ~445 us/token served and nobody could
    // say where the rest was. Each is net of the turnstile wait, which is its
    // own term.
    double prefill_embed_seconds   = 0.0;   // the embeddings gather, incl. its copy
    double prefill_forward_seconds = 0.0;   // the language model
    double prefill_blocks_seconds  = 0.0;   // KV page allocation
    double prefill_restore_seconds = 0.0;   // cache lookup, restore, row zeroing
    double cache_promote_seconds   = 0.0;   // host tier -> pages, when the hit was tiered
    double prefill_wait_seconds    = 0.0;   // waiting for the device
    double decode_seconds    = 0.0;
    // What the other lane cost this one: graph executions that had to wait for
    // their turn, and how long they waited (DESIGN.md §4.1). Zero on an idle
    // server, which is how a contended run is told from a solo one. The p95 is
    // over decode steps, because that is the stall a reader feels.
    int    stalled_steps       = 0;
    double stall_total_seconds = 0.0;
    double stall_p95_seconds   = 0.0;
    double stall_max_seconds   = 0.0;

    // Cumulative DFlash2 failures on the lane this request ran on, counted
    // since the lane last loaded (not reset per request): a lane whose
    // drafter keeps disabling itself request after request is a persistently
    // failing head, and that trend is only visible if the counter survives
    // the per-lane re-arm at the next dflash_reset. Zero on a backend with no
    // drafter, or a lane that has never failed.
    int dflash_lane_failures = 0;

    double prefill_rate() const {
        return prefill_seconds > 0.0 ? prompt_tokens / prefill_seconds : 0.0;
    }
    double decode_rate() const {
        return decode_seconds > 0.0 ? completion_tokens / decode_seconds : 0.0;
    }
};

// M11 step profile (DESIGN §7.0.2aa row, the M11 design note (not in the
// repository)): one line per drafting cycle on the paged serving path,
// naming every segment that had no timer before it (propose, verify's
// index build and infer/logits/hidden split, the accept loop, the
// turnstile wait). Gated on ARCINT_PROFILE_CYCLE, off by default. Declared
// here (defined in backend_stub.cpp, which every build links, matching
// finish_reason_name above) so the gating and formatting are each one
// definition and each testable without a GPU.
bool profile_cycle_enabled();

std::string format_profile_cycle_line(size_t past, size_t n, size_t accepted, double propose_ms,
                                      double propose_embed_ms, double propose_mask_ms,
                                      double propose_layer_ms, double propose_head_ms,
                                      double verify_embed_ms, double verify_index_ms,
                                      double verify_infer_ms, double verify_logits_ms,
                                      double verify_hidden_ms, double accept_ms, double wait_ms,
                                      double cycle_ms, int frag);

class Backend {
public:
    virtual ~Backend() = default;

    virtual const ModelStatus& status() const = 0;
    virtual Tokenizer&         tokenizer()    = 0;

    // Renders a chat request into the model's own prompt format. DESIGN.md §3.7
    // makes the artifact the single source of truth for the template, so this
    // is the backend's job: only it knows which template shipped with the
    // weights. The stub answers with ChatML because it has no artifact.
    virtual std::string render_chat(const ChatRequest& req) const = 0;
    // What the artifact's template can do, in llama.cpp's /props vocabulary:
    // a proxy reads `supports_preserve_reasoning` there to mark a model as
    // reasoning, and a discovering client shows its thinking knob only then.
    // Until 2026-08-30 arcint published nothing here and the fleet's local
    // models came up "non-reasoning" in every client.
    virtual nlohmann::json template_caps() const { return nlohmann::json::object(); }

    // `slot` is the lane this request owns for its whole life (DESIGN.md §4.1).
    // Every piece of mutable execution state — InferRequests, GDN checkpoint
    // rows, the KV block table, the MTP head's own KV — is indexed by it, which
    // is what makes two concurrent sequences independent rather than merely
    // interleaved.
    virtual FinishReason generate(const GenerationInput& in, int slot,
                                  const TokenCallback& on_piece, GenerationStats& stats) = 0;

    // KV pages not currently held by a sequence or by the prefix cache. Live,
    // so /health can show the pool draining and a refusal can quote it.
    virtual uint64_t free_blocks() const { return 0; }
    // The prefix cache's counters, for /health; empty when there is no cache.
    virtual PrefixCacheStats cache_stats() const { return {}; }
};

// M0: no OpenVINO, no weights, deterministic synthetic output.
// `delay_ms` inserts artificial per-token latency so that cancellation and
// streaming can be observed; zero for the fastest possible round-trip.
std::unique_ptr<Backend> make_stub_backend(const ModelEntry& entry, Quant quant, int n_ctx,
                                           int delay_ms = 0,
                                           const std::string& served_name = {});

#ifdef ARCINT_OPENVINO
struct Artifact;

// M1: the OpenVINO executor. Throws std::runtime_error with a readable message
// if the artifact cannot be compiled or validated.
struct Config;
std::unique_ptr<Backend> make_ov_backend(const Artifact& artifact, const Config& cfg, int n_ctx);
#endif

}  // namespace lgc

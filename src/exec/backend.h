#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include "core/chat.h"
#include "core/sampling.h"

// The seam between "arcint owns the serving" and "llama.cpp owns the math"
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

struct ModelStatus {
    // Two names. `id` is the served file's own name (the GGUF's file name; the
    // stub's "stub") and never moves. `served_id` is what the outside world is
    // told — the same thing unless --served-model-name says otherwise
    // (DESIGN.md §4.2). A roster or proxy that discovers names from /v1/models
    // pins itself to whatever it finds there, so that name has to be settable
    // without touching which file is asserted.
    std::string id;
    std::string served_id;
    // The weight format the file carries (llama.cpp's file-type name, "Q4_K -
    // Medium", "IQ2_XS - 2.3125 bpw"); empty when nothing is loaded (the stub),
    // which /v1/models and /props print as null.
    std::string quant;
    // The GGUF's general.architecture ("qwen35", "qwen35moe", ...); empty for
    // the stub.
    std::string arch;
    bool        loaded       = false;
    bool        stub         = false;
    int         n_ctx        = 0;   // what this server is running with
    int         n_ctx_train  = 0;   // what the model was trained for
    int         n_layer      = 0;
    bool        mtp_enabled  = false;
    bool        dflash_enabled = false;   // --llama-dflash-gguf: the DFlash2 drafter loaded
    int         dflash_drafts  = 0;       // its drafts a verify: --llama-dflash capped by its block
    uint64_t    weights_bytes = 0;

    // The KV precision actually served, as --llama-kv set it ("f16", "q8_0",
    // "q8_0:q4_0"; config.h's llama_kv_dtype): /props reads this instead of
    // guessing from cfg. Empty when nothing is loaded (the stub backend),
    // which is what lets /props print null rather than a default that was
    // never served.
    std::string kv_precision;
    // The model may open a think block with "<thinking>" or "[THINK]" as well
    // as "<think>": set by the llama engine for the Mistral Small 24B family
    // (Cydonia 24B, Magistral tunes). The Qwens keep "<think>" alone, so an
    // answer of theirs that merely starts with such text stays content.
    bool think_tags_extended = false;

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
    // When non-empty, THESE are the prompt and `prompt` is not tokenised
    // (CompletionRequest::prompt_ids; the KLD gate's replay of the reference
    // capture's token windows).
    std::vector<int> prompt_ids;
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
    // verify steps (one target decode of [last, drafts...] each): the
    // accepted drafts + steps over steps is the tokens a verify
    int    draft_steps       = 0;
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

};

// The stub's trained context: a plausible stand-in (the larger Qwen3 family's),
// so n_ctx_train and the context default have something to differ from.
inline constexpr int kStubNCtxTrain = 262144;

// The stub: no weights, no card, deterministic synthetic output, with a fixed
// identity (id "stub", `served_name` when given) and a plausible trained
// context, so the whole HTTP surface runs in the device-free suites.
// `delay_ms` inserts artificial per-token latency so that cancellation and
// streaming can be observed; zero for the fastest possible round-trip.
// `lane_ctx`: named lanes' caps (--lane-ctx), by slot; generation on a lane
// stops at its cap. Empty: every slot at n_ctx.
std::unique_ptr<Backend> make_stub_backend(int n_ctx, int delay_ms = 0,
                                           const std::string& served_name = {},
                                           const std::vector<int>& lane_ctx = {},
                                           const SamplerDefaults& sampler = {});

#ifdef ARCINT_LLAMA
// The libllama executor (ggml OpenCL), the GGUF in --gguf.
struct Config;
std::unique_ptr<Backend> make_llama_backend(const Config& cfg, int n_ctx);
#endif

}  // namespace lgc

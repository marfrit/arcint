#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
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
    // The vocabulary's size, for a scan over every id's piece (the loop-guard
    // door needs it to find the multi-newline tokens). Default 0: a tokenizer
    // that cannot enumerate offers no door mask and keeps today's behaviour.
    virtual int              vocab_size() const { return 0; }

    // The ids a tag string occupies when it is a SINGLE special token, empty
    // when the vocabulary spells it as ordinary text. The think-block mask
    // needs the difference: a tag that is one control token can be blocked in
    // the logits; a tag the model spells as text cannot, and is matched as text
    // instead. Default empty keeps every other Tokenizer working.
    virtual std::vector<int> special_ids(std::string_view tag) { (void)tag; return {}; }
};

// loop-guard door: the ids whose decoded piece is newline characters only,
// two or more of them (the double, triple ... newline tokens). Pure: one pass
// over the vocabulary the tokenizer owns, no cache and no shared state, so the
// backend can run it once at model load and several threads can call it at
// once. The single-newline id is not in the result.
inline std::vector<int> scan_loop_guard_door_ids(Tokenizer& tok) {
    std::vector<int> out;
    const int n = tok.vocab_size();
    for (int id = 0; id < n; ++id) {
        const std::string piece = tok.decode_one(id);
        size_t newlines = 0;
        bool   only     = true;
        for (char c : piece) {
            if (c == '\n') ++newlines;
            else { only = false; break; }
        }
        if (only && newlines >= 2) out.push_back(id);
    }
    return out;
}

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
    // --mmproj loaded: the chat endpoint takes image_url parts (docs/
    // milestone-1.2.0.md A2). The stub is false, so tests/roundtrip.sh's
    // text-only refusal exercises that branch.
    bool        mmproj_loaded = false;
    // The projector's own capabilities, from libmtmd's
    // `mtmd_get_cap_from_file` (EXPERIMENTAL at the pin): whether its clip
    // half takes images (`inp_vision`) and/or audio (`inp_audio`). /props
    // reports them so the image path and the audio-generation path can be told
    // apart from the projector alone, not only from --tts. False on the stub.
    bool        mmproj_vision = false;
    bool        mmproj_audio  = false;
    // --tts and a mmproj whose gen half supports audio generation: the speech
    // routes are served (docs/milestone-1.2.0.md A4). False on the stub and on
    // a build without --tts, which is how the route refuses clearly.
    bool        tts = false;
    // The enrolled voices this process may serve, enabled entries only (D5),
    // in the profile directory's name order. Names only — never a path, a
    // reference or bytes, which is why /props carries the count and
    // GET /v1/audio/voices the names.
    std::vector<std::string> voices;
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

enum class FinishReason { Stop, Length, Abort, Loop };

const char* finish_reason_name(FinishReason r);

// One text-to-speech request, in the OpenAI /v1/audio/speech shape minus the
// fields the API refuses (D5): no reference audio ever crosses this seam.
// `voice` is an enrolled name (ModelStatus::voices); the backend resolves it
// to its own server-side profile. `text` is what is spoken, `sampler` the
// request's own sampling (temperature/top_p/top_k/seed/...), applied to the
// backbone's codec-token sampling.
struct SpeechInput {
    std::string text;
    std::string voice;
    SamplerParams sampler;
};

// A request the backend cannot serve because of its own content (an image it
// cannot read or decode, too many or too large images, a prompt that with its
// image tokens exceeds the lane): the handlers answer it with 400, every other
// exception from generate() with 500. Thrown before any token is emitted.
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The prompt over the lane, counted with its image tokens: the handlers answer
// it with the text path's context_length_exceeded body (DESIGN §3.8).
struct ContextOverflowError : InputError {
    ContextOverflowError(int prompt_tokens_, int n_ctx_)
        : InputError("prompt over the lane"), prompt_tokens(prompt_tokens_), n_ctx(n_ctx_) {}
    int prompt_tokens;
    int n_ctx;
};

struct GenerationInput {
    // OpenAI image_url parts of this request, in the order their markers appear
    // in `prompt` (`docs/milestone-1.2.0.md §3`). Empty on a text request; the
    // backend's multimodal branch reads the bytes and encodes them.
    std::vector<ImagePart> images;

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

    // clean-turns step 2: the declared tools with their schemas, for the
    // tool-call grammar (SamplerParams::tool_grammar). Filled only when the
    // request's calls are parsed at all (tools declared, tool_choice not
    // "none"), so a request whose calls would not be parsed is never
    // constrained either. `parallel_tool_calls` as the request said (OpenAI
    // default true): the grammar then admits more than one call.
    std::vector<ToolSpec> tools;
    bool                  parallel_tool_calls = true;

    // clean-turns: the rendered prompt ended inside a think block, and whether
    // this artifact answers to the wider tag family. The API knows both from the
    // template (Prepared::think_open / think_tags_extended); the backend needs
    // them because the mask must engage before the FIRST sample, not after the
    // first emitted piece. SamplerParams::thinking_turn is the request's
    // enable_thinking after the operator default, so llava's answer row never
    // switches in a non-thinking turn.
    bool think_open          = false;
    bool think_tags_extended = false;
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
    // loop-guard: what the ladder did in this request. Zero on every request
    // without the switch -- the decode line and the response only print them
    // when they moved.
    int    loop_detections   = 0;
    int    loop_nudges       = 0;
    int    loop_slaps        = 0;
    int    loop_aborts       = 0;
    // loop-guard door: closes by the model's own draw while the door was open,
    // and draws whose top-k candidates held a masked multi-newline token.
    int    loop_self_closes  = 0;
    int    loop_door_masked  = 0;
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

    // clean-turns review B4: tool-call grammar desyncs on this request -- the
    // constraint lost sync with the generated text and the rest of the turn
    // ran unconstrained (fail open, so the turn can still end). Zero is the
    // normal value; anything else is printed on the decode line, where the
    // operator looks for the request's own numbers.
    int grammar_desyncs = 0;

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

    // The think tags' token ids the artifact's vocabulary holds as one special
    // token, per pair (ThinkBlockIds::open_close_ids), or empty when none. The
    // reasoning split and the sampler's tracker decide those roles by id; a
    // pair with -1 keeps the text behaviour. Filled once at load, not per
    // request.
    virtual ThinkBlockIds think_block_ids(bool extended_tags) const {
        (void)extended_tags;
        return {};
    }

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

    // One enrolled voice's waveform for `in.text` (docs/milestone-1.2.0.md A4).
    // `slot` is the lane. On success `wav` holds the container's bytes and
    // `error` is empty; on refusal or failure `error` says why and `wav` is
    // untouched. The container is named by the request's response_format
    // (prepare_speech admits only WAV).
    //
    // The default refuses: a backend with no TTS (the stub) says so here; the
    // route has already refused a build without --tts (prepare_speech).
    virtual bool generate_speech(const SpeechInput& in, int slot, std::string& wav,
                                 std::string& error, GenerationStats& stats) {
        (void)in; (void)slot; (void)wav; (void)stats;
        error = "this backend has no TTS";
        return false;
    }
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

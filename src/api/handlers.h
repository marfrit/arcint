#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "core/chat.h"
#include "exec/backend.h"

namespace lgc::api {

class ResponseStore;   // api/responses_state.h: the process's in-memory store

// A fixed set of lanes (DESIGN.md §4: /health reports free/total and queue
// depth). A lane is a memory reservation, not a queue position: N lanes means
// the startup arithmetic (§7.0.2a) reserved activations, GDN checkpoint rows
// and KV for N concurrent sequences, so an N+1st has nowhere to live. It is
// therefore refused with those numbers rather than queued behind a session
// that may decode for minutes — unless --queue-timeout says how long the
// caller may wait, in which case it waits that long first.
class SlotPool {
public:
    explicit SlotPool(int count);

    class Lease {
    public:
        Lease() = default;
        Lease(SlotPool* pool, int index) : pool_(pool), index_(index) {}
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept { *this = std::move(other); }
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();

        int index() const { return index_; }

    private:
        SlotPool* pool_  = nullptr;
        int       index_ = -1;
    };

    // Waits at most `timeout_seconds` (0 = do not wait at all) for a free lane.
    // A lease with index() < 0 means none came free. There is deliberately no
    // unbounded variant: an admission that can wait forever is the failure this
    // milestone replaced with a numbered refusal.
    Lease acquire_for(double timeout_seconds);
    // Non-blocking, for a continuation's slot affinity: index < 0 = invalid, 0
    // = taken, else the lease. Never waits and never picks another index.
    Lease try_acquire(int index);
    void  release(int index);

    int total() const;
    int free() const;
    int queue_depth() const;

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::vector<bool>       busy_;
    int                     waiting_ = 0;
};

// A named lane (DESIGN.md §4.2, amended 2026-10-07;
// docs/campaigns/lanes-agent-subagent.md): the name that picks it, its cap
// (the §3.8 400 is against this, not the process's n_ctx), the backend
// sequence it owns (the `slot` Backend::generate takes) and its own pool, so
// a request for one lane waits for, or is refused by, that lane alone.
struct Lane {
    std::string name;
    int         n_ctx = 0;
    int         seq   = 0;
    SlotPool*   pool  = nullptr;
};

struct Context {
    const Config* cfg     = nullptr;
    Backend*      backend = nullptr;
    SlotPool*     slots   = nullptr;
    // Named lanes, in --served-model-name order. Empty: one name, `slots`
    // admits and the model field is not binding (the behaviour before them).
    std::vector<Lane> lanes{};
    // The process's stop flag, owned by the HTTP server. A request prepared
    // while it is set carries it into the backend, which ends it at its next
    // token or ubatch boundary (DESIGN §3.7). Null outside a server (a test
    // calling the seam directly).
    std::atomic<bool>* stop_flag = nullptr;
    // The process's in-memory response store (docs/campaigns/feyntalk.md State,
    // W3). Null outside a server (a test calling the seam directly): then a
    // continuation cannot resolve and answers previous_response_not_found.
    std::shared_ptr<ResponseStore> responses;
};

struct HttpResult {
    int            status = 200;
    nlohmann::json body;
    // The request ended because its peer's connection is gone: the route
    // writes nothing then (there is nobody to receive it). Distinct from a
    // stop signal, which answers a 503 an alive client can still read.
    bool gone = false;
};

// The lanes --served-model-name A,B --lane-ctx CA,CB configure, a pool of one
// each, appended to `pools` (which must outlive the lanes). Empty without
// named lanes.
std::vector<Lane> make_lanes(const Config& cfg, std::vector<std::unique_ptr<SlotPool>>& pools);

// The lane a request's `model` picks. Without named lanes: -1, whatever the
// name (one process serves one model and there is nothing else it could
// mean). With them: the lane of that name; an empty name, or the artifact's
// canonical id, the first lane; any other name the 404.
std::optional<HttpResult> resolve_lane(const Context& ctx, const std::string& model, int& lane);

// The context a request on `lane` is admitted against (§3.8): the lane's cap,
// or the process's n_ctx for -1.
int lane_n_ctx(const Context& ctx, int lane);

// What the served decode line's "other" term is, after every segment that
// has its own timer is subtracted out. M11 (DESIGN §7.0.2aa row): on a
// drafting request this used to omit draft_verify_seconds and
// draft_propose_seconds, so the propose/verify/accept work of every
// drafting cycle (the served line's "graph" stays near-zero for those
// cycles — the forward cost is timed as `verify`, not `graph`) landed in
// `other` unlabelled instead of under its own name. `propose` and `verify`
// are exclusive terms (subtracted here); `re-forward` sits inside `graph` and
// `rollback` inside `other`, both zero on the paged path. draft_propose_seconds
// is written on the paged path only; on the stateful path propose remains part
// of `other`. Exposed here, not left static in handlers.cpp, so the
// accounting itself is unit-testable without a GPU.
double decode_other_seconds(const GenerationStats& stats);
// The decode line's conditional terms (review B4 follow-up): the suffix that
// lands on the line, extracted from log_stats so the wiring is unit-testable
// at the struct level.
std::string decode_suffix_for(const GenerationStats& stats);
// Whether a rendered prompt ends inside a think block: it ends with one tag
// family's opener and the newline the templates put after it. The chat path
// and /v1/completions ask this one function, built on the shared tag table, so
// they cannot disagree about the start state. With `extended_tags` every
// family's opener is checked -- a template that opens with a wider family's
// tag ends inside the block the same way; with the default family only the
// Qwen pair is checked, exactly as the chat path always did.
bool prompt_ends_inside_think_block(std::string_view prompt, bool extended_tags);

// loop-guard: the per-request counters as the response's `loop_guard` object.
// Exposed so the stats-to-JSON wiring is unit-testable without a card. A plain
// json: the body it lands in is one, so key order on the wire is alphabetical
// whatever type this returns; the seven keys and their values are the
// contract, and the handler tests read them off the serialized response.
nlohmann::json loop_guard_json(const GenerationStats& stats);

// Takes a lane for one request, or returns the 503 that says why not — with
// the lane counts in it (slots.total, slots.free), so "busy" is a number and not a mood.
std::optional<HttpResult> acquire_slot(const Context& ctx, SlotPool::Lease& out);
// The same for a request resolved to `lane` (resolve_lane; -1: no named
// lanes). `slot` is the backend sequence to generate on: the lease's index
// without named lanes, the lane's own sequence with them. A busy lane is
// waited for (--queue-timeout) or refused on its own; another lane is never
// taken instead.
std::optional<HttpResult> acquire_slot(const Context& ctx, int lane, SlotPool::Lease& out,
                                       int& slot, int affinity_slot = -1);

nlohmann::json health(const Context& ctx);
nlohmann::json props(const Context& ctx);
// GET /v1/models: one entry per served name, each with the context it is
// admitted against (a discovering proxy reads n_ctx from here, §4.2).
nlohmann::json models(const Context& ctx);

// Writes one SSE frame. Returns false when the client is gone, which aborts the
// request's work at the next boundary (DESIGN.md §3.7).
using SseWriter = std::function<bool(std::string_view)>;

// Whether the peer is still connected. The routes put it into the prepared
// request's GenerationInput (peer_alive): the generation calls it at its own
// boundaries and never writes to the stream to find out, and the input the
// backend runs is not copied to attach it.
using PeerProbe = std::function<bool()>;

struct PreparedChat {
    ChatRequest     req;
    GenerationInput input;
    ToolSchemas     schemas;
    bool            parse_tool_calls = false;
    bool            think_open       = false;  // the prompt ended inside a think block
    bool            think_tags_extended = false;  // also <thinking> / [THINK] (ModelStatus)
    ThinkBlockIds   think_ids;                   // the artifact's per-pair tag ids
    std::string     id;
    int64_t         created       = 0;
    int             prompt_tokens = 0;
    int             lane          = -1;  // resolve_lane's answer
    std::string     model;               // the name the response carries
};

struct PreparedCompletion {
    CompletionRequest req;
    GenerationInput   input;
    std::string       id;
    int64_t           created       = 0;
    int               prompt_tokens = 0;
    int               lane          = -1;
    std::string       model;
};

// Everything a /v1/audio/speech request needs after its own validations: the
// text, the enrolled voice name and the resolved sampler. No reference audio
// is in it (D5).
struct PreparedSpeech {
    SpeechRequest req;
    SpeechInput   input;
    int           lane  = -1;
    std::string   model;
};

// Everything that can be rejected happens here, before a single response byte
// is committed — including the context-overflow 400 (§3.8).
std::optional<HttpResult> prepare_chat(const Context& ctx, const nlohmann::json& body,
                                       PreparedChat& out);
// The half of prepare_chat that runs after the body is parsed: lane resolution,
// the template default, render_chat, the sampler and the think state. The
// Responses route maps its request into a ChatRequest and calls this same
// function (docs/campaigns/feyntalk.md W1), so there is one prompt builder.
std::optional<HttpResult> prepare_from_chat(const Context& ctx, ChatRequest req,
                                           PreparedChat& out);
std::optional<HttpResult> prepare_completion(const Context& ctx, const nlohmann::json& body,
                                             PreparedCompletion& out);
// /v1/audio/speech: the availability refusals ("this build has no TTS", no
// mmproj, no gen half), the D5 refusal of any reference-audio field, the
// unknown-voice 400 and the sampler.
std::optional<HttpResult> prepare_speech(const Context& ctx, const nlohmann::json& body,
                                         PreparedSpeech& out);

HttpResult run_chat(const Context& ctx, const PreparedChat& prep, int slot);
HttpResult run_completion(const Context& ctx, const PreparedCompletion& prep, int slot);

// What a speech request produced: the WAV bytes on success, an error envelope
// otherwise (HTTP's envelope, not the backend's own error string).
struct SpeechResult {
    int               status = 200;
    std::string       wav;
    nlohmann::json    error;
};
SpeechResult run_speech(const Context& ctx, const PreparedSpeech& prep, int slot);

// GET /v1/audio/voices: the enrolled names only (D5).
nlohmann::json voices(const Context& ctx);

void stream_chat(const Context& ctx, const PreparedChat& prep, int slot,
                 const SseWriter& write);
void stream_completion(const Context& ctx, const PreparedCompletion& prep, int slot,
                       const SseWriter& write);

}  // namespace lgc::api

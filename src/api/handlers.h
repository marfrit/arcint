#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "core/chat.h"
#include "exec/backend.h"

namespace lgc::api {

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
};

struct HttpResult {
    int            status = 200;
    nlohmann::json body;
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

// Takes a lane for one request, or returns the 503 that says why not — with
// the reservation arithmetic in it, so "busy" is a number and not a mood.
std::optional<HttpResult> acquire_slot(const Context& ctx, SlotPool::Lease& out);
// The same for a request resolved to `lane` (resolve_lane; -1: no named
// lanes). `slot` is the backend sequence to generate on: the lease's index
// without named lanes, the lane's own sequence with them. A busy lane is
// waited for (--queue-timeout) or refused on its own; another lane is never
// taken instead.
std::optional<HttpResult> acquire_slot(const Context& ctx, int lane, SlotPool::Lease& out,
                                       int& slot);

nlohmann::json health(const Context& ctx);
nlohmann::json props(const Context& ctx);
// GET /v1/models: one entry per served name, each with the context it is
// admitted against (a discovering proxy reads n_ctx from here, §4.2).
nlohmann::json models(const Context& ctx);

// Writes one SSE frame. Returns false when the client is gone, which aborts the
// request's work at the next boundary (DESIGN.md §3.7).
using SseWriter = std::function<bool(std::string_view)>;

struct PreparedChat {
    ChatRequest     req;
    GenerationInput input;
    ToolSchemas     schemas;
    bool            parse_tool_calls = false;
    bool            think_open       = false;  // the prompt ended inside a think block
    bool            think_tags_extended = false;  // also <thinking> / [THINK] (ModelStatus)
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

// Everything that can be rejected happens here, before a single response byte
// is committed — including the context-overflow 400 (§3.8).
std::optional<HttpResult> prepare_chat(const Context& ctx, const nlohmann::json& body,
                                       PreparedChat& out);
std::optional<HttpResult> prepare_completion(const Context& ctx, const nlohmann::json& body,
                                             PreparedCompletion& out);

HttpResult run_chat(const Context& ctx, const PreparedChat& prep, int slot);
HttpResult run_completion(const Context& ctx, const PreparedCompletion& prep, int slot);

void stream_chat(const Context& ctx, const PreparedChat& prep, int slot,
                 const SseWriter& write);
void stream_completion(const Context& ctx, const PreparedCompletion& prep, int slot,
                       const SseWriter& write);

}  // namespace lgc::api

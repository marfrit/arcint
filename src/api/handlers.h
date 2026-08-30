#pragma once

#include <condition_variable>
#include <functional>
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

struct Context {
    const Config* cfg     = nullptr;
    Backend*      backend = nullptr;
    SlotPool*     slots   = nullptr;
};

struct HttpResult {
    int            status = 200;
    nlohmann::json body;
};

// Takes a lane for one request, or returns the 503 that says why not — with
// the reservation arithmetic in it, so "busy" is a number and not a mood.
std::optional<HttpResult> acquire_slot(const Context& ctx, SlotPool::Lease& out);

nlohmann::json health(const Context& ctx);
nlohmann::json props(const Context& ctx);

// Writes one SSE frame. Returns false when the client is gone, which aborts the
// request's work at the next boundary (DESIGN.md §3.7).
using SseWriter = std::function<bool(std::string_view)>;

struct PreparedChat {
    ChatRequest     req;
    GenerationInput input;
    ToolSchemas     schemas;
    bool            parse_tool_calls = false;
    bool            think_open       = false;  // the prompt ended inside a think block
    std::string     id;
    int64_t         created       = 0;
    int             prompt_tokens = 0;
};

struct PreparedCompletion {
    CompletionRequest req;
    GenerationInput   input;
    std::string       id;
    int64_t           created       = 0;
    int               prompt_tokens = 0;
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

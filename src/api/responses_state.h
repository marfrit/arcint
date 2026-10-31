#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "api/handlers.h"
#include "api/responses.h"

// The in-memory response store (docs/campaigns/feyntalk.md "State (point 4)",
// W3). One per process, owned by Context; no disk, no restart survival.
namespace lgc::api {

struct StoredResponse {
    std::string id;
    int64_t     created_at   = 0;
    int64_t     completed_at = 0;
    std::string status;                       // completed / incomplete / failed
    int         lane = -1;                    // affinity: the lane that served it
    int         slot = -1;                    // the lease index (named lane: its seq)
    std::string previous_response_id;         // its own chain, "" for a first turn
    ChatRequest chat;                         // the input half (mapping, no assistant turn)
    nlohmann::json output_items = nlohmann::json::array();  // the output half
    int            prompt_tokens = 0;
    std::vector<int> prompt_ids;              // token mode's concatenation (W6)
    std::string    prompt;                    // the rendered prompt, for the hash check
    std::string    prefix_hash;              // sha256 of `prompt` (text mode)
    std::string    renderer_hash;            // W6
    nlohmann::json usage               = nullptr;
    nlohmann::json error               = nullptr;
    nlohmann::json incomplete_details  = nullptr;
    // Held by the in-flight registry too (W4); unused in W3.
    std::shared_ptr<std::atomic<bool>> cancel;
    int64_t  last_used = 0;                  // wall seconds, for the TTL
    uint64_t use_seq   = 0;                  // store-assigned LRU order
    uint64_t bytes     = 0;                  // what this entry retains
};

// Per-lane limits, LRU eviction by `last_used`, TTL sweep. Operator flags fill
// Limits; entries == 0 disables retention entirely (review R4).
// A response that is running right now (docs/campaigns/feyntalk.md Cancel, W4):
// the id registered before the first event, its cancel flag and the prepared
// request, so a cancel POST can answer the in-progress snapshot. `store=false`
// is registered too and removed when the request ends.
struct InFlightResponse {
    std::string                       id;
    int                               lane = -1;
    std::shared_ptr<std::atomic<bool>> cancel;
    std::shared_ptr<PreparedResponse> prep;   // alive for the snapshot, not copied
};

class ResponseStore {
public:
    struct Limits {
        int      entries = 32;
        uint64_t bytes   = 16ull * 1024 * 1024;
        int64_t  ttl_s   = 24 * 60 * 60;
    };

    explicit ResponseStore(Limits limits);

    // Stores (and evicts) unless entries is 0. Computes `bytes`.
    void put(std::shared_ptr<StoredResponse> entry);
    // Lane-scoped: a different lane is not found. Sweeps expired entries and
    // touches `last_used`.
    std::shared_ptr<StoredResponse> get(const std::string& id, int lane);
    void erase(const std::string& id);
    // Any stored entry, regardless of lane: the cancel POST has no lane.
    std::shared_ptr<StoredResponse> find_stored_any(const std::string& id);

    // The in-flight registry (W4): begin before the first event, end on every
    // exit. A cancel POST finds it here first, then a terminal stored entry.
    void begin_in_flight(std::shared_ptr<InFlightResponse> entry);
    std::shared_ptr<InFlightResponse> find_in_flight(const std::string& id);
    void end_in_flight(const std::string& id);
    // The TTL sweep, driven by `now`; put/get call it with the wall clock.
    void sweep(int64_t now);

    int      entries() const;
    uint64_t bytes() const;
    Limits   limits() const;

private:
    void evict_lane_locked(int lane, int64_t now);

    mutable std::mutex mutex_;
    Limits             limits_;
    uint64_t           use_counter_ = 0;
    std::unordered_map<std::string, std::shared_ptr<StoredResponse>> entries_;
    std::unordered_map<std::string, std::shared_ptr<InFlightResponse>> in_flight_;
};

// sha256 of a byte string as lowercase hex: the prefix hash the campaign names.
std::string sha256_hex(std::string_view bytes);

// Stores a finished turn when `store` is true and a store exists. The assistant
// turn is not appended to `chat`; the continuation rebuilds it from
// `output_items` (message text and function_call items).
void remember(const Context& ctx, const PreparedResponse& prep,
              const nlohmann::json& resource, int slot);

// A continuation the server treated as broken must not be silently continued
// again: evicts the entry it referenced.
void evict_continuation(const Context& ctx, const std::string& id);

}  // namespace lgc::api

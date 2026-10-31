#include "api/responses_state.h"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <limits>
#include <unordered_map>

#include "api/generate.h"
#include "util/log.h"
#include "util/sha256.h"

// The response store (docs/campaigns/feyntalk.md "State (point 4)", W3). No
// disk, no restart survival; one store per process, all access under one mutex.
namespace lgc::api {
namespace {

int64_t now_seconds_store() { return static_cast<int64_t>(std::time(nullptr)); }

// What an entry retains: the prompt ids dominate, the rendered prompt and the
// output items follow (the campaign's 16 MiB budget is stated in those terms).
uint64_t entry_bytes(const StoredResponse& e) {
    uint64_t n = 256;
    n += e.prompt.size();
    n += e.prompt_ids.size() * sizeof(int);
    for (const ChatMessage& m : e.chat.messages) {
        n += m.role.size() + m.content.size() + m.name.size() + m.tool_call_id.size();
        for (const ImagePart& img : m.images) n += img.url.size();
        for (const ToolCall& c : m.tool_calls)
            n += c.id.size() + c.name.size() + c.arguments.size();
    }
    for (const ToolSpec& t : e.chat.tools)
        n += t.name.size() + t.description.size() + t.parameters.dump().size();
    n += e.output_items.dump().size();
    n += e.prefix_hash.size() + e.renderer_hash.size() + e.previous_response_id.size();
    return n;
}

}  // namespace

std::string sha256_hex(std::string_view bytes) { return lgc::sha256_hex(bytes); }

ResponseStore::ResponseStore(Limits limits) : limits_(limits) {}

void ResponseStore::sweep(int64_t now) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (limits_.ttl_s <= 0) return;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (now - it->second->last_used > limits_.ttl_s) it = entries_.erase(it);
        else ++it;
    }
}

void ResponseStore::evict_lane_locked(int lane, int64_t) {
    if (limits_.entries <= 0) return;
    for (;;) {
        int      count = 0;
        uint64_t bytes = 0;
        for (const auto& [id, e] : entries_) {
            if (e->lane != lane) continue;
            ++count;
            bytes += e->bytes;
        }
        if (count <= limits_.entries && bytes <= limits_.bytes) return;

        std::shared_ptr<StoredResponse> oldest;
        for (const auto& [id, e] : entries_) {
            if (e->lane != lane) continue;
            // LRU by the store's monotonic use order: the wall clock has
            // one-second resolution, so three puts in one second would tie.
            if (!oldest || e->use_seq < oldest->use_seq) oldest = e;
        }
        if (!oldest) return;
        entries_.erase(oldest->id);
    }
}

void ResponseStore::put(std::shared_ptr<StoredResponse> entry) {
    if (!entry || limits_.entries <= 0) return;   // R4: retention disabled
    if (entry->prefix_hash.empty() && !entry->prompt.empty())
        entry->prefix_hash = sha256_hex(entry->prompt);
    entry->bytes     = entry_bytes(*entry);
    entry->last_used = now_seconds_store();

    std::lock_guard<std::mutex> guard(mutex_);
    const int64_t now = entry->last_used;
    if (limits_.ttl_s > 0) {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (now - it->second->last_used > limits_.ttl_s) it = entries_.erase(it);
            else ++it;
        }
    }
    const int lane = entry->lane;
    entry->use_seq = ++use_counter_;
    entries_[entry->id] = std::move(entry);
    evict_lane_locked(lane, now);
}

std::shared_ptr<StoredResponse> ResponseStore::get(const std::string& id, int lane) {
    if (limits_.entries <= 0) return nullptr;   // R4: nothing is retained
    std::lock_guard<std::mutex> guard(mutex_);
    const int64_t               now = now_seconds_store();
    if (limits_.ttl_s > 0) {
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (now - it->second->last_used > limits_.ttl_s) it = entries_.erase(it);
            else ++it;
        }
    }
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    if (it->second->lane != lane) return nullptr;   // wrong lane: not found
    it->second->last_used = now;
    it->second->use_seq   = ++use_counter_;
    return it->second;
}

void ResponseStore::erase(const std::string& id) {
    std::lock_guard<std::mutex> guard(mutex_);
    entries_.erase(id);
}

std::shared_ptr<StoredResponse> ResponseStore::find_stored_any(const std::string& id) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = entries_.find(id);
    return it == entries_.end() ? nullptr : it->second;
}

void ResponseStore::begin_in_flight(std::shared_ptr<InFlightResponse> entry) {
    if (!entry) return;
    std::lock_guard<std::mutex> guard(mutex_);
    in_flight_[entry->id] = std::move(entry);
}

std::shared_ptr<InFlightResponse> ResponseStore::find_in_flight(const std::string& id) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = in_flight_.find(id);
    return it == in_flight_.end() ? nullptr : it->second;
}

void ResponseStore::end_in_flight(const std::string& id) {
    std::lock_guard<std::mutex> guard(mutex_);
    in_flight_.erase(id);
}

int ResponseStore::entries() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int>(entries_.size());
}

uint64_t ResponseStore::bytes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    uint64_t n = 0;
    for (const auto& [id, e] : entries_) n += e->bytes;
    return n;
}

ResponseStore::Limits ResponseStore::limits() const { return limits_; }

void remember(const Context& ctx, const PreparedResponse& prep,
              const nlohmann::json& resource, int slot) {
    if (!prep.req.store || !ctx.responses || !resource.is_object()) return;
    auto e = std::make_shared<StoredResponse>();
    e->id        = resource.value("id", "");
    e->created_at = resource.value("created_at", 0);
    if (resource.contains("completed_at") && resource["completed_at"].is_number())
        e->completed_at = resource["completed_at"].get<int64_t>();
    e->status = resource.value("status", "completed");
    e->lane   = prep.chat.lane;
    e->slot   = slot;
    e->previous_response_id = prep.req.has_previous_response_id ? prep.req.previous_response_id : "";
    // The input half only: the continuation rebuilds the assistant turn from
    // output_items (message text and function_call items).
    e->chat         = prep.chat.req;
    e->output_items = resource.contains("output") ? resource["output"] : nlohmann::json::array();
    e->prompt_tokens = prep.chat.prompt_tokens;
    e->prompt_ids    = prep.chat.input.prompt_ids;
    e->prompt        = prep.chat.input.prompt;
    // feyntalk W3.5: a truncated turn served a prompt with replayed reasoning
    // dropped, while e->chat keeps the full history. The continuation re-renders
    // the FULL history and drops reasoning again under its own truncation
    // field, so the prefix check compares render(e->chat); the hash is taken
    // over that full render, not over the served (truncated) prompt.
    e->prefix_hash   = prep.chat.truncation.blocks_dropped > 0
                           ? sha256_hex(ctx.backend->render_chat(e->chat))
                           : sha256_hex(e->prompt);
    e->usage         = resource.contains("usage") ? resource["usage"] : nlohmann::json(nullptr);
    e->error         = resource.contains("error") ? resource["error"] : nlohmann::json(nullptr);
    e->incomplete_details = resource.contains("incomplete_details") ? resource["incomplete_details"]
                                                                    : nlohmann::json(nullptr);
    ctx.responses->put(std::move(e));
}

void evict_continuation(const Context& ctx, const std::string& id) {
    if (!ctx.responses || id.empty()) return;
    ctx.responses->erase(id);
}

}  // namespace lgc::api

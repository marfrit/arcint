#include "api/responses_state.h"

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <limits>
#include <unordered_map>

#include "api/generate.h"
#include "util/log.h"

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

// sha256, lowercase hex. The campaign names sha256 for prefix_hash; this is a
// self-contained implementation so no new dependency enters the public tree.
constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

std::string sha256_hex(std::string_view bytes) {
    // Message schedule and the compression function over 64-byte blocks.
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::string m(bytes);
    const uint64_t bit_len = static_cast<uint64_t>(m.size()) * 8;
    m.push_back('\x80');
    while (m.size() % 64 != 56) m.push_back('\0');
    for (int i = 7; i >= 0; --i)
        m.push_back(static_cast<char>((bit_len >> (i * 8)) & 0xff));

    for (size_t block = 0; block < m.size(); block += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const auto* p = reinterpret_cast<const unsigned char*>(m.data() + block + i * 4);
            w[i] = (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
                   (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + kSha256K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d  = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t v : h) {
        for (int i = 7; i >= 0; --i) out.push_back(kHex[(v >> (i * 4)) & 0xf]);
    }
    return out;
}

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

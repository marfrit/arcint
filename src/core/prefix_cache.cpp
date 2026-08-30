#include "core/prefix_cache.h"

#include <algorithm>
#include <cstring>

#include "util/sha256.h"

namespace lgc {
namespace {

// Two 64-bit halves of a SHA-256 over (key, previous hash, block tokens).
// Chaining means a hash identifies a whole prefix, not just one block, so two
// different prefixes cannot collide merely by sharing a final block.
void hash_block(uint64_t key, uint64_t& hi, uint64_t& lo, const int* tokens, size_t count) {
    Sha256 h;
    h.update(&key, sizeof(key));
    h.update(&hi, sizeof(hi));
    h.update(&lo, sizeof(lo));
    for (size_t i = 0; i < count; ++i) {
        const int32_t t = tokens[i];
        h.update(&t, sizeof(t));
    }

    const std::string hex = h.hex();
    hi = std::stoull(hex.substr(0, 16), nullptr, 16);
    lo = std::stoull(hex.substr(16, 16), nullptr, 16);
}

size_t blob_bytes(const PrefixCache::StateBlob& state) {
    size_t n = 0;
    for (const std::vector<uint8_t>& t : state) n += t.size();
    return n;
}

}  // namespace

void PrefixCache::chain_block(uint64_t key, uint64_t& hi, uint64_t& lo, const int* tokens,
                              size_t count) {
    hash_block(key, hi, lo, tokens, count);
}

PrefixCache::PrefixCache(size_t budget_bytes, int block_size, uint64_t key, size_t host_budget_bytes)
    : budget_(budget_bytes), host_budget_(host_budget_bytes), block_size_(std::max(1, block_size)),
      key_(key) {}

PrefixCache::~PrefixCache() { clear(); }

// The pages go back to the pool when the *last* holder lets go, not when the
// cache drops the entry: a lane that hit this prefix a moment ago may still be
// mapping it. Tying that to the shared_ptr rather than to the list makes the
// two impossible to get out of step.
PrefixCache::EntryPtr PrefixCache::make_entry() const {
    ReleaseFn fn = release_;
    return EntryPtr(new Entry(), [fn](Entry* e) {
        if (fn && !e->blocks.empty()) fn(e->blocks);
        delete e;
    });
}

PrefixCache::Hit PrefixCache::lookup(const std::vector<int>& tokens) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.lookups;
    Hit hit;
    if (entries_.empty() || tokens.size() <= static_cast<size_t>(block_size_)) return hit;

    // Walk the chain forward one block at a time, remembering the deepest entry
    // that both hashes and compares equal.
    uint64_t hi = 0, lo = 0;
    const size_t usable = tokens.size() - 1;  // one token must remain to be run

    auto best = entries_.end();
    size_t best_len = 0;

    for (size_t end = static_cast<size_t>(block_size_); end <= usable;
         end += static_cast<size_t>(block_size_)) {
        hash_block(key_, hi, lo, tokens.data() + end - static_cast<size_t>(block_size_),
                   static_cast<size_t>(block_size_));

        auto it = std::find_if(entries_.begin(), entries_.end(), [&](const EntryPtr& e) {
            return e->hi == hi && e->lo == lo;
        });
        // A depth with no entry is ordinary — nothing says every boundary was
        // ever stored — so keep chaining rather than giving up here.
        if (it == entries_.end()) continue;

        // Verify the tokens. A hash hit that disagrees on tokens is a collision,
        // and reusing it would silently answer a different prompt.
        if ((*it)->tokens.size() != end ||
            !std::equal((*it)->tokens.begin(), (*it)->tokens.end(), tokens.begin())) {
            ++stats_.collisions;
            continue;
        }
        best     = it;
        best_len = end;
    }

    if (best == entries_.end()) return hit;

    entries_.splice(entries_.begin(), entries_, best);  // most recently used
    hit.matched_tokens = best_len;
    hit.keep           = entries_.front();
    hit.state          = &entries_.front()->state;
    hit.blocks         = &entries_.front()->blocks;
    hit.tiered         = entries_.front()->tiered;

    ++stats_.hits;
    stats_.hit_tokens += best_len;
    return hit;
}

void PrefixCache::insert(const std::vector<int>& tokens, size_t prefix_len, StateBlob state,
                         std::vector<int32_t> blocks) {
    std::lock_guard<std::mutex> guard(mutex_);
    // The caller took a reference for the cache before calling; every path out
    // of here has to hand it back, or the pages leak and the pool shrinks by a
    // prefix on every rejected insert.
    auto give_back = [&] {
        if (release_ && !blocks.empty()) release_(blocks);
    };
    if (budget_ == 0 || prefix_len == 0) { give_back(); return; }
    if (prefix_len % static_cast<size_t>(block_size_) != 0) { give_back(); return; }
    if (prefix_len > tokens.size()) { give_back(); return; }

    uint64_t hi = 0, lo = 0;
    for (size_t end = static_cast<size_t>(block_size_); end <= prefix_len;
         end += static_cast<size_t>(block_size_)) {
        hash_block(key_, hi, lo, tokens.data() + end - static_cast<size_t>(block_size_),
                   static_cast<size_t>(block_size_));
    }

    for (const EntryPtr& e : entries_) {
        if (e->hi != hi || e->lo != lo) continue;
        // Same rule as lookup: a hash match is not identity. Without the token
        // check a collision would silently discard a genuinely different prefix
        // and never be counted.
        if (e->tokens.size() == prefix_len &&
            std::equal(e->tokens.begin(), e->tokens.end(), tokens.begin())) {
            // Already held, so the caller's reference for the cache is one
            // reference too many; hand it straight back.
            give_back();
            return;
        }
        ++stats_.collisions;
    }

    const size_t bytes = blob_bytes(state);
    // A single snapshot larger than the whole budget is not cacheable; storing
    // it would evict everything and then itself.
    if (bytes > budget_) {
        give_back();
        return;
    }

    EntryPtr entry = make_entry();
    entry->hi      = hi;
    entry->lo      = lo;
    entry->tokens.assign(tokens.begin(), tokens.begin() + static_cast<long>(prefix_len));
    entry->bytes   = bytes;
    entry->state   = std::move(state);
    entry->blocks  = std::move(blocks);

    evict_to_fit(bytes);
    bytes_ += bytes;
    stats_.blocks_held += entry->blocks.size();
    entries_.push_front(std::move(entry));

    ++stats_.inserts;
    stats_.bytes_resident = bytes_;
    stats_.entries        = entries_.size();
}

void PrefixCache::evict_to_fit(size_t incoming) {
    while (!entries_.empty() && bytes_ + incoming > budget_) {
        drop_back();
        ++stats_.evictions;
    }
    stats_.bytes_resident = bytes_;
    stats_.host_bytes     = host_bytes_;
    stats_.entries        = entries_.size();
    stats_.tiered_entries = static_cast<size_t>(std::count_if(
        entries_.begin(), entries_.end(), [](const EntryPtr& p) { return p->tiered; }));
}

void PrefixCache::drop_back() {
    bytes_ -= entries_.back()->bytes;
    host_bytes_ -= entries_.back()->host_bytes;
    stats_.blocks_held -= entries_.back()->blocks.size();
    entries_.pop_back();
}

bool PrefixCache::evict_oldest() {
    std::lock_guard<std::mutex> guard(mutex_);
    // Called for pages. Find the least recently used entry that still holds
    // any: a tiered entry has none to give, and the entry budget's own
    // eviction is evict_to_fit, not this.
    auto victim = entries_.end();
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        if (!(*it)->tiered) { victim = std::prev(it.base()); break; }
    }
    if (victim == entries_.end()) return false;
    Entry& e = **victim;
    if (host_budget_ > 0 && demote_ && !e.blocks.empty() && demote_(e)) {
        // Pages are on the host now and released; the entry stays hittable.
        stats_.blocks_held -= e.blocks.size();
        e.blocks.clear();
        e.tiered = true;
        host_bytes_ += e.host_bytes;
        ++stats_.demotions;
        ++stats_.evictions;
        // The host tier has its own budget; the least recently used tiered
        // entries go first, which may be this one if it alone is too large.
        while (host_bytes_ > host_budget_) {
            auto old = entries_.end();
            for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
                if ((*it)->tiered) { old = std::prev(it.base()); break; }
            }
            if (old == entries_.end()) break;
            bytes_ -= (*old)->bytes;
            host_bytes_ -= (*old)->host_bytes;
            entries_.erase(old);
        }
    } else {
        bytes_ -= e.bytes;
        stats_.blocks_held -= e.blocks.size();
        entries_.erase(victim);
        ++stats_.evictions;
    }
    stats_.bytes_resident = bytes_;
    stats_.host_bytes     = host_bytes_;
    stats_.entries        = entries_.size();
    stats_.tiered_entries = static_cast<size_t>(std::count_if(
        entries_.begin(), entries_.end(), [](const EntryPtr& p) { return p->tiered; }));
    return true;
}

bool PrefixCache::promote(const EntryRef& entry, std::vector<int32_t> blocks) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (EntryPtr& p : entries_) {
        if (p.get() != entry.get()) continue;
        if (!p->tiered) return false;
        host_bytes_ -= p->host_bytes;
        p->host_bytes = 0;
        p->host_kv.clear();
        p->host_kv.shrink_to_fit();
        p->blocks = std::move(blocks);
        p->tiered = false;
        stats_.blocks_held += p->blocks.size();
        ++stats_.promotions;
        stats_.host_bytes     = host_bytes_;
        stats_.tiered_entries = static_cast<size_t>(std::count_if(
            entries_.begin(), entries_.end(), [](const EntryPtr& q) { return q->tiered; }));
        return true;
    }
    return false;
}

bool PrefixCache::may_accept(size_t bytes) const {
    return budget_ > 0 && bytes > 0 && bytes <= budget_;
}

void PrefixCache::clear() {
    std::lock_guard<std::mutex> guard(mutex_);
    entries_.clear();
    bytes_                = 0;
    host_bytes_           = 0;
    stats_.bytes_resident = 0;
    stats_.host_bytes     = 0;
    stats_.entries        = 0;
    stats_.tiered_entries = 0;
    stats_.blocks_held    = 0;
}

PrefixCacheStats PrefixCache::stats() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return stats_;
}

}  // namespace lgc

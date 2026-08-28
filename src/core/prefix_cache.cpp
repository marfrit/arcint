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

PrefixCache::PrefixCache(size_t budget_bytes, int block_size, uint64_t key)
    : budget_(budget_bytes), block_size_(std::max(1, block_size)), key_(key) {}

PrefixCache::Hit PrefixCache::lookup(const std::vector<int>& tokens) {
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

        auto it = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) {
            return e.hi == hi && e.lo == lo;
        });
        // A depth with no entry is ordinary — nothing says every boundary was
        // ever stored — so keep chaining rather than giving up here.
        if (it == entries_.end()) continue;

        // Verify the tokens. A hash hit that disagrees on tokens is a collision,
        // and reusing it would silently answer a different prompt.
        if (it->tokens.size() != end ||
            !std::equal(it->tokens.begin(), it->tokens.end(), tokens.begin())) {
            ++stats_.collisions;
            continue;
        }
        best     = it;
        best_len = end;
    }

    if (best == entries_.end()) return hit;

    entries_.splice(entries_.begin(), entries_, best);  // most recently used
    hit.matched_tokens = best_len;
    hit.state          = &entries_.front().state;

    ++stats_.hits;
    stats_.hit_tokens += best_len;
    return hit;
}

void PrefixCache::insert(const std::vector<int>& tokens, size_t prefix_len, StateBlob state) {
    if (budget_ == 0 || prefix_len == 0) return;
    if (prefix_len % static_cast<size_t>(block_size_) != 0) return;
    if (prefix_len > tokens.size()) return;

    uint64_t hi = 0, lo = 0;
    for (size_t end = static_cast<size_t>(block_size_); end <= prefix_len;
         end += static_cast<size_t>(block_size_)) {
        hash_block(key_, hi, lo, tokens.data() + end - static_cast<size_t>(block_size_),
                   static_cast<size_t>(block_size_));
    }

    for (const Entry& e : entries_) {
        if (e.hi != hi || e.lo != lo) continue;
        // Same rule as lookup: a hash match is not identity. Without the token
        // check a collision would silently discard a genuinely different prefix
        // and never be counted.
        if (e.tokens.size() == prefix_len &&
            std::equal(e.tokens.begin(), e.tokens.end(), tokens.begin())) {
            return;  // already held
        }
        ++stats_.collisions;
    }

    Entry entry;
    entry.hi     = hi;
    entry.lo     = lo;
    entry.tokens.assign(tokens.begin(), tokens.begin() + static_cast<long>(prefix_len));
    entry.bytes  = blob_bytes(state);
    entry.state  = std::move(state);

    // A single snapshot larger than the whole budget is not cacheable; storing
    // it would evict everything and then itself.
    if (entry.bytes > budget_) return;

    evict_to_fit(entry.bytes);
    bytes_ += entry.bytes;
    entries_.push_front(std::move(entry));

    ++stats_.inserts;
    stats_.bytes_resident = bytes_;
    stats_.entries        = entries_.size();
}

void PrefixCache::evict_to_fit(size_t incoming) {
    while (!entries_.empty() && bytes_ + incoming > budget_) {
        bytes_ -= entries_.back().bytes;
        entries_.pop_back();
        ++stats_.evictions;
    }
    stats_.bytes_resident = bytes_;
    stats_.entries        = entries_.size();
}

void PrefixCache::clear() {
    entries_.clear();
    bytes_                = 0;
    stats_.bytes_resident = 0;
    stats_.entries        = 0;
}

}  // namespace lgc

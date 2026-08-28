#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <vector>

// DESIGN.md §3.4. A prefix hit restores the attention KV *and* the GDN
// recurrent state at the same block boundary — both or neither. On this
// architecture that is not an aspiration but the only honest option: a GDN
// state cannot be reconstructed from a KV page, so a half-restored prefix is a
// silently wrong answer.
//
// The state blobs are opaque here. Their layout belongs to the backend; this
// class only decides *which* prefix may be reused, and it verifies the tokens
// on a hit rather than trusting the hash (OV GenAI shipped a collision bug,
// their #3489).
namespace lgc {

struct PrefixCacheStats {
    uint64_t lookups        = 0;
    uint64_t hits           = 0;
    uint64_t hit_tokens     = 0;
    uint64_t inserts        = 0;
    uint64_t evictions      = 0;
    uint64_t collisions     = 0;  // hash matched, tokens did not
    size_t   bytes_resident = 0;
    size_t   entries        = 0;
};

class PrefixCache {
public:
    using StateBlob = std::vector<std::vector<uint8_t>>;

    // `block_size` is the granularity of a reusable prefix, in tokens; it is
    // deliberately the KV block size, so a checkpoint always lands on a page
    // boundary and reuse is exact by construction rather than by interpolation.
    PrefixCache(size_t budget_bytes, int block_size, uint64_t key);

    struct Hit {
        size_t           matched_tokens = 0;
        const StateBlob* state          = nullptr;
    };

    // Longest cached prefix of `tokens` that ends on a block boundary. Never
    // returns the whole sequence: at least one token must remain to be run, or
    // there would be no logits to sample from.
    Hit lookup(const std::vector<int>& tokens);

    // Stores the state after exactly `prefix_len` tokens. `prefix_len` must be
    // a multiple of the block size.
    void insert(const std::vector<int>& tokens, size_t prefix_len, StateBlob state);

    void            clear();
    PrefixCacheStats stats() const { return stats_; }
    int              block_size() const { return block_size_; }
    size_t           budget_bytes() const { return budget_; }

    // The keyed 128-bit chain hash over whole blocks, exposed for tests.
    static void chain_block(uint64_t key, uint64_t& hi, uint64_t& lo, const int* tokens,
                            size_t count);

private:
    struct Entry {
        uint64_t         hi = 0;
        uint64_t         lo = 0;
        std::vector<int> tokens;  // exactly prefix_len of them, for verification
        StateBlob        state;
        size_t           bytes = 0;
    };

    void evict_to_fit(size_t incoming);

    size_t   budget_;
    int      block_size_;
    uint64_t key_;
    size_t   bytes_ = 0;

    // Front is most recently used.
    std::list<Entry> entries_;
    PrefixCacheStats stats_;
};

}  // namespace lgc

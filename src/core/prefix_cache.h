#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// DESIGN.md §3.4. A prefix hit restores the attention KV *and* the GDN
// recurrent state at the same block boundary — both or neither. On this
// architecture that is not an aspiration but the only honest option: a GDN
// state cannot be reconstructed from a KV page, so a half-restored prefix is a
// silently wrong answer.
//
// The two halves are kept in different places, each for a measured reason. The
// GDN checkpoint is a fixed-size host blob (~32 MiB per row) and travels in the
// entry. The KV is large, already on the card, and is *not* copied: the entry
// holds references to the pages themselves, so a hit maps them rather than
// restoring them. That is what M6 replaces the single-slot pool-epoch tag with
// — with two lanes a page can be live in one sequence, cached, and mapped by
// the other at the same time, and only a refcount can say when it is free.
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
    uint64_t demotions      = 0;  // entries whose pages went to the host tier
    uint64_t promotions     = 0;  // tiered entries whose pages came back
    size_t   bytes_resident = 0;
    size_t   entries        = 0;
    size_t   tiered_entries = 0;  // entries holding host buffers instead of pages
    size_t   host_bytes     = 0;  // KV bytes parked in host memory
    size_t   blocks_held    = 0;  // KV pages this cache is keeping alive
};

class PrefixCache {
public:
    using StateBlob = std::vector<std::vector<uint8_t>>;

    // Called with an entry's pages when the last holder of that entry lets go,
    // which may be an eviction or the end of the last request that hit it.
    using ReleaseFn = std::function<void(const std::vector<int32_t>&)>;

    // `block_size` is the granularity of a reusable prefix, in tokens; it is
    // deliberately the KV block size, so a checkpoint always lands on a page
    // boundary and reuse is exact by construction rather than by interpolation.
    PrefixCache(size_t budget_bytes, int block_size, uint64_t key, size_t host_budget_bytes = 0);
    ~PrefixCache();

    // The pages of an entry go back to the pool through this. Set once, before
    // the cache is used; without it the cache is host-side only, which is what
    // the stateful reference path wants.
    void set_release(ReleaseFn fn) { release_ = std::move(fn); }

    // The host tier (DESIGN §4.4). An entry evicted for its pages is demoted
    // rather than dropped: the callback copies the pages into the entry's host
    // buffers and releases them, and the entry stays hittable. A hit on such an
    // entry is promoted by the backend -- pages allocated, buffers copied back
    // -- and then `promote()` records the new pages. The callback runs with the
    // cache locked; lookups wait for the copy, which is the point of it.
    struct Entry;
    using DemoteFn = std::function<bool(Entry&)>;
    void set_demote(DemoteFn fn) { demote_ = std::move(fn); }

    struct Entry {
        uint64_t             hi = 0;
        uint64_t             lo = 0;
        std::vector<int>     tokens;  // exactly prefix_len of them, for verification
        StateBlob            state;
        std::vector<int32_t> blocks;  // the KV pages of this prefix, refcounted
        size_t               bytes = 0;
        // Host tier: one buffer per KV pool tensor, pages in `blocks` order as
        // they were when demoted. Empty while the entry is resident.
        std::vector<std::vector<uint8_t>> host_kv;
        size_t                            host_bytes = 0;
        bool                              tiered     = false;
    };
    using EntryRef = std::shared_ptr<const Entry>;

    struct Hit {
        size_t                      matched_tokens = 0;
        const StateBlob*            state          = nullptr;
        const std::vector<int32_t>* blocks         = nullptr;
        // Holds the entry — and with it the pages — alive for as long as the
        // caller needs to take its own references. An entry evicted in the
        // window between lookup and use would otherwise hand its pages to the
        // other lane while this one was still reading them.
        EntryRef                    keep;
        bool                        tiered         = false;  // pages live on the host; promote first
    };
    // Records the pages a promoted entry now owns; the backend copied its host
    // buffers into them first. False if the entry is gone.
    bool promote(const EntryRef& entry, std::vector<int32_t> blocks);

    // Longest cached prefix of `tokens` that ends on a block boundary. Never
    // returns the whole sequence: at least one token must remain to be run, or
    // there would be no logits to sample from.
    Hit lookup(const std::vector<int>& tokens);

    // Stores the state after exactly `prefix_len` tokens, together with the KV
    // pages that hold that prefix. `prefix_len` must be a multiple of the block
    // size, which is also what makes every stored page a complete one — and a
    // complete page is never written again, so sharing it needs no copy.
    // The caller must already hold a reference for the cache's own use.
    void insert(const std::vector<int>& tokens, size_t prefix_len, StateBlob state,
                std::vector<int32_t> blocks = {});

    // Would an entry of this size survive insertion? Lets a caller skip an
    // expensive serialisation whose result would only be dropped.
    bool may_accept(size_t bytes) const;

    // Drops the least recently used entry. The pool calls this when it is out
    // of pages: cached pages are reclaimable, a live sequence's are not.
    bool evict_oldest();

    void             clear();
    PrefixCacheStats stats() const;
    int              block_size() const { return block_size_; }
    size_t           budget_bytes() const { return budget_; }

    // The keyed 128-bit chain hash over whole blocks, exposed for tests.
    static void chain_block(uint64_t key, uint64_t& hi, uint64_t& lo, const int* tokens,
                            size_t count);

private:
    using EntryPtr = std::shared_ptr<Entry>;

    EntryPtr make_entry() const;
    void     evict_to_fit(size_t incoming);   // caller holds the lock

    void     drop_back();                     // caller holds the lock
    size_t   budget_;
    size_t   host_budget_;
    int      block_size_;
    uint64_t key_;
    size_t   bytes_ = 0;

    // Off the hot path by construction: one lookup and at most one insert per
    // request, against a per-token decode loop that never touches this.
    mutable std::mutex mutex_;
    ReleaseFn          release_;
    DemoteFn           demote_;
    size_t             host_bytes_ = 0;

    // Front is most recently used.
    std::list<EntryPtr> entries_;
    PrefixCacheStats    stats_;
};

}  // namespace lgc

#include "core/prefix_cache.h"
#include "harness.h"

#include <numeric>

using namespace lgc;

namespace {

PrefixCache::StateBlob blob(size_t bytes, uint8_t fill) {
    return {std::vector<uint8_t>(bytes, fill)};
}

std::vector<int> seq(int n, int base = 0) {
    std::vector<int> v(static_cast<size_t>(n));
    std::iota(v.begin(), v.end(), base);
    return v;
}

constexpr size_t kMiB = 1024 * 1024;

}  // namespace

TEST(prefix_cache_disabled_by_a_zero_budget) {
    PrefixCache c(0, 16, 1);
    c.insert(seq(64), 32, blob(8, 1));
    CHECK_EQ(c.lookup(seq(64)).matched_tokens, 0u);
    CHECK_EQ(c.stats().inserts, 0u);
}

TEST(prefix_cache_hits_the_longest_block_aligned_prefix) {
    PrefixCache c(kMiB, 16, 7);
    const auto  tokens = seq(128);

    c.insert(tokens, 32, blob(64, 1));
    c.insert(tokens, 64, blob(64, 2));

    const auto hit = c.lookup(tokens);
    CHECK_EQ(hit.matched_tokens, 64u);
    CHECK(hit.state != nullptr);
    CHECK_EQ((*hit.state)[0][0], uint8_t{2});
}

TEST(prefix_cache_never_consumes_the_whole_prompt) {
    // Matching every token would leave no logits to sample from, so an entry
    // covering the entire prompt is simply unusable for that prompt.
    PrefixCache c(kMiB, 16, 7);
    const auto  tokens = seq(64);
    c.insert(tokens, 64, blob(8, 1));
    CHECK_EQ(c.lookup(tokens).matched_tokens, 0u);

    // One block shallower is usable, and leaves 16 tokens to run.
    c.insert(tokens, 48, blob(8, 2));
    const auto hit = c.lookup(tokens);
    CHECK_EQ(hit.matched_tokens, 48u);
    CHECK_EQ((*hit.state)[0][0], uint8_t{2});

    // The full-length entry is still there for a longer prompt that extends it.
    auto longer = tokens;
    longer.push_back(4242);
    CHECK_EQ(c.lookup(longer).matched_tokens, 64u);
}

TEST(prefix_cache_matches_a_shared_prefix_of_a_longer_prompt) {
    PrefixCache c(kMiB, 16, 7);
    c.insert(seq(64), 32, blob(8, 5));

    auto longer = seq(64);
    longer.insert(longer.end(), {999, 998, 997});
    const auto hit = c.lookup(longer);
    CHECK_EQ(hit.matched_tokens, 32u);
}

TEST(prefix_cache_misses_when_the_prefix_diverges) {
    PrefixCache c(kMiB, 16, 7);
    c.insert(seq(64), 32, blob(8, 5));

    auto other = seq(64);
    other[3]   = 12345;  // inside the first block
    CHECK_EQ(c.lookup(other).matched_tokens, 0u);
}

TEST(prefix_cache_divergence_after_the_first_block_still_reuses_the_first) {
    PrefixCache c(kMiB, 16, 7);
    const auto  tokens = seq(64);
    c.insert(tokens, 16, blob(8, 1));
    c.insert(tokens, 32, blob(8, 2));

    auto other = tokens;
    other[20]  = 4242;  // second block differs
    const auto hit = c.lookup(other);
    CHECK_EQ(hit.matched_tokens, 16u);
    CHECK_EQ((*hit.state)[0][0], uint8_t{1});
}

TEST(prefix_cache_reuses_a_prefix_stored_at_only_one_depth) {
    // Nothing stores every boundary, so a lookup must keep chaining past
    // depths that hold no entry instead of giving up at the first gap.
    PrefixCache c(kMiB, 16, 7);
    c.insert(seq(128), 96, blob(8, 9));

    const auto hit = c.lookup(seq(128));
    CHECK_EQ(hit.matched_tokens, 96u);
    CHECK_EQ((*hit.state)[0][0], uint8_t{9});
    CHECK_EQ(c.stats().collisions, 0u);
}

TEST(prefix_cache_rejects_a_misaligned_insert) {
    PrefixCache c(kMiB, 16, 7);
    c.insert(seq(64), 20, blob(8, 1));  // 20 is not a multiple of 16
    CHECK_EQ(c.stats().inserts, 0u);
}

TEST(prefix_cache_evicts_least_recently_used_under_budget) {
    PrefixCache c(200, 16, 7);

    c.insert(seq(64, 0), 16, blob(100, 1));
    c.insert(seq(64, 1000), 16, blob(100, 2));
    CHECK_EQ(c.stats().entries, 2u);

    // Touch the first so the second becomes least recently used.
    CHECK_EQ(c.lookup(seq(64, 0)).matched_tokens, 16u);

    c.insert(seq(64, 2000), 16, blob(100, 3));
    CHECK_EQ(c.stats().evictions, 1u);
    CHECK_EQ(c.lookup(seq(64, 0)).matched_tokens, 16u);     // survived
    CHECK_EQ(c.lookup(seq(64, 1000)).matched_tokens, 0u);   // evicted
}

TEST(prefix_cache_refuses_an_entry_larger_than_the_budget) {
    PrefixCache c(64, 16, 7);
    c.insert(seq(64), 16, blob(1024, 1));
    CHECK_EQ(c.stats().inserts, 0u);
    CHECK_EQ(c.stats().entries, 0u);
}

TEST(prefix_cache_does_not_store_the_same_prefix_twice) {
    PrefixCache c(kMiB, 16, 7);
    c.insert(seq(64), 32, blob(8, 1));
    c.insert(seq(64), 32, blob(8, 1));
    CHECK_EQ(c.stats().inserts, 1u);
}

TEST(prefix_cache_key_separates_otherwise_identical_chains) {
    uint64_t hi_a = 0, lo_a = 0, hi_b = 0, lo_b = 0;
    const auto tokens = seq(16);
    PrefixCache::chain_block(1, hi_a, lo_a, tokens.data(), tokens.size());
    PrefixCache::chain_block(2, hi_b, lo_b, tokens.data(), tokens.size());
    CHECK(hi_a != hi_b || lo_a != lo_b);
}

TEST(prefix_cache_chain_depends_on_order) {
    const auto a = seq(16);
    auto       b = a;
    std::swap(b[0], b[1]);

    uint64_t hi_a = 0, lo_a = 0, hi_b = 0, lo_b = 0;
    PrefixCache::chain_block(9, hi_a, lo_a, a.data(), a.size());
    PrefixCache::chain_block(9, hi_b, lo_b, b.data(), b.size());
    CHECK(hi_a != hi_b || lo_a != lo_b);
}

TEST(prefix_cache_clear_releases_everything) {
    PrefixCache c(kMiB, 16, 7);
    c.insert(seq(64), 32, blob(128, 1));
    c.clear();
    CHECK_EQ(c.stats().entries, 0u);
    CHECK_EQ(c.stats().bytes_resident, 0u);
    CHECK_EQ(c.lookup(seq(64)).matched_tokens, 0u);
}

// ------------------------------------------------------------ KV page ownership
//
// M6: an entry holds references to the KV pages of its prefix, so the pages of a
// cached prefix outlive the request that produced them and are shared with
// whichever lane hits it next. Every path out of the cache has to hand those
// references back exactly once — a leak here shrinks the pool by a prefix per
// rejected insert, which shows up much later as a lane that cannot get pages.

namespace {

// Records what the cache hands back, standing in for the block pool.
struct Released {
    std::vector<int32_t> ids;
    void operator()(const std::vector<int32_t>& b) { ids.insert(ids.end(), b.begin(), b.end()); }
};

}  // namespace

TEST(prefix_cache_returns_pages_when_it_refuses_an_entry) {
    Released    freed;
    PrefixCache c(kMiB, 16, 3);
    c.set_release([&](const std::vector<int32_t>& b) { freed(b); });

    // Not on a block boundary: refused, and the pages must come straight back.
    c.insert(seq(64), 20, blob(64, 1), {10, 11});
    CHECK_EQ(freed.ids, (std::vector<int32_t>{10, 11}));
    CHECK_EQ(c.stats().inserts, 0u);

    // A prefix longer than the token list: same.
    c.insert(seq(32), 64, blob(64, 1), {12});
    CHECK_EQ(freed.ids.size(), 3u);

    // A blob larger than the whole budget: same again.
    PrefixCache tiny(64, 16, 3);
    Released    tiny_freed;
    tiny.set_release([&](const std::vector<int32_t>& b) { tiny_freed(b); });
    tiny.insert(seq(64), 32, blob(4096, 1), {20, 21});
    CHECK_EQ(tiny_freed.ids, (std::vector<int32_t>{20, 21}));

    // And an insert of something already held returns the surplus reference
    // rather than double-holding the pages.
    c.insert(seq(64), 32, blob(64, 1), {30, 31});
    CHECK_EQ(c.stats().inserts, 1u);
    const size_t before = freed.ids.size();
    c.insert(seq(64), 32, blob(64, 1), {32, 33});
    CHECK_EQ(c.stats().inserts, 1u);
    CHECK_EQ(freed.ids.size(), before + 2);
}

TEST(prefix_cache_holds_pages_until_the_last_user_lets_go) {
    Released    freed;
    PrefixCache c(kMiB, 16, 3);
    c.set_release([&](const std::vector<int32_t>& b) { freed(b); });

    c.insert(seq(128), 64, blob(64, 1), {1, 2, 3, 4});
    CHECK_EQ(c.stats().blocks_held, 4u);

    // A lane is holding the entry. Evicting it must not free the pages out from
    // under that lane — the hit keeps them alive.
    PrefixCache::Hit hit = c.lookup(seq(128));
    CHECK_EQ(hit.matched_tokens, 64u);
    CHECK(hit.blocks != nullptr);
    CHECK_EQ(*hit.blocks, (std::vector<int32_t>{1, 2, 3, 4}));

    CHECK(c.evict_oldest());
    CHECK_EQ(freed.ids.size(), 0u);   // still mapped by the hit

    hit = PrefixCache::Hit{};         // the lane is done
    CHECK_EQ(freed.ids, (std::vector<int32_t>{1, 2, 3, 4}));
}

TEST(prefix_cache_returns_pages_when_it_is_cleared) {
    Released    freed;
    PrefixCache c(kMiB, 16, 3);
    c.set_release([&](const std::vector<int32_t>& b) { freed(b); });

    c.insert(seq(128), 32, blob(64, 1), {7});
    c.insert(seq(128), 64, blob(64, 2), {8});
    CHECK_EQ(c.stats().blocks_held, 2u);

    c.clear();
    CHECK_EQ(freed.ids.size(), 2u);
    CHECK_EQ(c.stats().blocks_held, 0u);
}

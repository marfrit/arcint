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

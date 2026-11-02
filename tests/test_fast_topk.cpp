// LLAMA_SAMPLER_FAST_TOPK (default 0): llama_top_k_from_logits selects the k
// largest logits straight from the row and repeats the heap steps of libstdc++
// std::partial_sort, so entries and the order of equal logits are the same as
// the full-array path. The red case is a comparator/step change: make the heap
// tie branch `>` into `>=` (or drop the tie's index order) and the equal-logit
// case below fails. Device-free: CPU logits only, no context, no card.
#ifdef ARCINT_LLAMA

#include "harness.h"

#include "llama.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// deterministic xorshift: the same arrays on every run
struct rng {
    uint64_t s;
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<uint32_t>(s);
    }
    float unit() { return static_cast<float>(next() >> 8) / 16777216.0f; }
};

bool same(const llama_token_data& a, const llama_token_data& b) {
    // the logit is copied, never recomputed: compare its bytes (NaN included)
    return a.id == b.id && std::memcmp(&a.logit, &b.logit, sizeof(float)) == 0;
}

// the old path: the whole row as a candidate array, then std::partial_sort
std::vector<llama_token_data> full_array_topk(const std::vector<float>& logits, int32_t k) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        cur[i] = llama_token_data{ static_cast<llama_token>(i), logits[i], 0.0f };
    }
    const auto comp = [](const llama_token_data& a, const llama_token_data& b) { return a.logit > b.logit; };
    std::partial_sort(cur.begin(), cur.begin() + k, cur.end(), comp);
    cur.resize(k);
    return cur;
}

void check_array(const std::vector<float>& logits, int32_t k) {
    const std::vector<llama_token_data> want = full_array_topk(logits, k);
    std::vector<llama_token_data> got(k);
    llama_top_k_from_logits(logits.data(), static_cast<int32_t>(logits.size()), k, got.data());
    CHECK_EQ(got.size(), want.size());
    for (int32_t i = 0; i < k; ++i) {
        CHECK_EQ(got[i].id, want[i].id);
        CHECK(same(got[i], want[i]));
    }
}

}  // namespace

TEST(fast_topk_matches_partial_sort_random) {
    rng g{0x9e3779b97f4a7c15ull};
    const size_t sizes[] = {64, 1024, 4096};
    const int32_t ks[] = {1, 2, 10, 128, 2048};
    for (size_t n : sizes) {
        for (int32_t k_in : ks) {
            const int32_t k = std::min<int32_t>(k_in, static_cast<int32_t>(n) - 1);
            for (int rep = 0; rep < 40; ++rep) {
                std::vector<float> l(n);
                for (float& v : l) {
                    v = g.unit() * 20.0f - 10.0f;
                }
                check_array(l, k);
            }
        }
    }
}

TEST(fast_topk_matches_partial_sort_long_vocab) {
    // 248320 is the served vocabulary: three shapes, the full row once each
    rng g{0x123456789abcdef1ull};
    const size_t n = 248320;
    for (int32_t k : {1, 10, 128}) {
        std::vector<float> l(n);
        for (float& v : l) {
            v = g.unit() * 40.0f - 20.0f;
        }
        check_array(l, k);
    }
}

TEST(fast_topk_ties_keep_the_partial_sort_order) {
    // many equal logits, mixed with a few larger ones: the tie order is the
    // libstdc++ heap's, and it has to match token for token
    const size_t n = 4096;
    std::vector<float> l(n, 1.0f);
    for (size_t i = 0; i < n; i += 7) {
        l[i] = 2.0f;
    }
    for (int32_t k : {1, 3, 10, 128, 2048}) {
        check_array(l, k);
    }
}

TEST(fast_topk_all_equal) {
    std::vector<float> l(1024, -3.5f);
    for (int32_t k : {1, 4, 128, 1023}) {
        check_array(l, k);
    }
}

TEST(fast_topk_nan_and_inf) {
    const size_t n = 2048;
    std::vector<float> l(n);
    for (size_t i = 0; i < n; ++i) {
        l[i] = static_cast<float>(i % 17) - 8.0f;
    }
    l[3]   = std::numeric_limits<float>::quiet_NaN();
    l[17]  = std::numeric_limits<float>::quiet_NaN();
    l[100] = std::numeric_limits<float>::infinity();
    l[101] = -std::numeric_limits<float>::infinity();
    for (int32_t k : {1, 5, 128}) {
        check_array(l, k);
    }
    // a NaN inside the first k and larger values after it
    std::vector<float> m(512, 0.0f);
    m[0] = std::numeric_limits<float>::quiet_NaN();
    m[1] = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 2; i < m.size(); ++i) {
        m[i] = static_cast<float>(i);
    }
    for (int32_t k : {1, 2, 10, 128}) {
        check_array(m, k);
    }
}

#endif  // ARCINT_LLAMA

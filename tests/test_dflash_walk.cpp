// The DFlash2 lattice walk (src/exec/dflash_walk.h) on synthetic lattices:
// the path follows each pick's transition row, not a row's own best score;
// the p_min stop; the step's budget; a lattice too narrow for its candidates.
#include "exec/dflash_walk.h"
#include "harness.h"

#include <vector>

using namespace lgc;

namespace {

constexpr int    kTopK   = 3;
constexpr size_t kStride = 16;   // wider than 3 + 9, as the drafter's rows are (n_embd, zero-padded)

// one block position: its candidate ids and its transition rows, one per
// predecessor candidate (trans[p][j]: the score of candidate j after candidate p)
struct Pos {
    std::vector<int>                ids;
    std::vector<std::vector<float>> trans;
};

std::vector<float> lattice(const std::vector<Pos>& block) {
    std::vector<float> r(block.size() * kStride, 0.0f);
    for (size_t i = 0; i < block.size(); ++i) {
        float* row = r.data() + i * kStride;
        for (int j = 0; j < kTopK; ++j) row[j] = static_cast<float>(block[i].ids[static_cast<size_t>(j)]);
        for (int p = 0; p < kTopK; ++p)
            for (int j = 0; j < kTopK; ++j)
                row[kTopK + p * kTopK + j] = block[i].trans[static_cast<size_t>(p)][static_cast<size_t>(j)];
    }
    return r;
}

// Row 0 is the anchor's, never read (filled with ids a walk starting there
// would draft). Row 1's predecessor is the anchor: its transition rows are
// all the same, and their best is candidate 1 (11). Row 2 after 11 (its
// candidate 1) is best at candidate 2 (22), while its first transition row
// is best at 20; row 3 after 22 (candidate 2) is best at 32, its first
// transition row at 31. So the right path [11, 22, 32] differs from a walk
// that reads one transition row a position whatever came before ([11, 20,
// 31]), from one that reads the transition matrix transposed (the scores of
// candidate p after each candidate: [10, ...]), and from one that starts at
// the anchor's row.
std::vector<Pos> block_of_four() {
    const std::vector<float> same = {1.0f, 5.0f, 2.0f};
    return {
        {{90, 91, 92}, {{0, 0, 9}, {0, 0, 9}, {0, 0, 9}}},
        {{10, 11, 12}, {same, same, same}},
        {{20, 21, 22}, {{9, 0, 0}, {0, 0, 7}, {0, 8, 0}}},
        {{30, 31, 32}, {{0, 6, 0}, {6, 0, 0}, {0, 0, 6}}},
    };
}

std::vector<int> walk(const std::vector<float>& l, float p_min = 0.0f, int n_max = 7, size_t stride = kStride) {
    return dflash_walk(l.data(), l.size() / kStride, stride, kTopK, p_min, n_max);
}

}  // namespace

TEST(dflash_walk_follows_the_picks_transition_rows) {
    const std::vector<float> l = lattice(block_of_four());
    CHECK(walk(l) == std::vector<int>({11, 22, 32}));
}

TEST(dflash_walk_is_not_the_rows_own_best) {
    // what the mutants would draft, spelled out: the right path is neither
    const std::vector<float> l = lattice(block_of_four());
    const std::vector<int>   got = walk(l);
    CHECK(got != std::vector<int>({11, 20, 31}));   // one transition row a position
    CHECK(got.empty() || got[0] != 10);             // the matrix read transposed
    CHECK(got.empty() || got[0] != 92);             // the walk started at the anchor's row
}

TEST(dflash_walk_budget_keeps_the_prefix) {
    const std::vector<float> l = lattice(block_of_four());
    CHECK(walk(l, 0.0f, 2) == std::vector<int>({11, 22}));
    CHECK(walk(l, 0.0f, 1) == std::vector<int>({11}));
    CHECK(walk(l, 0.0f, 0).empty());
    CHECK(walk(l, 0.0f, 7) == walk(l, 0.0f, 3));   // a block of 4 drafts 3 at most
}

TEST(dflash_walk_min_p_stops_at_an_unsure_pick) {
    std::vector<Pos> b = block_of_four();
    // after 11, row 2 is torn between 21 and 22: softmax at the pick ~0.52
    b[2].trans[1] = {0.0f, 3.0f, 3.1f};
    const std::vector<float> l = lattice(b);
    CHECK(walk(l) == std::vector<int>({11, 22, 32}));           // no stop: the block whole
    CHECK(walk(l, 0.6f) == std::vector<int>({11}));             // the unsure pick and what follows stay out
    CHECK(walk(l, 0.5f) == std::vector<int>({11, 22, 32}));     // 0.52 passes 0.5
    // row 1: exp(1-5) + exp(2-5) puts the pick at ~0.936
    CHECK(walk(l, 0.95f).empty());
}

TEST(dflash_walk_ties_take_the_first_candidate) {
    std::vector<Pos> b = block_of_four();
    b[1].trans = {{4, 4, 1}, {4, 4, 1}, {4, 4, 1}};   // 10 and 11 tie: std::max_element's first, as llama.cpp
    const std::vector<float> l = lattice(b);
    const std::vector<int>   w = walk(l);
    CHECK(!w.empty());
    CHECK_EQ(w[0], 10);
    CHECK_EQ(w[1], 20);   // and the next row follows candidate 0's transition row
}

TEST(dflash_walk_refuses_a_lattice_too_narrow) {
    const std::vector<float> l = lattice(block_of_four());
    CHECK(walk(l, 0.0f, 7, 11).empty());   // 3 + 9 floats a row do not fit 11
    CHECK(dflash_walk(nullptr, 4, kStride, kTopK, 0.0f, 7).empty());
    CHECK(dflash_walk(l.data(), 4, kStride, 0, 0.0f, 7).empty());
    CHECK(dflash_walk(l.data(), 1, kStride, kTopK, 0.0f, 7).empty());   // the anchor alone drafts nothing
}

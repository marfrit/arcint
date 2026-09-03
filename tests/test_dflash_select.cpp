// M11 (the M11 design note (not in the repository) V): the DFlash2 selector's scoring, pulled out of
// backend_ov.cpp into src/core/dflash_select.{h,cpp} so the lattice and the
// two selection strategies (today's greedy, and the new exact Viterbi) can be
// checked without OpenVINO or a card. See dflash_select.h for the score
// model: score(row, prev, j) = unary[row][j] + lambda * dot(pred_key(prev),
// hp[row] elementwise* succ_key[row][j]).
//
// Findings fixed here, from ):
//   -- a 3-row lattice where even a 1-step-lookahead heuristic (not only
//         plain greedy) fails to find Viterbi's optimum, plus a seeded
//         randomized brute-force cross-check.
//   -- LatticeRow now carries hp/succ_key (not just the precomputed
//         row_key), so tests can exercise kGreedy's legacy association order
//         directly; a hand-traced non-integer-float test locks it in.
//   -- a zero-candidate row must truncate identically in both modes.
#include "core/dflash_select.h"
#include "harness.h"

#include <limits>
#include <random>

using namespace lgc;

namespace {

// Builds one row, computing row_key = hp elementwise* succ_key so tests only
// ever state hp/succ_key once (matching what backend_ov.cpp's dflash_select
// does) rather than hand-multiplying row_key separately.
LatticeRow make_row(std::vector<int> token, std::vector<float> unary,
                     std::vector<std::vector<float>> pred_key, std::vector<float> hp,
                     std::vector<std::vector<float>> succ_key) {
    LatticeRow row;
    row.token    = std::move(token);
    row.unary    = std::move(unary);
    row.pred_key = std::move(pred_key);
    row.hp       = std::move(hp);
    row.succ_key = std::move(succ_key);
    row.row_key.assign(row.token.size(), std::vector<float>(row.hp.size()));
    for (size_t j = 0; j < row.token.size(); ++j) {
        for (size_t d = 0; d < row.hp.size(); ++d) row.row_key[j][d] = row.hp[d] * row.succ_key[j][d];
    }
    return row;
}

// Recomputes the additive score of an arbitrary token path through `lattice`
// (not necessarily one dflash_select_path would produce), by matching each
// row's chosen token id back to its candidate index and using row_key (the
// same ingredient kViterbi reads) -- independent of dflash_select_path's own
// implementation, used to brute-force verify it.
float path_score(const Lattice& lattice, const std::vector<int>& path, float lambda) {
    float                      total = 0.0f;
    const std::vector<float>* prev_key = &lattice.anchor_key;
    for (size_t r = 0; r < lattice.rows.size(); ++r) {
        const LatticeRow& row = lattice.rows[r];
        size_t            idx = row.token.size();
        for (size_t j = 0; j < row.token.size(); ++j) {
            if (row.token[j] == path[r]) {
                idx = j;
                break;
            }
        }
        // A path built from this lattice's own candidates always finds idx.
        float dot = 0.0f;
        for (size_t d = 0; d < prev_key->size() && d < row.row_key[idx].size(); ++d) {
            dot += (*prev_key)[d] * row.row_key[idx][d];
        }
        total += row.unary[idx] + lambda * dot;
        prev_key = &row.pred_key[idx];
    }
    return total;
}

// Builds the 2-row poisoning lattice: row 0's locally-best candidate (higher
// unary) leads into a row 1 where every continuation is bad, while row 0's
// locally-worse candidate unlocks a much better row 1. r = 1. anchor_key is
// 0 so row 0 has no bilinear term (isolates the unary comparison there);
// row 1's hp is 1 so its succ_key values equal what row_key would have been.
Lattice poisoning_lattice_2row() {
    Lattice L;
    L.anchor_key = {0.0f};
    L.rows.push_back(make_row({0, 1}, {1.0f, 0.9f}, {{1.0f}, {-1.0f}}, {0.0f}, {{0.0f}, {0.0f}}));
    // From token 0's key (1.0): 1*0.5=0.5, 1*-100=-100 -- capped low.
    // From token 1's key (-1.0): -1*0.5=-0.5, -1*-100=100 -- huge.
    L.rows.push_back(make_row({2, 3}, {0.0f, 0.0f}, {{0.0f}, {0.0f}}, {1.0f}, {{0.5f}, {-100.0f}}));
    return L;
}

}  // namespace

// (a) RED: greedy commits to row 0's locally-best candidate and is stuck with
// a capped row 1; Viterbi sees the whole lattice and takes row 0's
// locally-worse candidate because it unlocks a far better row 1. Brute-forced
// over all 2x2 = 4 paths so the assertion does not depend on hand arithmetic
// alone, and both selectors' outputs are asserted so the test fails if
// Viterbi were greedy in disguise (same path as greedy).
TEST(dflash_select_viterbi_beats_greedy_on_a_poisoned_lattice) {
    const Lattice L      = poisoning_lattice_2row();
    const float   lambda = 1.0f;

    const std::vector<int> greedy_path = dflash_select_path(L, DflashSelectMode::kGreedy, lambda);
    const std::vector<int> viterbi_path =
        dflash_select_path(L, DflashSelectMode::kViterbi, lambda);

    CHECK_EQ(greedy_path, std::vector<int>({0, 2}));
    CHECK_EQ(viterbi_path, std::vector<int>({1, 3}));
    CHECK(greedy_path != viterbi_path);

    // Brute force: every path through {0,1} x {2,3}.
    float       brute_best  = -std::numeric_limits<float>::infinity();
    std::vector<int> brute_path;
    for (int a : {0, 1}) {
        for (int b : {2, 3}) {
            const std::vector<int> p{a, b};
            const float            s = path_score(L, p, lambda);
            if (s > brute_best) {
                brute_best = s;
                brute_path = p;
            }
        }
    }
    CHECK_EQ(brute_path, viterbi_path);
    CHECK(path_score(L, viterbi_path, lambda) > path_score(L, greedy_path, lambda));
}

// (a2) Review follow-up: the 2-row lattice above does not separate
// Viterbi from a "greedy with 1-step lookahead" heuristic -- on a 2-row
// lattice, 1-step lookahead from row 0 already sees all the way to the end,
// so it is exhaustive there. This 3-row lattice hides the trap two rows deep:
//
//   row0: A(unary 2.0) beats B(unary 1.0); anchor_key=0 so row 0's bilinear
//         term is 0 and the row-0 comparison is pure unary.
//   row1: candidates C, D score {1.0, -1.0} from A and {-1.0, 1.0} from B --
//         a 1-step lookahead from row 0 sees the SAME best value (1.0) down
//         either branch (max(1.0,-1.0) == max(-1.0,1.0)), so it cannot see a
//         reason to prefer B over A from here: lookahead-1 also picks A.
//   row2: reachable via C capped at {0.5, -100}, but via D at {-0.5, 100} --
//         the trap. A 1-step lookahead freshly re-applied AT row1 (comparing
//         C's own score + best row2 via C against D's own score + best row2
//         via D) DOES see this and picks D over C, partially recovering --
//         but it can never undo the wrong row-0 commitment to A.
//
// Hand-computed scores for all 8 paths (A/B x C/D x E/F):
//   ACE=3.5  ACF=-97.0  ADE=0.5  ADF=101.0
//   BCE=0.5  BCF=-100.0 BDE=1.5  BDF=102.0  <- true optimum
//
// So: plain greedy picks A,C,E (3.5) -- the worst of the three. A fresh
// per-row 1-step-lookahead heuristic picks A (row 0, wrongly) then
// self-corrects to D,F at rows 1-2, landing on A,D,F (101.0) -- better than
// greedy but still short of the true optimum. Viterbi finds B,D,F (102.0).
// This is asserted directly (Viterbi's score exceeds the lookahead-1 path's
// score, not just greedy's), so the test does not just show "beats greedy in
// disguise" but "beats a real 1-ply search too".
TEST(dflash_select_viterbi_beats_a_one_step_lookahead_on_a_3row_poisoned_lattice) {
    Lattice L;
    L.anchor_key = {0.0f};
    L.rows.push_back(make_row({10, 11}, {2.0f, 1.0f}, {{1.0f}, {-1.0f}}, {0.0f}, {{0.0f}, {0.0f}}));
    L.rows.push_back(make_row({20, 21}, {0.0f, 0.0f}, {{1.0f}, {-1.0f}}, {1.0f}, {{1.0f}, {-1.0f}}));
    L.rows.push_back(make_row({30, 31}, {0.0f, 0.0f}, {{0.0f}, {0.0f}}, {1.0f}, {{0.5f}, {-100.0f}}));

    const float lambda = 1.0f;
    const std::vector<int> greedy_path  = dflash_select_path(L, DflashSelectMode::kGreedy, lambda);
    const std::vector<int> viterbi_path = dflash_select_path(L, DflashSelectMode::kViterbi, lambda);
    const std::vector<int> lookahead1_path{10, 21, 31};   // A, D, F -- hand-derived above

    CHECK_EQ(greedy_path, std::vector<int>({10, 20, 30}));
    CHECK_EQ(viterbi_path, std::vector<int>({11, 21, 31}));
    CHECK_NEAR(path_score(L, greedy_path, lambda), 3.5, 1e-6);
    CHECK_NEAR(path_score(L, lookahead1_path, lambda), 101.0, 1e-6);
    CHECK_NEAR(path_score(L, viterbi_path, lambda), 102.0, 1e-6);
    CHECK(path_score(L, viterbi_path, lambda) > path_score(L, lookahead1_path, lambda));

    // Brute force over all 2^3 = 8 paths, independent of dflash_select_path.
    float            brute_best = -std::numeric_limits<float>::infinity();
    std::vector<int> brute_path;
    for (int a : {10, 11}) {
        for (int c : {20, 21}) {
            for (int e : {30, 31}) {
                const std::vector<int> p{a, c, e};
                const float            s = path_score(L, p, lambda);
                if (s > brute_best) {
                    brute_best = s;
                    brute_path = p;
                }
            }
        }
    }
    CHECK_EQ(brute_path, viterbi_path);
}

// (a3): a seeded randomized cross-check against an independent brute
// force, well past what two hand-built lattices can cover. 200 lattices x
// 5 rows x 4 candidates x r=3 = 1024 paths each; fixed seed for a
// reproducible red/green (a genuine implementation bug reliably fails this,
// as 
// the correct implementation but would trivially find some against a
// lookahead-bounded one).
TEST(dflash_select_viterbi_matches_brute_force_on_random_lattices) {
    constexpr size_t kRows   = 5;
    constexpr size_t kCand   = 4;
    constexpr size_t kRank   = 3;
    constexpr size_t kTrials = 200;
    constexpr float   kLambda = 1.0f;

    std::mt19937                          rng(0xD11A5EEDu);
    std::uniform_real_distribution<float> val(-5.0f, 5.0f);

    size_t total_combos = 1;
    for (size_t i = 0; i < kRows; ++i) total_combos *= kCand;

    for (size_t trial = 0; trial < kTrials; ++trial) {
        Lattice L;
        L.anchor_key.resize(kRank);
        for (float& x : L.anchor_key) x = val(rng);

        int next_token = 0;
        for (size_t r = 0; r < kRows; ++r) {
            std::vector<int>                token(kCand);
            std::vector<float>              unary(kCand);
            std::vector<std::vector<float>> pred_key(kCand, std::vector<float>(kRank));
            std::vector<float>              hp(kRank);
            std::vector<std::vector<float>> succ_key(kCand, std::vector<float>(kRank));
            for (float& x : hp) x = val(rng);
            for (size_t j = 0; j < kCand; ++j) {
                token[j] = next_token++;
                unary[j] = val(rng);
                for (size_t d = 0; d < kRank; ++d) {
                    pred_key[j][d] = val(rng);
                    succ_key[j][d] = val(rng);
                }
            }
            L.rows.push_back(make_row(token, unary, pred_key, hp, succ_key));
        }

        const std::vector<int> viterbi = dflash_select_path(L, DflashSelectMode::kViterbi, kLambda);

        float            brute_best = -std::numeric_limits<float>::infinity();
        std::vector<int> brute_path;
        for (size_t combo = 0; combo < total_combos; ++combo) {
            size_t           rem = combo;
            std::vector<int> path(kRows);
            for (size_t r = 0; r < kRows; ++r) {
                const size_t j = rem % kCand;
                rem /= kCand;
                path[r] = L.rows[r].token[j];
            }
            const float s = path_score(L, path, kLambda);
            if (s > brute_best) {
                brute_best = s;
                brute_path = path;
            }
        }

        if (viterbi != brute_path) {
            t::fail(__FILE__, __LINE__,
                    std::string("trial ") + std::to_string(trial) +
                        ": viterbi != brute force\n           viterbi: " + t::show(viterbi) +
                        "\n           brute:   " + t::show(brute_path));
        }
    }
}

// (b) Review follow-up: kGreedy at lambda=1 must reproduce the LEGACY
// selector's exact association order -- t[d] = pred_key(prev)[d] * hp[d],
// THEN score = unary[j]; score += t[d] * succ_key[j][d] accumulated over d in
// order (dflash_select.h's header comment; backend_ov.cpp's pre-M11
// dflash_select ran precisely this sequence). This is the point of this
// test, not merely "some candidate wins": unary is tied (0.25 == 0.25) so
// the bilinear term alone decides, and every value is an exactly
// representable binary fraction (halves and quarters) so the hand trace
// below is exact in real arithmetic and therefore exact in float32 too --
// deliberately, so this test is hand-verifiable without any rounding
// uncertainty. (Because every value is exactly representable, this
// particular test cannot by itself catch a pure re-association that leaves
// the real-number result unchanged; what it does verify, independently of
// dflash_select.cpp's own source, is that kGreedy runs the documented
// SEQUENCE -- right operand order, right accumulator start, right vector in
// each role -- not a differently-shaped computation that would give a
// different winner here.)
//
// anchor_key (this row's "prev" key) = [1.5, -0.5], hp = [2.0, 0.5]:
//   t = [1.5*2.0, -0.5*0.5] = [3.0, -0.25]
// candidate 100: succ_key=[1.0,-2.0]:
//   score = 0.25; += 3.0*1.0 = 3.0 -> 3.25; += -0.25*-2.0 = 0.5 -> 3.75
// candidate 101: succ_key=[-1.0,2.0]:
//   score = 0.25; += 3.0*-1.0 = -3.0 -> -2.75; += -0.25*2.0 = -0.5 -> -3.25
// candidate 100 wins (3.75 > -3.25).
TEST(dflash_select_greedy_matches_the_legacy_association_order) {
    Lattice L;
    L.anchor_key = {1.5f, -0.5f};
    L.rows.push_back(make_row({100, 101}, {0.25f, 0.25f}, {{0.0f, 0.0f}, {0.0f, 0.0f}},
                              {2.0f, 0.5f}, {{1.0f, -2.0f}, {-1.0f, 2.0f}}));

    const std::vector<int> path = dflash_select_path(L, DflashSelectMode::kGreedy, 1.0f);
    CHECK_EQ(path, std::vector<int>({100}));
}

// (c) lambda=0 ignores the bilinear term entirely: both selectors reduce to
// an independent per-row argmax of the unary score, which for the lattice
// below picks a DIFFERENT path than lambda=1's (token 11, not 10, is row 0's
// unary-argmax). hp=[1,1] on every row makes succ_key equal to what row_key
// would have been, reproducing the original 3-row/K=3/r=2 hand-computation
// this test was built from.
TEST(dflash_select_lambda_zero_ignores_the_bilinear_term) {
    Lattice L;
    L.anchor_key = {1.0f, 0.0f};

    L.rows.push_back(make_row({10, 11, 12}, {2.0f, 2.5f, 1.0f},
                              {{1.0f, 1.0f}, {0.0f, 2.0f}, {-1.0f, 1.0f}}, {1.0f, 1.0f},
                              {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}}));
    L.rows.push_back(make_row({20, 21, 22}, {0.5f, 0.5f, 3.0f},
                              {{0.0f, 0.0f}, {1.0f, -1.0f}, {2.0f, 2.0f}}, {1.0f, 1.0f},
                              {{1.0f, 0.0f}, {2.0f, 0.0f}, {0.0f, -1.0f}}));
    L.rows.push_back(make_row({30, 31, 32}, {1.0f, 0.0f, 0.0f},
                              {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}, {1.0f, 1.0f},
                              {{1.0f, 1.0f}, {2.0f, 2.0f}, {-1.0f, -1.0f}}));

    const std::vector<int> expected{11, 22, 30};   // per-row unary argmax
    CHECK_EQ(dflash_select_path(L, DflashSelectMode::kGreedy, 0.0f), expected);
    CHECK_EQ(dflash_select_path(L, DflashSelectMode::kViterbi, 0.0f), expected);

    // And it really is different from lambda=1's path -- lambda has an effect.
    CHECK(dflash_select_path(L, DflashSelectMode::kGreedy, 1.0f) != expected);
}

// (d) A smaller top-k restricts which candidates the selector can reach at
// all: the same row scored with the full candidate set picks a token that
// simply is not offered once the set shrinks, and the selector then picks
// the best of what remains rather than, say, silently keeping the old best.
TEST(dflash_select_topk_restricts_the_reachable_candidates) {
    Lattice full;
    full.anchor_key = {0.0f};   // bilinear term is 0 regardless: unary decides
    full.rows.push_back(
        make_row({1, 2, 3}, {5.0f, 9.0f, 7.0f}, {{0.0f}, {0.0f}, {0.0f}}, {0.0f},
                 {{0.0f}, {0.0f}, {0.0f}}));

    CHECK_EQ(dflash_select_path(full, DflashSelectMode::kGreedy, 1.0f), std::vector<int>({2}));
    CHECK_EQ(dflash_select_path(full, DflashSelectMode::kViterbi, 1.0f), std::vector<int>({2}));

    // Same scenario, but the top-k cut before token 2 (unary 9, the winner
    // above) ever reached the lattice -- only tokens 1 and 3 are offered.
    Lattice restricted;
    restricted.anchor_key = {0.0f};
    restricted.rows.push_back(
        make_row({1, 3}, {5.0f, 7.0f}, {{0.0f}, {0.0f}}, {0.0f}, {{0.0f}, {0.0f}}));

    CHECK_EQ(dflash_select_path(restricted, DflashSelectMode::kGreedy, 1.0f),
             std::vector<int>({3}));
    CHECK_EQ(dflash_select_path(restricted, DflashSelectMode::kViterbi, 1.0f),
             std::vector<int>({3}));
}

TEST(dflash_select_empty_lattice_is_an_empty_path) {
    Lattice L;
    L.anchor_key = {0.0f};
    CHECK(dflash_select_path(L, DflashSelectMode::kGreedy, 1.0f).empty());
    CHECK(dflash_select_path(L, DflashSelectMode::kViterbi, 1.0f).empty());
}

//  A row with zero candidates is a caller bug (never reachable from
// backend_ov.cpp's dflash_select: top-k has k>=1 by --dflash-topk's [1,64]
// validation, and the vocab is always far larger), but the two modes must
// fail the SAME way rather than one truncating cleanly and the other
// indexing into an empty vector. Both must return the path up to but not
// including the empty row.
TEST(dflash_select_zero_candidate_row_truncates_both_modes_identically) {
    Lattice L;
    L.anchor_key = {0.0f};
    L.rows.push_back(make_row({1, 2}, {5.0f, 3.0f}, {{0.0f}, {0.0f}}, {0.0f}, {{0.0f}, {0.0f}}));
    L.rows.push_back(LatticeRow{});   // no candidates at all

    CHECK_EQ(dflash_select_path(L, DflashSelectMode::kGreedy, 1.0f), std::vector<int>({1}));
    CHECK_EQ(dflash_select_path(L, DflashSelectMode::kViterbi, 1.0f), std::vector<int>({1}));

    // And when the very FIRST row is empty, both modes return nothing.
    Lattice empty_first;
    empty_first.anchor_key = {0.0f};
    empty_first.rows.push_back(LatticeRow{});
    empty_first.rows.push_back(
        make_row({1, 2}, {5.0f, 3.0f}, {{0.0f}, {0.0f}}, {0.0f}, {{0.0f}, {0.0f}}));

    CHECK(dflash_select_path(empty_first, DflashSelectMode::kGreedy, 1.0f).empty());
    CHECK(dflash_select_path(empty_first, DflashSelectMode::kViterbi, 1.0f).empty());
}

#pragma once

#include <cstddef>
#include <vector>

// The DFlash2 selector's scoring model, extracted into a pure function so it
// is unit-testable without OpenVINO (M11, the M11 design note (not in the repository)-1.2, DESIGN.md
// §7.0.2r). backend_ov.cpp's dflash_select builds a Lattice from one verify
// cycle's target-lm_head logits and codebook lookups, then calls
// dflash_select_path here to trace one token id per row.
//
// The score a row's candidate j gets, given the token `prev` chosen for the
// row before it (or the anchor for row 0), is
//
//     score(row, prev, j) = unary[row][j] + lambda * dot(pred_key(prev), hp[row] elementwise* succ_key[row][j])
//
// where pred_key(t) = predecessor_codebook[t] (rank-r, a function of the
// token id alone), hp[row] = P . draft_hidden[row] (rank-r, one vector per
// row, shared by every candidate in it) and succ_key[row][j] =
// successor_codebook[token[row][j]] (rank-r, per candidate). This is the
// same trilinear term dflash_select in backend_ov.cpp computes as
// (pred_cb[prev] * hp) . succ_cb[cand].
//
// Bit-exactness (review finding, ): kGreedy at lambda=1
// reproduces the legacy selector's floating-point association order exactly
// -- t[d] = pred_key(prev)[d] * hp[row][d] (elementwise, prev-dependent,
// computed fresh per row since `prev` is whatever the row before it actually
// picked), then score = unary[row][j]; score += lambda * (t[d] * succ_key[j][d])
// accumulated over d in order -- the same sequence of operations backend_ov's
// pre-M11 dflash_select ran, so at lambda=1.0f (where `1.0f * x == x` is
// exact in IEEE 754) it produces bit-identical scores and therefore
// bit-identical decisions. kViterbi has no such legacy to match: it
// precomputes row_key[row][j] = hp[row] elementwise* succ_key[row][j] once
// per row (independent of `prev`, since it must be scored against every
// candidate of the row before it, not just one), and its own association
// order is unconstrained.
namespace lgc {

// One row of the lattice: up to K candidate tokens at one draft position.
struct LatticeRow {
    std::vector<int>                token;      // candidate token ids, size K
    std::vector<float>              unary;      // target lm_head logit, size K
    // rank-r key used when this row's candidate j is ITSELF the predecessor
    // of the next row (predecessor_codebook[token[j]]). Unused on the last row.
    std::vector<std::vector<float>> pred_key;
    // rank-r vector for this row alone (P . draft_hidden[row]), shared by
    // every candidate in the row. Used by kGreedy to reproduce the legacy
    // association order (see the header comment above); size r.
    std::vector<float>              hp;
    // rank-r key per candidate (successor_codebook[token[j]]), kept SEPARATE
    // from hp (rather than pre-multiplied) so kGreedy can multiply in the
    // legacy order at score time. Size K, each entry rank r.
    std::vector<std::vector<float>> succ_key;
    // rank-r key per candidate: hp elementwise* succ_key[j], precomputed once
    // per row for kViterbi's convenience (it evaluates every candidate of
    // this row against every candidate of the row before it, so precomputing
    // this product here is what keeps the DP's inner loop to one dot product
    // per pair instead of an elementwise multiply plus a dot). NOT used by
    // kGreedy -- see the bit-exactness note above.
    std::vector<std::vector<float>> row_key;
};

// A lattice for one verify cycle's draft: `rows.size()` positions, each with
// up to K candidates. `anchor_key` is the rank-r predecessor key of the
// already-committed token the first row transitions from
// (predecessor_codebook[anchor]).
struct Lattice {
    std::vector<float>      anchor_key;
    std::vector<LatticeRow> rows;
};

enum class DflashSelectMode { kGreedy, kViterbi };

// Selects one path (one token id per row) through `lattice` under the
// additive score above.
//
// kGreedy commits row by row: at each row it takes the candidate with the
// highest score given the PREVIOUS row's actual pick, and never reconsiders
// that pick -- today's production behaviour (backend_ov.cpp's dflash_select
// at lambda=1, bit-identical there -- see the header comment). Ties (equal
// score) keep the earliest candidate index, via a strict `>` comparison,
// matching the legacy loop.
//
// kViterbi finds the exact maximum-total-score path over the whole lattice:
// the standard DP,
//
//     dp[row][j] = unary[row][j] + lambda * max_i(dp[row-1][i] + dot(pred_key(row-1, i), row_key[row][j]))
//
// i.e. the predecessor's OWN accumulated score dp[row-1][i] is added
// unscaled, and only the fresh transition term is weighted by lambda.
// Backpointers are tracked and the best final row's path traced back. Ties
// in the DP and in the final argmax also keep the earliest index.
//
// An empty `lattice.rows` returns an empty path. A row with zero candidates
// truncates the path at that row (both modes: the path returned covers every
// row strictly before the first empty one, and nothing from it or after) --
// the same shape as a verify cycle whose lattice ran out of rows, so a caller
// bug there fails safe identically in both modes rather than only in one.
std::vector<int> dflash_select_path(const Lattice& lattice, DflashSelectMode mode, float lambda);

}  // namespace lgc

#include "core/dflash_select.h"

#include <limits>

namespace lgc {
namespace {

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float acc = 0.0f;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t d = 0; d < n; ++d) acc += a[d] * b[d];
    return acc;
}

// Review follow-up: the first row with zero candidates, or
// lattice.rows.size() if every row has at least one. Both selectors truncate
// the path there identically rather than disagreeing on how a caller bug
// (an empty candidate set is never expected in practice -- top-k with k>=1
// over a real vocabulary) fails.
size_t first_empty_row(const Lattice& lattice) {
    for (size_t r = 0; r < lattice.rows.size(); ++r) {
        if (lattice.rows[r].token.empty()) return r;
    }
    return lattice.rows.size();
}

// kGreedy, legacy association order (review follow-up): `prev_key` is
// whatever the row before actually picked (or the anchor at row 0), and the
// multiplication `t[d] = prev_key[d] * row.hp[d]` happens fresh per row --
// this is prev-dependent and therefore cannot be precomputed into row_key,
// which is why kGreedy reads hp/succ_key rather than row_key. At lambda=1.0f
// this reproduces backend_ov.cpp's pre-M11 dflash_select bit-for-bit: same
// t[d] = pc[d]*hp[d], same score = unary[j] starting value, same
// score += t[d]*sc[d] accumulation order over d.
std::vector<int> select_greedy(const Lattice& lattice, float lambda) {
    const size_t      n = first_empty_row(lattice);
    std::vector<int>  path;
    path.reserve(n);
    const std::vector<float>* prev_key = &lattice.anchor_key;
    std::vector<float> t;
    for (size_t ridx = 0; ridx < n; ++ridx) {
        const LatticeRow& row = lattice.rows[ridx];
        const size_t       k = row.token.size();
        const size_t       r = row.hp.size();
        t.assign(r, 0.0f);
        for (size_t d = 0; d < r && d < prev_key->size(); ++d) {
            t[d] = (*prev_key)[d] * row.hp[d];
        }
        size_t best_idx = 0;
        float  best     = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < k; ++j) {
            float score = row.unary[j];
            const std::vector<float>& sc = row.succ_key[j];
            for (size_t d = 0; d < r; ++d) score += lambda * (t[d] * sc[d]);
            if (score > best) {
                best     = score;
                best_idx = j;
            }
        }
        path.push_back(row.token[best_idx]);
        prev_key = &row.pred_key[best_idx];
    }
    return path;
}

// kViterbi: exact max-score path over the lattice. No legacy to match, so it
// uses row_key (hp elementwise* succ_key, precomputed once per row) rather
// than recomputing the elementwise product per (prev, candidate) pair.
std::vector<int> select_viterbi(const Lattice& lattice, float lambda) {
    const size_t rows = first_empty_row(lattice);
    if (rows == 0) return {};

    // dp[j] / backptr[j] for the row currently being filled; prev_row is
    // what the transition into this row reads its pred_key from.
    std::vector<float>               dp;
    std::vector<std::vector<size_t>> backptr(rows);

    const LatticeRow& row0 = lattice.rows[0];
    const size_t       k0  = row0.token.size();
    dp.resize(k0);
    for (size_t j = 0; j < k0; ++j) {
        dp[j] = row0.unary[j] + lambda * dot(lattice.anchor_key, row0.row_key[j]);
    }

    const LatticeRow* prev_row = &row0;
    for (size_t r = 1; r < rows; ++r) {
        const LatticeRow& row = lattice.rows[r];
        const size_t      k   = row.token.size();
        std::vector<float> next_dp(k);
        backptr[r].resize(k);
        for (size_t j = 0; j < k; ++j) {
            float  best     = -std::numeric_limits<float>::infinity();
            size_t best_idx = 0;
            for (size_t p = 0; p < prev_row->token.size(); ++p) {
                // The predecessor's OWN accumulated score dp[p] is added
                // unscaled; only the fresh transition term is weighted by
                // lambda (
                // backwards).
                const float cand =
                    dp[p] + row.unary[j] + lambda * dot(prev_row->pred_key[p], row.row_key[j]);
                if (cand > best) {
                    best     = cand;
                    best_idx = p;
                }
            }
            next_dp[j]    = best;
            backptr[r][j] = best_idx;
        }
        dp       = std::move(next_dp);
        prev_row = &row;
    }

    // Final argmax over the last processed row, then trace back.
    size_t best_last = 0;
    float  best      = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < dp.size(); ++j) {
        if (dp[j] > best) {
            best      = dp[j];
            best_last = j;
        }
    }

    std::vector<size_t> chosen(rows);
    chosen[rows - 1] = best_last;
    for (size_t r = rows - 1; r-- > 0;) {
        chosen[r] = backptr[r + 1][chosen[r + 1]];
    }

    std::vector<int> path(rows);
    for (size_t r = 0; r < rows; ++r) path[r] = lattice.rows[r].token[chosen[r]];
    return path;
}

}  // namespace

std::vector<int> dflash_select_path(const Lattice& lattice, DflashSelectMode mode, float lambda) {
    if (mode == DflashSelectMode::kViterbi) return select_viterbi(lattice, lambda);
    return select_greedy(lattice, lambda);
}

}  // namespace lgc

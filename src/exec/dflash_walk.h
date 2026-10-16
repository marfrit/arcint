// The DFlash2 selector's lattice walk (the libllama engine's
// --llama-dflash-gguf): llama.cpp's draft-dflash at the pin,
// common/speculative.cpp:1335-1366 (the walk) and src/models/dflash.cpp:550-639
// (the lattice it reads). Pure: no context, so the walk is testable on
// synthetic lattices (tests/test_dflash_walk.cpp), as verify_walk.h is.
//
// The drafter's selector packs one row a block position into its nextn
// output, n_embd floats wide:
//   row[0 .. k)               the position's top-k candidate token ids (as floats)
//   row[k + p*k .. k + p*k+k) the scores of those k candidates after candidate
//                             p of the previous position (its transition row:
//                             successor . (predecessor * gate) + the candidate's logit)
// Row 0 is the anchor's (id_last) and is not read. At row 1 the predecessor is
// the anchor itself, one id, so every transition row there is the same and
// the walk reads the first. From row 2 on, the predecessor is the candidate
// the walk picked at the row before.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace lgc {

// The greedy walk over a block's lattice rows (rows[0] is the anchor's row),
// `stride` floats apart, `top_k` candidates a position. A row's pick is the
// argmax of its transition row from the previous pick; with p_min > 0 the
// walk stops at the first row whose pick has less than p_min of the softmax
// over its k scores (that pick is not drafted). At most n_max drafts:
// llama.cpp drafts the whole block and common_speculative_draft truncates the
// result to the step's budget, which keeps the same prefix. Empty when the
// lattice cannot hold top_k + top_k^2 floats a row.
inline std::vector<int> dflash_walk(const float* rows, size_t n_rows, size_t stride, int top_k, float p_min,
                                    int n_max) {
    std::vector<int> out;
    if (rows == nullptr || top_k <= 0 || n_max <= 0 ||
        static_cast<size_t>(top_k) + static_cast<size_t>(top_k) * static_cast<size_t>(top_k) > stride)
        return out;
    const size_t k    = static_cast<size_t>(top_k);
    size_t       pred = 0;
    for (size_t i = 1; i < n_rows && static_cast<int>(out.size()) < n_max; ++i) {
        const float* row    = rows + i * stride;
        const float* scores = row + k + pred * k;
        pred                = static_cast<size_t>(std::max_element(scores, scores + k) - scores);
        if (p_min > 0.0f) {
            // softmax(scores) at the argmax: 1 / sum(exp(s_j - s_max))
            float sum = 0.0f;
            for (size_t j = 0; j < k; ++j) sum += std::exp(scores[j] - scores[pred]);
            if (1.0f / sum < p_min) break;
        }
        out.push_back(static_cast<int>(row[pred]));
    }
    return out;
}

}  // namespace lgc

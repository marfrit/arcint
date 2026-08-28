#include "core/drafter.h"

#include <algorithm>

namespace lgc {

std::vector<int> NgramDrafter::draft(const std::vector<int>& tokens, size_t max_tokens) {
    const size_t n = std::min(max_tokens, max_draft_);
    if (n == 0 || ngram_ == 0 || tokens.size() <= ngram_) return {};

    const size_t tail = tokens.size() - ngram_;

    // Search backwards: the most recent occurrence is the best guess, and it is
    // also the cheapest to find for the common case of an immediate repeat.
    for (size_t start = tail; start-- > 0;) {
        if (!std::equal(tokens.begin() + static_cast<long>(start),
                        tokens.begin() + static_cast<long>(start + ngram_),
                        tokens.begin() + static_cast<long>(tail))) {
            continue;
        }
        // start <= tokens.size() - ngram_ - 1, so there is always at least one
        // token after the match to copy.
        const size_t after = start + ngram_;
        const size_t take = std::min(n, tokens.size() - after);
        return {tokens.begin() + static_cast<long>(after),
                tokens.begin() + static_cast<long>(after + take)};
    }
    return {};
}

}  // namespace lgc

// A draft vocabulary file (--llama-mtp-vocab): token ids as little-endian
// int32 (Strata's data/draft_vocab.bin) or a JSON list of integers
// (HyperQwen's prepare/draft_vocab_ids.json). Pure, so it is testable
// without a model (tests/test_token_ids.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace lgc {

// The ids in `data`, sorted, without duplicates; false with `err` set for a
// malformed file, an id outside [0, n_vocab) or an empty list. json: the
// data is a JSON list, else int32s.
inline bool parse_token_ids(const std::string& data, bool json, int n_vocab, std::vector<int32_t>& out,
                            std::string& err) {
    out.clear();
    if (json) {
        const size_t a = data.find('['), b = data.rfind(']');
        if (a == std::string::npos || b == std::string::npos || b < a) {
            err = "not a JSON list";
            return false;
        }
        size_t i = a + 1;
        while (i < b) {
            const char c = data[i];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',') {
                ++i;
                continue;
            }
            if (c < '0' || c > '9') {
                err = "a JSON list of non-negative integers expected";
                return false;
            }
            int64_t v = 0;
            while (i < b && data[i] >= '0' && data[i] <= '9') {
                v = v * 10 + (data[i] - '0');
                if (v > INT32_MAX) {
                    err = "a token id out of range";
                    return false;
                }
                ++i;
            }
            out.push_back(static_cast<int32_t>(v));
        }
    } else {
        if (data.size() % 4 != 0) {
            err = "an int32 file's size is a multiple of 4";
            return false;
        }
        out.resize(data.size() / 4);
        if (!out.empty()) std::memcpy(out.data(), data.data(), data.size());
    }
    for (const int32_t v : out) {
        if (v < 0 || v >= n_vocab) {
            err = "token id " + std::to_string(v) + " outside the vocabulary of " + std::to_string(n_vocab);
            return false;
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (out.empty()) {
        err = "no token ids";
        return false;
    }
    return true;
}

}  // namespace lgc

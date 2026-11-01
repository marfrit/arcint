// The one-pass tokenisation, shared by the llama tokenizer and its test.
//
// The old encode() tokenised the text twice: once with a null output to get
// the count (a full pass), then once into an exactly-sized buffer (audit 15).
// The first buffer is an upper-bound guess, not a bound: in SPM/UGM
// vocabularies the space prefix after every special fragment can add a token
// (llama-vocab.cpp, the SPM/UGM branch), so k bytes can give k+1 tokens. The
// `-need` retry is what makes the result exact; the caller's +8 slack only
// reduces how often it runs. The ids are the same as the two-pass path makes:
// the first call already produces them.
#ifndef LGC_TOKENIZE_ONCE_H
#define LGC_TOKENIZE_ONCE_H

#include <cstdint>
#include <string_view>
#include <vector>

static_assert(sizeof(int) == sizeof(int32_t), "the token ids are int32_t");

// fn(text, len, out, cap) writes ids and returns the count, or -need when cap
// is too small (llama_tokenize's contract with a short buffer).
template <typename Fn>
std::vector<int> tokenize_once(Fn fn, std::string_view text, size_t initial) {
    std::vector<int> out(initial);
    int n = fn(text.data(), static_cast<int32_t>(text.size()), out.data(), static_cast<int32_t>(out.size()));
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = fn(text.data(), static_cast<int32_t>(text.size()), out.data(), static_cast<int32_t>(out.size()));
    }
    if (n < 0 || static_cast<size_t>(n) > out.size()) return {};
    out.resize(static_cast<size_t>(n));
    return out;
}

#endif  // LGC_TOKENIZE_ONCE_H

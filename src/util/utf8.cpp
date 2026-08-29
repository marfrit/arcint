#include "util/utf8.h"

namespace lgc::utf8 {
namespace {

constexpr size_t kMaxSequence = 4;

inline bool is_continuation(unsigned char b) { return (b & 0xC0) == 0x80; }

}  // namespace

size_t sequence_length(unsigned char lead) {
    if (lead < 0x80)             return 1;
    if ((lead & 0xE0) == 0xC0)   return 2;
    if ((lead & 0xF0) == 0xE0)   return 3;
    if ((lead & 0xF8) == 0xF0)   return 4;
    return 1;  // continuation byte or invalid lead — not a sequence start
}

size_t complete_prefix(std::string_view s) {
    const size_t n = s.size();
    if (n == 0) return 0;

    // Walk back over at most three continuation bytes looking for the lead.
    for (size_t back = 0; back < kMaxSequence && back < n; ++back) {
        const size_t        i = n - 1 - back;
        const unsigned char b = static_cast<unsigned char>(s[i]);
        if (is_continuation(b)) continue;

        const size_t need = sequence_length(b);
        if (need == 1)      return n;  // ASCII, or invalid lead we refuse to wait on
        if (i + need <= n)  return n;  // the sequence is fully present
        return i;                      // truncated tail — hold it back
    }

    // Four or more trailing continuation bytes: malformed. Emitting is better
    // than stalling the stream forever.
    return n;
}

bool is_valid(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        if (b >= 0xF8 || is_continuation(b)) return false;

        const size_t need = sequence_length(b);
        if (i + need > s.size()) return false;
        for (size_t k = 1; k < need; ++k) {
            if (!is_continuation(static_cast<unsigned char>(s[i + k]))) return false;
        }

        // Reject overlong encodings and surrogates; both are invalid UTF-8 and
        // both are things a broken detokenizer can emit.
        if (need == 2 && b < 0xC2) return false;
        if (need == 3 && b == 0xE0 && static_cast<unsigned char>(s[i + 1]) < 0xA0) return false;
        if (need == 3 && b == 0xED && static_cast<unsigned char>(s[i + 1]) >= 0xA0) return false;
        if (need == 4 && b == 0xF0 && static_cast<unsigned char>(s[i + 1]) < 0x90) return false;
        if (need == 4 && b == 0xF4 && static_cast<unsigned char>(s[i + 1]) >= 0x90) return false;
        if (need == 4 && b > 0xF4) return false;

        i += need;
    }
    return true;
}

size_t count_codepoints(std::string_view s) {
    size_t n = 0;
    for (char c : s) {
        if (!is_continuation(static_cast<unsigned char>(c))) ++n;
    }
    return n;
}

std::string Streamer::push(std::string_view bytes) {
    pending_.append(bytes);

    const size_t cut = complete_prefix(pending_);
    if (cut == 0) return {};

    std::string out = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return out;
}

std::string Streamer::flush() {
    std::string out;
    out.swap(pending_);
    return out;
}

}  // namespace lgc::utf8

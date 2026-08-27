#pragma once

#include <cstddef>
#include <string>
#include <string_view>

// Incremental, UTF-8-safe detokenization support (DESIGN.md §3.7): a multi-byte
// code point must never be split across two SSE chunks.
namespace lgc::utf8 {

// Byte length of the sequence a lead byte starts. Returns 1 for ASCII, for
// continuation bytes, and for invalid lead bytes — callers treat "1" as
// "nothing to wait for".
size_t sequence_length(unsigned char lead);

// Length of the longest prefix of `s` that ends on a complete code point.
// A trailing truncated sequence is excluded; malformed input never causes an
// unbounded hold-back (at most 3 bytes are ever withheld).
size_t complete_prefix(std::string_view s);

bool   is_valid(std::string_view s);
size_t count_codepoints(std::string_view s);

// Buffers the tail of a byte stream until it forms whole code points.
class Streamer {
public:
    // Appends `bytes` and returns the portion that is safe to emit now.
    std::string push(std::string_view bytes);

    // Returns everything still held, complete or not. For end of stream: at
    // that point a truncated sequence is the model's output, not our bug, and
    // swallowing it would lose bytes.
    std::string flush();

    bool               empty() const { return pending_.empty(); }
    std::string_view   pending() const { return pending_; }

private:
    std::string pending_;
};

}  // namespace lgc::utf8

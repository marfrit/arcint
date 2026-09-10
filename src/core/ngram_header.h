#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// FIX D Link 2: parse the 24-byte ARCINGRM header the synthetic
// per_layer_token_embd generator writes (`tools/synthetic_ngram_table.py`,
// `docs/design-qwen-flash-next.md` §"Link 1"). The C++ layout below is a
// byte-for-byte match against the Python emitter's `build_header`:
//
//     offset  size  field
//     0       8     magic "ARCINGRM" (ASCII, no NUL)
//     8       4     ggml_type id (u32 LE): 2=Q4_0, 3=Q4_1, 8=Q8_0
//     12      4     n_cols (u32 LE), must be a multiple of 32
//     16      4     n_rows (u32 LE)
//     20      4     reserved (u32, 0)
//
// This header is what Link 2 (loader admission, this section) reads at
// load time to decide whether to admit the on-disk table under
// `--flash-next-ngram`. A future real-generation entry point in the
// Python side will write the same header; the two implementations
// share the layout and nothing else.
namespace lgc::ngram {

inline constexpr std::string_view kMagic{"ARCINGRM"};
inline constexpr std::size_t      kHeaderBytes = 24;
inline constexpr std::size_t      kMagicBytes  = 8;
inline constexpr uint32_t         kBlockElements = 32;

struct Header {
    uint32_t ggml_type = 0;   // 2 = Q4_0, 3 = Q4_1, 8 = Q8_0
    uint32_t n_cols    = 0;
    uint32_t n_rows    = 0;
};

// Parse a 24-byte header. On success, returns the empty string and fills
// `out`. On failure, returns a message naming the exact field that
// failed (matches `tools/synthetic_ngram_table.py::parse_header`'s
// refusal messages so a caller does not have to guess which side of
// the wire refused the file).
std::string parse_header(const uint8_t* bytes, std::size_t n_bytes, Header& out);

// Bytes per 32-element block for a known ggml_type. Returns 0 for an
// unrecognised type -- callers that want a refusal test that instead.
std::size_t bytes_per_block(uint32_t ggml_type);

// Total payload bytes for `(ggml_type, n_cols, n_rows)`, header
// excluded. Overflow-safe on the 32-bit inputs the header carries
// (n_rows * blocks_per_row * bytes_per_block stays inside uint64_t).
uint64_t payload_bytes(const Header& h);

// Total on-disk bytes: header + payload.
uint64_t file_bytes(const Header& h);

}  // namespace lgc::ngram

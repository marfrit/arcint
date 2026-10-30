#pragma once

// The loan-KL driver's rules as plain data, so arcint's device-free unit test
// (tests/test_loan_book.cpp) can see them without a card, the same way patch
// 0068's llama-loan-book.h is checked. The driver is
// contrib/llama-loan-kld/loan_kld.cpp, built inside the llama.cpp checkout; the
// gate is docs/campaigns/prefill-slot-loan.md, round 13.

#include <cstdint>
#include <cstring>

namespace loan_kld_book {

// The lend threshold. A loan is wanted only when the chunk's tokens exceed the
// resident chunk arcint reserves (slot_loan.h's kResidentChunk, 2,048): the
// prompt's reserve has to beat the resident one, else the empty lend/refill
// pair costs a reserve and a refill for nothing. The served measured arm asks
// 8,192 (--llama-loan-chunk 8192) against the 18,802-token prompt.
constexpr uint32_t kResidentChunk = 2048;

// what the driver asks the context to reserve: the chunk's tokens, never more
// than the context
inline uint32_t chunk_tokens(uint32_t n_ctx) {
    return n_ctx;
}

inline bool loan_wanted(uint32_t n_ctx) {
    return chunk_tokens(n_ctx) > kResidentChunk;
}

// A capture row window, from llama.cpp's perplexity loop (`code`,
// tools/perplexity/perplexity.cpp: `first = n_ctx/2`, rows n_ctx - 1 - first):
// the driver writes the same rows, so capture_vs_capture.py reads both files
// position for position.
inline int first_row(int n_ctx) {
    return n_ctx / 2;
}

inline int n_rows(int n_ctx) {
    return n_ctx - 1 - first_row(n_ctx);
}

// the per-row width of llama's capture, budget for the 16-bit quantised
// log-probs plus the 4-word header (`code`, perplexity.cpp's `nv`)
inline int row_width(int n_vocab) {
    return 2 * ((n_vocab + 1) / 2) + 4;
}

// Round 14: the driver writes rows straight from llama_get_logits(ctx), so the
// only host row it holds is ONE row (Flash-Next's 248,320-token vocab:
// 248,324 uint16 = ~0.47 MiB). The round-13 form sized n_rows x nv (~2.03 GiB)
// and copied n_ctx x n_vocab floats (~8.14 GiB, zero-filled, half used).
inline size_t row_buffer_bytes(int n_vocab) {
    return (size_t) row_width(n_vocab) * sizeof(uint16_t);
}

// The context's logits reservation. llama.cpp's default is n_batch, so a
// 1 x 8,192 window at this vocab reserved 8,192 x 248,320 f32 (~8.14 GiB) while
// the capture asks n_rows rows; n_ctx - n_ctx/2 is the number that suffices
// (`code`: the graph outputs positions first..n_ctx-2, one row each).
inline uint32_t output_rows(uint32_t n_ctx) {
    return n_ctx - n_ctx / 2;
}

// Round 15: the loan's sizing takes n_outputs = min(measure_ub, n_outputs_max)
// (`code`, llama-context.cpp's slot_loan_begin). At this vocab 4,096 output rows
// put ~4.07 GB of logits in the lent graph's sched compute buffer (~3.79 GiB),
// which the 6.20 GiB lend's plan (`need + 0.375 GiB <= lend`) refuses; the same
// rows are also the context's output reservation, a plain CPU buffer on OpenCL
// (the device has no host buffer type) -- ~4.07 GB of anon RSS. So the driver
// caps n_outputs_max at a row window and scores the served 8,192-token ubatch in
// several decodes under one loan: each replay is the same 8,192-token graph and
// starts from an empty KV, and a causal row at position p depends only on tokens
// 0..p, so every row equals llama-perplexity's. `requested` is --rows-per-decode
// (default 2048: two decodes, 2048 + 2047 rows at n_ctx 8,192).
inline uint32_t decode_width(int n_rows, uint32_t requested) {
    if (n_rows <= 0) return 0;
    return requested < (uint32_t) n_rows ? requested : (uint32_t) n_rows;
}

inline int n_decodes(int n_rows, uint32_t width) {
    if (n_rows <= 0 || width == 0) return 0;
    return (n_rows + (int) width - 1) / (int) width;
}

// the rows of decode `p`: width for every window but the last
inline int rows_in_decode(int n_rows, uint32_t width, int p) {
    const int left = n_rows - p * (int) width;
    if (left <= 0) return 0;
    return left < (int) width ? left : (int) width;
}

// Round 14, perplexity.cpp:72-77: the capture's quantiser rounds with the
// f32 + 12582912 trick, which is half-to-even; std::lround is half away from
// zero, so the driver could not reproduce llama-perplexity's rows (the control
// arm compares the two, row for row). Transcribed, not re-derived.
inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// The served shape the loan arm must run (round 14): arcint's chunk is one
// 8,192-token ubatch under a 2,048-token resident reserve and an 8,192-token
// loan reserve. A driver at -ub 2,048 under the 8,192 loan reserve instead runs
// four ubatches of 2,048 -- a different graph, not the unit's. The off arm's
// split is the perplexity tool's own and is allowed (the bar is answer-level).
inline bool served_shape(uint32_t n_ubatch, uint32_t chunk) {
    return n_ubatch >= chunk;
}

}  // namespace loan_kld_book

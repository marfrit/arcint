#pragma once

// FIX D (0.5.0, docs/design-qwen-flash-next.md "FIX D — N-gram table:
// host-offload + dequantise-on-gather"): the gather-with-dequant kernel for
// Qwen Flash Next's `per_layer_token_embd` table.
//
// That table is 51.2 G elements, 160-wide rows, in one of the three plain
// 32-element-block ggml quant formats (Q4_0/Q4_1/Q8_0) -- NOT K-quant
// (Q4_K/etc, 256-element superblocks): 160 is not a multiple of 256, so the
// existing FullyConnectedKQuant op / K-quant decoders (kquant_op.h,
// core/gguf_dequant.cpp) do not apply. It is too large to hold on either
// card (q4_1, the shipped precision's up-cast, is ~30.5 GiB alone) so it
// stays host-resident, and only the rows a forward pass actually looks up
// (`ggml_get_rows` in the reference implementation) are ever dequantized --
// this file is that gather-with-dequant path, host side.
//
// Two implementations, checked byte-exact against each other in
// tests/test_ngram_gather.cpp:
//   - gather_dequant_scalar: calls lgc::gguf::dequantize_row per row --
//     already reviewed, already tested (tests/test_gguf.cpp), reused here
//     as the reference rather than re-derived.
//   - the AVX2 path (dequant_block_q{4_0,4_1,8_0}_avx2 + gather_dequant):
//     the fast path this kernel exists for.
//
// AVX2 only. The dev host (docs/design-qwen-flash-next.md's "the dev
// container") is a Zen 3 part: AVX2 yes, AVX-512 NO -- AVX-512 there is a
// SIGILL at runtime, not a graceful degrade, so nothing in this file may
// ever emit it. The AVX2 functions below are compiled with
// `__attribute__((target("avx2")))` (GCC/Clang function multiversioning),
// not a blanket `-mavx2` on the translation unit, so this header is safe to
// include from a binary built with no `-march`/`-mavx2` flag at all -- the
// AVX2 code generation is scoped to exactly the functions that opt in, and
// `gather_dequant` below never calls them without first checking
// `cpu_has_avx2()` at runtime (cpuid, not a compile-time assumption). No
// `fma` target string anywhere in this file: every block dequant is an
// explicit multiply then an explicit add (`_mm256_mul_ps` then
// `_mm256_add_ps`), matching the scalar reference's own separate operations
// term for term so the two paths round the same way -- an FMA'd multiply-
// add rounds ONCE where a separate multiply-then-add rounds TWICE, and the
// two are not always bit-identical.
//
// tests/test_ngram_gather.cpp is the red-first case this file exists to
// make pass: a synthetic table, hand-quantized into all three formats,
// gathered by both paths, asserted byte-exact.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/gguf.h"
#include "core/gguf_dequant.h"
#include "util/log.h"

#if defined(__x86_64__) || defined(_M_X64)
#define ARCINT_NGRAM_HAVE_X86 1
#include <immintrin.h>
#endif

namespace lgc::ngram {

// Runtime AVX2 detection (cpuid via the compiler's builtin, not a compile-
// time `#ifdef __AVX2__` -- the whole point is that this binary may be
// built without `-mavx2` and still choose the fast path on a CPU that has
// it). Never assume; every call site in this file that reaches into the
// `_avx2` functions below gates on this first.
inline bool cpu_has_avx2() {
#if defined(ARCINT_NGRAM_HAVE_X86)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
#endif
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

// Only these three formats are this kernel's scope (docs/design-qwen-flash-
// next.md FIX D: "32-element-block formats (Q4_0/Q4_1/Q8_0) on 160-wide
// rows"). Anything else -- including the K-quant types core/gguf_dequant.cpp
// already covers -- is out of scope for the AVX2 path and falls back to the
// scalar reference (which, via lgc::gguf::dequantize_row, handles every type
// that reader knows).
inline bool avx2_kernel_supports_type(int32_t ggml_type) {
    switch (static_cast<lgc::gguf::GgmlType>(ggml_type)) {
        case lgc::gguf::GgmlType::Q4_0:
        case lgc::gguf::GgmlType::Q4_1:
        case lgc::gguf::GgmlType::Q8_0: return true;
        default: return false;
    }
}

namespace detail {

inline uint16_t read_u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

#if defined(ARCINT_NGRAM_HAVE_X86)

// Free helper functions, not lambdas: a lambda's call operator is a
// separate function whose own target attribute GCC does NOT inherit from
// the enclosing function it is defined in (measured on this file's first
// draft, built and run on the dev host -- "inlining failed in call to
// 'always_inline' ...: target specific option mismatch" on every intrinsic
// called from inside a lambda body, even though the lambda was only ever
// invoked from inside a target("avx2") function). Explicit `target("avx2")`
// free functions below do not have this problem.
__attribute__((target("avx2"))) inline void widen_epu8_sub8_mul_store(__m128i nib8, float dv_scalar,
                                                                       float* dst) {
    const __m256i eight = _mm256_set1_epi32(8);
    __m256i       w32   = _mm256_cvtepu8_epi32(nib8);  // low 8 bytes -> 8 int32
    w32                 = _mm256_sub_epi32(w32, eight);
    const __m256 dv     = _mm256_set1_ps(dv_scalar);
    const __m256 yv     = _mm256_mul_ps(_mm256_cvtepi32_ps(w32), dv);
    _mm256_storeu_ps(dst, yv);
}

__attribute__((target("avx2"))) inline void widen_epu8_mul_add_store(__m128i nib8, float dv_scalar,
                                                                      float mv_scalar, float* dst) {
    const __m256i w32 = _mm256_cvtepu8_epi32(nib8);
    const __m256  dv  = _mm256_set1_ps(dv_scalar);
    const __m256  mv  = _mm256_set1_ps(mv_scalar);
    const __m256  yv  = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(w32), dv), mv);
    _mm256_storeu_ps(dst, yv);
}

__attribute__((target("avx2"))) inline void widen_epi8_mul_store(__m128i i8x8, float dv_scalar,
                                                                  float* dst) {
    const __m256i w32 = _mm256_cvtepi8_epi32(i8x8);  // sign-extend, low 8 bytes
    const __m256  dv  = _mm256_set1_ps(dv_scalar);
    const __m256  yv  = _mm256_mul_ps(_mm256_cvtepi32_ps(w32), dv);
    _mm256_storeu_ps(dst, yv);
}

// One Q4_0 block: 2 bytes f16 scale `d`, 16 bytes of 32 packed 4-bit
// nibbles. y[l] = (nibble(l) - 8) * d, for l in [0, 32) -- the low nibble of
// byte l lands at y[l], the high nibble at y[l+16] (ggml's own packing;
// core/gguf_dequant.cpp's dequantize_row_q4_0, added alongside this file,
// is the scalar transcription this mirrors).
__attribute__((target("avx2"))) inline void dequant_block_q4_0_avx2(const uint8_t* block,
                                                                     float* y) {
    const float    d  = lgc::gguf::f16_to_f32(read_u16(block));
    const uint8_t* qs = block + 2;

    const __m128i v       = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i mask0f  = _mm_set1_epi8(0x0F);
    const __m128i lo_nib  = _mm_and_si128(v, mask0f);
    const __m128i hi_nib  = _mm_and_si128(_mm_srli_epi16(v, 4), mask0f);

    widen_epu8_sub8_mul_store(lo_nib, d, y + 0);                          // bytes 0..7  -> y[0..8)
    widen_epu8_sub8_mul_store(_mm_srli_si128(lo_nib, 8), d, y + 8);       // bytes 8..15 -> y[8..16)
    widen_epu8_sub8_mul_store(hi_nib, d, y + 16);                        // bytes 0..7  -> y[16..24)
    widen_epu8_sub8_mul_store(_mm_srli_si128(hi_nib, 8), d, y + 24);      // bytes 8..15 -> y[24..32)
}

// One Q4_1 block: 2 bytes f16 `d`, 2 bytes f16 `m`, 16 bytes of nibbles.
// y[l] = nibble(l) * d + m -- no -8 offset (Q4_1 is min-offset, not
// centered).
__attribute__((target("avx2"))) inline void dequant_block_q4_1_avx2(const uint8_t* block,
                                                                     float* y) {
    const float    d  = lgc::gguf::f16_to_f32(read_u16(block));
    const float    m  = lgc::gguf::f16_to_f32(read_u16(block + 2));
    const uint8_t* qs = block + 4;

    const __m128i v      = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    const __m128i lo_nib = _mm_and_si128(v, mask0f);
    const __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(v, 4), mask0f);

    widen_epu8_mul_add_store(lo_nib, d, m, y + 0);
    widen_epu8_mul_add_store(_mm_srli_si128(lo_nib, 8), d, m, y + 8);
    widen_epu8_mul_add_store(hi_nib, d, m, y + 16);
    widen_epu8_mul_add_store(_mm_srli_si128(hi_nib, 8), d, m, y + 24);
}

// One Q8_0 block: 2 bytes f16 `d`, 32 signed int8. y[l] = qs[l] * d, l in
// [0, 32) directly (no split reorder, unlike Q4_0/Q4_1).
__attribute__((target("avx2"))) inline void dequant_block_q8_0_avx2(const uint8_t* block,
                                                                     float* y) {
    const float d = lgc::gguf::f16_to_f32(read_u16(block));
    const auto* qs = reinterpret_cast<const int8_t*>(block + 2);

    const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16));

    widen_epi8_mul_store(v0, d, y + 0);
    widen_epi8_mul_store(_mm_srli_si128(v0, 8), d, y + 8);
    widen_epi8_mul_store(v1, d, y + 16);
    widen_epi8_mul_store(_mm_srli_si128(v1, 8), d, y + 24);
}

#endif  // ARCINT_NGRAM_HAVE_X86

}  // namespace detail

// Gathers `indices.size()` rows of `n_cols` elements from `table` (row-major,
// `row_stride_bytes` bytes per row -- callers pass
// `lgc::gguf::type_info(ggml_type).type_size * (n_cols /
// type_info(ggml_type).block_size)`, i.e. the exact packed row size, no
// padding) into `out` (row-major, `n_cols` floats per row, pre-sized by the
// caller to `indices.size() * n_cols`), via the scalar reference path --
// this is what gather_dequant below falls back to, and the byte-exact
// target the AVX2 path is checked against.
inline void gather_dequant_scalar(int32_t ggml_type, const uint8_t* table,
                                   size_t row_stride_bytes, size_t n_cols,
                                   const std::vector<uint32_t>& indices, float* out) {
    for (size_t i = 0; i < indices.size(); ++i) {
        const uint8_t* row = table + static_cast<size_t>(indices[i]) * row_stride_bytes;
        lgc::gguf::dequantize_row(ggml_type, row, n_cols, out + i * n_cols);
    }
}

// The fast path. Falls back to gather_dequant_scalar whenever the AVX2
// kernel does not apply: no AVX2 on this CPU, a type the AVX2 kernel does
// not cover, or `n_cols` not a whole multiple of the format's block size
// (32 for all three formats this file knows -- the n-gram table's own rows
// are 160-wide, 5 whole blocks, so this only matters for a test fixture
// deliberately shaped to hit it). `force_scalar` exists for tests: run both
// paths on the identical input and diff.
inline void gather_dequant(int32_t ggml_type, const uint8_t* table, size_t row_stride_bytes,
                            size_t n_cols, const std::vector<uint32_t>& indices, float* out,
                            bool force_scalar = false) {
    const bool avx2_path =
        !force_scalar && cpu_has_avx2() && avx2_kernel_supports_type(ggml_type);
    const size_t block_size = lgc::gguf::type_info(ggml_type).block_size;
    if (!avx2_path || block_size == 0 || n_cols % block_size != 0) {
        gather_dequant_scalar(ggml_type, table, row_stride_bytes, n_cols, indices, out);
        return;
    }

#if defined(ARCINT_NGRAM_HAVE_X86)
    const size_t block_bytes = lgc::gguf::type_info(ggml_type).type_size;
    const auto   type        = static_cast<lgc::gguf::GgmlType>(ggml_type);
    for (size_t i = 0; i < indices.size(); ++i) {
        const uint8_t* row = table + static_cast<size_t>(indices[i]) * row_stride_bytes;
        float*         y   = out + i * n_cols;
        for (size_t b = 0; b * block_size < n_cols; ++b) {
            const uint8_t* blk = row + b * block_bytes;
            float*         yb  = y + b * block_size;
            switch (type) {
                case lgc::gguf::GgmlType::Q4_0: detail::dequant_block_q4_0_avx2(blk, yb); break;
                case lgc::gguf::GgmlType::Q4_1: detail::dequant_block_q4_1_avx2(blk, yb); break;
                case lgc::gguf::GgmlType::Q8_0: detail::dequant_block_q8_0_avx2(blk, yb); break;
                default:
                    throw std::runtime_error(log::format(
                        "ngram: gather_dequant: %s reached the AVX2 branch without an AVX2 kernel "
                        "(avx2_kernel_supports_type is out of sync with this switch)",
                        lgc::gguf::type_name(ggml_type).c_str()));
            }
        }
    }
#else
    // Unreachable: avx2_path is false on any non-x86 build (cpu_has_avx2()
    // always returns false there), so the scalar branch above always fires
    // first. Kept as a defensive fallback rather than relying on that.
    gather_dequant_scalar(ggml_type, table, row_stride_bytes, n_cols, indices, out);
#endif
}

}  // namespace lgc::ngram

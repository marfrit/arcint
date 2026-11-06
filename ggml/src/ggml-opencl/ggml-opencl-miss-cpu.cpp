// patch 0056: the CPU half of the expert cache's miss path, several tokens at once.
//
// Strata's AVX2 expert rows (src/kernels/cpu/iq_avx2.cpp, q2_avx2.cpp; MIT, Copyright (c) 2026 Niko1221
// and the Strata contributors): a 32-value chunk of weights is decoded once - the grid lookups, one sign
// vector, one scale vector - and every token of the verify window applies it with a load, a sign, a
// maddubs, a madd and an add. ggml-cpu's dot products are single-token, and its Q2_0 one has no SIMD form
// (~0.7 GB/s a thread here against Strata's 6.4). Two departures: the weights are read from the bank's
// planes (patch 0022, ggml_cl_lb_fields: each field of every block back to back per expert slice) instead
// of GGUF blocks, and the activations' quantizers are written here (ggml-cpu's rules: q8_K for the i-quants,
// Strata's 32-value chunks with the -1 offset's correction for Q2_0, ggml-cpu's AVX2 q8_0 for IQ4_NL).
// IQ4_NL (Strata's iq4nl_rows) reads the bank's split form instead: the tensor's scales and its quants as
// two arrays over every expert (ggml_cl_usm_set_tensor), and keeps ggml-cpu's two accumulators so its rows
// equal ggml_vec_dot_iq4_nl_q8_0's bit for bit.
// IQ2_XS has its own row loop, as in Strata (row_dot_iq2xs): ggml's vectorized sign decode and scale unpack.
// IQ3_S: Strata's Fmt32<21> decode with ggml's vectorized grid index and a gather; one token in ggml's AVX2 loop.
//
// Compiled for AVX2 + FMA + F16C by the target pragma below; ggml_cl_miss_cpu_ok() checks the CPU at run time.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "ggml-opencl-miss-cpu.h"

#include <cstring>

// plain C++, every host: the slow path's IQ4_NL rows (any CPU, any build)
void ggml_cl_miss_cpu_join_iq4nl(const void * d, const void * q, int64_t i0, int64_t n, void * dst) {
    const size_t qb = sizeof(block_iq4_nl) - sizeof(ggml_half);
    for (int64_t i = 0; i < n; ++i) {
        block_iq4_nl * o = (block_iq4_nl *) dst + i;
        memcpy(&o->d, (const char *) d + (size_t) (i0 + i) * sizeof(ggml_half), sizeof(ggml_half));
        memcpy(o->qs, (const char *) q + (size_t) (i0 + i) * qb, qb);
    }
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define GGML_CL_MISS_CPU 1
namespace {
// IQ2_XXS's and IQ3_XXS's 7-bit sign index -> 8 sign bytes (the 8th bit by parity), as 64-bit words. Built at load,
// before the target pragma below: a static initializer runs on every host, AVX2 or not.
struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            const uint8_t s = ksigns_iq2xs[i];
            uint64_t w = 0;
            for (int b = 0; b < 8; ++b) w |= (uint64_t) ((s >> b) & 1 ? 0xFF : 0x01) << (8 * b);
            v[i] = w;
        }
    }
};
static const EvenSigns even_signs;
}  // namespace

bool ggml_cl_miss_cpu_ok(void) {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return ok;
}

bool ggml_cl_miss_cpu_type(ggml_type t) {
    return t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ3_XXS ||
           t == GGML_TYPE_IQ3_S || t == GGML_TYPE_Q2_0 || t == GGML_TYPE_IQ4_NL;
}

// AVX2 + FMA + F16C for everything below; ggml_cl_miss_cpu_ok() checks the CPU at run time
#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif
#endif

#if GGML_CL_MISS_CPU
#include <immintrin.h>

// patch 0075: the level, set once when the cache registers a layer
// (GGML_OPENCL_CPU_MISS_PREFETCH: 0 off, 1 the rows' block-ahead hints, 2 also
// the page-touch pass at publish). A hint cannot change a row's bits; it names
// the NEXT BLOCK's field 1 (the payload plane) two blocks ahead — 64-128 bytes,
// inside the hardware prefetcher's reach — and fields 0 (the scale) and 2 (the
// signs/qh plane) are unhinted (`code`), so a hint's gain, if any, is expected
// in the run's spread rather than a step.
static int g_cl_miss_pf = 0;
#define GGML_CL_MISS_PF(p) do { if (g_cl_miss_pf >= 1) __builtin_prefetch((const void *) (p), 0, 3); } while (0)
static const int GGML_CL_MISS_PF_AHEAD = 2;   // blocks

namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t * p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t * p) { uint16_t v; memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t * p) { uint64_t v; memcpy(&v, p, 8); return v; }

// ggml's bit_selector sign vector: byte i of the 32 is -1 where bit i of m is set, +1 elsewhere
inline __m256i sgn_vec(uint32_t m) {
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    const __m256i bsel = _mm256_set1_epi64x((long long) 0x8040201008040201ULL);
    const __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int) m), shuf);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b, bsel), bsel), _mm256_set1_epi8(1));
}
inline __m256i sc16(int a, int b) { return _mm256_set_m128i(_mm_set1_epi16((short) b), _mm_set1_epi16((short) a)); }
inline __m256i sc32(int s) { return _mm256_set1_epi16((short) s); }


inline float hsum8(__m256 v) {
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

// one weight matrix (an expert slice) as fields: in planes, field q of block i at f[q] + i * size[q];
// in GGUF blocks, at base + i * block + off[q]
struct Fields {
    const uint8_t * f[5];
    size_t          step[5];
    inline const uint8_t * at(int q, int64_t i) const { return f[q] + (size_t) i * step[q]; }
};

inline bool fields_of(ggml_type t, const uint8_t * slice, bool planar, int64_t ne0, int64_t ne1, Fields & F) {
    struct fd { int off, size; } fl[5];
    int nf = 0;
    switch (t) {
        case GGML_TYPE_IQ2_XXS: fl[0] = { 0, 2 }; fl[1] = { 2, 64 }; nf = 2; break;
        case GGML_TYPE_IQ2_XS:  fl[0] = { 0, 2 }; fl[1] = { 2, 64 }; fl[2] = { 66, 8 }; nf = 3; break;
        case GGML_TYPE_IQ2_S:   fl[0] = { 0, 2 }; fl[1] = { 2, 32 }; fl[2] = { 34, 32 }; fl[3] = { 66, 8 }; fl[4] = { 74, 8 }; nf = 5; break;
        // IQ3_XXS: the bank keeps it in GGUF blocks (not in ggml_cl_lb_fields); planar only for symmetry
        case GGML_TYPE_IQ3_XXS: fl[0] = { 0, 2 }; fl[1] = { 2, 64 }; fl[2] = { 66, 32 }; nf = 3; break;
        case GGML_TYPE_IQ3_S:   fl[0] = { 0, 2 }; fl[1] = { 2, 64 }; fl[2] = { 66, 8 }; fl[3] = { 74, 32 }; fl[4] = { 106, 4 }; nf = 5; break;
        case GGML_TYPE_Q2_0:    fl[0] = { 0, 2 }; fl[1] = { 2, 16 }; nf = 2; break;
        default: return false;
    }
    const size_t blk = ggml_type_size(t);
    const int64_t nblk = ne0 * ne1 / ggml_blck_size(t);
    size_t plane = 0;
    for (int q = 0; q < nf; ++q) {
        if (planar) { F.f[q] = slice + plane; F.step[q] = (size_t) fl[q].size; plane += (size_t) nblk * fl[q].size; }
        else        { F.f[q] = slice + fl[q].off; F.step[q] = blk; }
    }
    return true;
}

// ---- the i-quants against q8_K activations (Strata's Fmt32 decoders, fields from planes)

// IQ2_S: d, grid bytes [32], sign bytes [32], qh [8], scales [8]; chunk = 64-value group j, 32-value half
inline void dec_iq2s(const Fields & F, int64_t i, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
    const uint8_t * qs = F.at(1, i) + 8 * j;
    const uint8_t   h  = F.at(3, i)[2 * j + half];
    const int o = 4 * half;
    g = _mm256_set_epi64x((long long) iq2s_grid[qs[o + 3] | ((h << 2) & 0x300)],
                          (long long) iq2s_grid[qs[o + 2] | ((h << 4) & 0x300)],
                          (long long) iq2s_grid[qs[o + 1] | ((h << 6) & 0x300)],
                          (long long) iq2s_grid[qs[o] | ((h << 8) & 0x300)]);
    const uint64_t m = u64(F.at(2, i) + 8 * j);
    sgn = sgn_vec(half ? (uint32_t) (m >> 32) : (uint32_t) m);
    const uint8_t sb = F.at(4, i)[2 * j + half];
    sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
}

// IQ2_XXS: d, qs [64] (per 64-value group: 4 grid bytes, then 4 x 7-bit sign index + 4-bit scale)
inline void dec_iq2xxs(const Fields & F, int64_t i, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
    const uint8_t * q = F.at(1, i) + 16 * j + 8 * half;
    const uint32_t w0 = u32(q), w1 = u32(q + 4);
    g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                          (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
    sgn = _mm256_set_epi64x((long long) even_signs.v[(w1 >> 21) & 127], (long long) even_signs.v[(w1 >> 14) & 127],
                            (long long) even_signs.v[(w1 >> 7) & 127], (long long) even_signs.v[w1 & 127]);
    sc = sc32(2 * (int) (w1 >> 28) + 1);
}

// IQ3_XXS: d, grid bytes [64] (8 per 32-value half, 4 values each), scales_and_signs [32] (per half one u32:
// 4 x 7-bit sign index + 4-bit scale); the scale is (2 s + 1) like IQ2_XXS, the block constant 1/4 (row_iq)
inline void dec_iq3xxs(const Fields & F, int64_t i, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
    const uint8_t * q = F.at(1, i) + 16 * j + 8 * half;
    g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                         (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
    const uint32_t w = u32(F.at(2, i) + 8 * j + 4 * half);
    sgn = _mm256_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127],
                            (long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
    sc = sc32(2 * (int) (w >> 28) + 1);
}

// IQ3_S: d, qs [64] (low 8 bits of the 9-bit grid index), qh [8] (bit k of byte 2j+half = 9th bit of grid
// byte k of that half), signs [32], scales [4] (low nibble half 0, high nibble half 1 of group j).
// The 8 indices of a half are built in one register (ggml-cpu's AVX2 IQ3_S: a widened load, a sllv of qh and a
// mask) and looked up with one gather, instead of Strata's 8 scalar shifts, ors and loads: 4 tokens 323 -> 226 us
// for a 2560 x 640 matrix (-O3, one thread; the gather alone 255 -> 226).
inline void dec_iq3s(const Fields & F, int64_t i, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
    const uint8_t * q = F.at(1, i) + 16 * j + 8 * half;
    const uint32_t  h = F.at(2, i)[2 * j + half];
    const __m256i ix = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) q)),
        _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32((int) h), _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8)), _mm256_set1_epi32(256)));
    g = _mm256_i32gather_epi32((const int *) iq3s_grid, ix, 4);
    sgn = sgn_vec(u32(F.at(3, i) + 8 * j + 4 * half));
    const uint8_t s = F.at(4, i)[j];
    sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
}

// ggml's block constant, the factor on d: 1/8 for the IQ2 types, 1/4 for IQ3_XXS (ggml_vec_dot_iq3_xxs_q8_K),
// 1 for IQ3_S (ggml_vec_dot_iq3_s_q8_K applies none; Strata's Fmt32<21>::K = 1)
template <int TY> constexpr float iq_k() {
    return TY == GGML_TYPE_IQ3_XXS ? 0.25f : TY == GGML_TYPE_IQ3_S ? 1.0f : 0.125f;
}

template <int TY, int NT>
inline void row_iq(const Fields & F, int64_t i0, int nblocks, const block_q8_K * const * y, float * res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        const int64_t i = i0 + b;
        GGML_CL_MISS_PF(F.at(1, i + GGML_CL_MISS_PF_AHEAD));
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                if constexpr (TY == GGML_TYPE_IQ2_S)        dec_iq2s(F, i, j, half, g, sgn, sc);
                else if constexpr (TY == GGML_TYPE_IQ3_XXS) dec_iq3xxs(F, i, j, half, g, sgn, sc);
                else if constexpr (TY == GGML_TYPE_IQ3_S)   dec_iq3s(F, i, j, half, g, sgn, sc);
                else                                        dec_iq2xxs(F, i, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i *) (y[t][b].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc));
                }
            }
        }
        const float dx = h2f(u16(F.at(0, i))) * iq_k<TY>();
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][b].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// IQ2_XS: d, qs [32 x u16: 9-bit grid index + 7-bit sign index], scales [8: two 4-bit scales a 32 values].
// Strata's row_dot_iq2xs (ggml_vec_dot_iq2_xs_q8_K's decode): per 128 values one 32-byte load gives the 16
// grid indices and the 16 sign bytes (the 8th bit by parity: bits 0-3 of the index xor bits 4-6, then a
// pshufb of the parity table); the 8 scale bytes become the 16 half-scales 2 * s + 1 interleaved [a0, b0,
// a1, b1, ...] once a block. Every token pays a load, a sign, a maddubs, a madd and an add per 32 values,
// into two alternating accumulators. The scale is ggml's: d x 0.125 x sum(ls x grid x q8).
template <int NT>
inline void row_iq2xs(const Fields & F, int64_t i0, int nblocks, const block_q8_K * const * y, float * res) {
    static const uint8_t bit_sel[32] = {
        1, 2, 4, 8, 16, 32, 64, 0x80, 1, 2, 4, 8, 16, 32, 64, 0x80,
        1, 2, 4, 8, 16, 32, 64, 0x80, 1, 2, 4, 8, 16, 32, 64, 0x80 };
    static const uint8_t shuf_even[32] = {   // sign bytes of u16 0-3 of a 128-bit lane
        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6 };
    static const uint8_t shuf_odd[32] = {    // sign bytes of u16 4-7
        8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14 };
    static const uint8_t bit_help[32] = {    // 0x80 where the 4-bit value has odd popcount
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00 };
    static const uint8_t sc_shuf[128] = {    // 32-value half H -> scale bytes 2H (lanes 0-7), 2H + 1 (lanes 8-15)
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,     10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 };
    const __m256i bsel  = _mm256_loadu_si256((const __m256i *) bit_sel);
    const __m256i shev  = _mm256_loadu_si256((const __m256i *) shuf_even);
    const __m256i shod  = _mm256_loadu_si256((const __m256i *) shuf_odd);
    const __m256i bhelp = _mm256_loadu_si256((const __m256i *) bit_help);
    const __m256i m511  = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    const __m256i one8 = _mm256_set1_epi8(1);

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        const int64_t i = i0 + b;
        GGML_CL_MISS_PF(F.at(1, i + GGML_CL_MISS_PF_AHEAD));
        const uint8_t * qs = F.at(1, i);
        __m128i st = _mm_set1_epi64x((long long) u64(F.at(2, i)));
        st = _mm_unpacklo_epi8(_mm_and_si128(st, m4), _mm_and_si128(_mm_srli_epi16(st, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(st, 1), m1);
        __m256i acc[NT][2];
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_si256();
        for (int jj = 0; jj < 2; ++jj) {   // 128 values (16 u16) a step
            const __m256i q2  = _mm256_loadu_si256((const __m256i *) (qs + 32 * jj));
            const __m256i p7  = _mm256_srli_epi16(q2, 9);    // the 7 sign bits, in the even bytes
            const __m256i p3  = _mm256_srli_epi16(q2, 13);   // sign index bits 4-6
            const __m256i fsb = _mm256_or_si256(p7, _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3)));
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb));       // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1));  // u16 8-15
            alignas(32) uint16_t gi[16];
            _mm256_store_si256((__m256i *) gi, _mm256_and_si256(q2, m511));
            for (int h = 0; h < 4; ++h) {
                const __m256i g = _mm256_set_epi64x((long long) iq2xs_grid[gi[4 * h + 3]], (long long) iq2xs_grid[gi[4 * h + 2]],
                                                    (long long) iq2xs_grid[gi[4 * h + 1]], (long long) iq2xs_grid[gi[4 * h]]);
                const __m256i sb  = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc  = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_loadu_si128((const __m128i *) sc_shuf + 4 * jj + h)));
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i *) (y[t][b].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h & 1] = _mm256_add_epi32(acc[t][h & 1], _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc));
                }
            }
        }
        const float dx = h2f(u16(F.at(0, i))) * iq_k<GGML_TYPE_IQ2_XS>();
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][b].d), _mm256_cvtepi32_ps(_mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// IQ3_XXS, one token (NT = 1): the decode above in the shape of ggml-cpu's AVX2 loop (ggml_vec_dot_iq3_xxs_q8_K),
// a 64-value group a step and the block's pointers taken once - row_iq's half-at-a-time loop is slower than ggml's
// for a single token, faster for two and more. The integer sums are exact, so only the float steps set the result:
// one fmadd a block of d x d8 (the 1/4 folded into d, a power of two) and the final horizontal sum, as in row_iq.
template <int NT>
inline void row_iq3xxs(const Fields & F, int64_t i0, int nblocks, const block_q8_K * const * y, float * res) {
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        const int64_t i = i0 + b;
        GGML_CL_MISS_PF(F.at(1, i + GGML_CL_MISS_PF_AHEAD));
        const uint8_t * q3  = F.at(1, i);
        const uint8_t * gas = F.at(2, i);
        __m256i s1[NT], s2[NT];
        for (int t = 0; t < NT; ++t) { s1[t] = _mm256_setzero_si256(); s2[t] = _mm256_setzero_si256(); }
        for (int ib = 0; ib < QK_K / 32; ib += 2, q3 += 16, gas += 8) {
            const __m256i g1 = _mm256_set_epi32((int) iq3xxs_grid[q3[7]], (int) iq3xxs_grid[q3[6]], (int) iq3xxs_grid[q3[5]], (int) iq3xxs_grid[q3[4]],
                                                (int) iq3xxs_grid[q3[3]], (int) iq3xxs_grid[q3[2]], (int) iq3xxs_grid[q3[1]], (int) iq3xxs_grid[q3[0]]);
            const __m256i g2 = _mm256_set_epi32((int) iq3xxs_grid[q3[15]], (int) iq3xxs_grid[q3[14]], (int) iq3xxs_grid[q3[13]], (int) iq3xxs_grid[q3[12]],
                                                (int) iq3xxs_grid[q3[11]], (int) iq3xxs_grid[q3[10]], (int) iq3xxs_grid[q3[9]], (int) iq3xxs_grid[q3[8]]);
            const uint32_t w1 = u32(gas), w2 = u32(gas + 4);
            const __m256i sg1 = _mm256_set_epi64x((long long) even_signs.v[(w1 >> 21) & 127], (long long) even_signs.v[(w1 >> 14) & 127],
                                                  (long long) even_signs.v[(w1 >> 7) & 127], (long long) even_signs.v[w1 & 127]);
            const __m256i sg2 = _mm256_set_epi64x((long long) even_signs.v[(w2 >> 21) & 127], (long long) even_signs.v[(w2 >> 14) & 127],
                                                  (long long) even_signs.v[(w2 >> 7) & 127], (long long) even_signs.v[w2 & 127]);
            const __m256i sc1 = sc32(2 * (int) (w1 >> 28) + 1), sc2 = sc32(2 * (int) (w2 >> 28) + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][b].qs + 32 * ib;
                const __m256i y1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), sg1);
                const __m256i y2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32)), sg2);
                s1[t] = _mm256_add_epi32(s1[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g1, y1), sc1));
                s2[t] = _mm256_add_epi32(s2[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g2, y2), sc2));
            }
        }
        const float dx = h2f(u16(F.at(0, i))) * iq_k<GGML_TYPE_IQ3_XXS>();
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][b].d), _mm256_cvtepi32_ps(_mm256_add_epi32(s1[t], s2[t])), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(accf[t]);
}

// IQ3_S, one token: ggml-cpu's AVX2 loop (ggml_vec_dot_iq3_s_q8_K) over the fields - a 64-value group a step, the
// 16 grid indices from one 16-byte load, the signs xor/sub, two integer accumulators, one fmadd a block - with
// three savings: one broadcast of the two qh bytes (sllv for the low, srlv for the high), one of the 8 sign
// bytes (two shuffles), the 4 scale bytes read once a block. row_iq's half-at-a-time decode is ~6 % slower than
// ggml's for one token and faster for two and more. Integer sums are exact; the float steps are row_iq's.
inline void row_iq3s_1(const Fields & F, int64_t i0, int nblocks, const block_q8_K * y, float * res) {
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    const __m256i mask1b = _mm256_add_epi8(mask1, _mm256_set1_epi8(4));
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);
    const __m256i idx_shift_hi = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    typedef union { __m256i vec[2]; uint32_t index[16]; } index_t;
    index_t idx;
    __m256 accumf = _mm256_setzero_ps();
    for (int b = 0; b < nblocks; ++b) {
        const int64_t i = i0 + b;
        GGML_CL_MISS_PF(F.at(1, i + GGML_CL_MISS_PF_AHEAD));
        const float d = h2f(u16(F.at(0, i))) * iq_k<GGML_TYPE_IQ3_S>() * y[b].d;
        const uint8_t * qs = F.at(1, i);
        const uint8_t * qh = F.at(2, i);
        const uint8_t * signs = F.at(3, i);
        const uint32_t sc4 = u32(F.at(4, i));
        const int8_t * q8 = y[b].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) qs)); qs += 16;
            const __m256i hh = _mm256_set1_epi32((int) u16(qh + ib32));
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(hh, idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_srlv_epi32(hh, idx_shift_hi), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(
                    (int) iq3s_grid[idx.index[7]], (int) iq3s_grid[idx.index[6]], (int) iq3s_grid[idx.index[5]], (int) iq3s_grid[idx.index[4]],
                    (int) iq3s_grid[idx.index[3]], (int) iq3s_grid[idx.index[2]], (int) iq3s_grid[idx.index[1]], (int) iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                    (int) iq3s_grid[idx.index[15]], (int) iq3s_grid[idx.index[14]], (int) iq3s_grid[idx.index[13]], (int) iq3s_grid[idx.index[12]],
                    (int) iq3s_grid[idx.index[11]], (int) iq3s_grid[idx.index[10]], (int) iq3s_grid[idx.index[ 9]], (int) iq3s_grid[idx.index[ 8]]);
            const __m256i sgb = _mm256_set1_epi64x((long long) u64(signs));
            __m256i aux256 = _mm256_and_si256(_mm256_shuffle_epi8(sgb, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(sgb, mask1b), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 8;
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = (sc4 >> (4 * ib32)) & 0xf;
            const uint16_t ls2 = (sc4 >> (4 * ib32 + 4)) & 0xf;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    res[0] = hsum8(accumf);
}

// ---- Q2_0 against 32-value int8 chunks (Strata's q2_avx2.cpp): codes 0..3, the -1 offset as a correction

struct ActQ2 {   // a token's activation: q[n], per 32 values scale, scale x sum (hx)
    int8_t * q; float * scale; float * hx;
};

inline void unpack64(const uint8_t * codes, __m256i & lo, __m256i & hi) {
    const __m128i b = _mm_loadu_si128((const __m128i *) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));
}

template <int NT>
inline void row_q2(const Fields & F, int64_t i0, int nblocks, const ActQ2 * a, float * res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nblocks; ++b) {
        const int64_t i = i0 + b;
        GGML_CL_MISS_PF(F.at(1, i + GGML_CL_MISS_PF_AHEAD));
        const float d = h2f(u16(F.at(0, i)));
        __m256i lo, hi;
        unpack64(F.at(1, i), lo, hi);
        for (int t = 0; t < NT; ++t) {
            const int8_t * q = a[t].q + b * 64;
            const __m256i s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *) q)), ones);
            const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *) (q + 32))), ones);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t].scale[2 * b]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t].scale[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * (a[t].hx[2 * b] + a[t].hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum8(acc[t]) - corr[t];
}

// ---- IQ4_NL against Q8_0 activations (Strata's iq4nl_rows), from the bank's split scales and quants
//
// The nibbles' kvalues_iq4nl lookup and |w| (maddubs' unsigned operand) once a block; each token then
// costs a sign, a maddubs, a madd and an fmadd. ggml_vec_dot_iq4_nl_q8_0's float order is kept: even
// blocks in one accumulator, odd blocks in another, their sum reduced, an odd last block added as a scalar.
// block ib's 32 weights: values 0..15 from the low nibbles, 16..31 from the high ones; and |w|
inline void dec_iq4nl(const uint8_t * q, int ib, __m128i values, __m256i & q4, __m256i & aq) {
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m128i bits = _mm_loadu_si128((const __m128i *) (q + 16 * ib));
    const __m128i lo = _mm_and_si128(bits, m4b);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);
    q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)), _mm_shuffle_epi8(values, hi), 1);
    aq = _mm256_sign_epi8(q4, q4);
}

// one row against NT tokens; dy[t][ib]: token t's block scales as floats
template <int NT>
inline void row_iq4nl(const uint8_t * d, const uint8_t * q, int nb, const block_q8_0 * const * y, const float * const * dy,
                      float * res) {
    const __m128i values = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0[NT], acc1[NT];
#pragma GCC unroll 4
    for (int t = 0; t < NT; ++t) { acc0[t] = _mm256_setzero_ps(); acc1[t] = _mm256_setzero_ps(); }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        GGML_CL_MISS_PF(q + 16 * (ib + GGML_CL_MISS_PF_AHEAD));
        __m256i q4, aq;
        dec_iq4nl(q, ib, values, q4, aq);
        __m256 dx = _mm256_set1_ps(h2f(u16(d + 2 * ib)));
#pragma GCC unroll 4
        for (int t = 0; t < NT; ++t) {
            const __m256i q8 = _mm256_loadu_si256((const __m256i *) y[t][ib].qs);
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
            acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_broadcast_ss(dy[t] + ib), dx), _mm256_cvtepi32_ps(p), acc0[t]);
        }
        dec_iq4nl(q, ib + 1, values, q4, aq);
        dx = _mm256_set1_ps(h2f(u16(d + 2 * (ib + 1))));
#pragma GCC unroll 4
        for (int t = 0; t < NT; ++t) {
            const __m256i q8 = _mm256_loadu_si256((const __m256i *) y[t][ib + 1].qs);
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
            acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_broadcast_ss(dy[t] + ib + 1), dx), _mm256_cvtepi32_ps(p), acc1[t]);
        }
    }
#pragma GCC unroll 4
    for (int t = 0; t < NT; ++t) res[t] = hsum8(_mm256_add_ps(acc0[t], acc1[t]));
    if (ib < nb) {
        __m256i q4, aq;
        dec_iq4nl(q, ib, values, q4, aq);
        const float dx = h2f(u16(d + 2 * ib));
        for (int t = 0; t < NT; ++t) {
            const __m256i q8 = _mm256_loadu_si256((const __m256i *) y[t][ib].qs);
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
            const __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
            const __m128i s2 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
            const int sumi = _mm_cvtsi128_si32(_mm_add_epi32(s2, _mm_shuffle_epi32(s2, 1)));
            res[t] += (dy[t][ib] * dx) * (float) sumi;
        }
    }
}

template <int NT>
void rows_iq4nl_nt(const uint8_t * d, const uint8_t * q, int64_t blk0, int nb, const void * const * act,
                   float * const * out, int r0, int r1) {
    // the tokens' f16 block scales converted once a call, not once a row
    float dys[4 * 256];
    std::vector<float> dyh;
    float * dyb = dys;
    if (nb > 256) { dyh.resize((size_t) NT * nb); dyb = dyh.data(); }
    float res[NT];
    const block_q8_0 * y[NT];
    const float * dy[NT];
    for (int k = 0; k < NT; ++k) {
        y[k] = (const block_q8_0 *) act[k];
        float * f = dyb + (size_t) k * nb;
        for (int ib = 0; ib < nb; ++ib) f[ib] = h2f(y[k][ib].d);
        dy[k] = f;
    }
    for (int r = r0; r < r1; ++r) {
        const int64_t i = blk0 + (int64_t) r * nb;
        row_iq4nl<NT>(d + 2 * i, q + 16 * i, nb, y, dy, res);
#pragma GCC unroll 4
        for (int k = 0; k < NT; ++k) out[k][r] = res[k];
    }
}

template <int NT>
void rows_nt(ggml_type t, const Fields & F, int64_t ne0, const void * const * act, float * const * out, int r0, int r1) {
    float res[NT];
    if (t == GGML_TYPE_Q2_0) {
        const int nb = (int) (ne0 / 64);
        ActQ2 a[NT];
        for (int k = 0; k < NT; ++k) {
            const char * p = (const char *) act[k];
            a[k].q = (int8_t *) p; a[k].scale = (float *) (p + ne0); a[k].hx = a[k].scale + ne0 / 32;
        }
        for (int r = r0; r < r1; ++r) {
            row_q2<NT>(F, (int64_t) r * nb, nb, a, res);
            for (int k = 0; k < NT; ++k) out[k][r] = res[k];
        }
        return;
    }
    const int nb = (int) (ne0 / QK_K);
    const block_q8_K * y[NT];
    for (int k = 0; k < NT; ++k) y[k] = (const block_q8_K *) act[k];
    for (int r = r0; r < r1; ++r) {
        if (t == GGML_TYPE_IQ2_S)        row_iq<GGML_TYPE_IQ2_S, NT>(F, (int64_t) r * nb, nb, y, res);
        else if (t == GGML_TYPE_IQ2_XS)  row_iq2xs<NT>(F, (int64_t) r * nb, nb, y, res);
        else if (t == GGML_TYPE_IQ3_XXS) {
            if constexpr (NT == 1) row_iq3xxs<1>(F, (int64_t) r * nb, nb, y, res);
            else                   row_iq<GGML_TYPE_IQ3_XXS, NT>(F, (int64_t) r * nb, nb, y, res);
        }
        else if (t == GGML_TYPE_IQ3_S) {
            if constexpr (NT == 1) row_iq3s_1(F, (int64_t) r * nb, nb, y[0], res);
            else                   row_iq<GGML_TYPE_IQ3_S, NT>(F, (int64_t) r * nb, nb, y, res);
        }
        else                             row_iq<GGML_TYPE_IQ2_XXS, NT>(F, (int64_t) r * nb, nb, y, res);
        for (int k = 0; k < NT; ++k) out[k][r] = res[k];
    }
}

}  // namespace

size_t ggml_cl_miss_cpu_act_size(ggml_type t, int64_t n) {
    if (t == GGML_TYPE_IQ4_NL) return (size_t) (n / QK8_0) * sizeof(block_q8_0);
    return t == GGML_TYPE_Q2_0 ? (size_t) n + (size_t) (n / 32) * 2 * sizeof(float) : (size_t) (n / QK_K) * sizeof(block_q8_K);
}

void ggml_cl_miss_cpu_quant(ggml_type t, const float * x, int64_t n, void * y) {
    if (t == GGML_TYPE_Q2_0) {
        // Strata's act_quant_q8_1: per 32 values s = amax / 127, q = round-half-away(x / s), hx = s x sum(q)
        int8_t * q = (int8_t *) y;
        float * scale = (float *) ((char *) y + n), * hx = scale + n / 32;
        for (int64_t c = 0; c < n / 32; ++c) {
            const float * xb = x + 32 * c;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, fabsf(xb[j]));
            const float s = amax > 0.f ? amax / 127.f : 0.f, inv = s > 0.f ? 1.f / s : 0.f;
            int sum = 0;
            for (int j = 0; j < 32; ++j) {
                const float v = xb[j] * inv;
                int qi = (int) (v + (v >= 0.f ? 0.5f : -0.5f));
                qi = qi < -127 ? -127 : qi > 127 ? 127 : qi;
                q[32 * c + j] = (int8_t) qi; sum += qi;
            }
            scale[c] = s; hx[c] = s * (float) sum;
        }
        return;
    }
    if (t == GGML_TYPE_IQ4_NL) {
        // ggml-cpu's AVX2 quantize_row_q8_0 (IQ4_NL's from_float there, not quantize_row_q8_0_ref):
        // d = amax / 127, x * (127 / amax) rounded half to even
        block_q8_0 * b = (block_q8_0 *) y;
        const __m256 sign = _mm256_set1_ps(-0.0f);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (int64_t i = 0; i < n / QK8_0; ++i, x += QK8_0) {
            __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
            __m256 ma = _mm256_andnot_ps(sign, v0);
            ma = _mm256_max_ps(ma, _mm256_andnot_ps(sign, v1));
            ma = _mm256_max_ps(ma, _mm256_andnot_ps(sign, v2));
            ma = _mm256_max_ps(ma, _mm256_andnot_ps(sign, v3));
            __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(ma, 1), _mm256_castps256_ps128(ma));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            const float amax = _mm_cvtss_f32(m4);
            b[i].d = (ggml_half) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(amax / 127.f), 0), 0);
            const __m256 mul = _mm256_set1_ps(amax != 0.0f ? 127.f / amax : 0.0f);
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST));
            const __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST));
            const __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST));
            const __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST));
            const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
            _mm256_storeu_si256((__m256i *) b[i].qs, _mm256_permutevar8x32_epi32(p, perm));
        }
        return;
    }
    // ggml's quantize_row_q8_K_ref
    block_q8_K * b = (block_q8_K *) y;
    for (int64_t i = 0; i < n / QK_K; ++i, x += QK_K) {
        float amax = 0.f, mx = 0.f;
        for (int j = 0; j < QK_K; ++j) { const float a = fabsf(x[j]); if (a > amax) { amax = a; mx = x[j]; } }
        if (!amax) { b[i].d = 0.f; memset(b[i].qs, 0, QK_K); memset(b[i].bsums, 0, sizeof(b[i].bsums)); continue; }
        const float iscale = -127.f / mx;
        for (int j = 0; j < QK_K; ++j) { const int v = (int) nearbyintf(iscale * x[j]); b[i].qs[j] = (int8_t) (v > 127 ? 127 : v); }
        for (int j = 0; j < QK_K / 16; ++j) { int s = 0; for (int l = 0; l < 16; ++l) s += b[i].qs[16 * j + l]; b[i].bsums[j] = (int16_t) s; }
        b[i].d = 1.f / iscale;
    }
}

void ggml_cl_miss_cpu_rows(ggml_type t, const void * slice, bool planar, int64_t ne0, int64_t ne1,
                           const void * const * act, int nt, float * const * out, int r0, int r1) {
    Fields F;
    // a type without a field table here (IQ4_NL: ggml_cl_miss_cpu_rows_iq4nl) must not reach this entry:
    // returning would leave the caller's rows unwritten
    if (!fields_of(t, (const uint8_t *) slice, planar, ne0, ne1, F)) {
        GGML_ABORT("ggml_cl_miss_cpu_rows: no field table for %s", ggml_type_name(t));
    }
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: rows_nt<1>(t, F, ne0, act + t0, out + t0, r0, r1); break;
            case 2: rows_nt<2>(t, F, ne0, act + t0, out + t0, r0, r1); break;
            case 3: rows_nt<3>(t, F, ne0, act + t0, out + t0, r0, r1); break;
            default: rows_nt<4>(t, F, ne0, act + t0, out + t0, r0, r1); break;
        }
    }
}

void ggml_cl_miss_cpu_set_prefetch(int level) {
    g_cl_miss_pf = level < 0 ? 0 : level;
}

int ggml_cl_miss_cpu_prefetch_ahead(void) {
    return GGML_CL_MISS_PF_AHEAD;
}

// patch 0075's cold-page half: one byte a 4 KiB page of a slice. The premise is
// NOT Strata's: a bank here is USM host memory (clHostMemAllocINTEL) and every
// byte is written at load by ggml_cl_usm_set_tensor (`code`: ggml-opencl.cpp,
// the planar loop and the IQ4_NL split copy), so the pages are resident before
// the first miss and drop_caches evicts page cache, not this memory. Strata's
// cold reads are file-backed rows, a materially different build. The prediction
// is therefore major faults near 0 and this pass buying nothing — the window's
// fault count decides (`measured-here` owed). The reads are volatile: the point
// would be the fault, not the value.
void ggml_cl_miss_cpu_touch(const void * p, size_t bytes) {
    const volatile unsigned char * q = (const volatile unsigned char *) p;
    unsigned char acc = 0;
    for (size_t o = 0; o < bytes; o += 4096) acc ^= q[o];
    (void) acc;
}

void ggml_cl_miss_cpu_rows_iq4nl(const void * d, const void * q, int64_t blk0, int64_t ne0, int64_t ne1,
                                 const void * const * act, int nt, float * const * out, int r0, int r1) {
    (void) ne1;
    const int nb = (int) (ne0 / QK4_NL);
    const uint8_t * dp = (const uint8_t *) d, * qp = (const uint8_t *) q;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
            case 1: rows_iq4nl_nt<1>(dp, qp, blk0, nb, act + t0, out + t0, r0, r1); break;
            case 2: rows_iq4nl_nt<2>(dp, qp, blk0, nb, act + t0, out + t0, r0, r1); break;
            case 3: rows_iq4nl_nt<3>(dp, qp, blk0, nb, act + t0, out + t0, r0, r1); break;
            default: rows_iq4nl_nt<4>(dp, qp, blk0, nb, act + t0, out + t0, r0, r1); break;
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#else
bool   ggml_cl_miss_cpu_ok(void) { return false; }
bool   ggml_cl_miss_cpu_type(ggml_type) { return false; }
size_t ggml_cl_miss_cpu_act_size(ggml_type, int64_t) { return 0; }
void   ggml_cl_miss_cpu_quant(ggml_type, const float *, int64_t, void *) {}
void   ggml_cl_miss_cpu_rows(ggml_type, const void *, bool, int64_t, int64_t, const void * const *, int, float * const *, int, int) {}
void   ggml_cl_miss_cpu_set_prefetch(int) {}
int    ggml_cl_miss_cpu_prefetch_ahead(void) { return 0; }
void   ggml_cl_miss_cpu_touch(const void *, size_t) {}
void   ggml_cl_miss_cpu_rows_iq4nl(const void *, const void *, int64_t, int64_t, int64_t, const void * const *, int, float * const *, int, int) {}
#endif

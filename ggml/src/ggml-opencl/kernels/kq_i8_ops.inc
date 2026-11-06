// kq_i8_ops.inc -- the int8 DPAS expert GEMM's device-side helpers and its
// dispatch rule, shared verbatim by the int8 DPAS kernel (mul_mm_kq_i8.cl, by a
// source prepend at build time) and the device-free test
// (tests/test-opencl-kq-i8.cpp). One text, so a red mutation in the test is a
// mutation of the kernel.
//
// Evidence classes for the claims here are in contrib/llama.cpp/README.md
// (0071) and docs/campaigns/int8-dpas-prompt-experts.md.
//
// Layout references (all `code`): ggml's dequantize_row_iq3_s /
// dequantize_row_iq4_nl in ggml/src/ggml-quants.c; the plane forms the tile
// kernel already uses (patch 0022) in mul_mm_kq_f16.cl; Strata's ESIMD
// decoders (esimd_moe.dp.cpp, gate_up/down) for the signed int8 pieces and the
// q8_1 activation form.

#ifndef KQ_I8_OPS_INC
#define KQ_I8_OPS_INC

#ifdef __cplusplus
#include <cstdint>
typedef std::uint32_t kq_u32;
typedef std::uint8_t  kq_u8;
typedef std::int8_t   kq_s8;
typedef std::int16_t  kq_i16;
#define KQ_CONST const
#else
// OpenCL C: uint / uchar / char / short exist.
typedef uint   kq_u32;
typedef uchar  kq_u8;
typedef char   kq_s8;
typedef short  kq_i16;
#define KQ_CONST const constant
#endif

// little-endian 32-bit word from four bytes (the plane fields are not 4-byte
// aligned in every tensor, so no uint loads)
static inline kq_u32 kq_i8_u32(const kq_u8 *b) {
    return (kq_u32)b[0] | ((kq_u32)b[1] << 8) | ((kq_u32)b[2] << 16) | ((kq_u32)b[3] << 24);
}

// Four sign bits of a IQ2-family nibble -> 0xFF in the bytes to negate, then
// negate by (g ^ mask) + mask: one add and one xor a byte, no carry or borrow
// crosses a byte because no grid value is 0 (IQ3_S grid values are 1..15).
static inline kq_u32 kq_i8_signs4(kq_u32 g, kq_u32 nib) {
    const kq_u32 m = ((nib * 0x00204081u) & 0x01010101u);
    return (g ^ (m * 255u)) + m;
}

// IQ3_S: one 32-weight block of one row as eight packed dwords of signed int8
// values (values 4i..4i+3 in w[i]), from the planes the tile kernel holds:
// q0/q1 the 8 qs bytes, qh the high-index byte, s0 the 4 sign bytes, grid =
// iq3s_grid (512 uint32 entries from ggml-common.h / ggml-quants.c, each four
// codebook bytes). Same arithmetic as dequantize_row_iq3_s: for r = 0..7, qsw
// byte r&3 of q0/q1, its 9th index bit is qh bit r; signs nibble r&1 of signs
// byte r/2 (low nibble for the even r, high for the odd one).
static inline void kq_i8_decode_iq3s(kq_u32 q0, kq_u32 q1, kq_u32 s0, kq_u32 qh,
                                     KQ_CONST kq_u32 *grid, kq_u32 w[8]) {
    for (int r = 0; r < 8; ++r) {
        const kq_u32 qsw = r < 4 ? q0 : q1;
        const kq_u32 idx = ((qsw >> (8 * (r & 3))) & 0xFFu) | (((qh >> r) & 1u) << 8);
        const kq_u32 nib = (s0 >> (8 * (r >> 1))) >> (4 * (r & 1));
        w[r] = kq_i8_signs4(grid[idx], nib & 0xFu);
    }
}

// IQ4_NL: one 32-value block of one row as eight packed dwords. The 16 stored
// bytes are 4-bit codes; dequantize_row_iq4_nl's order is values k = j and
// k = 16 + j for stored byte j (low nibble first), so dwords 0..3 are the low
// nibbles and dwords 4..7 the high ones. kv = kvalues_iq4nl as 16 signed
// bytes; no rounding, the codebook is already int8.
static inline void kq_i8_decode_iq4nl(kq_u32 qd0, kq_u32 qd1, kq_u32 qd2, kq_u32 qd3,
                                      KQ_CONST kq_s8 *kv, kq_u32 w[8]) {
    const kq_u32 qd[4] = { qd0, qd1, qd2, qd3 };
    for (int j = 0; j < 4; ++j) {
        const kq_u32 q = qd[j];
        const kq_u32 l0 = (kq_u32)(kq_u8)kv[q & 0xFu],      l1 = (kq_u32)(kq_u8)kv[(q >> 8) & 0xFu];
        const kq_u32 l2 = (kq_u32)(kq_u8)kv[(q >> 16) & 0xFu], l3 = (kq_u32)(kq_u8)kv[(q >> 24) & 0xFu];
        const kq_u32 h0 = (kq_u32)(kq_u8)kv[(q >> 4) & 0xFu],  h1 = (kq_u32)(kq_u8)kv[(q >> 12) & 0xFu];
        const kq_u32 h2 = (kq_u32)(kq_u8)kv[(q >> 20) & 0xFu], h3 = (kq_u32)(kq_u8)kv[(q >> 28) & 0xFu];
        w[j]     = l0 | (l1 << 8) | (l2 << 16) | (l3 << 24);
        w[j + 4] = h0 | (h1 << 8) | (h2 << 16) | (h3 << 24);
    }
}

// IQ3_S's per-32 weight scale from a 256-block's 4 scale bytes: nibble p&1 of
// byte p>>1, factor d * (1 + 2 sc) (dequantize_row_iq3_s).
static inline int kq_i8_iq3s_sc(kq_u8 scale_byte, int p) {
    return (scale_byte >> (4 * (p & 1))) & 0xFu;
}

// q8_1 activation quantization of one 32-value block, llama.cpp's q8_1 grid
// (ggml quantize_row_q8_1): scale = amax / 127, code = round(x / scale) with
// round half away from zero. Strata's kernels build the same values in ESIMD
// (esimd_moe.dp.cpp quantize_rows: adj = +-0.5, convert). The integer sum q8_1
// also stores is NOT needed by this kernel form: each 32-block's int32 DPAS
// product is converted to float and scaled by (weight scale x activation
// scale), which replaces the integer mad on a per-super-block accumulator.
static inline float kq_i8_amax32(const float *x) {
    float a = 0.f;
    for (int i = 0; i < 32; ++i) {
        const float v = x[i] < 0.f ? -x[i] : x[i];
        a = v > a ? v : a;
    }
    return a;
}

static inline void kq_i8_quant32(const float *x, kq_s8 q[32], float *scale) {
    const float amax = kq_i8_amax32(x);
    const float d = amax / 127.f;
    const float id = amax > 0.f ? 127.f / amax : 0.f;
    for (int i = 0; i < 32; ++i) {
        const float y = x[i] * id;
        q[i] = (kq_s8)(y < 0.f ? y - 0.5f : y + 0.5f);   // truncation toward zero
    }
    *scale = d;
}

// ---- the block indices ----
// The int8 kernel's block loop counts kb in ELEMENTS of K (k0 += 256,
// kb += 32) while the fp16 tile kernel indexes its 32-weight sub-block j32 =
// kb / 32 and reads it through LB_BI(T, r, nbr, j32) = r * nbr + j32 /
// LB_SUBS(T), LB_P(T, j32) = j32 % LB_SUBS(T) (LB_SUBS(IQ3_S) = 8). The
// earlier int8 kernel read kb as if it were that 32-block index, so the
// weight reads landed outside the block on the last experts (round 3's card
// defect). These helpers are the fp16 path's arithmetic, one text for the
// kernels and the test.
static inline int kq_i8_iq3s_sb(int kb) { return kb / 256; }          // IQ3_S super-block of an element offset
static inline int kq_i8_iq3s_p(int kb)  { return (kb / 32) & 7; }     // its 32-sub-block, LB_P's p
static inline int kq_i8_iq4nl_blk(int kb) { return kb / 32; }         // IQ4_NL's 32-block index, LB_BI's j32

// the activation column of a route pair p, the fp16 tile kernel's form:
// (i % ne11) + t * ne11 with i = p % n_used (the slot) and t = p / n_used (the
// token). With a broadcast b (ne11 = 1, the served gate/up) a token's n_used
// pairs share one column, so the int8 activation is stored once a column and
// not once a pair (the fp16 path's size, n_cols = ne11 * ne12, not n_pairs).
static inline int kq_i8_col(int p, int n_used, int ne11) {
    const int t = n_used > 0 ? p / n_used : 0;
    const int i = n_used > 0 ? p % n_used : 0;
    return ne11 > 0 ? (i % ne11) + t * ne11 : t;
}

// ---- host-only: the dispatch rule, the format list and the launch sizing ----
// Nothing below is called by the kernels: it is the backend's and the test's
// decision text. It sits behind #ifdef __cplusplus so the OpenCL program never
// sees it -- a string literal here is __constant char *, which an OpenCL C
// pointer cannot take (round 6's build blocker: Intel's compiler and clang-18
// both refused the assembly, and the silent fallback measured the fp16 tile).
#ifdef __cplusplus

// ---- the dispatch rule (host-only) ----
// GGML_OPENCL_KQ_MM_ID_I8: unset -> 0, the fp16 tile kernel (this patch keeps
// int8 opt-in; a later commit flips the default to auto and records the B60's
// KL/task/speed first). 0 -> fp16; 1 -> int8 wherever the format is handled;
// auto -> 2, int8 while the expert group's mean rows per expert are at or
// under the crossover. Returns 0 fp16, 1 int8, 2 auto. The s8 DPAS is a
// sub-group 16 primitive and the arm needs its kernels, so a sub-group 8
// device or a missing kernels pointer is fp16 whatever the switch says.
// Host-only text: the test includes it as C++, the OpenCL program never sees it.
static inline int kq_i8_mode(const char *env, int sg, int kernels) {
    if (sg != 16 || kernels == 0) return 0;
    if (env != 0 && env[0] != 0) {
        if (env[0] == 'a' || env[0] == 'A') return 2;
        long v = 0;
        for (const char *c = env; *c >= '0' && *c <= '9' && v < 1000000; ++c) v = v * 10 + (*c - '0');
        return v != 0 ? 1 : 0;
    }
    return 0;
}

// ---- the per-format list ----
// GGML_OPENCL_KQ_MM_ID_I8_FMT: a comma or whitespace separated list of iq3_s
// and iq4_nl (case-insensitive), the formats the int8 arm may take. Unset,
// empty, whitespace-only or a value with no token is both, the behaviour when
// the arm is on. A format not in the list takes the fp16 tile; a layer whose gate/up are
// int8 while its down is fp16 reroutes through the cache key's rc.i8, so the
// route is per call and correct.
static inline char kq_i8_lower(char c) {
    return c >= 'A' && c <= 'Z' ? (char)(c + 32) : c;
}

static inline int kq_i8_sep(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',';
}

static inline int kq_i8_fmt_allowed(const char *fmt, int is_iq4nl) {
    if (fmt == 0) return 1;                     // unset: both formats
    const char *name = is_iq4nl ? "iq4_nl" : "iq3_s";
    const char *c = fmt;
    int seen = 0;
    for (;;) {
        while (kq_i8_sep(*c)) ++c;              // commas and whitespace, \n and \r too
        if (*c == 0) return seen ? 0 : 1;       // no token at all -> both; else no match
        seen = 1;
        const char *t = c;
        while (*c && !kq_i8_sep(*c)) ++c;
        const char *n = name;
        const char *p = t;
        while (p < c && *n != 0 && kq_i8_lower(*p) == *n) { ++p; ++n; }
        if (p == c && *n == 0) return 1;
    }
}

// whether the int8 program is built at all: the same switch the dispatch reads
// (unset and 0 are off), so the default path does not pay a program compile a
// unit start. The OpenCL program is only worth building when the arm can run.
// the program is gated before its kernels exist, so the parse is asked with
// the kernels present: the program gate is "would the arm run here".
static inline int kq_i8_program_wanted(const char *env) {
    return kq_i8_mode(env, 16, 1) != 0 ? 1 : 0;
}

// The group's mean routed pairs (rows) per expert. n_expert is the matrix's
// expert count; min(n_expert, n_pairs) is the upper bound of the experts a
// call touches, so the mean is never understated (Strata reads the real cnt
// per expert; a prompt's routing fills all 512 experts of Flash-Next, so the
// bound is tight there).
static inline long kq_i8_mean_pairs(long n_pairs, int n_expert) {
    const long d = n_expert < n_pairs ? n_expert : n_pairs;
    return d > 0 ? n_pairs / d : 0;
}

// auto: int8 at or under the crossover (Strata's measured 130-250 rows per
// expert on a B70; the B60's value is measured, GGML_OPENCL_KQ_MM_ID_I8_CROSS).
static inline int kq_i8_auto_use(long mean_pairs, int cross) {
    return mean_pairs <= (long)cross ? 1 : 0;
}

#endif  // __cplusplus

#endif  // KQ_I8_OPS_INC

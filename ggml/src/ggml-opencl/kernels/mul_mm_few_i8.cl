#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// ARC-LAB 0115: Q4_0, Q8_0, IQ4_XS and IQ4_NL weights times 2 to 16 f32
// columns (a speculative verify, a few lanes) on int8 DPAS, each weight read
// once for all of them. The shape of mul_mm_kq_few.cl (the K-quants' few-column
// product): a lane is a weight row, its 32 codes of a block the b operand
// (lane = column of the product), the tokens the a operand, 8 a DPAS; the NK
// sub-groups of a work-group take every NK-th chunk of K and add their sums in
// local memory. The activation is the 1-4-column matvec's (kernel_quantize_q8_1):
// per 32 values two int8 terms, the codes and the codes of what they leave,
// each with its scale, so the a operand of a block is two DPAS. The weights go
// in as their exact int8 values: Q8_0's codes as they are, Q4_0's n - 8, the
// IQ4 types' kvalues_iq4nl. Per block and token group:
//   acc += ws * (xd0 * dot(a0, b) + xd1 * dot(a1, b))
// ws the weight's block scale, xd0 / xd1 the activation's. A block's dot is at
// most 32 x 127 x 128 in magnitude: exact in int32 and in f32.
//
// int8 DPAS operands (cl_intel_subgroup_matrix_multiply_accumulate, k32):
//   SG 8 (Xe-HPG): a = int   per token, lane i holds k = 4i .. 4i + 3
//   SG 16 (Xe2):   a = short per token, lane i holds k = 2i, 2i + 1
//   b = int8 per lane: lane n is weight row n, its 32 k as 8 dwords, k ascending
//   result int8: lane n is row n, component m is token m
// Activation layout (kernel_mul_mm_fewq_prep): xq [G][k / 32][2][8][32] bytes,
// so that a term of a block for 8 tokens is one 256-byte sub-group block read;
// xd [2][k / 32][TOK] floats.
//
// Defines: FEWQ_ONE_TERM drops the residual term (one DPAS a block);
// FEWQ_IQ4_MUX decodes the IQ4 codes with 0103's select tree instead of the
// byte-pair table in local memory; FEWQ_RED leaves out every eighth block of K
// (the red case of the suite).

#ifndef SG
#define SG 16
#endif
#ifndef R
#define R 1
#endif
#ifndef G
#define G 1
#endif
#ifndef NK
#define NK 4
#endif
#define TOK (8 * G)
#define QK_K 256

#if SG == 8
typedef int8 ai8_t;
#define AREAD(p) as_int8(intel_sub_group_block_read8((global const uint *)(p)))
#else
typedef short8 ai8_t;
#define AREAD(p) as_short8(intel_sub_group_block_read_us8((global const ushort *)(p)))
#endif
#define DPAS8(a, b) intel_sub_group_i8_i8_matrix_mad_k32(a, b, (int8)(0))

// ---- the activations ----
// One work-item per (token, 32-block), the token fastest. x: f32 columns of k
// values at src + offs + t * nbx. Columns from ncols to TOK are zero.
kernel void kernel_mul_mm_fewq_prep(global const char * src, ulong offs, ulong nbx, global char * xq,
                                    global float * xd, int k, int ncols) {
    const int n32 = k / 32;
    const int gid = get_global_id(0);
    const int t = gid % TOK, kb = gid / TOK;
    if (kb >= n32) return;
    float v[32];
    float amax = 0.f;
    if (t < ncols) {
        global const float * p = (global const float *)(src + offs + (ulong)t * nbx) + (ulong)kb * 32;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 f = vload4(i, p);
            v[4 * i] = f.s0; v[4 * i + 1] = f.s1; v[4 * i + 2] = f.s2; v[4 * i + 3] = f.s3;
            amax = fmax(amax, fmax(fmax(fabs(f.s0), fabs(f.s1)), fmax(fabs(f.s2), fabs(f.s3))));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = 0.f;
    }
    // kernel_quantize_q8_1's two terms
    const float d  = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    char q[32];
    float rmax = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = convert_char_sat_rte(v[i] * id);
        rmax = fmax(rmax, fabs(v[i] - d * (float)q[i]));
    }
    const float d2  = rmax / 127.f;
    const float id2 = d2 > 0.f ? 1.f / d2 : 0.f;
    char q2[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) q2[i] = convert_char_sat_rte((v[i] - d * (float)q[i]) * id2);
    global char * o = xq + (((ulong)(t >> 3) * n32 + kb) * 16 + (t & 7)) * 32;
    #pragma unroll
    for (int i = 0; i < 32; i += 8) {
        vstore8((char8)(q[i], q[i + 1], q[i + 2], q[i + 3], q[i + 4], q[i + 5], q[i + 6], q[i + 7]), 0, o + i);
        vstore8((char8)(q2[i], q2[i + 1], q2[i + 2], q2[i + 3], q2[i + 4], q2[i + 5], q2[i + 6], q2[i + 7]), 0, o + 256 + i);
    }
    xd[(ulong)kb * TOK + t] = d;
    xd[((ulong)n32 + kb) * TOK + t] = d2;
}

// ---- the shared parts ----
#define FQ_PROLOGUE                                                                             \
    const int lane = get_sub_group_local_id(), sgk = get_sub_group_id();                        \
    const int row0 = get_group_id(0) * (R * SG);                                                \
    const int n32 = ne00 / 32;                                                                  \
    ulong rw[R];                                                                                \
    _Pragma("unroll")                                                                           \
    for (int rr = 0; rr < R; ++rr) rw[rr] = (ulong)min(row0 + rr * SG + lane, ne01 - 1);        \
    float8 acc[R][G];                                                                           \
    _Pragma("unroll")                                                                           \
    for (int rr = 0; rr < R; ++rr)                                                              \
        _Pragma("unroll")                                                                       \
        for (int g = 0; g < G; ++g) acc[rr][g] = (float8)(0.f);

#ifdef FEWQ_RED
#define FQ_KEEP(kb) (((kb) & 7) != 7)
#else
#define FQ_KEEP(kb) 1
#endif

#ifdef FEWQ_ONE_TERM
#define FQ_RES_LOAD(g, p, kb)
#define FQ_DOT(g, b) (convert_float8(DPAS8(a0_[g], b)) * x0_[g])
#else
#define FQ_RES_LOAD(g, p, kb)                                                                   \
    a1_[g] = AREAD((p) + 256);                                                                  \
    x1_[g] = vload8(0, xd + ((ulong)n32 + (kb)) * TOK + 8 * (g));
#define FQ_DOT(g, b) fma(convert_float8(DPAS8(a1_[g], b)), x1_[g], convert_float8(DPAS8(a0_[g], b)) * x0_[g])
#endif

// block KB of K for every row set: BQ is row set rr's b operand (an int8), WS
// its block scale; the activation's terms are read once for all row sets
#define FQ_STEP(KB, BQ, WS)                                                                     \
    {                                                                                           \
        const int kb_ = (KB);                                                                   \
        if (FQ_KEEP(kb_)) {                                                                     \
            ai8_t a0_[G], a1_[G];                                                               \
            float8 x0_[G], x1_[G];                                                              \
            _Pragma("unroll")                                                                   \
            for (int g = 0; g < G; ++g) {                                                       \
                global const char * p_ = xq + ((ulong)g * n32 + kb_) * 512;                     \
                a0_[g] = AREAD(p_);                                                             \
                x0_[g] = vload8(0, xd + (ulong)kb_ * TOK + 8 * g);                              \
                FQ_RES_LOAD(g, p_, kb_)                                                         \
            }                                                                                   \
            _Pragma("unroll")                                                                   \
            for (int rr = 0; rr < R; ++rr) {                                                    \
                const int8  b_  = (BQ);                                                         \
                const float ws_ = (WS);                                                         \
                _Pragma("unroll")                                                               \
                for (int g = 0; g < G; ++g) acc[rr][g] = fma(FQ_DOT(g, b_), (float8)(ws_), acc[rr][g]); \
            }                                                                                   \
        }                                                                                       \
    }

// the NK sub-groups' partial sums added in local memory; token t of row r
// to dst + offd + t * nb1, float r (mul_mm_kq_few.cl's FEW_EPILOGUE)
#define FQ_EPILOGUE                                                                             \
    local float red[NK * R * G * 8 * SG];                                                       \
    _Pragma("unroll")                                                                           \
    for (int rr = 0; rr < R; ++rr)                                                              \
        _Pragma("unroll")                                                                       \
        for (int g = 0; g < G; ++g)                                                             \
            vstore8(acc[rr][g], 0, red + (((sgk * R + rr) * G + g) * SG + lane) * 8);           \
    barrier(CLK_LOCAL_MEM_FENCE);                                                               \
    for (int u = get_local_id(0); u < R * G * 8 * SG; u += NK * SG) {                           \
        float s = 0.f;                                                                          \
        for (int j = 0; j < NK; ++j) s += red[j * R * G * 8 * SG + u];                          \
        const int m = u & 7, l = (u >> 3) % SG, rg = (u >> 3) / SG, g = rg % G, rr = rg / G;    \
        const int t = g * 8 + m, row = row0 + rr * SG + l;                                      \
        if (t < ncols && row < ne01)                                                            \
            ((global float *)(dst + offd + (ulong)t * nb1))[row] = s;                           \
    }

// component i (a constant after unrolling) of a float8 / a uint16's dword quad
#define F8(v, i) ((i) == 0 ? (v).s0 : (i) == 1 ? (v).s1 : (i) == 2 ? (v).s2 : (i) == 3 ? (v).s3 : \
                  (i) == 4 ? (v).s4 : (i) == 5 ? (v).s5 : (i) == 6 ? (v).s6 : (v).s7)
#define U4(v, i) ((i) == 0 ? (v).s0123 : (i) == 1 ? (v).s4567 : (i) == 2 ? (v).s89ab : (v).scdef)

// ---- the decodes: a 32-block's 16 code bytes (low nibbles weights 0..15,
// high nibbles 16..31) to the int8 b operand ----
// Q4_0: n - 8 bytewise, (n + 0x78) ^ 0x80 (no carry between bytes)
#define Q40_S8(t) as_int((((t) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
static inline int8 fq_q40_b(uint4 w) {
    int8 b;
    b.s0 = Q40_S8(w.s0);      b.s1 = Q40_S8(w.s1);      b.s2 = Q40_S8(w.s2);      b.s3 = Q40_S8(w.s3);
    b.s4 = Q40_S8(w.s0 >> 4); b.s5 = Q40_S8(w.s1 >> 4); b.s6 = Q40_S8(w.s2 >> 4); b.s7 = Q40_S8(w.s3 >> 4);
    return b;
}

constant char fq_kvalues[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

#ifdef FEWQ_IQ4_MUX
// 0103's id_kv4: kvalues_iq4nl of the nibble at bits b .. b + 3 of each byte, a
// select tree over the code's bit planes, no memory read
static inline uint fq_mask(uint w, int b) {
    const uint m = (w >> b) & 0x01010101u;
    return (m << 8) - m;
}
#define FQ_K(v) ((uint)(uchar)(v) * 0x01010101u)
#define FQ_SEL(x, y, m) ((x) ^ (((x) ^ (y)) & (m)))
static inline uint fq_kv4(uint w, int b) {
    const uint m0 = fq_mask(w, b), m1 = fq_mask(w, b + 1), m2 = fq_mask(w, b + 2), m3 = fq_mask(w, b + 3);
    const uint a0 = FQ_SEL(FQ_K(-127), FQ_K(-104), m0), a1 = FQ_SEL(FQ_K(-83), FQ_K(-65), m0);
    const uint a2 = FQ_SEL(FQ_K(-49), FQ_K(-35), m0), a3 = FQ_SEL(FQ_K(-22), FQ_K(-10), m0);
    const uint a4 = FQ_SEL(FQ_K(1), FQ_K(13), m0), a5 = FQ_SEL(FQ_K(25), FQ_K(38), m0);
    const uint a6 = FQ_SEL(FQ_K(53), FQ_K(69), m0), a7 = FQ_SEL(FQ_K(89), FQ_K(113), m0);
    const uint b0 = FQ_SEL(a0, a1, m1), b1 = FQ_SEL(a2, a3, m1), b2 = FQ_SEL(a4, a5, m1), b3 = FQ_SEL(a6, a7, m1);
    const uint c0 = FQ_SEL(b0, b1, m2), c1 = FQ_SEL(b2, b3, m2);
    return FQ_SEL(c0, c1, m3);
}
#define FQ_IQ4_TABLE
static inline int8 fq_iq4_b(uint4 w) {
    int8 b;
    b.s0 = as_int(fq_kv4(w.s0, 0)); b.s1 = as_int(fq_kv4(w.s1, 0)); b.s2 = as_int(fq_kv4(w.s2, 0)); b.s3 = as_int(fq_kv4(w.s3, 0));
    b.s4 = as_int(fq_kv4(w.s0, 4)); b.s5 = as_int(fq_kv4(w.s1, 4)); b.s6 = as_int(fq_kv4(w.s2, 4)); b.s7 = as_int(fq_kv4(w.s3, 4));
    return b;
}
#define FQ_IQ4_B(w) fq_iq4_b(w)
#else
// 0103's byte-pair table, in local memory (32-bit addresses; 0035): byte b ->
// kvalues_iq4nl[b & 15] in byte 0, kvalues_iq4nl[b >> 4] in byte 1. Four
// lookups give a dword's 8 weights.
#define FQ_IQ4_TABLE                                                                            \
    local uint kvp[256];                                                                        \
    for (int i_ = get_local_id(0); i_ < 256; i_ += get_local_size(0))                           \
        kvp[i_] = (uint)(uchar)fq_kvalues[i_ & 15] | ((uint)(uchar)fq_kvalues[i_ >> 4] << 8);   \
    barrier(CLK_LOCAL_MEM_FENCE);
#define FQ_PAIR(w, lo, hi) {                                                                    \
        const uint t0 = kvt[(w) & 0xFFu], t1 = kvt[((w) >> 8) & 0xFFu];                         \
        const uint t2 = kvt[((w) >> 16) & 0xFFu], t3 = kvt[(w) >> 24];                          \
        lo = as_int((t0 & 0xFFu) | ((t1 & 0xFFu) << 8) | ((t2 & 0xFFu) << 16) | (t3 << 24));    \
        hi = as_int(((t0 >> 8) & 0xFFu) | (t1 & 0xFF00u) | ((t2 & 0xFF00u) << 8) | ((t3 & 0xFF00u) << 16)); \
    }
static inline int8 fq_iq4_b(uint4 w, local const uint * kvt) {
    int8 b;
    FQ_PAIR(w.s0, b.s0, b.s4)
    FQ_PAIR(w.s1, b.s1, b.s5)
    FQ_PAIR(w.s2, b.s2, b.s6)
    FQ_PAIR(w.s3, b.s3, b.s7)
    return b;
}
#define FQ_IQ4_B(w) fq_iq4_b(w, kvp)
#endif

// ---- the kernels: (q at offq, d) are the weight's planes: Q4_0, Q8_0 and
// IQ4_NL the code and the f16 scale plane (a block's codes at row * k / 32 +
// block), IQ4_XS ggml's blocks as stored (136 bytes a 256, d unused) ----

// the flat 4-bit types: a chunk of 8 blocks is a row's 128 code bytes, two
// 64-byte reads, and 8 scales; a K that is not a multiple of 256 ends block by
// block
#define FQ_FLAT4_BODY(DECODE)                                                                   \
    q += offq;                                                                                  \
    const int nch = n32 / 8;                                                                    \
    for (int c = sgk; c < nch; c += NK) {                                                       \
        uint16 w0[R], w1[R];                                                                    \
        float8 dv[R];                                                                           \
        _Pragma("unroll")                                                                       \
        for (int rr = 0; rr < R; ++rr) {                                                        \
            const ulong b0 = rw[rr] * n32 + (ulong)c * 8;                                       \
            w0[rr] = vload16(0, (global const uint *)(q + b0 * 16));                            \
            w1[rr] = vload16(0, (global const uint *)(q + b0 * 16 + 64));                       \
            dv[rr] = vload_half8(0, d + b0);                                                    \
        }                                                                                       \
        _Pragma("unroll")                                                                       \
        for (int j = 0; j < 8; ++j) {                                                           \
            FQ_STEP(c * 8 + j, DECODE(U4(j < 4 ? w0[rr] : w1[rr], j & 3)), F8(dv[rr], j))       \
        }                                                                                       \
    }                                                                                           \
    for (int kb = nch * 8 + sgk; kb < n32; kb += NK) {                                          \
        FQ_STEP(kb, DECODE(vload4(0, (global const uint *)(q + (rw[rr] * n32 + kb) * 16))),     \
                vload_half(rw[rr] * n32 + kb, d))                                               \
    }

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_fewq_q4_0(global const uchar * q, ulong offq, global const half * d,
                                    global const char * xq, global const float * xd, global char * dst,
                                    ulong offd, ulong nb1, int ne00, int ne01, int ncols) {
    FQ_PROLOGUE
    FQ_FLAT4_BODY(fq_q40_b)
    FQ_EPILOGUE
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_fewq_iq4_nl(global const uchar * q, ulong offq, global const half * d,
                                      global const char * xq, global const float * xd, global char * dst,
                                      ulong offd, ulong nb1, int ne00, int ne01, int ncols) {
    FQ_PROLOGUE
    FQ_IQ4_TABLE
    FQ_FLAT4_BODY(FQ_IQ4_B)
    FQ_EPILOGUE
}

// Q8_0: a block's 32 codes are the b operand as they are; a pair of blocks is
// one 64-byte read
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_fewq_q8_0(global const uchar * q, ulong offq, global const half * d,
                                    global const char * xq, global const float * xd, global char * dst,
                                    ulong offd, ulong nb1, int ne00, int ne01, int ncols) {
    FQ_PROLOGUE
    q += offq;
    const int nch = n32 / 8;
    for (int c = sgk; c < nch; c += NK) {
        float8 dv[R];
        _Pragma("unroll")
        for (int rr = 0; rr < R; ++rr) dv[rr] = vload_half8(0, d + rw[rr] * n32 + (ulong)c * 8);
        _Pragma("unroll")
        for (int jp = 0; jp < 4; ++jp) {
            uint16 w[R];
            _Pragma("unroll")
            for (int rr = 0; rr < R; ++rr) w[rr] = vload16(0, (global const uint *)(q + (rw[rr] * n32 + (ulong)c * 8 + 2 * jp) * 32));
            FQ_STEP(c * 8 + 2 * jp,     as_int8(w[rr].lo), F8(dv[rr], 2 * jp))
            FQ_STEP(c * 8 + 2 * jp + 1, as_int8(w[rr].hi), F8(dv[rr], 2 * jp + 1))
        }
    }
    for (int kb = nch * 8 + sgk; kb < n32; kb += NK) {
        FQ_STEP(kb, as_int8(vload8(0, (global const uint *)(q + (rw[rr] * n32 + kb) * 32))), vload_half(rw[rr] * n32 + kb, d))
    }
    FQ_EPILOGUE
}

// IQ4_XS: a super-block (d, scales_h, scales_l[4], qs[128]) is a row's 136
// bytes, an 8-byte head and two 64-byte reads; sub-block p's 6-bit scale is
// scales_l's nibble p and scales_h's bits 2p, 2p + 1, minus 32 (dequantize_row_iq4_xs)
static inline float fq_iq4xs_ws(uint2 hd, int p) {
    const float dd = convert_float(as_half((ushort)(hd.s0 & 0xFFFFu)));
    const uint  sl = (hd.s1 >> (8 * (p >> 1))) & 0xFFu;
    const int   ls = (int)((sl >> (4 * (p & 1))) & 0xFu) | (int)(((hd.s0 >> (16 + 2 * p)) & 3u) << 4);
    return dd * (float)(ls - 32);
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_fewq_iq4_xs(global const uchar * q, ulong offq, global const half * d,
                                      global const char * xq, global const float * xd, global char * dst,
                                      ulong offd, ulong nb1, int ne00, int ne01, int ncols) {
    FQ_PROLOGUE
    FQ_IQ4_TABLE
    q += offq;
    const int nb = ne00 / QK_K;
    for (int ib = sgk; ib < nb; ib += NK) {
        uint2 hd[R];
        uint16 w0[R], w1[R];
        _Pragma("unroll")
        for (int rr = 0; rr < R; ++rr) {
            global const uchar * bp = q + (rw[rr] * nb + ib) * 136;
            hd[rr] = vload2(0, (global const uint *)bp);
            w0[rr] = vload16(0, (global const uint *)(bp + 8));
            w1[rr] = vload16(0, (global const uint *)(bp + 72));
        }
        _Pragma("unroll")
        for (int p = 0; p < 8; ++p) {
            FQ_STEP(ib * 8 + p, FQ_IQ4_B(U4(p < 4 ? w0[rr] : w1[rr], p & 3)), fq_iq4xs_ws(hd[rr], p))
        }
    }
    FQ_EPILOGUE
}

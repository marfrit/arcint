#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// K-quant weights times a few f32 columns (a speculative verify, a few
// lanes: 2 to 8 G tokens) on the XMX units, each weight read once for all of
// them. A lane is a weight row: its codes, as halves 1024 + q built from the
// code bytes by integer operations, are the fp16 DPAS b operand (lane =
// column of the product); the tokens are the a operand, 8 a DPAS, read from
// a tiled f16 copy of the activations (kernel_mul_mm_few_prep). A
// sub-block's DPAS chain starts at -1024 S, S the tokens' sums of the f16
// values, so it ends at sum q x; then xsc (D sc chain - M m S), xsc the
// activations' block scale. A sub-group takes R groups of SG rows; the NK
// sub-groups of a work-group take every NK-th super-block and add their sums
// in local memory.
//
// fp16 DPAS operands (cl_intel_subgroup_matrix_multiply_accumulate):
//   SG 8 (Xe-HPG): a = int   per row, lane i holds k = 2i, 2i+1
//   SG 16 (Xe2):   a = short per row, lane i holds k = i
//   b = int8 per lane: lane n is column n, its 16 k as 8 half pairs
//   result float8: lane n is column n, component m is row m

#define QK_K 256
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

#if SG == 8
typedef int8 a8_t;
#define LOAD_AG(p) as_int8(intel_sub_group_block_read8((global const uint *)(p)))
#else
typedef short8 a8_t;
#define LOAD_AG(p) as_short8(intel_sub_group_block_read_us8((global const ushort *)(p)))
#endif
#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(a, b, c)

static inline uint kbyte(uint3 w, int k) {
    const uint v = k < 4 ? w.s0 : (k < 8 ? w.s1 : w.s2);
    return (v >> (8 * (k & 3))) & 0xFFu;
}

// sub-block k's scale and min from a K-quant block's 12 scale bytes
static inline void scale_min(uint3 w, int k, float * sc, float * m) {
    if (k < 4) {
        *sc = kbyte(w, k) & 63; *m = kbyte(w, k + 4) & 63;
    } else {
        *sc = (kbyte(w, k + 4) & 0xF) | ((kbyte(w, k - 4) >> 6) << 4);
        *m  = (kbyte(w, k + 4) >> 4) | ((kbyte(w, k) >> 6) << 4);
    }
}

// 16 Q4_K / Q5_K codes as halves 1024 + q: byte pairs (0, 2) and (1, 3) of
// each word (the activations are staged with the middle two of every four
// halves swapped to match); nibble sh, fifth bit hb of the qh bytes
static inline int8 b16_q45_codes(uint4 w, uint4 h, int sh, int hb, const int q5) {
    const uint wv[4] = { w.s0, w.s1, w.s2, w.s3 }, hv[4] = { h.s0, h.s1, h.s2, h.s3 };
    uint r[8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        uint lo = ((wv[u] >> sh) & 0x000F000Fu) | 0x64006400u;
        uint hi = ((wv[u] >> (8 + sh)) & 0x000F000Fu) | 0x64006400u;
        if (q5) {
            lo |= ((hv[u] >> hb) & 0x00010001u) << 4;
            hi |= ((hv[u] >> (8 + hb)) & 0x00010001u) << 4;
        }
        r[2 * u] = lo;
        r[2 * u + 1] = hi;
    }
    return as_int8((uint8)(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]));
}

// 16 Q6_K codes as halves 1024 + q (q the 6-bit code, 32 above the weight)
static inline int8 b16_q6_codes(uint4 w, uint4 h, int sh, int hb) {
    const uint wv[4] = { w.s0, w.s1, w.s2, w.s3 }, hv[4] = { h.s0, h.s1, h.s2, h.s3 };
    uint r[8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        r[2 * u]     = ((wv[u] >> sh) & 0x000F000Fu) | (((hv[u] >> hb) & 0x00030003u) << 4) | 0x64006400u;
        r[2 * u + 1] = ((wv[u] >> (8 + sh)) & 0x000F000Fu) | (((hv[u] >> (8 + hb)) & 0x00030003u) << 4) | 0x64006400u;
    }
    return as_int8((uint8)(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]));
}

// a half's value from its bits, by integer operations: the sums S must be
// of exactly the halves the DPAS multiplies (the chain's 1024 offset
// multiplies any difference by a thousand), and with ggml's
// -cl-unsafe-math-optimizations -cl-finite-math-only the compiler sums the
// unrounded floats for (float) h (4-9 % off, measured-here)
static inline float half_bits_to_float(ushort h) {
    const uint  e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const float v = e == 0 ? (float)m * 0x1p-24f : as_float(((e + 112u) << 23) | (m << 13));
    return (h & 0x8000) ? -v : v;
}

// ---- the activations ----
// One work-item per (token, 32-block), the token fastest. x: f32 columns of
// k values at src + offs + t * nbx. xa: half [G][k/16][8][16] (the middle
// two of every four swapped), xs: f32 [k/16][TOK] the sums of the f16
// values, xsc: f32 [k/32][TOK] the block's scale, a power of two keeping it
// below 2^15 (1 for nearly every block). Columns from ncols to TOK are
// zero.
kernel void kernel_mul_mm_few_prep(global const char * src, ulong offs, ulong nbx, global half * xa,
                                   global float * xs, global float * xsc, int k, int ncols) {
    const int t = get_global_id(0), kb = get_global_id(1);
    if (t >= TOK || kb >= k / 32) return;
    float v[32];
    float amax = 0.f;
    if (t < ncols) {
        global const float4 * p = (global const float4 *)(src + offs + (ulong)t * nbx) + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 f = p[i];
            v[4 * i] = f.s0; v[4 * i + 1] = f.s1; v[4 * i + 2] = f.s2; v[4 * i + 3] = f.s3;
            amax = fmax(amax, fmax(fmax(fabs(f.s0), fabs(f.s1)), fmax(fabs(f.s2), fabs(f.s3))));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = 0.f;
    }
    // 2^e from the exponent bits, so that the block stays below 2^15
    const int   e   = max(0, ((as_int(amax) >> 23) & 0xFF) - 127 - 14);
    const float sc  = as_float((127 + e) << 23), isc = as_float((127 - e) << 23);
    #pragma unroll
    for (int hb = 0; hb < 2; ++hb) {
        const int k16 = 2 * kb + hb;
        half h[16];
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            // subnormal halves to zero, so S and the DPAS agree whether or
            // not the XMX units flush them
            ushort u = as_ushort(convert_half_rte(v[16 * hb + i] * isc));
            u = ((u >> 10) & 0x1F) == 0 ? (ushort)0 : u;
            h[i] = as_half(u);
            s += half_bits_to_float(u);
        }
        global half * dst = xa + (((ulong)(t >> 3) * (k / 16) + k16) * 8 + (t & 7)) * 16;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            dst[i] = h[i]; dst[i + 1] = h[i + 2]; dst[i + 2] = h[i + 1]; dst[i + 3] = h[i + 3];
        }
        xs[(ulong)k16 * TOK + t] = s;
    }
    xsc[(ulong)kb * TOK + t] = sc;
}

#define FEW_PROLOGUE                                                                            \
    const int lane = get_sub_group_local_id(), sgk = get_sub_group_id();                        \
    const int row0 = get_group_id(0) * (R * SG);                                                \
    const int nb = ne00 / QK_K, k16 = ne00 / 16;                                                \
    ulong rb[R];                                                                                \
    _Pragma("unroll")                                                                           \
    for (int rr = 0; rr < R; ++rr) rb[rr] = (ulong)min(row0 + rr * SG + lane, ne01 - 1) * nb;   \
    float8 acc[R][G];                                                                           \
    _Pragma("unroll")                                                                           \
    for (int rr = 0; rr < R; ++rr)                                                              \
        _Pragma("unroll")                                                                       \
        for (int g = 0; g < G; ++g) acc[rr][g] = (float8)(0.f);

// the NK sub-groups' partial sums added in local memory; token t of row r
// to dst + offd + t * nb1, float r
#define FEW_EPILOGUE                                                                            \
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

// Q4_K (Q5 = 0) and Q5_K (Q5 = 1): per super-block a lane's 128 code bytes
// as two 64-byte lines; sub-blocks 2 c4 and 2 c4 + 1 are the low and high
// nibbles of bytes 32 c4 .. 32 c4 + 31
#define FEW_Q45_BODY(QH, Q5)                                                                    \
    FEW_PROLOGUE                                                                                \
    for (int ib = sgk; ib < nb; ib += NK) {                                                     \
        uint3 sw[R];                                                                            \
        float D[R], M[R];                                                                       \
        uint16 w0[R], w1[R];                                                                    \
        uint8 hq[R];                                                                            \
        _Pragma("unroll")                                                                       \
        for (int rr = 0; rr < R; ++rr) {                                                        \
            const ulong b = rb[rr] + ib;                                                        \
            sw[rr] = vload3(0, (global const uint *)(s + b * 12));                              \
            D[rr]  = vload_half(0, d + b);                                                      \
            M[rr]  = vload_half(0, dm + b);                                                     \
            w0[rr] = vload16(0, (global const uint *)(q + b * 128));                            \
            w1[rr] = vload16(0, (global const uint *)(q + b * 128 + 64));                       \
            hq[rr] = Q5 ? vload8(0, (global const uint *)(QH + b * 32)) : (uint8)(0);           \
        }                                                                                       \
        _Pragma("unroll")                                                                       \
        for (int c4 = 0; c4 < 4; ++c4) {                                                        \
            float8 sl[G], sh[G], cl[G], ch[G];                                                  \
            _Pragma("unroll")                                                                   \
            for (int g = 0; g < G; ++g) {                                                       \
                const int kb = ib * 16 + 4 * c4;                                                \
                sl[g] = vload8(0, xs + (ulong)kb * TOK + 8 * g) + vload8(0, xs + (ulong)(kb + 1) * TOK + 8 * g); \
                sh[g] = vload8(0, xs + (ulong)(kb + 2) * TOK + 8 * g) + vload8(0, xs + (ulong)(kb + 3) * TOK + 8 * g); \
                cl[g] = vload8(0, xsc + (ulong)(kb / 2) * TOK + 8 * g);                         \
                ch[g] = vload8(0, xsc + (ulong)(kb / 2 + 1) * TOK + 8 * g);                     \
            }                                                                                   \
            float8 tl[R][G], th[R][G];                                                          \
            _Pragma("unroll")                                                                   \
            for (int rr = 0; rr < R; ++rr)                                                      \
                _Pragma("unroll")                                                               \
                for (int g = 0; g < G; ++g) { tl[rr][g] = -1024.f * sl[g]; th[rr][g] = -1024.f * sh[g]; } \
            _Pragma("unroll")                                                                   \
            for (int hh = 0; hh < 2; ++hh) {                                                    \
                a8_t alo[G], ahi[G];                                                            \
                _Pragma("unroll")                                                               \
                for (int g = 0; g < G; ++g) {                                                   \
                    alo[g] = LOAD_AG(xa + (((ulong)g * k16 + ib * 16 + 4 * c4 + hh) * 8) * 16); \
                    ahi[g] = LOAD_AG(xa + (((ulong)g * k16 + ib * 16 + 4 * c4 + 2 + hh) * 8) * 16); \
                }                                                                               \
                _Pragma("unroll")                                                               \
                for (int rr = 0; rr < R; ++rr) {                                                \
                    const uint16 wl = c4 < 2 ? w0[rr] : w1[rr];                                 \
                    const int o = 2 * (c4 & 1) + hh;                                            \
                    const uint4 w = o == 0 ? wl.s0123 : o == 1 ? wl.s4567 : o == 2 ? wl.s89ab : wl.scdef; \
                    const uint4 h = hh == 0 ? hq[rr].s0123 : hq[rr].s4567;                      \
                    const int8 bl = b16_q45_codes(w, h, 0, 2 * c4, Q5);                         \
                    const int8 bh = b16_q45_codes(w, h, 4, 2 * c4 + 1, Q5);                     \
                    _Pragma("unroll")                                                           \
                    for (int g = 0; g < G; ++g) {                                               \
                        tl[rr][g] = DPAS(alo[g], bl, tl[rr][g]);                                \
                        th[rr][g] = DPAS(ahi[g], bh, th[rr][g]);                                \
                    }                                                                           \
                }                                                                               \
            }                                                                                   \
            _Pragma("unroll")                                                                   \
            for (int rr = 0; rr < R; ++rr) {                                                    \
                float sc0, m0, sc1, m1;                                                         \
                scale_min(sw[rr], 2 * c4, &sc0, &m0);                                           \
                scale_min(sw[rr], 2 * c4 + 1, &sc1, &m1);                                       \
                const float ka0 = D[rr] * sc0, kb0 = M[rr] * m0;                                \
                const float ka1 = D[rr] * sc1, kb1 = M[rr] * m1;                                \
                _Pragma("unroll")                                                               \
                for (int g = 0; g < G; ++g) {                                                   \
                    acc[rr][g] = fma(fma(tl[rr][g], (float8)(ka0), sl[g] * -kb0), cl[g], acc[rr][g]); \
                    acc[rr][g] = fma(fma(th[rr][g], (float8)(ka1), sh[g] * -kb1), ch[g], acc[rr][g]); \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    FEW_EPILOGUE

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_few_q4_K_f16(global const uchar * q, global const uchar * s, global const half * d,
                                       global const half * dm, global const half * xa, global const float * xs,
                                       global const float * xsc, global char * dst, ulong offd, ulong nb1,
                                       int ne00, int ne01, int ncols) {
    FEW_Q45_BODY(q, 0)
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_few_q5_K_f16(global const uchar * q, global const uchar * qh, global const uchar * s,
                                       global const half * d, global const half * dm, global const half * xa,
                                       global const float * xs, global const float * xsc, global char * dst,
                                       ulong offd, ulong nb1, int ne00, int ne01, int ncols) {
    FEW_Q45_BODY(qh, 1)
}

// Q6_K: k16 step 2 (4 n + p) + hh of a super-block holds weights 128 n +
// 32 p + 16 hh + i, from ql bytes 64 n + 32 (p & 1) + 16 hh (nibble p / 2)
// and qh bytes 32 n + 16 hh (bits 2 p), at the int8 scale 8 n + 2 p + hh
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_few_q6_K_f16(global const uchar * ql, global const uchar * qh, global const char * s,
                                       global const half * d, global const half * xa, global const float * xs,
                                       global const float * xsc, global char * dst, ulong offd, ulong nb1,
                                       int ne00, int ne01, int ncols) {
    FEW_PROLOGUE
    for (int ib = sgk; ib < nb; ib += NK) {
        float D[R];
        uint4 sc4[R];
        _Pragma("unroll")
        for (int rr = 0; rr < R; ++rr) {
            D[rr]   = vload_half(0, d + rb[rr] + ib);
            sc4[rr] = vload4(0, (global const uint *)(s + (rb[rr] + ib) * 16));
        }
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            #pragma unroll
            for (int pl = 0; pl < 2; ++pl) {
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    // p = pl (nibble 0) and p = pl + 2 (nibble 1): the same bytes
                    const int s0 = 2 * (4 * n + pl) + hh, s1 = 2 * (4 * n + pl + 2) + hh;   // k16 steps
                    a8_t a0[G], a1[G];
                    float8 S0[G], S1[G], c0[G], c1[G];
                    _Pragma("unroll")
                    for (int g = 0; g < G; ++g) {
                        a0[g] = LOAD_AG(xa + (((ulong)g * k16 + ib * 16 + s0) * 8) * 16);
                        a1[g] = LOAD_AG(xa + (((ulong)g * k16 + ib * 16 + s1) * 8) * 16);
                        S0[g] = vload8(0, xs + (ulong)(ib * 16 + s0) * TOK + 8 * g);
                        S1[g] = vload8(0, xs + (ulong)(ib * 16 + s1) * TOK + 8 * g);
                        c0[g] = vload8(0, xsc + (ulong)(ib * 8 + s0 / 2) * TOK + 8 * g);
                        c1[g] = vload8(0, xsc + (ulong)(ib * 8 + s1 / 2) * TOK + 8 * g);
                    }
                    _Pragma("unroll")
                    for (int rr = 0; rr < R; ++rr) {
                        const ulong b = rb[rr] + ib;
                        const uint4 w = vload4(0, (global const uint *)(ql + b * 128 + 64 * n + 32 * pl + 16 * hh));
                        const uint4 h = vload4(0, (global const uint *)(qh + b * 64 + 32 * n + 16 * hh));
                        const int8 b0 = b16_q6_codes(w, h, 0, 2 * pl);
                        const int8 b1 = b16_q6_codes(w, h, 4, 2 * (pl + 2));
                        const uint sv = (s0 >> 2) == 0 ? sc4[rr].s0 : (s0 >> 2) == 1 ? sc4[rr].s1 : (s0 >> 2) == 2 ? sc4[rr].s2 : sc4[rr].s3;
                        const uint sv1 = (s1 >> 2) == 0 ? sc4[rr].s0 : (s1 >> 2) == 1 ? sc4[rr].s1 : (s1 >> 2) == 2 ? sc4[rr].s2 : sc4[rr].s3;
                        const float ka0 = D[rr] * (float)(char)((sv >> (8 * (s0 & 3))) & 0xFF);
                        const float ka1 = D[rr] * (float)(char)((sv1 >> (8 * (s1 & 3))) & 0xFF);
                        _Pragma("unroll")
                        for (int g = 0; g < G; ++g) {
                            const float8 t0 = DPAS(a0[g], b0, -1056.f * S0[g]);
                            const float8 t1 = DPAS(a1[g], b1, -1056.f * S1[g]);
                            acc[rr][g] = fma(t0 * ka0, c0[g], acc[rr][g]);
                            acc[rr][g] = fma(t1 * ka1, c1[g], acc[rr][g]);
                        }
                    }
                }
            }
        }
    }
    FEW_EPILOGUE
}

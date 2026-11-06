#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

// K-quant x f16 matrix-matrix products on the XMX units (fp16 DPAS): a
// work-group dequantizes a tile of weight rows, one 256-weight super-block at
// a time, to f16 in local memory, laid out so that one sub-group block read
// is one DPAS a operand; the activations are f16 columns (converted from f32
// by kernel_cvt_f32_f16, with a power-of-two scale per column). The accumulation is f32 in the DPAS, with no
// per-sub-block scaling in the inner loop: the scales are applied once, in
// the dequantization -- the cuBLAS-style dequantize-then-GEMM of llama.cpp's
// CUDA backend, tiled through local memory.
//
// fp16 DPAS operands (cl_intel_subgroup_matrix_multiply_accumulate; the
// emulation in bashbaug/SimpleOpenCLSamples 20_matrixexperiments-bf16):
//   SG 8 (Alchemist): a = int   per row, lane i holds k = 2i, 2i+1
//   SG 16 (Xe2):      a = short per row, lane i holds k = i
//   b = int8 per lane: lane n is column n, its 16 k as 8 half pairs
//   result float8: lane n is column n, component m is row m
// Local layout: [row group][k16 block][row m][k 0..15] halves, so the a
// operand of (row group, k16 block) is 128 consecutive halves.
//
// A work-group is NSGM x NSGN sub-groups; each takes RG*8 rows x TG*SG
// columns; the work-group tile is MT = NSGM*RG*8 rows x NT = NSGN*TG*SG.

#define QK_K 256
#ifndef SG
#define SG 16
#endif
#ifndef RG
#define RG 4
#endif
#ifndef TG
#define TG 2
#endif
#ifndef NSGM
#define NSGM 2
#endif
#ifndef NSGN
#define NSGN 2
#endif
#define MT (NSGM * RG * 8)
#define NT (NSGN * TG * SG)
#define NTHR (NSGM * NSGN * SG)

#if SG == 8
typedef int8 a8_t;
#define LOAD_A(p) as_int8(intel_sub_group_block_read8((local const uint *)(p)))
#else
typedef short8 a8_t;
#define LOAD_A(p) as_short8(intel_sub_group_block_read_us8((local const ushort *)(p)))
#endif
#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(a, b, c)

// ggml-common.h tables (byte-exact copies)
constant uint iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

constant ulong ksigns64[128] = {
    0x0000000000000000, 0xff000000000000ff, 0xff0000000000ff00, 0x000000000000ffff, 0xff00000000ff0000, 0x0000000000ff00ff, 0x0000000000ffff00, 0xff00000000ffffff,
    0xff000000ff000000, 0x00000000ff0000ff, 0x00000000ff00ff00, 0xff000000ff00ffff, 0x00000000ffff0000, 0xff000000ffff00ff, 0xff000000ffffff00, 0x00000000ffffffff,
    0xff0000ff00000000, 0x000000ff000000ff, 0x000000ff0000ff00, 0xff0000ff0000ffff, 0x000000ff00ff0000, 0xff0000ff00ff00ff, 0xff0000ff00ffff00, 0x000000ff00ffffff,
    0x000000ffff000000, 0xff0000ffff0000ff, 0xff0000ffff00ff00, 0x000000ffff00ffff, 0xff0000ffffff0000, 0x000000ffffff00ff, 0x000000ffffffff00, 0xff0000ffffffffff,
    0xff00ff0000000000, 0x0000ff00000000ff, 0x0000ff000000ff00, 0xff00ff000000ffff, 0x0000ff0000ff0000, 0xff00ff0000ff00ff, 0xff00ff0000ffff00, 0x0000ff0000ffffff,
    0x0000ff00ff000000, 0xff00ff00ff0000ff, 0xff00ff00ff00ff00, 0x0000ff00ff00ffff, 0xff00ff00ffff0000, 0x0000ff00ffff00ff, 0x0000ff00ffffff00, 0xff00ff00ffffffff,
    0x0000ffff00000000, 0xff00ffff000000ff, 0xff00ffff0000ff00, 0x0000ffff0000ffff, 0xff00ffff00ff0000, 0x0000ffff00ff00ff, 0x0000ffff00ffff00, 0xff00ffff00ffffff,
    0xff00ffffff000000, 0x0000ffffff0000ff, 0x0000ffffff00ff00, 0xff00ffffff00ffff, 0x0000ffffffff0000, 0xff00ffffffff00ff, 0xff00ffffffffff00, 0x0000ffffffffffff,
    0xffff000000000000, 0x00ff0000000000ff, 0x00ff00000000ff00, 0xffff00000000ffff, 0x00ff000000ff0000, 0xffff000000ff00ff, 0xffff000000ffff00, 0x00ff000000ffffff,
    0x00ff0000ff000000, 0xffff0000ff0000ff, 0xffff0000ff00ff00, 0x00ff0000ff00ffff, 0xffff0000ffff0000, 0x00ff0000ffff00ff, 0x00ff0000ffffff00, 0xffff0000ffffffff,
    0x00ff00ff00000000, 0xffff00ff000000ff, 0xffff00ff0000ff00, 0x00ff00ff0000ffff, 0xffff00ff00ff0000, 0x00ff00ff00ff00ff, 0x00ff00ff00ffff00, 0xffff00ff00ffffff,
    0xffff00ffff000000, 0x00ff00ffff0000ff, 0x00ff00ffff00ff00, 0xffff00ffff00ffff, 0x00ff00ffffff0000, 0xffff00ffffff00ff, 0xffff00ffffffff00, 0x00ff00ffffffffff,
    0x00ffff0000000000, 0xffffff00000000ff, 0xffffff000000ff00, 0x00ffff000000ffff, 0xffffff0000ff0000, 0x00ffff0000ff00ff, 0x00ffff0000ffff00, 0xffffff0000ffffff,
    0xffffff00ff000000, 0x00ffff00ff0000ff, 0x00ffff00ff00ff00, 0xffffff00ff00ffff, 0x00ffff00ffff0000, 0xffffff00ffff00ff, 0xffffff00ffffff00, 0x00ffff00ffffffff,
    0xffffffff00000000, 0x00ffffff000000ff, 0x00ffffff0000ff00, 0xffffffff0000ffff, 0x00ffffff00ff0000, 0xffffffff00ff00ff, 0xffffffff00ffff00, 0x00ffffff00ffffff,
    0x00ffffffff000000, 0xffffffffff0000ff, 0xffffffffff00ff00, 0x00ffffffff00ffff, 0xffffffffffff0000, 0x00ffffffffff00ff, 0x00ffffffffffff00, 0xffffffffffffffff,
};

constant char kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// f32 -> f16 by columns of k values, one work-group per column. A column
// whose largest magnitude exceeds 16384 is stored divided by a power of two
// (exact) that brings it under, and the GEMM multiplies its outputs back:
// f16 overflows at 65504 (ggml's MUL_MAT_ID cells use activations of 1e5).
kernel void kernel_cvt_f32_f16(global const char * src, ulong offs, global half * dst, global float * scale, int k) {
#if SG == 8
    const int col = get_group_id(0);
    global const float * x = (global const float *)(src + offs) + (ulong)col * k;
    global half * y = dst + (ulong)col * k;
    const int lid = get_local_id(0), lsz = get_local_size(0);
    /* float4 only where the 16-byte load and 8-byte store are aligned */
    const bool vec = ((offs & 15ul) == 0ul) && ((k & 3) == 0);
    float amax = 0.f;
    if (vec) {
        for (int i = lid * 4; i < k; i += lsz * 4) {
            const float4 v = vload4(0, x + i);
            amax = fmax(amax, fmax(fmax(fabs(v.s0), fabs(v.s1)), fmax(fabs(v.s2), fabs(v.s3))));
        }
    } else {
        for (int i = lid; i < k; i += lsz) amax = fmax(amax, fabs(x[i]));
    }
    amax = work_group_reduce_max(amax);
    const float sc = amax > 16384.f ? ldexp(1.f, (int)ceil(log2(amax / 16384.f))) : 1.f;
    const float isc = 1.f / sc;
    if (vec) {
        for (int i = lid * 4; i < k; i += lsz * 4) {
            const float4 v = vload4(0, x + i) * isc;
            vstore4(convert_half4(v), 0, y + i);
        }
    } else {
        for (int i = lid; i < k; i += lsz) vstore_half_rte(x[i] * isc, 0, y + i);
    }
    if (lid == 0) scale[col] = sc;
#else
    const int col = get_group_id(0);
    global const float * x = (global const float *)(src + offs) + (ulong)col * k;
    global half * y = dst + (ulong)col * k;
    float amax = 0.f;
    for (int i = get_local_id(0); i < k; i += get_local_size(0)) amax = fmax(amax, fabs(x[i]));
    amax = work_group_reduce_max(amax);
    const float sc = amax > 16384.f ? ldexp(1.f, (int)ceil(log2(amax / 16384.f))) : 1.f;
    const float isc = 1.f / sc;
    for (int i = get_local_id(0); i < k; i += get_local_size(0)) vstore_half_rte(x[i] * isc, 0, y + i);
    if (get_local_id(0) == 0) scale[col] = sc;
#endif
}

static inline uint kbyte(uint3 w, int k) {
    const uint v = k < 4 ? w.s0 : (k < 8 ? w.s1 : w.s2);
    return (v >> (8 * (k & 3))) & 0xFFu;
}

static inline void scale_min(uint3 w, int k, float * sc, float * m) {
    if (k < 4) {
        *sc = kbyte(w, k) & 63; *m = kbyte(w, k + 4) & 63;
    } else {
        *sc = (kbyte(w, k + 4) & 0xF) | ((kbyte(w, k - 4) >> 6) << 4);
        *m  = (kbyte(w, k + 4) >> 4) | ((kbyte(w, k) >> 6) << 4);
    }
}

// weight (row r of the tile, k) in the local tile
#define LOFF(r, k) (((((r) >> 3) * 16 + ((k) >> 4)) * 8 + ((r) & 7)) * 16 + ((k) & 15))
// weight (row r of the tile, chunk-local k in 0..127) in a half-super-block
// buffer: [row group][k16 0..7][row m][k 0..15]
#define LOFFC(r, k) (((((r) >> 3) * 8 + ((k) >> 4)) * 8 + ((r) & 7)) * 16 + ((k) & 15))

// 16 consecutive weights k0 .. k0+15 (k0 % 16 == 0) of tile row r
static inline void put16(local half * wt, int r, int k0, float16 v) {
#if SG == 8
    vstore_half16(v, 0, wt + LOFFC(r, k0));
#else
    vstore_half16(v, 0, wt + LOFF(r, k0));
#endif
}

#if SG == 8
static inline float16 nib16(uint4 w, int hi, half a, half nb, uint4 h, int hbit) {
    // 16 4-bit codes (low or high nibbles of 16 bytes), plus an optional 5th
    // bit from h (bit hbit of each byte; hbit < 0: none): a * q - b as one
    // f32 fma, the coefficients a and nb = -b rounded to half first
    const uint4 q = hi ? (w >> 4) & 0x0F0F0F0Fu : w & 0x0F0F0F0Fu;
    uint4 q5 = q;
    if (hbit >= 0) q5 |= ((h >> hbit) & 0x01010101u) << 4;
    const uchar16 c = as_uchar16(q5);
    return fma(convert_float16(c), (float16)(a), (float16)(nb));
}
#else
static inline float16 nib16(uint4 w, int hi, float a, float b, uint4 h, int hbit) {
    const uint4 q = hi ? (w >> 4) & 0x0F0F0F0Fu : w & 0x0F0F0F0Fu;
    uint4 q5 = q;
    if (hbit >= 0) q5 |= ((h >> hbit) & 0x01010101u) << 4;
    const uchar16 c = as_uchar16(q5);
    return a * convert_float16(c) - b;
}
#endif

// ---- dequantization of the tile's super-block ib into local memory ----

#if SG == 8
// Q4_K / Q5_K: unit (row r, pair j of the chunk) = 64 weights, sub-blocks 2j
// and 2j+1; the chunk holds sub-blocks 4*ch..4*ch+3
static inline void deq_q45_K(local half * wt, global const uchar * q, global const uchar * qh, global const uchar * s,
                             global const half * d, global const half * dm, const int q5,
                             ulong wrow0, int row0, int ne01, int nb, int ib) {
#if MT <= 32
    const int sb = ib >> 1, ch = ib & 1;
    for (int rr = get_local_linear_id(); rr < MT; rr += NTHR) {
        const int r = min(row0 + rr, ne01 - 1);
        const ulong b = (wrow0 + r) * nb + sb;
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));
        float sc[4], mm[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) scale_min(sw, 4 * ch + t, &sc[t], &mm[t]);
        const float D = vload_half(0, d + b), M = vload_half(0, dm + b);
        half ha[4], nbv[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) { ha[t] = convert_half(D * sc[t]); nbv[t] = convert_half(-M * mm[t]); }
        uint4 h0 = (uint4)(0), h1 = (uint4)(0);
        if (q5) {
            const uint8 hw = vload8(0, (global const uint *)(qh + b * 32));
            h0 = (uint4)(hw.s0, hw.s1, hw.s2, hw.s3);
            h1 = (uint4)(hw.s4, hw.s5, hw.s6, hw.s7);
        }
        #pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int j = 2 * ch + jj;
            const uint8 ww = vload8(0, (global const uint *)(q + b * 128 + 32 * j));
            const uint4 w0 = (uint4)(ww.s0, ww.s1, ww.s2, ww.s3);
            const uint4 w1 = (uint4)(ww.s4, ww.s5, ww.s6, ww.s7);
            const int k0 = 64 * jj;
            put16(wt, rr, k0,      nib16(w0, 0, ha[2 * jj],     nbv[2 * jj],     h0, q5 ? 4 * ch + 2 * jj : -1));
            put16(wt, rr, k0 + 16, nib16(w1, 0, ha[2 * jj],     nbv[2 * jj],     h1, q5 ? 4 * ch + 2 * jj : -1));
            put16(wt, rr, k0 + 32, nib16(w0, 1, ha[2 * jj + 1], nbv[2 * jj + 1], h0, q5 ? 4 * ch + 2 * jj + 1 : -1));
            put16(wt, rr, k0 + 48, nib16(w1, 1, ha[2 * jj + 1], nbv[2 * jj + 1], h1, q5 ? 4 * ch + 2 * jj + 1 : -1));
        }
    }
#else
    for (int u = get_local_linear_id(); u < MT * 2; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 1, jj = u & 1, j = 2 * ch + jj;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong b = (wrow0 + r) * nb + sb;
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));
        float sc0, m0, sc1, m1;
        scale_min(sw, 2 * j, &sc0, &m0);
        scale_min(sw, 2 * j + 1, &sc1, &m1);
        const float D = vload_half(0, d + b), M = vload_half(0, dm + b);
        const float a0 = D * sc0, b0 = M * m0, a1 = D * sc1, b1 = M * m1;
        const half ha0 = convert_half(a0), nb0 = convert_half(-b0);
        const half ha1 = convert_half(a1), nb1 = convert_half(-b1);
        const uint8 ww = vload8(0, (global const uint *)(q + b * 128 + 32 * j));
        const uint4 w0 = (uint4)(ww.s0, ww.s1, ww.s2, ww.s3);
        const uint4 w1 = (uint4)(ww.s4, ww.s5, ww.s6, ww.s7);
        uint4 h0 = (uint4)(0), h1 = (uint4)(0);
        if (q5) {
            const uint8 hw = vload8(0, (global const uint *)(qh + b * 32));
            h0 = (uint4)(hw.s0, hw.s1, hw.s2, hw.s3);
            h1 = (uint4)(hw.s4, hw.s5, hw.s6, hw.s7);
        }
        const int k0 = 64 * jj;
        put16(wt, rr, k0,      nib16(w0, 0, ha0, nb0, h0, q5 ? 2 * j : -1));
        put16(wt, rr, k0 + 16, nib16(w1, 0, ha0, nb0, h1, q5 ? 2 * j : -1));
        put16(wt, rr, k0 + 32, nib16(w0, 1, ha1, nb1, h0, q5 ? 2 * j + 1 : -1));
        put16(wt, rr, k0 + 48, nib16(w1, 1, ha1, nb1, h1, q5 ? 2 * j + 1 : -1));
    }
#endif
}
#else
// Q4_K / Q5_K: unit (row r, pair j) = 64 weights, sub-blocks 2j and 2j+1
static inline void deq_q45_K(local half * wt, global const uchar * q, global const uchar * qh, global const uchar * s,
                             global const half * d, global const half * dm, const int q5,
                             ulong wrow0, int row0, int ne01, int nb, int ib) {
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int rr = u >> 2, j = u & 3;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong b = (wrow0 + r) * nb + ib;
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));
        float sc0, m0, sc1, m1;
        scale_min(sw, 2 * j, &sc0, &m0);
        scale_min(sw, 2 * j + 1, &sc1, &m1);
        const float D = vload_half(0, d + b), M = vload_half(0, dm + b);
        const float a0 = D * sc0, b0 = M * m0, a1 = D * sc1, b1 = M * m1;
        const uint4 w0 = vload4(0, (global const uint *)(q + b * 128 + 32 * j));
        const uint4 w1 = vload4(0, (global const uint *)(q + b * 128 + 32 * j + 16));
        uint4 h0 = (uint4)(0), h1 = (uint4)(0);
        if (q5) {
            h0 = vload4(0, (global const uint *)(qh + b * 32));
            h1 = vload4(0, (global const uint *)(qh + b * 32 + 16));
        }
        put16(wt, rr, 64 * j,      nib16(w0, 0, a0, b0, h0, q5 ? 2 * j : -1));
        put16(wt, rr, 64 * j + 16, nib16(w1, 0, a0, b0, h1, q5 ? 2 * j : -1));
        put16(wt, rr, 64 * j + 32, nib16(w0, 1, a1, b1, h0, q5 ? 2 * j + 1 : -1));
        put16(wt, rr, 64 * j + 48, nib16(w1, 1, a1, b1, h1, q5 ? 2 * j + 1 : -1));
    }
}
#endif

// Q6_K: unit (row r, 32-block kb = part p of half n) = 32 weights; the chunk
// holds blocks 4*ch..4*ch+3
#if SG == 8
static inline void deq_q6_K(local half * wt, global const uchar * ql, global const uchar * qh, global const char * s,
                            global const half * d, ulong wrow0, int row0, int ne01, int nb, int ib) {
#if MT <= 32
    const int sb = ib >> 1, ch = ib & 1;
    for (int rr = get_local_linear_id(); rr < MT; rr += NTHR) {
        const int r = min(row0 + rr, ne01 - 1);
        const ulong b = (wrow0 + r) * nb + sb;
        const float D = vload_half(0, d + b);
        #pragma unroll
        for (int uu = 0; uu < 4; ++uu) {
            const int kb = 4 * ch + uu;
            const int n = kb >> 2, p = kb & 3;
            const int is = 8 * n + 2 * p;
            const float a0 = D * (float)s[b * 16 + is], a1 = D * (float)s[b * 16 + is + 1];
            const half ha0 = convert_half(a0), off0 = convert_half(-32.f * a0);
            const half ha1 = convert_half(a1), off1 = convert_half(-32.f * a1);
            for (int hh = 0; hh < 2; ++hh) {
                const uint4 w = vload4(0, (global const uint *)(ql + b * 128 + 64 * n + 32 * (p & 1) + 16 * hh));
                const uint4 h = vload4(0, (global const uint *)(qh + b * 64 + 32 * n + 16 * hh));
                const uint4 c = ((w >> (4 * (p >> 1))) & 0x0F0F0F0Fu) | (((h >> (2 * p)) & 0x03030303u) << 4);
                const float16 v = fma(convert_float16(as_uchar16(c)),
                                      (float16)(hh ? ha1 : ha0), (float16)(hh ? off1 : off0));
                put16(wt, rr, 32 * uu + 16 * hh, v);
            }
        }
    }
#else
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 2, kb = 4 * ch + (u & 3);
        const int n = kb >> 2, p = kb & 3;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong b = (wrow0 + r) * nb + sb;
        const float D = vload_half(0, d + b);
        const int is = 8 * n + 2 * p;
        const float a0 = D * (float)s[b * 16 + is], a1 = D * (float)s[b * 16 + is + 1];
        const half ha0 = convert_half(a0), off0 = convert_half(-32.f * a0);
        const half ha1 = convert_half(a1), off1 = convert_half(-32.f * a1);
        for (int hh = 0; hh < 2; ++hh) {
            const uint4 w = vload4(0, (global const uint *)(ql + b * 128 + 64 * n + 32 * (p & 1) + 16 * hh));
            const uint4 h = vload4(0, (global const uint *)(qh + b * 64 + 32 * n + 16 * hh));
            const uint4 c = ((w >> (4 * (p >> 1))) & 0x0F0F0F0Fu) | (((h >> (2 * p)) & 0x03030303u) << 4);
            const float16 v = fma(convert_float16(as_uchar16(c)),
                                  (float16)(hh ? ha1 : ha0), (float16)(hh ? off1 : off0));
            put16(wt, rr, 32 * (u & 3) + 16 * hh, v);
        }
    }
#endif
}
#else
static inline void deq_q6_K(local half * wt, global const uchar * ql, global const uchar * qh, global const char * s,
                            global const half * d, ulong wrow0, int row0, int ne01, int nb, int ib) {
    for (int u = get_local_linear_id(); u < MT * 8; u += NTHR) {
        const int rr = u >> 3, kb = u & 7;
        const int n = kb >> 2, p = kb & 3;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong b = (wrow0 + r) * nb + ib;
        const float D = vload_half(0, d + b);
        const int is = 8 * n + 2 * p;
        const float a0 = D * (float)s[b * 16 + is], a1 = D * (float)s[b * 16 + is + 1];
        for (int hh = 0; hh < 2; ++hh) {
            const uint4 w = vload4(0, (global const uint *)(ql + b * 128 + 64 * n + 32 * (p & 1) + 16 * hh));
            const uint4 h = vload4(0, (global const uint *)(qh + b * 64 + 32 * n + 16 * hh));
            const uint4 c = ((w >> (4 * (p >> 1))) & 0x0F0F0F0Fu) | (((h >> (2 * p)) & 0x03030303u) << 4);
            const float16 v = (hh ? a1 : a0) * (convert_float16(as_uchar16(c)) - 32.f);
            put16(wt, rr, 32 * kb + 16 * hh, v);
        }
    }
}
#endif

// IQ3_XXS (98-byte blocks, 2-byte aligned at q + off): unit (row r, 32-block
// kb) = 32 weights (dequantize_row_iq3_xxs)
#if SG == 8
static inline void deq_iq3_xxs(local half * wt, global const uchar * q, ulong wrow0, int row0, int ne01, int nb, int ib) {
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 2, kb = 4 * ch + (u & 3);
        const int r = min(row0 + rr, ne01 - 1);
        global const uchar * bp = q + ((wrow0 + r) * nb + sb) * 98;
        global const ushort * h = (global const ushort *)bp;
        const float d = vload_half(0, (global const half *)bp);
        const ushort4 gi = vload4(0, h + 1 + 4 * kb);
        const uint aux = (uint)h[1 + 32 + 2 * kb] | ((uint)h[1 + 32 + 2 * kb + 1] << 16);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        const uchar idx[8] = { gi.s0 & 0xFF, gi.s0 >> 8, gi.s1 & 0xFF, gi.s1 >> 8,
                               gi.s2 & 0xFF, gi.s2 >> 8, gi.s3 & 0xFF, gi.s3 >> 8 };
        float v[32];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const ulong m = ksigns64[(aux >> (7 * l)) & 127];
            const uchar4 g1 = as_uchar4(iq3xxs_grid[idx[2 * l]]), g2 = as_uchar4(iq3xxs_grid[idx[2 * l + 1]]);
            const uchar8 mb = as_uchar8(m);
            v[8 * l + 0] = (mb.s0 ? -db : db) * g1.s0; v[8 * l + 1] = (mb.s1 ? -db : db) * g1.s1;
            v[8 * l + 2] = (mb.s2 ? -db : db) * g1.s2; v[8 * l + 3] = (mb.s3 ? -db : db) * g1.s3;
            v[8 * l + 4] = (mb.s4 ? -db : db) * g2.s0; v[8 * l + 5] = (mb.s5 ? -db : db) * g2.s1;
            v[8 * l + 6] = (mb.s6 ? -db : db) * g2.s2; v[8 * l + 7] = (mb.s7 ? -db : db) * g2.s3;
        }
        put16(wt, rr, 32 * (u & 3),      vload16(0, v));
        put16(wt, rr, 32 * (u & 3) + 16, vload16(1, v));
    }
}
#else
static inline void deq_iq3_xxs(local half * wt, global const uchar * q, ulong wrow0, int row0, int ne01, int nb, int ib) {
    for (int u = get_local_linear_id(); u < MT * 8; u += NTHR) {
        const int rr = u >> 3, kb = u & 7;
        const int r = min(row0 + rr, ne01 - 1);
        global const uchar * bp = q + ((wrow0 + r) * nb + ib) * 98;
        global const ushort * h = (global const ushort *)bp;
        const float d = vload_half(0, (global const half *)bp);
        const ushort4 gi = vload4(0, h + 1 + 4 * kb);
        const uint aux = (uint)h[1 + 32 + 2 * kb] | ((uint)h[1 + 32 + 2 * kb + 1] << 16);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        const uchar idx[8] = { gi.s0 & 0xFF, gi.s0 >> 8, gi.s1 & 0xFF, gi.s1 >> 8,
                               gi.s2 & 0xFF, gi.s2 >> 8, gi.s3 & 0xFF, gi.s3 >> 8 };
        float v[32];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const ulong m = ksigns64[(aux >> (7 * l)) & 127];
            const uchar4 g1 = as_uchar4(iq3xxs_grid[idx[2 * l]]), g2 = as_uchar4(iq3xxs_grid[idx[2 * l + 1]]);
            const uchar8 mb = as_uchar8(m);
            v[8 * l + 0] = (mb.s0 ? -db : db) * g1.s0; v[8 * l + 1] = (mb.s1 ? -db : db) * g1.s1;
            v[8 * l + 2] = (mb.s2 ? -db : db) * g1.s2; v[8 * l + 3] = (mb.s3 ? -db : db) * g1.s3;
            v[8 * l + 4] = (mb.s4 ? -db : db) * g2.s0; v[8 * l + 5] = (mb.s5 ? -db : db) * g2.s1;
            v[8 * l + 6] = (mb.s6 ? -db : db) * g2.s2; v[8 * l + 7] = (mb.s7 ? -db : db) * g2.s3;
        }
        put16(wt, rr, 32 * kb,      vload16(0, v));
        put16(wt, rr, 32 * kb + 16, vload16(1, v));
    }
}
#endif

// 16 code bytes -> 32 values a * kvalue (low nibbles 0..15, high 16..31)
static inline void iq4_put32(local half * wt, int rr, int k0, uint4 qw, float a) {
    const uchar16 c = as_uchar16(qw);
    float lo[16], hi[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) { lo[j] = a * kvalues_iq4nl[c[j] & 0xF]; hi[j] = a * kvalues_iq4nl[c[j] >> 4]; }
    put16(wt, rr, k0, vload16(0, lo));
    put16(wt, rr, k0 + 16, vload16(0, hi));
}

// IQ4_XS (136-byte blocks at q + off): unit (row r, 32-block kb)
#if SG == 8
static inline void deq_iq4_xs(local half * wt, global const uchar * q, ulong wrow0, int row0, int ne01, int nb, int ib) {
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 2, kb = 4 * ch + (u & 3);
        const int r = min(row0 + rr, ne01 - 1);
        global const uchar * bp = q + ((wrow0 + r) * nb + sb) * 136;
        const float d = vload_half(0, (global const half *)bp);
        const uint sh = ((global const ushort *)bp)[1];
        const int ls = ((bp[4 + kb / 2] >> (4 * (kb & 1))) & 0xF) | (((sh >> (2 * kb)) & 3) << 4);
        iq4_put32(wt, rr, 32 * (u & 3), vload4(0, (global const uint *)(bp + 8 + 16 * kb)), d * (float)(ls - 32));
    }
}
#else
static inline void deq_iq4_xs(local half * wt, global const uchar * q, ulong wrow0, int row0, int ne01, int nb, int ib) {
    for (int u = get_local_linear_id(); u < MT * 8; u += NTHR) {
        const int rr = u >> 3, kb = u & 7;
        const int r = min(row0 + rr, ne01 - 1);
        global const uchar * bp = q + ((wrow0 + r) * nb + ib) * 136;
        const float d = vload_half(0, (global const half *)bp);
        const uint sh = ((global const ushort *)bp)[1];
        const int ls = ((bp[4 + kb / 2] >> (4 * (kb & 1))) & 0xF) | (((sh >> (2 * kb)) & 3) << 4);
        iq4_put32(wt, rr, 32 * kb, vload4(0, (global const uint *)(bp + 8 + 16 * kb)), d * (float)(ls - 32));
    }
}
#endif

// IQ4_NL flat planes (16 code bytes, one half per 32 weights), rows of n32
// blocks: unit (row r, 32-block kb of super-block ib), those past K skipped
#if SG == 8
static inline void deq_iq4_nl(local half * wt, global const uchar * q, global const half * dp, ulong wrow0, int row0, int ne01, int n32, int ib) {
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 2, kb = 4 * ch + (u & 3);
        if (sb * 8 + kb >= n32) continue;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong blk = (wrow0 + r) * n32 + sb * 8 + kb;
        iq4_put32(wt, rr, 32 * (u & 3), vload4(0, (global const uint *)(q + blk * 16)), vload_half(blk, dp));
    }
}
#else
static inline void deq_iq4_nl(local half * wt, global const uchar * q, global const half * dp, ulong wrow0, int row0, int ne01, int n32, int ib) {
    for (int u = get_local_linear_id(); u < MT * 8; u += NTHR) {
        const int rr = u >> 3, kb = u & 7;
        if (ib * 8 + kb >= n32) continue;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong blk = (wrow0 + r) * n32 + ib * 8 + kb;
        iq4_put32(wt, rr, 32 * kb, vload4(0, (global const uint *)(q + blk * 16)), vload_half(blk, dp));
    }
}
#endif

// ARC-LAB 0096: dense Q4_0, IQ4_NL's planes with the value n - 8
static inline void q40_put32(local half * wt, int rr, int k0, uint4 qw, float a) {
    const uchar16 c = as_uchar16(qw);
    put16(wt, rr, k0, convert_float16(as_char16(c & (uchar16)(15)) - (char16)(8)) * a);
    put16(wt, rr, k0 + 16, convert_float16(as_char16(c >> (uchar16)(4)) - (char16)(8)) * a);
}
#if SG == 8
static inline void deq_q4_0(local half * wt, global const uchar * q, global const half * dp, ulong wrow0, int row0, int ne01, int n32, int ib) {
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 2, kb = 4 * ch + (u & 3);
        if (sb * 8 + kb >= n32) continue;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong blk = (wrow0 + r) * n32 + sb * 8 + kb;
        q40_put32(wt, rr, 32 * (u & 3), vload4(0, (global const uint *)(q + blk * 16)), vload_half(blk, dp));
    }
}
#else
static inline void deq_q4_0(local half * wt, global const uchar * q, global const half * dp, ulong wrow0, int row0, int ne01, int n32, int ib) {
    for (int u = get_local_linear_id(); u < MT * 8; u += NTHR) {
        const int rr = u >> 3, kb = u & 7;
        if (ib * 8 + kb >= n32) continue;
        const int r = min(row0 + rr, ne01 - 1);
        const ulong blk = (wrow0 + r) * n32 + ib * 8 + kb;
        q40_put32(wt, rr, 32 * kb, vload4(0, (global const uint *)(q + blk * 16)), vload_half(blk, dp));
    }
}
#endif

constant ulong iq2xxs_grid[256] = {
    0x0808080808080808UL, 0x080808080808082bUL, 0x0808080808081919UL, 0x0808080808082b08UL,
    0x0808080808082b2bUL, 0x0808080808190819UL, 0x0808080808191908UL, 0x08080808082b0808UL,
    0x08080808082b082bUL, 0x08080808082b2b08UL, 0x08080808082b2b2bUL, 0x0808080819080819UL,
    0x0808080819081908UL, 0x0808080819190808UL, 0x0808080819192b08UL, 0x08080808192b0819UL,
    0x08080808192b1908UL, 0x080808082b080808UL, 0x080808082b08082bUL, 0x080808082b082b2bUL,
    0x080808082b2b082bUL, 0x0808081908080819UL, 0x0808081908081908UL, 0x0808081908190808UL,
    0x0808081908191919UL, 0x0808081919080808UL, 0x080808192b081908UL, 0x080808192b192b08UL,
    0x0808082b08080808UL, 0x0808082b0808082bUL, 0x0808082b082b082bUL, 0x0808082b2b08082bUL,
    0x0808190808080819UL, 0x0808190808081908UL, 0x0808190808190808UL, 0x08081908082b0819UL,
    0x08081908082b1908UL, 0x0808190819080808UL, 0x080819081908082bUL, 0x0808190819082b08UL,
    0x08081908192b0808UL, 0x080819082b080819UL, 0x080819082b081908UL, 0x080819082b190808UL,
    0x080819082b2b1908UL, 0x0808191908080808UL, 0x080819190808082bUL, 0x0808191908082b08UL,
    0x08081919082b0808UL, 0x080819191908192bUL, 0x08081919192b2b19UL, 0x080819192b080808UL,
    0x080819192b190819UL, 0x0808192b08082b19UL, 0x0808192b08190808UL, 0x0808192b19080808UL,
    0x0808192b2b081908UL, 0x0808192b2b2b1908UL, 0x08082b0808080808UL, 0x08082b0808081919UL,
    0x08082b0808082b08UL, 0x08082b0808191908UL, 0x08082b08082b2b08UL, 0x08082b0819080819UL,
    0x08082b0819081908UL, 0x08082b0819190808UL, 0x08082b081919082bUL, 0x08082b082b082b08UL,
    0x08082b1908081908UL, 0x08082b1919080808UL, 0x08082b2b0808082bUL, 0x08082b2b08191908UL,
    0x0819080808080819UL, 0x0819080808081908UL, 0x0819080808190808UL, 0x08190808082b0819UL,
    0x0819080819080808UL, 0x08190808192b0808UL, 0x081908082b081908UL, 0x081908082b190808UL,
    0x081908082b191919UL, 0x0819081908080808UL, 0x0819081908082b08UL, 0x08190819082b0808UL,
    0x0819081919190808UL, 0x0819081919192b2bUL, 0x081908192b080808UL, 0x0819082b082b1908UL,
    0x0819082b19081919UL, 0x0819190808080808UL, 0x0819190808082b08UL, 0x08191908082b0808UL,
    0x08191908082b1919UL, 0x0819190819082b19UL, 0x081919082b080808UL, 0x0819191908192b08UL,
    0x08191919192b082bUL, 0x0819192b08080808UL, 0x0819192b0819192bUL, 0x08192b0808080819UL,
    0x08192b0808081908UL, 0x08192b0808190808UL, 0x08192b0819080808UL, 0x08192b082b080819UL,
    0x08192b1908080808UL, 0x08192b1908081919UL, 0x08192b192b2b0808UL, 0x08192b2b19190819UL,
    0x082b080808080808UL, 0x082b08080808082bUL, 0x082b080808082b2bUL, 0x082b080819081908UL,
    0x082b0808192b0819UL, 0x082b08082b080808UL, 0x082b08082b08082bUL, 0x082b0819082b2b19UL,
    0x082b081919082b08UL, 0x082b082b08080808UL, 0x082b082b0808082bUL, 0x082b190808080819UL,
    0x082b190808081908UL, 0x082b190808190808UL, 0x082b190819080808UL, 0x082b19081919192bUL,
    0x082b191908080808UL, 0x082b191919080819UL, 0x082b1919192b1908UL, 0x082b192b2b190808UL,
    0x082b2b0808082b08UL, 0x082b2b08082b0808UL, 0x082b2b082b191908UL, 0x082b2b2b19081908UL,
    0x1908080808080819UL, 0x1908080808081908UL, 0x1908080808190808UL, 0x1908080808192b08UL,
    0x19080808082b0819UL, 0x19080808082b1908UL, 0x1908080819080808UL, 0x1908080819082b08UL,
    0x190808081919192bUL, 0x19080808192b0808UL, 0x190808082b080819UL, 0x190808082b081908UL,
    0x190808082b190808UL, 0x1908081908080808UL, 0x19080819082b0808UL, 0x19080819192b0819UL,
    0x190808192b080808UL, 0x190808192b081919UL, 0x1908082b08080819UL, 0x1908082b08190808UL,
    0x1908082b19082b08UL, 0x1908082b1919192bUL, 0x1908082b192b2b08UL, 0x1908190808080808UL,
    0x1908190808082b08UL, 0x19081908082b0808UL, 0x190819082b080808UL, 0x190819082b192b19UL,
    0x190819190819082bUL, 0x19081919082b1908UL, 0x1908192b08080808UL, 0x19082b0808080819UL,
    0x19082b0808081908UL, 0x19082b0808190808UL, 0x19082b0819080808UL, 0x19082b0819081919UL,
    0x19082b1908080808UL, 0x19082b1919192b08UL, 0x19082b19192b0819UL, 0x19082b192b08082bUL,
    0x19082b2b19081919UL, 0x19082b2b2b190808UL, 0x1919080808080808UL, 0x1919080808082b08UL,
    0x1919080808190819UL, 0x1919080808192b19UL, 0x19190808082b0808UL, 0x191908082b080808UL,
    0x191908082b082b08UL, 0x1919081908081908UL, 0x191908191908082bUL, 0x191908192b2b1908UL,
    0x1919082b2b190819UL, 0x191919082b190808UL, 0x191919082b19082bUL, 0x1919191908082b2bUL,
    0x1919192b08080819UL, 0x1919192b19191908UL, 0x19192b0808080808UL, 0x19192b0808190819UL,
    0x19192b0808192b19UL, 0x19192b08192b1908UL, 0x19192b1919080808UL, 0x19192b2b08082b08UL,
    0x192b080808081908UL, 0x192b080808190808UL, 0x192b080819080808UL, 0x192b0808192b2b08UL,
    0x192b081908080808UL, 0x192b081919191919UL, 0x192b082b08192b08UL, 0x192b082b192b0808UL,
    0x192b190808080808UL, 0x192b190808081919UL, 0x192b191908190808UL, 0x192b19190819082bUL,
    0x192b19192b081908UL, 0x192b2b081908082bUL, 0x2b08080808080808UL, 0x2b0808080808082bUL,
    0x2b08080808082b2bUL, 0x2b08080819080819UL, 0x2b0808082b08082bUL, 0x2b08081908081908UL,
    0x2b08081908192b08UL, 0x2b08081919080808UL, 0x2b08082b08190819UL, 0x2b08190808080819UL,
    0x2b08190808081908UL, 0x2b08190808190808UL, 0x2b08190808191919UL, 0x2b08190819080808UL,
    0x2b081908192b0808UL, 0x2b08191908080808UL, 0x2b0819191908192bUL, 0x2b0819192b191908UL,
    0x2b08192b08082b19UL, 0x2b08192b19080808UL, 0x2b08192b192b0808UL, 0x2b082b080808082bUL,
    0x2b082b1908081908UL, 0x2b082b2b08190819UL, 0x2b19080808081908UL, 0x2b19080808190808UL,
    0x2b190808082b1908UL, 0x2b19080819080808UL, 0x2b1908082b2b0819UL, 0x2b1908190819192bUL,
    0x2b1908192b080808UL, 0x2b19082b19081919UL, 0x2b19190808080808UL, 0x2b191908082b082bUL,
    0x2b19190819081908UL, 0x2b19191919190819UL, 0x2b192b082b080819UL, 0x2b192b19082b0808UL,
    0x2b2b08080808082bUL, 0x2b2b080819190808UL, 0x2b2b08082b081919UL, 0x2b2b081908082b19UL,
    0x2b2b082b08080808UL, 0x2b2b190808192b08UL, 0x2b2b2b0819190808UL, 0x2b2b2b1908081908UL,
};

// IQ2_XS: 512 grid entries of 8 unsigned bytes (ggml-common.h iq2xs_grid)
constant ulong iq2xs_grid[512] = {
    0x0808080808080808UL, 0x080808080808082bUL, 0x0808080808081919UL, 0x0808080808082b08UL,
    0x0808080808082b2bUL, 0x0808080808190819UL, 0x0808080808191908UL, 0x080808080819192bUL,
    0x0808080808192b19UL, 0x08080808082b0808UL, 0x08080808082b082bUL, 0x08080808082b1919UL,
    0x08080808082b2b08UL, 0x0808080819080819UL, 0x0808080819081908UL, 0x080808081908192bUL,
    0x0808080819082b19UL, 0x0808080819190808UL, 0x080808081919082bUL, 0x0808080819191919UL,
    0x0808080819192b08UL, 0x08080808192b0819UL, 0x08080808192b1908UL, 0x080808082b080808UL,
    0x080808082b08082bUL, 0x080808082b081919UL, 0x080808082b082b08UL, 0x080808082b190819UL,
    0x080808082b191908UL, 0x080808082b192b19UL, 0x080808082b2b0808UL, 0x0808081908080819UL,
    0x0808081908081908UL, 0x080808190808192bUL, 0x0808081908082b19UL, 0x0808081908190808UL,
    0x080808190819082bUL, 0x0808081908191919UL, 0x0808081908192b08UL, 0x0808081908192b2bUL,
    0x08080819082b0819UL, 0x08080819082b1908UL, 0x0808081919080808UL, 0x080808191908082bUL,
    0x0808081919081919UL, 0x0808081919082b08UL, 0x0808081919190819UL, 0x0808081919191908UL,
    0x08080819192b0808UL, 0x08080819192b2b08UL, 0x080808192b080819UL, 0x080808192b081908UL,
    0x080808192b190808UL, 0x0808082b08080808UL, 0x0808082b0808082bUL, 0x0808082b08081919UL,
    0x0808082b08082b08UL, 0x0808082b08190819UL, 0x0808082b08191908UL, 0x0808082b082b0808UL,
    0x0808082b19080819UL, 0x0808082b19081908UL, 0x0808082b19190808UL, 0x0808082b19191919UL,
    0x0808082b2b080808UL, 0x0808082b2b082b2bUL, 0x0808190808080819UL, 0x0808190808081908UL,
    0x080819080808192bUL, 0x0808190808082b19UL, 0x0808190808190808UL, 0x080819080819082bUL,
    0x0808190808191919UL, 0x0808190808192b08UL, 0x08081908082b0819UL, 0x08081908082b1908UL,
    0x0808190819080808UL, 0x080819081908082bUL, 0x0808190819081919UL, 0x0808190819082b08UL,
    0x0808190819190819UL, 0x0808190819191908UL, 0x080819081919192bUL, 0x08081908192b0808UL,
    0x080819082b080819UL, 0x080819082b081908UL, 0x080819082b190808UL, 0x0808191908080808UL,
    0x080819190808082bUL, 0x0808191908081919UL, 0x0808191908082b08UL, 0x0808191908190819UL,
    0x0808191908191908UL, 0x08081919082b0808UL, 0x0808191919080819UL, 0x0808191919081908UL,
    0x0808191919190808UL, 0x08081919192b0819UL, 0x080819192b080808UL, 0x0808192b08080819UL,
    0x0808192b08081908UL, 0x0808192b08190808UL, 0x0808192b082b192bUL, 0x0808192b19080808UL,
    0x0808192b1908082bUL, 0x0808192b2b081908UL, 0x08082b0808080808UL, 0x08082b080808082bUL,
    0x08082b0808081919UL, 0x08082b0808082b08UL, 0x08082b0808082b2bUL, 0x08082b0808190819UL,
    0x08082b0808191908UL, 0x08082b08082b0808UL, 0x08082b08082b1919UL, 0x08082b0819080819UL,
    0x08082b0819081908UL, 0x08082b0819190808UL, 0x08082b0819192b08UL, 0x08082b082b080808UL,
    0x08082b082b2b0808UL, 0x08082b082b2b2b2bUL, 0x08082b1908080819UL, 0x08082b1908081908UL,
    0x08082b1908190808UL, 0x08082b1919080808UL, 0x08082b192b080819UL, 0x08082b192b082b19UL,
    0x08082b2b08080808UL, 0x08082b2b082b0808UL, 0x08082b2b082b2b08UL, 0x08082b2b2b19192bUL,
    0x08082b2b2b2b0808UL, 0x0819080808080819UL, 0x0819080808081908UL, 0x081908080808192bUL,
    0x0819080808082b19UL, 0x0819080808190808UL, 0x081908080819082bUL, 0x0819080808191919UL,
    0x0819080808192b08UL, 0x08190808082b0819UL, 0x08190808082b1908UL, 0x0819080819080808UL,
    0x081908081908082bUL, 0x0819080819081919UL, 0x0819080819082b08UL, 0x0819080819190819UL,
    0x0819080819191908UL, 0x08190808192b0808UL, 0x08190808192b2b2bUL, 0x081908082b080819UL,
    0x081908082b081908UL, 0x081908082b190808UL, 0x0819081908080808UL, 0x081908190808082bUL,
    0x0819081908081919UL, 0x0819081908082b08UL, 0x0819081908190819UL, 0x0819081908191908UL,
    0x08190819082b0808UL, 0x0819081919080819UL, 0x0819081919081908UL, 0x0819081919190808UL,
    0x081908192b080808UL, 0x081908192b191908UL, 0x081908192b19192bUL, 0x0819082b08080819UL,
    0x0819082b08081908UL, 0x0819082b0808192bUL, 0x0819082b08190808UL, 0x0819082b19080808UL,
    0x0819082b192b0808UL, 0x0819190808080808UL, 0x081919080808082bUL, 0x0819190808081919UL,
    0x0819190808082b08UL, 0x0819190808190819UL, 0x0819190808191908UL, 0x08191908082b0808UL,
    0x0819190819080819UL, 0x0819190819081908UL, 0x0819190819082b19UL, 0x0819190819190808UL,
    0x08191908192b1908UL, 0x081919082b080808UL, 0x0819191908080819UL, 0x0819191908081908UL,
    0x0819191908190808UL, 0x0819191919080808UL, 0x0819192b08080808UL, 0x0819192b08191908UL,
    0x0819192b19082b19UL, 0x08192b0808080819UL, 0x08192b0808081908UL, 0x08192b0808190808UL,
    0x08192b080819082bUL, 0x08192b0819080808UL, 0x08192b0819191908UL, 0x08192b082b08192bUL,
    0x08192b1908080808UL, 0x08192b1908081919UL, 0x08192b19192b192bUL, 0x08192b2b19190819UL,
    0x08192b2b2b2b2b19UL, 0x082b080808080808UL, 0x082b08080808082bUL, 0x082b080808081919UL,
    0x082b080808082b08UL, 0x082b080808082b2bUL, 0x082b080808190819UL, 0x082b080808191908UL,
    0x082b0808082b0808UL, 0x082b080819080819UL, 0x082b080819081908UL, 0x082b080819190808UL,
    0x082b08082b080808UL, 0x082b08082b2b0808UL, 0x082b081908080819UL, 0x082b081908081908UL,
    0x082b081908190808UL, 0x082b081919080808UL, 0x082b081919082b08UL, 0x082b0819192b1919UL,
    0x082b082b08080808UL, 0x082b082b082b082bUL, 0x082b082b2b080808UL, 0x082b082b2b2b2b08UL,
    0x082b190808080819UL, 0x082b190808081908UL, 0x082b190808190808UL, 0x082b1908082b2b19UL,
    0x082b190819080808UL, 0x082b191908080808UL, 0x082b191919080819UL, 0x082b19191919082bUL,
    0x082b19192b192b19UL, 0x082b192b08080819UL, 0x082b192b08192b2bUL, 0x082b192b2b2b192bUL,
    0x082b2b0808080808UL, 0x082b2b0808082b08UL, 0x082b2b0808082b2bUL, 0x082b2b08082b0808UL,
    0x082b2b0819191919UL, 0x082b2b082b082b08UL, 0x082b2b082b2b082bUL, 0x082b2b19192b2b08UL,
    0x082b2b192b190808UL, 0x082b2b2b08082b08UL, 0x082b2b2b082b0808UL, 0x082b2b2b2b08082bUL,
    0x082b2b2b2b082b08UL, 0x082b2b2b2b082b2bUL, 0x1908080808080819UL, 0x1908080808081908UL,
    0x190808080808192bUL, 0x1908080808082b19UL, 0x1908080808190808UL, 0x190808080819082bUL,
    0x1908080808191919UL, 0x1908080808192b08UL, 0x19080808082b0819UL, 0x19080808082b1908UL,
    0x1908080819080808UL, 0x190808081908082bUL, 0x1908080819081919UL, 0x1908080819082b08UL,
    0x1908080819082b2bUL, 0x1908080819190819UL, 0x1908080819191908UL, 0x19080808192b0808UL,
    0x19080808192b1919UL, 0x190808082b080819UL, 0x190808082b081908UL, 0x190808082b190808UL,
    0x1908081908080808UL, 0x190808190808082bUL, 0x1908081908081919UL, 0x1908081908082b08UL,
    0x1908081908190819UL, 0x1908081908191908UL, 0x19080819082b0808UL, 0x1908081919080819UL,
    0x1908081919081908UL, 0x1908081919190808UL, 0x190808192b080808UL, 0x190808192b081919UL,
    0x190808192b2b082bUL, 0x1908082b08080819UL, 0x1908082b08081908UL, 0x1908082b08190808UL,
    0x1908082b0819082bUL, 0x1908082b082b2b19UL, 0x1908082b19080808UL, 0x1908190808080808UL,
    0x190819080808082bUL, 0x1908190808081919UL, 0x1908190808082b08UL, 0x1908190808190819UL,
    0x1908190808191908UL, 0x1908190808192b19UL, 0x19081908082b0808UL, 0x1908190819080819UL,
    0x1908190819081908UL, 0x1908190819190808UL, 0x190819082b080808UL, 0x190819082b191908UL,
    0x1908191908080819UL, 0x1908191908081908UL, 0x1908191908190808UL, 0x19081919082b1908UL,
    0x1908191919080808UL, 0x190819192b192b2bUL, 0x1908192b08080808UL, 0x1908192b08082b2bUL,
    0x1908192b19081908UL, 0x1908192b19190808UL, 0x19082b0808080819UL, 0x19082b0808081908UL,
    0x19082b0808190808UL, 0x19082b0819080808UL, 0x19082b0819081919UL, 0x19082b0819191908UL,
    0x19082b08192b082bUL, 0x19082b1908080808UL, 0x19082b1908190819UL, 0x19082b1919081908UL,
    0x19082b1919190808UL, 0x19082b19192b2b19UL, 0x19082b2b08081908UL, 0x1919080808080808UL,
    0x191908080808082bUL, 0x1919080808081919UL, 0x1919080808082b08UL, 0x1919080808190819UL,
    0x1919080808191908UL, 0x19190808082b0808UL, 0x19190808082b2b08UL, 0x1919080819080819UL,
    0x1919080819081908UL, 0x1919080819190808UL, 0x191908082b080808UL, 0x1919081908080819UL,
    0x1919081908081908UL, 0x1919081908190808UL, 0x1919081908191919UL, 0x1919081919080808UL,
    0x191908191908082bUL, 0x1919082b08080808UL, 0x1919082b19081908UL, 0x1919082b2b2b2b2bUL,
    0x1919190808080819UL, 0x1919190808081908UL, 0x1919190808190808UL, 0x19191908082b0819UL,
    0x1919190819080808UL, 0x19191908192b0808UL, 0x191919082b080819UL, 0x191919082b2b0819UL,
    0x1919191908080808UL, 0x1919191908082b08UL, 0x191919192b080808UL, 0x191919192b082b08UL,
    0x1919192b082b0819UL, 0x1919192b192b2b08UL, 0x1919192b2b2b0819UL, 0x19192b0808080808UL,
    0x19192b0808191908UL, 0x19192b0819080819UL, 0x19192b0819190808UL, 0x19192b082b192b19UL,
    0x19192b1908192b2bUL, 0x19192b1919080808UL, 0x19192b191908082bUL, 0x19192b2b2b081919UL,
    0x192b080808080819UL, 0x192b080808081908UL, 0x192b080808190808UL, 0x192b080819080808UL,
    0x192b080819191908UL, 0x192b0808192b082bUL, 0x192b08082b08192bUL, 0x192b08082b2b2b19UL,
    0x192b081908080808UL, 0x192b082b082b1908UL, 0x192b082b19082b2bUL, 0x192b082b2b19082bUL,
    0x192b190808080808UL, 0x192b19080819192bUL, 0x192b191908190808UL, 0x192b191919080808UL,
    0x192b191919081919UL, 0x192b19192b2b1908UL, 0x192b2b0808080819UL, 0x192b2b08192b2b2bUL,
    0x192b2b19082b1919UL, 0x192b2b2b0808192bUL, 0x192b2b2b19191908UL, 0x192b2b2b192b082bUL,
    0x2b08080808080808UL, 0x2b0808080808082bUL, 0x2b08080808081919UL, 0x2b08080808082b08UL,
    0x2b08080808190819UL, 0x2b08080808191908UL, 0x2b080808082b0808UL, 0x2b080808082b2b2bUL,
    0x2b08080819080819UL, 0x2b08080819081908UL, 0x2b08080819190808UL, 0x2b0808082b080808UL,
    0x2b0808082b08082bUL, 0x2b0808082b2b2b08UL, 0x2b0808082b2b2b2bUL, 0x2b08081908080819UL,
    0x2b08081908081908UL, 0x2b0808190808192bUL, 0x2b08081908190808UL, 0x2b08081919080808UL,
    0x2b08081919190819UL, 0x2b08081919192b19UL, 0x2b08082b08080808UL, 0x2b08082b082b0808UL,
    0x2b08082b2b080808UL, 0x2b08082b2b08082bUL, 0x2b08082b2b2b0808UL, 0x2b08082b2b2b2b08UL,
    0x2b08190808080819UL, 0x2b08190808081908UL, 0x2b08190808190808UL, 0x2b0819080819082bUL,
    0x2b08190808191919UL, 0x2b08190819080808UL, 0x2b081908192b0808UL, 0x2b0819082b082b19UL,
    0x2b08191908080808UL, 0x2b08191919081908UL, 0x2b0819192b2b1919UL, 0x2b08192b08192b08UL,
    0x2b08192b192b2b2bUL, 0x2b082b0808080808UL, 0x2b082b0808082b08UL, 0x2b082b08082b1919UL,
    0x2b082b0819192b2bUL, 0x2b082b082b080808UL, 0x2b082b082b08082bUL, 0x2b082b082b2b2b08UL,
    0x2b082b190808192bUL, 0x2b082b2b082b082bUL, 0x2b082b2b2b080808UL, 0x2b082b2b2b082b08UL,
    0x2b082b2b2b19192bUL, 0x2b082b2b2b2b2b08UL, 0x2b19080808080819UL, 0x2b19080808081908UL,
    0x2b19080808190808UL, 0x2b19080819080808UL, 0x2b1908081919192bUL, 0x2b1908082b081908UL,
    0x2b19081908080808UL, 0x2b190819082b082bUL, 0x2b190819192b1908UL, 0x2b19082b1919192bUL,
    0x2b19082b2b082b19UL, 0x2b19190808080808UL, 0x2b19190808081919UL, 0x2b19190819081908UL,
    0x2b19190819190808UL, 0x2b19190819192b08UL, 0x2b191919082b2b19UL, 0x2b1919192b190808UL,
    0x2b1919192b19082bUL, 0x2b19192b19080819UL, 0x2b192b0819190819UL, 0x2b192b082b2b192bUL,
    0x2b192b1919082b19UL, 0x2b192b2b08191919UL, 0x2b192b2b192b0808UL, 0x2b2b080808080808UL,
    0x2b2b08080808082bUL, 0x2b2b080808082b08UL, 0x2b2b080808082b2bUL, 0x2b2b0808082b0808UL,
    0x2b2b0808082b2b2bUL, 0x2b2b08082b2b0808UL, 0x2b2b081919190819UL, 0x2b2b081919192b19UL,
    0x2b2b08192b2b192bUL, 0x2b2b082b08080808UL, 0x2b2b082b0808082bUL, 0x2b2b082b08082b08UL,
    0x2b2b082b082b2b2bUL, 0x2b2b082b2b080808UL, 0x2b2b082b2b2b0808UL, 0x2b2b190819080808UL,
    0x2b2b19082b191919UL, 0x2b2b192b192b1919UL, 0x2b2b192b2b192b08UL, 0x2b2b2b0808082b2bUL,
    0x2b2b2b08082b0808UL, 0x2b2b2b08082b082bUL, 0x2b2b2b08082b2b08UL, 0x2b2b2b082b2b0808UL,
    0x2b2b2b082b2b2b08UL, 0x2b2b2b1908081908UL, 0x2b2b2b192b081908UL, 0x2b2b2b192b08192bUL,
    0x2b2b2b2b082b2b08UL, 0x2b2b2b2b082b2b2bUL, 0x2b2b2b2b2b190819UL, 0x2b2b2b2b2b2b2b2bUL,
};

constant ulong iq2s_grid[1024] = {
    0x0808080808080808UL, 0x080808080808082bUL, 0x0808080808081919UL, 0x0808080808082b08UL,
    0x0808080808082b2bUL, 0x0808080808190819UL, 0x0808080808191908UL, 0x080808080819192bUL,
    0x0808080808192b19UL, 0x08080808082b0808UL, 0x08080808082b082bUL, 0x08080808082b1919UL,
    0x08080808082b2b08UL, 0x0808080819080819UL, 0x0808080819081908UL, 0x080808081908192bUL,
    0x0808080819082b19UL, 0x0808080819190808UL, 0x080808081919082bUL, 0x0808080819191919UL,
    0x0808080819192b08UL, 0x08080808192b0819UL, 0x08080808192b1908UL, 0x08080808192b192bUL,
    0x08080808192b2b19UL, 0x080808082b080808UL, 0x080808082b08082bUL, 0x080808082b081919UL,
    0x080808082b082b08UL, 0x080808082b190819UL, 0x080808082b191908UL, 0x080808082b2b0808UL,
    0x080808082b2b1919UL, 0x080808082b2b2b2bUL, 0x0808081908080819UL, 0x0808081908081908UL,
    0x080808190808192bUL, 0x0808081908082b19UL, 0x0808081908190808UL, 0x080808190819082bUL,
    0x0808081908191919UL, 0x0808081908192b08UL, 0x08080819082b0819UL, 0x08080819082b1908UL,
    0x0808081919080808UL, 0x080808191908082bUL, 0x0808081919081919UL, 0x0808081919082b08UL,
    0x0808081919190819UL, 0x0808081919191908UL, 0x080808191919192bUL, 0x0808081919192b19UL,
    0x08080819192b0808UL, 0x08080819192b1919UL, 0x08080819192b2b08UL, 0x080808192b080819UL,
    0x080808192b081908UL, 0x080808192b190808UL, 0x080808192b19082bUL, 0x080808192b191919UL,
    0x080808192b2b0819UL, 0x080808192b2b1908UL, 0x0808082b08080808UL, 0x0808082b0808082bUL,
    0x0808082b08081919UL, 0x0808082b08082b08UL, 0x0808082b08190819UL, 0x0808082b08191908UL,
    0x0808082b082b0808UL, 0x0808082b082b2b2bUL, 0x0808082b19080819UL, 0x0808082b19081908UL,
    0x0808082b1908192bUL, 0x0808082b19082b19UL, 0x0808082b19190808UL, 0x0808082b19191919UL,
    0x0808082b2b080808UL, 0x0808082b2b081919UL, 0x0808082b2b082b2bUL, 0x0808082b2b191908UL,
    0x0808082b2b2b082bUL, 0x0808190808080819UL, 0x0808190808081908UL, 0x080819080808192bUL,
    0x0808190808082b19UL, 0x0808190808190808UL, 0x080819080819082bUL, 0x0808190808191919UL,
    0x0808190808192b08UL, 0x08081908082b0819UL, 0x08081908082b1908UL, 0x08081908082b192bUL,
    0x08081908082b2b19UL, 0x0808190819080808UL, 0x080819081908082bUL, 0x0808190819081919UL,
    0x0808190819082b08UL, 0x0808190819082b2bUL, 0x0808190819190819UL, 0x0808190819191908UL,
    0x080819081919192bUL, 0x0808190819192b19UL, 0x08081908192b0808UL, 0x08081908192b082bUL,
    0x08081908192b1919UL, 0x080819082b080819UL, 0x080819082b081908UL, 0x080819082b08192bUL,
    0x080819082b082b19UL, 0x080819082b190808UL, 0x080819082b191919UL, 0x080819082b192b08UL,
    0x080819082b2b0819UL, 0x080819082b2b1908UL, 0x0808191908080808UL, 0x080819190808082bUL,
    0x0808191908081919UL, 0x0808191908082b08UL, 0x0808191908082b2bUL, 0x0808191908190819UL,
    0x0808191908191908UL, 0x080819190819192bUL, 0x0808191908192b19UL, 0x08081919082b0808UL,
    0x08081919082b1919UL, 0x08081919082b2b08UL, 0x0808191919080819UL, 0x0808191919081908UL,
    0x080819191908192bUL, 0x0808191919082b19UL, 0x0808191919190808UL, 0x080819191919082bUL,
    0x0808191919191919UL, 0x0808191919192b08UL, 0x08081919192b0819UL, 0x08081919192b1908UL,
    0x080819192b080808UL, 0x080819192b08082bUL, 0x080819192b081919UL, 0x080819192b082b08UL,
    0x080819192b190819UL, 0x080819192b191908UL, 0x080819192b2b0808UL, 0x0808192b08080819UL,
    0x0808192b08081908UL, 0x0808192b0808192bUL, 0x0808192b08082b19UL, 0x0808192b08190808UL,
    0x0808192b08191919UL, 0x0808192b19080808UL, 0x0808192b19081919UL, 0x0808192b19082b08UL,
    0x0808192b19190819UL, 0x0808192b19191908UL, 0x0808192b192b0808UL, 0x0808192b2b080819UL,
    0x0808192b2b081908UL, 0x0808192b2b190808UL, 0x08082b0808080808UL, 0x08082b080808082bUL,
    0x08082b0808081919UL, 0x08082b0808082b08UL, 0x08082b0808190819UL, 0x08082b0808191908UL,
    0x08082b080819192bUL, 0x08082b0808192b19UL, 0x08082b08082b0808UL, 0x08082b08082b1919UL,
    0x08082b08082b2b2bUL, 0x08082b0819080819UL, 0x08082b0819081908UL, 0x08082b081908192bUL,
    0x08082b0819082b19UL, 0x08082b0819190808UL, 0x08082b081919082bUL, 0x08082b0819191919UL,
    0x08082b0819192b08UL, 0x08082b08192b0819UL, 0x08082b08192b1908UL, 0x08082b082b080808UL,
    0x08082b082b081919UL, 0x08082b082b191908UL, 0x08082b082b2b2b2bUL, 0x08082b1908080819UL,
    0x08082b1908081908UL, 0x08082b1908190808UL, 0x08082b190819082bUL, 0x08082b1908191919UL,
    0x08082b1908192b08UL, 0x08082b19082b0819UL, 0x08082b1919080808UL, 0x08082b1919081919UL,
    0x08082b1919082b08UL, 0x08082b1919190819UL, 0x08082b1919191908UL, 0x08082b19192b0808UL,
    0x08082b192b080819UL, 0x08082b192b190808UL, 0x08082b2b08080808UL, 0x08082b2b08190819UL,
    0x08082b2b08191908UL, 0x08082b2b082b082bUL, 0x08082b2b082b2b08UL, 0x08082b2b082b2b2bUL,
    0x08082b2b19190808UL, 0x08082b2b2b192b19UL, 0x0819080808080819UL, 0x0819080808081908UL,
    0x081908080808192bUL, 0x0819080808082b19UL, 0x0819080808190808UL, 0x081908080819082bUL,
    0x0819080808191919UL, 0x0819080808192b08UL, 0x08190808082b0819UL, 0x08190808082b1908UL,
    0x08190808082b192bUL, 0x0819080819080808UL, 0x081908081908082bUL, 0x0819080819081919UL,
    0x0819080819082b08UL, 0x0819080819190819UL, 0x0819080819191908UL, 0x081908081919192bUL,
    0x0819080819192b19UL, 0x08190808192b0808UL, 0x08190808192b082bUL, 0x08190808192b1919UL,
    0x08190808192b2b08UL, 0x081908082b080819UL, 0x081908082b081908UL, 0x081908082b08192bUL,
    0x081908082b190808UL, 0x081908082b191919UL, 0x081908082b192b08UL, 0x081908082b2b0819UL,
    0x081908082b2b1908UL, 0x0819081908080808UL, 0x081908190808082bUL, 0x0819081908081919UL,
    0x0819081908082b08UL, 0x0819081908082b2bUL, 0x0819081908190819UL, 0x0819081908191908UL,
    0x081908190819192bUL, 0x0819081908192b19UL, 0x08190819082b0808UL, 0x08190819082b082bUL,
    0x08190819082b1919UL, 0x08190819082b2b08UL, 0x0819081919080819UL, 0x0819081919081908UL,
    0x081908191908192bUL, 0x0819081919082b19UL, 0x0819081919190808UL, 0x081908191919082bUL,
    0x0819081919191919UL, 0x0819081919192b08UL, 0x08190819192b0819UL, 0x08190819192b1908UL,
    0x081908192b080808UL, 0x081908192b08082bUL, 0x081908192b081919UL, 0x081908192b082b08UL,
    0x081908192b190819UL, 0x081908192b191908UL, 0x0819082b08080819UL, 0x0819082b08081908UL,
    0x0819082b08082b19UL, 0x0819082b08190808UL, 0x0819082b08191919UL, 0x0819082b082b0819UL,
    0x0819082b082b1908UL, 0x0819082b19080808UL, 0x0819082b19081919UL, 0x0819082b19190819UL,
    0x0819082b19191908UL, 0x0819082b2b080819UL, 0x0819082b2b081908UL, 0x0819082b2b190808UL,
    0x0819190808080808UL, 0x081919080808082bUL, 0x0819190808081919UL, 0x0819190808082b08UL,
    0x0819190808190819UL, 0x0819190808191908UL, 0x081919080819192bUL, 0x0819190808192b19UL,
    0x08191908082b0808UL, 0x08191908082b1919UL, 0x08191908082b2b08UL, 0x0819190819080819UL,
    0x0819190819081908UL, 0x081919081908192bUL, 0x0819190819082b19UL, 0x0819190819190808UL,
    0x081919081919082bUL, 0x0819190819191919UL, 0x0819190819192b08UL, 0x08191908192b0819UL,
    0x08191908192b1908UL, 0x081919082b080808UL, 0x081919082b08082bUL, 0x081919082b081919UL,
    0x081919082b082b08UL, 0x081919082b190819UL, 0x081919082b191908UL, 0x081919082b2b0808UL,
    0x0819191908080819UL, 0x0819191908081908UL, 0x081919190808192bUL, 0x0819191908082b19UL,
    0x0819191908190808UL, 0x081919190819082bUL, 0x0819191908191919UL, 0x0819191908192b08UL,
    0x08191919082b0819UL, 0x08191919082b1908UL, 0x0819191919080808UL, 0x081919191908082bUL,
    0x0819191919081919UL, 0x0819191919082b08UL, 0x0819191919190819UL, 0x0819191919191908UL,
    0x08191919192b0808UL, 0x081919192b080819UL, 0x081919192b081908UL, 0x081919192b190808UL,
    0x0819192b08080808UL, 0x0819192b08081919UL, 0x0819192b08082b08UL, 0x0819192b08190819UL,
    0x0819192b08191908UL, 0x0819192b082b0808UL, 0x0819192b19080819UL, 0x0819192b19081908UL,
    0x0819192b19190808UL, 0x0819192b2b080808UL, 0x0819192b2b2b2b2bUL, 0x08192b0808080819UL,
    0x08192b0808081908UL, 0x08192b080808192bUL, 0x08192b0808082b19UL, 0x08192b0808190808UL,
    0x08192b0808191919UL, 0x08192b0808192b08UL, 0x08192b08082b0819UL, 0x08192b0819080808UL,
    0x08192b081908082bUL, 0x08192b0819081919UL, 0x08192b0819082b08UL, 0x08192b0819190819UL,
    0x08192b0819191908UL, 0x08192b08192b0808UL, 0x08192b082b080819UL, 0x08192b082b081908UL,
    0x08192b1908080808UL, 0x08192b190808082bUL, 0x08192b1908081919UL, 0x08192b1908082b08UL,
    0x08192b1908190819UL, 0x08192b1908191908UL, 0x08192b19082b0808UL, 0x08192b1919080819UL,
    0x08192b1919081908UL, 0x08192b1919190808UL, 0x08192b19192b2b19UL, 0x08192b192b2b082bUL,
    0x08192b2b08081908UL, 0x08192b2b08190808UL, 0x08192b2b19080808UL, 0x08192b2b1919192bUL,
    0x082b080808080808UL, 0x082b08080808082bUL, 0x082b080808081919UL, 0x082b080808082b08UL,
    0x082b080808190819UL, 0x082b080808191908UL, 0x082b08080819192bUL, 0x082b080808192b19UL,
    0x082b0808082b0808UL, 0x082b0808082b1919UL, 0x082b0808082b2b2bUL, 0x082b080819080819UL,
    0x082b080819081908UL, 0x082b080819190808UL, 0x082b08081919082bUL, 0x082b080819191919UL,
    0x082b0808192b1908UL, 0x082b08082b080808UL, 0x082b08082b082b2bUL, 0x082b08082b191908UL,
    0x082b08082b2b2b2bUL, 0x082b081908080819UL, 0x082b081908081908UL, 0x082b081908190808UL,
    0x082b08190819082bUL, 0x082b081908191919UL, 0x082b0819082b0819UL, 0x082b081919080808UL,
    0x082b08191908082bUL, 0x082b081919081919UL, 0x082b081919190819UL, 0x082b081919191908UL,
    0x082b0819192b0808UL, 0x082b08192b080819UL, 0x082b08192b081908UL, 0x082b08192b190808UL,
    0x082b082b08080808UL, 0x082b082b08082b2bUL, 0x082b082b082b082bUL, 0x082b082b082b2b08UL,
    0x082b082b082b2b2bUL, 0x082b082b19081908UL, 0x082b082b19190808UL, 0x082b082b2b082b08UL,
    0x082b082b2b082b2bUL, 0x082b082b2b2b2b08UL, 0x082b190808080819UL, 0x082b190808081908UL,
    0x082b19080808192bUL, 0x082b190808082b19UL, 0x082b190808190808UL, 0x082b190808191919UL,
    0x082b190808192b08UL, 0x082b1908082b0819UL, 0x082b1908082b1908UL, 0x082b190819080808UL,
    0x082b19081908082bUL, 0x082b190819081919UL, 0x082b190819082b08UL, 0x082b190819190819UL,
    0x082b190819191908UL, 0x082b1908192b0808UL, 0x082b19082b080819UL, 0x082b19082b081908UL,
    0x082b19082b190808UL, 0x082b191908080808UL, 0x082b191908081919UL, 0x082b191908082b08UL,
    0x082b191908190819UL, 0x082b191908191908UL, 0x082b1919082b0808UL, 0x082b191919080819UL,
    0x082b191919081908UL, 0x082b191919190808UL, 0x082b1919192b192bUL, 0x082b19192b080808UL,
    0x082b192b08080819UL, 0x082b192b08081908UL, 0x082b192b08190808UL, 0x082b192b19080808UL,
    0x082b192b19192b19UL, 0x082b2b0808080808UL, 0x082b2b0808081919UL, 0x082b2b0808190819UL,
    0x082b2b0808191908UL, 0x082b2b0819080819UL, 0x082b2b0819081908UL, 0x082b2b0819190808UL,
    0x082b2b082b082b2bUL, 0x082b2b082b2b2b2bUL, 0x082b2b1908080819UL, 0x082b2b1908081908UL,
    0x082b2b1908190808UL, 0x082b2b192b191919UL, 0x082b2b2b08082b2bUL, 0x082b2b2b082b082bUL,
    0x082b2b2b192b1908UL, 0x082b2b2b2b082b08UL, 0x082b2b2b2b082b2bUL, 0x1908080808080819UL,
    0x1908080808081908UL, 0x190808080808192bUL, 0x1908080808082b19UL, 0x1908080808190808UL,
    0x190808080819082bUL, 0x1908080808191919UL, 0x1908080808192b08UL, 0x1908080808192b2bUL,
    0x19080808082b0819UL, 0x19080808082b1908UL, 0x19080808082b192bUL, 0x1908080819080808UL,
    0x190808081908082bUL, 0x1908080819081919UL, 0x1908080819082b08UL, 0x1908080819082b2bUL,
    0x1908080819190819UL, 0x1908080819191908UL, 0x190808081919192bUL, 0x1908080819192b19UL,
    0x19080808192b0808UL, 0x19080808192b082bUL, 0x19080808192b1919UL, 0x190808082b080819UL,
    0x190808082b081908UL, 0x190808082b190808UL, 0x190808082b191919UL, 0x190808082b192b08UL,
    0x190808082b2b0819UL, 0x190808082b2b1908UL, 0x1908081908080808UL, 0x190808190808082bUL,
    0x1908081908081919UL, 0x1908081908082b08UL, 0x1908081908190819UL, 0x1908081908191908UL,
    0x190808190819192bUL, 0x1908081908192b19UL, 0x19080819082b0808UL, 0x19080819082b082bUL,
    0x19080819082b1919UL, 0x1908081919080819UL, 0x1908081919081908UL, 0x190808191908192bUL,
    0x1908081919082b19UL, 0x1908081919190808UL, 0x190808191919082bUL, 0x1908081919191919UL,
    0x1908081919192b08UL, 0x19080819192b0819UL, 0x19080819192b1908UL, 0x190808192b080808UL,
    0x190808192b08082bUL, 0x190808192b081919UL, 0x190808192b082b08UL, 0x190808192b190819UL,
    0x190808192b191908UL, 0x190808192b2b0808UL, 0x1908082b08080819UL, 0x1908082b08081908UL,
    0x1908082b08190808UL, 0x1908082b0819082bUL, 0x1908082b08191919UL, 0x1908082b08192b08UL,
    0x1908082b082b1908UL, 0x1908082b19080808UL, 0x1908082b19081919UL, 0x1908082b19082b08UL,
    0x1908082b19190819UL, 0x1908082b19191908UL, 0x1908082b192b0808UL, 0x1908082b2b080819UL,
    0x1908082b2b081908UL, 0x1908190808080808UL, 0x190819080808082bUL, 0x1908190808081919UL,
    0x1908190808082b08UL, 0x1908190808082b2bUL, 0x1908190808190819UL, 0x1908190808191908UL,
    0x190819080819192bUL, 0x1908190808192b19UL, 0x19081908082b0808UL, 0x19081908082b082bUL,
    0x19081908082b1919UL, 0x19081908082b2b08UL, 0x1908190819080819UL, 0x1908190819081908UL,
    0x190819081908192bUL, 0x1908190819082b19UL, 0x1908190819190808UL, 0x190819081919082bUL,
    0x1908190819191919UL, 0x1908190819192b08UL, 0x19081908192b0819UL, 0x19081908192b1908UL,
    0x190819082b080808UL, 0x190819082b08082bUL, 0x190819082b081919UL, 0x190819082b082b08UL,
    0x190819082b190819UL, 0x190819082b191908UL, 0x190819082b2b0808UL, 0x1908191908080819UL,
    0x1908191908081908UL, 0x190819190808192bUL, 0x1908191908082b19UL, 0x1908191908190808UL,
    0x190819190819082bUL, 0x1908191908191919UL, 0x1908191908192b08UL, 0x19081919082b0819UL,
    0x19081919082b1908UL, 0x1908191919080808UL, 0x190819191908082bUL, 0x1908191919081919UL,
    0x1908191919082b08UL, 0x1908191919190819UL, 0x1908191919191908UL, 0x19081919192b0808UL,
    0x19081919192b2b2bUL, 0x190819192b080819UL, 0x190819192b081908UL, 0x190819192b190808UL,
    0x1908192b08080808UL, 0x1908192b0808082bUL, 0x1908192b08081919UL, 0x1908192b08082b08UL,
    0x1908192b08190819UL, 0x1908192b08191908UL, 0x1908192b082b0808UL, 0x1908192b19080819UL,
    0x1908192b19081908UL, 0x1908192b19190808UL, 0x1908192b2b080808UL, 0x1908192b2b2b1919UL,
    0x19082b0808080819UL, 0x19082b0808081908UL, 0x19082b0808082b19UL, 0x19082b0808190808UL,
    0x19082b080819082bUL, 0x19082b0808191919UL, 0x19082b0808192b08UL, 0x19082b08082b0819UL,
    0x19082b08082b1908UL, 0x19082b0819080808UL, 0x19082b081908082bUL, 0x19082b0819081919UL,
    0x19082b0819082b08UL, 0x19082b0819190819UL, 0x19082b0819191908UL, 0x19082b08192b0808UL,
    0x19082b082b081908UL, 0x19082b082b190808UL, 0x19082b1908080808UL, 0x19082b190808082bUL,
    0x19082b1908081919UL, 0x19082b1908082b08UL, 0x19082b1908190819UL, 0x19082b1908191908UL,
    0x19082b19082b0808UL, 0x19082b1919080819UL, 0x19082b1919081908UL, 0x19082b1919190808UL,
    0x19082b192b080808UL, 0x19082b192b19192bUL, 0x19082b2b08080819UL, 0x19082b2b08081908UL,
    0x19082b2b08190808UL, 0x19082b2b19080808UL, 0x1919080808080808UL, 0x191908080808082bUL,
    0x1919080808081919UL, 0x1919080808082b08UL, 0x1919080808190819UL, 0x1919080808191908UL,
    0x191908080819192bUL, 0x1919080808192b19UL, 0x19190808082b0808UL, 0x19190808082b082bUL,
    0x19190808082b1919UL, 0x19190808082b2b08UL, 0x1919080819080819UL, 0x1919080819081908UL,
    0x191908081908192bUL, 0x1919080819082b19UL, 0x1919080819190808UL, 0x191908081919082bUL,
    0x1919080819191919UL, 0x1919080819192b08UL, 0x19190808192b0819UL, 0x19190808192b1908UL,
    0x191908082b080808UL, 0x191908082b08082bUL, 0x191908082b081919UL, 0x191908082b082b08UL,
    0x191908082b190819UL, 0x191908082b191908UL, 0x1919081908080819UL, 0x1919081908081908UL,
    0x191908190808192bUL, 0x1919081908082b19UL, 0x1919081908190808UL, 0x191908190819082bUL,
    0x1919081908191919UL, 0x1919081908192b08UL, 0x19190819082b0819UL, 0x19190819082b1908UL,
    0x1919081919080808UL, 0x191908191908082bUL, 0x1919081919081919UL, 0x1919081919082b08UL,
    0x1919081919190819UL, 0x1919081919191908UL, 0x19190819192b0808UL, 0x191908192b080819UL,
    0x191908192b081908UL, 0x191908192b190808UL, 0x1919082b08080808UL, 0x1919082b08081919UL,
    0x1919082b08082b08UL, 0x1919082b08190819UL, 0x1919082b08191908UL, 0x1919082b082b0808UL,
    0x1919082b19080819UL, 0x1919082b19081908UL, 0x1919082b19190808UL, 0x1919082b192b2b19UL,
    0x1919082b2b080808UL, 0x1919190808080819UL, 0x1919190808081908UL, 0x191919080808192bUL,
    0x1919190808082b19UL, 0x1919190808190808UL, 0x191919080819082bUL, 0x1919190808191919UL,
    0x1919190808192b08UL, 0x19191908082b0819UL, 0x19191908082b1908UL, 0x1919190819080808UL,
    0x191919081908082bUL, 0x1919190819081919UL, 0x1919190819082b08UL, 0x1919190819190819UL,
    0x1919190819191908UL, 0x19191908192b0808UL, 0x191919082b080819UL, 0x191919082b081908UL,
    0x191919082b190808UL, 0x1919191908080808UL, 0x191919190808082bUL, 0x1919191908081919UL,
    0x1919191908082b08UL, 0x1919191908190819UL, 0x1919191908191908UL, 0x19191919082b0808UL,
    0x1919191919080819UL, 0x1919191919081908UL, 0x1919191919190808UL, 0x191919192b080808UL,
    0x1919192b08080819UL, 0x1919192b08081908UL, 0x1919192b08190808UL, 0x1919192b082b192bUL,
    0x1919192b19080808UL, 0x19192b0808080808UL, 0x19192b080808082bUL, 0x19192b0808081919UL,
    0x19192b0808082b08UL, 0x19192b0808190819UL, 0x19192b0808191908UL, 0x19192b08082b0808UL,
    0x19192b0819080819UL, 0x19192b0819081908UL, 0x19192b0819190808UL, 0x19192b0819192b2bUL,
    0x19192b082b080808UL, 0x19192b1908080819UL, 0x19192b1908081908UL, 0x19192b1908190808UL,
    0x19192b1919080808UL, 0x19192b2b08080808UL, 0x19192b2b08192b19UL, 0x19192b2b2b081919UL,
    0x19192b2b2b2b2b08UL, 0x192b080808080819UL, 0x192b080808081908UL, 0x192b08080808192bUL,
    0x192b080808190808UL, 0x192b08080819082bUL, 0x192b080808191919UL, 0x192b080808192b08UL,
    0x192b0808082b0819UL, 0x192b0808082b1908UL, 0x192b080819080808UL, 0x192b080819081919UL,
    0x192b080819082b08UL, 0x192b080819190819UL, 0x192b080819191908UL, 0x192b0808192b0808UL,
    0x192b08082b081908UL, 0x192b08082b190808UL, 0x192b081908080808UL, 0x192b08190808082bUL,
    0x192b081908081919UL, 0x192b081908082b08UL, 0x192b081908190819UL, 0x192b081908191908UL,
    0x192b0819082b0808UL, 0x192b081919080819UL, 0x192b081919081908UL, 0x192b081919190808UL,
    0x192b08192b080808UL, 0x192b08192b192b19UL, 0x192b082b08081908UL, 0x192b082b08190808UL,
    0x192b082b19080808UL, 0x192b082b1919192bUL, 0x192b082b2b2b0819UL, 0x192b190808080808UL,
    0x192b190808081919UL, 0x192b190808082b08UL, 0x192b190808190819UL, 0x192b190808191908UL,
    0x192b1908082b0808UL, 0x192b190819080819UL, 0x192b190819081908UL, 0x192b190819190808UL,
    0x192b19082b080808UL, 0x192b191908080819UL, 0x192b191908081908UL, 0x192b191908190808UL,
    0x192b191919080808UL, 0x192b191919082b2bUL, 0x192b1919192b2b08UL, 0x192b19192b19082bUL,
    0x192b192b08080808UL, 0x192b192b2b191908UL, 0x192b2b0808080819UL, 0x192b2b0808081908UL,
    0x192b2b0808190808UL, 0x192b2b08192b1919UL, 0x192b2b082b192b08UL, 0x192b2b1908080808UL,
    0x192b2b19082b2b2bUL, 0x192b2b2b1908082bUL, 0x192b2b2b2b2b0819UL, 0x2b08080808080808UL,
    0x2b0808080808082bUL, 0x2b08080808081919UL, 0x2b08080808082b08UL, 0x2b08080808190819UL,
    0x2b08080808191908UL, 0x2b08080808192b19UL, 0x2b080808082b0808UL, 0x2b080808082b1919UL,
    0x2b08080819080819UL, 0x2b08080819081908UL, 0x2b08080819190808UL, 0x2b0808081919082bUL,
    0x2b08080819191919UL, 0x2b08080819192b08UL, 0x2b080808192b0819UL, 0x2b0808082b080808UL,
    0x2b0808082b081919UL, 0x2b0808082b190819UL, 0x2b0808082b191908UL, 0x2b08081908080819UL,
    0x2b08081908081908UL, 0x2b08081908082b19UL, 0x2b08081908190808UL, 0x2b0808190819082bUL,
    0x2b08081908191919UL, 0x2b08081908192b08UL, 0x2b080819082b0819UL, 0x2b080819082b1908UL,
    0x2b08081919080808UL, 0x2b0808191908082bUL, 0x2b08081919081919UL, 0x2b08081919082b08UL,
    0x2b08081919190819UL, 0x2b08081919191908UL, 0x2b0808192b080819UL, 0x2b0808192b081908UL,
    0x2b0808192b190808UL, 0x2b0808192b2b2b19UL, 0x2b08082b08080808UL, 0x2b08082b08081919UL,
    0x2b08082b08082b2bUL, 0x2b08082b08190819UL, 0x2b08082b08191908UL, 0x2b08082b19080819UL,
    0x2b08082b19081908UL, 0x2b08082b19190808UL, 0x2b08190808080819UL, 0x2b08190808081908UL,
    0x2b0819080808192bUL, 0x2b08190808082b19UL, 0x2b08190808190808UL, 0x2b0819080819082bUL,
    0x2b08190808191919UL, 0x2b08190808192b08UL, 0x2b081908082b0819UL, 0x2b08190819080808UL,
    0x2b0819081908082bUL, 0x2b08190819081919UL, 0x2b08190819082b08UL, 0x2b08190819190819UL,
    0x2b08190819191908UL, 0x2b081908192b0808UL, 0x2b0819082b080819UL, 0x2b0819082b081908UL,
    0x2b0819082b190808UL, 0x2b08191908080808UL, 0x2b0819190808082bUL, 0x2b08191908081919UL,
    0x2b08191908082b08UL, 0x2b08191908190819UL, 0x2b08191908191908UL, 0x2b081919082b0808UL,
    0x2b08191919080819UL, 0x2b08191919081908UL, 0x2b08191919190808UL, 0x2b0819192b080808UL,
    0x2b0819192b082b2bUL, 0x2b08192b08080819UL, 0x2b08192b08081908UL, 0x2b08192b08190808UL,
    0x2b08192b082b2b19UL, 0x2b08192b19080808UL, 0x2b082b0808080808UL, 0x2b082b0808081919UL,
    0x2b082b0808190819UL, 0x2b082b0808191908UL, 0x2b082b0819080819UL, 0x2b082b0819081908UL,
    0x2b082b0819190808UL, 0x2b082b082b2b082bUL, 0x2b082b1908080819UL, 0x2b082b1908081908UL,
    0x2b082b1919080808UL, 0x2b082b19192b1919UL, 0x2b082b2b082b082bUL, 0x2b082b2b19192b08UL,
    0x2b082b2b19192b2bUL, 0x2b082b2b2b08082bUL, 0x2b082b2b2b2b082bUL, 0x2b19080808080819UL,
    0x2b19080808081908UL, 0x2b19080808082b19UL, 0x2b19080808190808UL, 0x2b1908080819082bUL,
    0x2b19080808191919UL, 0x2b19080808192b08UL, 0x2b190808082b1908UL, 0x2b19080819080808UL,
    0x2b1908081908082bUL, 0x2b19080819081919UL, 0x2b19080819082b08UL, 0x2b19080819190819UL,
    0x2b19080819191908UL, 0x2b190808192b0808UL, 0x2b1908082b080819UL, 0x2b1908082b081908UL,
    0x2b1908082b190808UL, 0x2b19081908080808UL, 0x2b19081908081919UL, 0x2b19081908190819UL,
    0x2b19081908191908UL, 0x2b19081919080819UL, 0x2b19081919081908UL, 0x2b19081919190808UL,
    0x2b19081919192b2bUL, 0x2b19082b08080819UL, 0x2b19082b08081908UL, 0x2b19082b08190808UL,
    0x2b19082b19080808UL, 0x2b19082b2b2b192bUL, 0x2b19190808080808UL, 0x2b1919080808082bUL,
    0x2b19190808081919UL, 0x2b19190808082b08UL, 0x2b19190808190819UL, 0x2b19190808191908UL,
    0x2b191908082b0808UL, 0x2b19190819080819UL, 0x2b19190819081908UL, 0x2b19190819190808UL,
    0x2b1919082b080808UL, 0x2b1919082b19192bUL, 0x2b19191908080819UL, 0x2b19191908081908UL,
    0x2b19191908190808UL, 0x2b19191919080808UL, 0x2b1919192b192b08UL, 0x2b1919192b2b0819UL,
    0x2b19192b08080808UL, 0x2b19192b1908192bUL, 0x2b19192b192b1908UL, 0x2b192b0808080819UL,
    0x2b192b0808081908UL, 0x2b192b0808190808UL, 0x2b192b08082b192bUL, 0x2b192b0819080808UL,
    0x2b192b082b2b2b19UL, 0x2b192b1908080808UL, 0x2b192b1919082b19UL, 0x2b192b191919082bUL,
    0x2b192b2b2b190808UL, 0x2b2b080808080808UL, 0x2b2b080808081919UL, 0x2b2b080808082b2bUL,
    0x2b2b080808191908UL, 0x2b2b0808082b082bUL, 0x2b2b0808082b2b2bUL, 0x2b2b080819080819UL,
    0x2b2b080819081908UL, 0x2b2b080819190808UL, 0x2b2b08082b2b082bUL, 0x2b2b08082b2b2b2bUL,
    0x2b2b081919080808UL, 0x2b2b0819192b1919UL, 0x2b2b082b0808082bUL, 0x2b2b082b08082b2bUL,
    0x2b2b082b082b082bUL, 0x2b2b082b082b2b08UL, 0x2b2b082b082b2b2bUL, 0x2b2b082b2b08082bUL,
    0x2b2b082b2b082b08UL, 0x2b2b082b2b082b2bUL, 0x2b2b082b2b2b2b08UL, 0x2b2b190808080819UL,
    0x2b2b190808081908UL, 0x2b2b190808190808UL, 0x2b2b190819080808UL, 0x2b2b19082b082b19UL,
    0x2b2b19082b2b1908UL, 0x2b2b191908080808UL, 0x2b2b191908192b19UL, 0x2b2b192b19190819UL,
    0x2b2b2b0808082b2bUL, 0x2b2b2b08082b2b08UL, 0x2b2b2b082b2b082bUL, 0x2b2b2b1919191908UL,
    0x2b2b2b192b08192bUL, 0x2b2b2b2b08082b08UL, 0x2b2b2b2b08082b2bUL, 0x2b2b2b2b082b0808UL,
    0x2b2b2b2b082b082bUL, 0x2b2b2b2b082b2b08UL, 0x2b2b2b2b2b082b08UL, 0x2b2b2b2b2b2b2b2bUL,
};

constant ulong iq1s_grid[2048] = {
    0xffffffffffffffffUL, 0xffffffffffffff01UL, 0xffffffffffff0000UL, 0xffffffffffff01ffUL,
    0xffffffffffff0101UL, 0xffffffffff00ff00UL, 0xffffffffff000000UL, 0xffffffffff01ffffUL,
    0xffffffffff01ff01UL, 0xffffffffff0101ffUL, 0xffffffffff010101UL, 0xffffffff00ff0000UL,
    0xffffffff0000ff00UL, 0xffffffff000000ffUL, 0xffffffff00000001UL, 0xffffffff00010000UL,
    0xffffffff01ffffffUL, 0xffffffff01ffff01UL, 0xffffffff01ff01ffUL, 0xffffffff01ff0101UL,
    0xffffffff01000000UL, 0xffffffff0101ffffUL, 0xffffffff0101ff01UL, 0xffffffff010101ffUL,
    0xffffffff01010101UL, 0xffffff00ffff00ffUL, 0xffffff00ffff0000UL, 0xffffff00ff00ff00UL,
    0xffffff00ff0000ffUL, 0xffffff00ff000001UL, 0xffffff00ff000100UL, 0xffffff00ff000101UL,
    0xffffff00ff010000UL, 0xffffff0000ffff00UL, 0xffffff0000ff0001UL, 0xffffff0000ff0100UL,
    0xffffff000000ff01UL, 0xffffff0000000000UL, 0xffffff0000000101UL, 0xffffff000001ff00UL,
    0xffffff00000100ffUL, 0xffffff0000010001UL, 0xffffff00000101ffUL, 0xffffff0001ff0000UL,
    0xffffff000100ff00UL, 0xffffff00010000ffUL, 0xffffff0001000001UL, 0xffffff0001010000UL,
    0xffffff01ffffffffUL, 0xffffff01ffffff01UL, 0xffffff01ffff01ffUL, 0xffffff01ffff0101UL,
    0xffffff01ff000000UL, 0xffffff01ff01ffffUL, 0xffffff01ff01ff01UL, 0xffffff01ff0101ffUL,
    0xffffff01ff010101UL, 0xffffff0100ff0000UL, 0xffffff010000ff00UL, 0xffffff0100000100UL,
    0xffffff01000100ffUL, 0xffffff0100010100UL, 0xffffff0101ffffffUL, 0xffffff0101ffff01UL,
    0xffffff0101ff01ffUL, 0xffffff0101ff0101UL, 0xffffff010100ff00UL, 0xffffff0101000000UL,
    0xffffff0101000100UL, 0xffffff010101ffffUL, 0xffffff010101ff01UL, 0xffffff01010101ffUL,
    0xffffff0101010101UL, 0xffff00ffff00ff00UL, 0xffff00ffff0000ffUL, 0xffff00ffff000001UL,
    0xffff00ffff010000UL, 0xffff00ff00ffff00UL, 0xffff00ff00ff0100UL, 0xffff00ff00000000UL,
    0xffff00ff00000101UL, 0xffff00ff000100ffUL, 0xffff00ff00010000UL, 0xffff00ff0100ff00UL,
    0xffff00ff01000100UL, 0xffff00ff01010000UL, 0xffff0000ffffff00UL, 0xffff0000ffff00ffUL,
    0xffff0000ffff0000UL, 0xffff0000ffff0001UL, 0xffff0000ff000000UL, 0xffff0000ff0001ffUL,
    0xffff0000ff000101UL, 0xffff0000ff010100UL, 0xffff000000ffffffUL, 0xffff000000ff0000UL,
    0xffff000000ff0101UL, 0xffff00000000ffffUL, 0xffff00000000ff00UL, 0xffff0000000000ffUL,
    0xffff000000000000UL, 0xffff000000000001UL, 0xffff000000000100UL, 0xffff00000001ffffUL,
    0xffff00000001ff01UL, 0xffff000000010000UL, 0xffff0000000101ffUL, 0xffff000000010101UL,
    0xffff000001ffff00UL, 0xffff00000100ff00UL, 0xffff000001000000UL, 0xffff0000010001ffUL,
    0xffff000001000101UL, 0xffff00000101ff00UL, 0xffff0000010100ffUL, 0xffff000001010000UL,
    0xffff000001010001UL, 0xffff000001010100UL, 0xffff0001ff0000ffUL, 0xffff0001ff000100UL,
    0xffff000100ffff00UL, 0xffff000100ff00ffUL, 0xffff00010000ffffUL, 0xffff00010000ff01UL,
    0xffff000100000000UL, 0xffff0001000001ffUL, 0xffff00010001ffffUL, 0xffff00010001ff00UL,
    0xffff000100010001UL, 0xffff000100010100UL, 0xffff000101ff0000UL, 0xffff00010100ff00UL,
    0xffff0001010000ffUL, 0xffff000101000100UL, 0xffff01ffffffffffUL, 0xffff01ffffffff01UL,
    0xffff01ffffff01ffUL, 0xffff01ffffff0101UL, 0xffff01ffff000000UL, 0xffff01ffff01ffffUL,
    0xffff01ffff01ff01UL, 0xffff01ffff0101ffUL, 0xffff01ffff010101UL, 0xffff01ff00ff0000UL,
    0xffff01ff0000ff00UL, 0xffff01ff00000001UL, 0xffff01ff00010000UL, 0xffff01ff01ffffffUL,
    0xffff01ff01ffff01UL, 0xffff01ff01ff01ffUL, 0xffff01ff01ff0101UL, 0xffff01ff01000000UL,
    0xffff01ff0101ffffUL, 0xffff01ff0101ff01UL, 0xffff01ff010101ffUL, 0xffff01ff01010101UL,
    0xffff0100ffff0000UL, 0xffff0100ff00ff00UL, 0xffff0100ff0000ffUL, 0xffff0100ff000100UL,
    0xffff0100ff0100ffUL, 0xffff0100ff010000UL, 0xffff010000ffff00UL, 0xffff01000000ffffUL,
    0xffff01000000ff00UL, 0xffff010000000000UL, 0xffff01000001ff00UL, 0xffff0100000100ffUL,
    0xffff010000010100UL, 0xffff01000100ff00UL, 0xffff0100010000ffUL, 0xffff010001000001UL,
    0xffff010001000100UL, 0xffff010001010000UL, 0xffff0101ffffffffUL, 0xffff0101ffffff01UL,
    0xffff0101ffff01ffUL, 0xffff0101ffff0101UL, 0xffff0101ff000000UL, 0xffff0101ff01ffffUL,
    0xffff0101ff01ff01UL, 0xffff0101ff0101ffUL, 0xffff0101ff010101UL, 0xffff010100ff0000UL,
    0xffff01010000ff00UL, 0xffff010100000100UL, 0xffff01010001ff00UL, 0xffff010100010000UL,
    0xffff010101ffffffUL, 0xffff010101ffff01UL, 0xffff010101ff0000UL, 0xffff010101ff01ffUL,
    0xffff010101ff0101UL, 0xffff010101000000UL, 0xffff01010101ffffUL, 0xffff01010101ff01UL,
    0xffff0101010101ffUL, 0xffff010101010101UL, 0xff00ffffff00ffffUL, 0xff00ffffff00ff00UL,
    0xff00ffffff0000ffUL, 0xff00ffffff000100UL, 0xff00ffffff0100ffUL, 0xff00ffffff010000UL,
    0xff00ffff00ffff00UL, 0xff00ffff00ff00ffUL, 0xff00ffff0000ffffUL, 0xff00ffff00000000UL,
    0xff00ffff000001ffUL, 0xff00ffff0001ff00UL, 0xff00ffff000100ffUL, 0xff00ffff00010000UL,
    0xff00ffff00010100UL, 0xff00ffff0100ff00UL, 0xff00ffff010000ffUL, 0xff00ffff01000001UL,
    0xff00ffff0101ff00UL, 0xff00ffff01010000UL, 0xff00ff00ffffff00UL, 0xff00ff00ffff00ffUL,
    0xff00ff00ffff0001UL, 0xff00ff00ffff0100UL, 0xff00ff00ff00ffffUL, 0xff00ff00ff00ff01UL,
    0xff00ff00ff000000UL, 0xff00ff00ff0001ffUL, 0xff00ff00ff01ff00UL, 0xff00ff00ff0100ffUL,
    0xff00ff00ff010100UL, 0xff00ff0000ff0000UL, 0xff00ff0000ff0101UL, 0xff00ff000000ffffUL,
    0xff00ff000000ff00UL, 0xff00ff000000ff01UL, 0xff00ff00000000ffUL, 0xff00ff0000000000UL,
    0xff00ff0000000001UL, 0xff00ff0000000100UL, 0xff00ff000001ffffUL, 0xff00ff0000010000UL,
    0xff00ff0001ff00ffUL, 0xff00ff000100ff01UL, 0xff00ff0001000000UL, 0xff00ff000101ff00UL,
    0xff00ff00010100ffUL, 0xff00ff01ff00ff00UL, 0xff00ff01ff0000ffUL, 0xff00ff01ff000001UL,
    0xff00ff01ff010000UL, 0xff00ff0100ffffffUL, 0xff00ff0100ff0001UL, 0xff00ff0100ff0100UL,
    0xff00ff010000ff01UL, 0xff00ff0100000000UL, 0xff00ff01000001ffUL, 0xff00ff0100000101UL,
    0xff00ff01000100ffUL, 0xff00ff0100010001UL, 0xff00ff0101ff0000UL, 0xff00ff010100ff00UL,
    0xff00ff01010000ffUL, 0xff00ff0101000001UL, 0xff00ff0101010000UL, 0xff0000ffffffff00UL,
    0xff0000ffffff0001UL, 0xff0000ffffff0100UL, 0xff0000ffff0000ffUL, 0xff0000ffff000000UL,
    0xff0000ffff0001ffUL, 0xff0000ffff000100UL, 0xff0000ffff01ff00UL, 0xff0000ffff010001UL,
    0xff0000ff00ffff00UL, 0xff0000ff00ff0000UL, 0xff0000ff00ff0001UL, 0xff0000ff00ff01ffUL,
    0xff0000ff00ff0101UL, 0xff0000ff0000ff00UL, 0xff0000ff000000ffUL, 0xff0000ff00000000UL,
    0xff0000ff00000001UL, 0xff0000ff00000100UL, 0xff0000ff0001ff01UL, 0xff0000ff00010000UL,
    0xff0000ff000101ffUL, 0xff0000ff01ff00ffUL, 0xff0000ff01ff0100UL, 0xff0000ff0100ffffUL,
    0xff0000ff010000ffUL, 0xff0000ff01000000UL, 0xff0000ff010001ffUL, 0xff0000ff01000100UL,
    0xff0000ff01000101UL, 0xff0000ff0101ff00UL, 0xff0000ff010100ffUL, 0xff0000ff01010000UL,
    0xff0000ff01010100UL, 0xff000000ffffff01UL, 0xff000000ffff0000UL, 0xff000000ffff0101UL,
    0xff000000ff00ff00UL, 0xff000000ff0000ffUL, 0xff000000ff000000UL, 0xff000000ff000001UL,
    0xff000000ff000100UL, 0xff000000ff01ffffUL, 0xff000000ff01ff01UL, 0xff000000ff010000UL,
    0xff000000ff0101ffUL, 0xff000000ff010101UL, 0xff00000000ffff00UL, 0xff00000000ff00ffUL,
    0xff00000000ff0000UL, 0xff00000000ff0001UL, 0xff0000000000ff00UL, 0xff0000000000ff01UL,
    0xff000000000000ffUL, 0xff00000000000000UL, 0xff00000000000001UL, 0xff00000000000100UL,
    0xff00000000000101UL, 0xff0000000001ff00UL, 0xff000000000100ffUL, 0xff00000000010000UL,
    0xff00000000010001UL, 0xff00000000010100UL, 0xff00000001ffffffUL, 0xff00000001ffff01UL,
    0xff00000001ff00ffUL, 0xff00000001ff0000UL, 0xff00000001ff01ffUL, 0xff00000001ff0101UL,
    0xff0000000100ffffUL, 0xff0000000100ff00UL, 0xff000000010000ffUL, 0xff00000001000000UL,
    0xff00000001000001UL, 0xff00000001000100UL, 0xff00000001000101UL, 0xff0000000101ffffUL,
    0xff0000000101ff01UL, 0xff00000001010000UL, 0xff000001ffffff00UL, 0xff000001ffff00ffUL,
    0xff000001ffff0000UL, 0xff000001ffff0001UL, 0xff000001ff000000UL, 0xff000001ff000001UL,
    0xff000001ff0001ffUL, 0xff000001ff000101UL, 0xff000001ff01ff00UL, 0xff000001ff010001UL,
    0xff00000100ffffffUL, 0xff00000100ffff01UL, 0xff00000100ff00ffUL, 0xff00000100ff0000UL,
    0xff00000100ff01ffUL, 0xff00000100ff0101UL, 0xff0000010000ff00UL, 0xff00000100000000UL,
    0xff00000100000001UL, 0xff000001000001ffUL, 0xff00000100000100UL, 0xff0000010001ff00UL,
    0xff000001000100ffUL, 0xff00000100010000UL, 0xff000001000101ffUL, 0xff00000100010100UL,
    0xff00000100010101UL, 0xff00000101ff0001UL, 0xff00000101ff0101UL, 0xff0000010100ff01UL,
    0xff00000101000000UL, 0xff000001010100ffUL, 0xff00000101010100UL, 0xff0001ffff00ff00UL,
    0xff0001ffff000001UL, 0xff0001ffff010000UL, 0xff0001ff00ffff00UL, 0xff0001ff00ff00ffUL,
    0xff0001ff00ff0001UL, 0xff0001ff00ff0100UL, 0xff0001ff0000ffffUL, 0xff0001ff00000000UL,
    0xff0001ff000001ffUL, 0xff0001ff00000101UL, 0xff0001ff0001ffffUL, 0xff0001ff0001ff00UL,
    0xff0001ff000100ffUL, 0xff0001ff00010001UL, 0xff0001ff00010100UL, 0xff0001ff01ff0000UL,
    0xff0001ff0100ff00UL, 0xff0001ff010000ffUL, 0xff0001ff01010000UL, 0xff000100ff00ffffUL,
    0xff000100ff00ff01UL, 0xff000100ff000000UL, 0xff000100ff000101UL, 0xff000100ff01ff00UL,
    0xff000100ff010000UL, 0xff00010000ffff01UL, 0xff00010000ff00ffUL, 0xff00010000ff0000UL,
    0xff00010000ff01ffUL, 0xff0001000000ff00UL, 0xff000100000000ffUL, 0xff00010000000000UL,
    0xff00010000000001UL, 0xff00010000000100UL, 0xff00010000000101UL, 0xff0001000001ffffUL,
    0xff00010000010000UL, 0xff00010000010101UL, 0xff00010001ff0100UL, 0xff0001000100ff00UL,
    0xff0001000100ff01UL, 0xff00010001000000UL, 0xff000100010001ffUL, 0xff0001000101ff00UL,
    0xff00010001010001UL, 0xff00010001010100UL, 0xff000101ffff0100UL, 0xff000101ff000001UL,
    0xff000101ff0100ffUL, 0xff000101ff010001UL, 0xff00010100ff00ffUL, 0xff00010100ff0001UL,
    0xff00010100ff0100UL, 0xff0001010000ffffUL, 0xff0001010000ff01UL, 0xff00010100000000UL,
    0xff000101000001ffUL, 0xff0001010001ff00UL, 0xff00010100010001UL, 0xff00010100010100UL,
    0xff00010101ff0000UL, 0xff0001010100ff00UL, 0xff00010101000001UL, 0xff00010101000101UL,
    0xff01ffffffffffffUL, 0xff01ffffffffff01UL, 0xff01ffffffff01ffUL, 0xff01ffffffff0101UL,
    0xff01ffffff000000UL, 0xff01ffffff01ffffUL, 0xff01ffffff01ff01UL, 0xff01ffffff010000UL,
    0xff01ffffff0101ffUL, 0xff01ffffff010101UL, 0xff01ffff00ff0000UL, 0xff01ffff0000ff00UL,
    0xff01ffff00000100UL, 0xff01ffff0001ff00UL, 0xff01ffff00010000UL, 0xff01ffff01ffffffUL,
    0xff01ffff01ffff01UL, 0xff01ffff01ff01ffUL, 0xff01ffff01ff0101UL, 0xff01ffff01000000UL,
    0xff01ffff0101ffffUL, 0xff01ffff0101ff01UL, 0xff01ffff01010000UL, 0xff01ffff010101ffUL,
    0xff01ffff01010101UL, 0xff01ff00ffff0000UL, 0xff01ff00ff00ff00UL, 0xff01ff00ff0000ffUL,
    0xff01ff00ff000100UL, 0xff01ff00ff010000UL, 0xff01ff0000ffff01UL, 0xff01ff0000ff00ffUL,
    0xff01ff0000ff0100UL, 0xff01ff0000000000UL, 0xff01ff00000001ffUL, 0xff01ff0000000101UL,
    0xff01ff000001ff00UL, 0xff01ff00000100ffUL, 0xff01ff0000010000UL, 0xff01ff0000010001UL,
    0xff01ff0001ff0000UL, 0xff01ff000100ffffUL, 0xff01ff0001000001UL, 0xff01ff0001000100UL,
    0xff01ff0001010000UL, 0xff01ff01ffffff00UL, 0xff01ff01ffff01ffUL, 0xff01ff01ffff0101UL,
    0xff01ff01ff00ff00UL, 0xff01ff01ff000000UL, 0xff01ff01ff01ffffUL, 0xff01ff01ff01ff01UL,
    0xff01ff01ff0101ffUL, 0xff01ff01ff010101UL, 0xff01ff0100ff0000UL, 0xff01ff010000ff00UL,
    0xff01ff0100000001UL, 0xff01ff0100000100UL, 0xff01ff0100010000UL, 0xff01ff0101ffff00UL,
    0xff01ff0101ff01ffUL, 0xff01ff0101ff0101UL, 0xff01ff010100ff00UL, 0xff01ff0101000000UL,
    0xff01ff010101ffffUL, 0xff01ff010101ff01UL, 0xff01ff01010101ffUL, 0xff01ff0101010101UL,
    0xff0100ffffff0000UL, 0xff0100ffff0000ffUL, 0xff0100ffff000001UL, 0xff0100ffff000100UL,
    0xff0100ffff010000UL, 0xff0100ff00ff00ffUL, 0xff0100ff00ff0000UL, 0xff0100ff00ff0001UL,
    0xff0100ff00ff0100UL, 0xff0100ff0000ff01UL, 0xff0100ff00000000UL, 0xff0100ff000001ffUL,
    0xff0100ff00000101UL, 0xff0100ff00010001UL, 0xff0100ff01ff0000UL, 0xff0100ff0100ff00UL,
    0xff0100ff010000ffUL, 0xff0100ff01000100UL, 0xff0100ff0101ff00UL, 0xff0100ff01010000UL,
    0xff010000ffff0100UL, 0xff010000ff000000UL, 0xff010000ff01ff00UL, 0xff010000ff010100UL,
    0xff01000000ffffffUL, 0xff01000000ff0000UL, 0xff01000000ff01ffUL, 0xff0100000000ff00UL,
    0xff010000000000ffUL, 0xff01000000000000UL, 0xff01000000000100UL, 0xff0100000001ff01UL,
    0xff01000000010000UL, 0xff010000000101ffUL, 0xff01000001ff0100UL, 0xff0100000100ffffUL,
    0xff010000010000ffUL, 0xff01000001000000UL, 0xff010000010001ffUL, 0xff01000001000101UL,
    0xff0100000101ff00UL, 0xff010000010100ffUL, 0xff01000001010001UL, 0xff01000001010100UL,
    0xff010001ffff0000UL, 0xff010001ff00ffffUL, 0xff010001ff00ff01UL, 0xff010001ff000100UL,
    0xff010001ff010000UL, 0xff01000100ffff00UL, 0xff01000100ff0100UL, 0xff01000100000000UL,
    0xff0100010001ffffUL, 0xff0100010001ff00UL, 0xff01000100010100UL, 0xff01000101ff00ffUL,
    0xff01000101ff0001UL, 0xff0100010100ffffUL, 0xff01000101000101UL, 0xff0101ffffffffffUL,
    0xff0101ffffffff01UL, 0xff0101ffffff01ffUL, 0xff0101ffffff0101UL, 0xff0101ffff000000UL,
    0xff0101ffff01ffffUL, 0xff0101ffff01ff01UL, 0xff0101ffff0101ffUL, 0xff0101ffff010101UL,
    0xff0101ff00ff0000UL, 0xff0101ff0000ff00UL, 0xff0101ff000000ffUL, 0xff0101ff00010000UL,
    0xff0101ff01ffffffUL, 0xff0101ff01ffff01UL, 0xff0101ff01ff01ffUL, 0xff0101ff01ff0101UL,
    0xff0101ff0101ffffUL, 0xff0101ff0101ff01UL, 0xff0101ff010101ffUL, 0xff0101ff01010101UL,
    0xff010100ffff0100UL, 0xff010100ff00ff00UL, 0xff010100ff0000ffUL, 0xff010100ff000100UL,
    0xff010100ff010000UL, 0xff01010000ff0001UL, 0xff01010000ff0100UL, 0xff0101000000ff01UL,
    0xff01010000000000UL, 0xff0101000001ff00UL, 0xff010100000100ffUL, 0xff01010000010001UL,
    0xff01010000010100UL, 0xff01010001ff0000UL, 0xff0101000100ffffUL, 0xff01010001000001UL,
    0xff01010001000100UL, 0xff010100010100ffUL, 0xff01010001010000UL, 0xff010101ffffffffUL,
    0xff010101ffffff01UL, 0xff010101ffff01ffUL, 0xff010101ffff0101UL, 0xff010101ff01ffffUL,
    0xff010101ff01ff01UL, 0xff010101ff0101ffUL, 0xff010101ff010101UL, 0xff01010100ff0000UL,
    0xff0101010000ff00UL, 0xff01010100000001UL, 0xff01010100000100UL, 0xff01010100010000UL,
    0xff01010101ffffffUL, 0xff01010101ffff01UL, 0xff01010101ff01ffUL, 0xff01010101ff0101UL,
    0xff01010101000000UL, 0xff0101010101ffffUL, 0xff0101010101ff01UL, 0xff010101010101ffUL,
    0xff01010101010101UL, 0x00ffffffffff0000UL, 0x00ffffffff00ff00UL, 0x00ffffffff000001UL,
    0x00ffffffff010000UL, 0x00ffffff00ff0100UL, 0x00ffffff0000ff01UL, 0x00ffffff00000000UL,
    0x00ffffff000001ffUL, 0x00ffffff00000101UL, 0x00ffffff0001ff00UL, 0x00ffffff000100ffUL,
    0x00ffffff00010001UL, 0x00ffffff010000ffUL, 0x00ffffff01000100UL, 0x00ffffff0101ff00UL,
    0x00ffffff01010001UL, 0x00ffff00ffffffffUL, 0x00ffff00ffffff00UL, 0x00ffff00ffff00ffUL,
    0x00ffff00ffff0001UL, 0x00ffff00ffff0100UL, 0x00ffff00ff00ff01UL, 0x00ffff00ff000000UL,
    0x00ffff00ff000001UL, 0x00ffff00ff0001ffUL, 0x00ffff00ff000101UL, 0x00ffff00ff01ff00UL,
    0x00ffff00ff010001UL, 0x00ffff00ff010100UL, 0x00ffff0000ff0000UL, 0x00ffff0000ff01ffUL,
    0x00ffff0000ff0101UL, 0x00ffff000000ff00UL, 0x00ffff00000000ffUL, 0x00ffff0000000000UL,
    0x00ffff0000000001UL, 0x00ffff0000000100UL, 0x00ffff0000000101UL, 0x00ffff0000010000UL,
    0x00ffff00000101ffUL, 0x00ffff0000010101UL, 0x00ffff0001ffff00UL, 0x00ffff0001ff00ffUL,
    0x00ffff0001ff0001UL, 0x00ffff000100ffffUL, 0x00ffff000100ff01UL, 0x00ffff0001000000UL,
    0x00ffff000101ffffUL, 0x00ffff000101ff00UL, 0x00ffff000101ff01UL, 0x00ffff01ffff0000UL,
    0x00ffff01ff00ff00UL, 0x00ffff01ff0000ffUL, 0x00ffff01ff000001UL, 0x00ffff01ff010000UL,
    0x00ffff0100ffff00UL, 0x00ffff010000ff01UL, 0x00ffff0100000000UL, 0x00ffff0100000101UL,
    0x00ffff01000100ffUL, 0x00ffff0100010100UL, 0x00ffff0101ff0100UL, 0x00ffff01010000ffUL,
    0x00ffff0101010000UL, 0x00ff00ffffffff00UL, 0x00ff00ffff000000UL, 0x00ff00ffff000100UL,
    0x00ff00ffff010100UL, 0x00ff00ff00ff0000UL, 0x00ff00ff00ff01ffUL, 0x00ff00ff00ff0101UL,
    0x00ff00ff0000ff00UL, 0x00ff00ff000000ffUL, 0x00ff00ff00000000UL, 0x00ff00ff00000001UL,
    0x00ff00ff0001ff00UL, 0x00ff00ff0001ff01UL, 0x00ff00ff00010000UL, 0x00ff00ff000101ffUL,
    0x00ff00ff00010101UL, 0x00ff00ff01ffff00UL, 0x00ff00ff01ff0001UL, 0x00ff00ff01ff0100UL,
    0x00ff00ff0100ffffUL, 0x00ff00ff0100ff01UL, 0x00ff00ff01000000UL, 0x00ff00ff0101ffffUL,
    0x00ff00ff0101ff00UL, 0x00ff00ff01010100UL, 0x00ff0000ffffff00UL, 0x00ff0000ffffff01UL,
    0x00ff0000ffff0000UL, 0x00ff0000ffff0101UL, 0x00ff0000ff00ff00UL, 0x00ff0000ff0000ffUL,
    0x00ff0000ff000000UL, 0x00ff0000ff000001UL, 0x00ff0000ff000100UL, 0x00ff0000ff01ffffUL,
    0x00ff0000ff010000UL, 0x00ff0000ff010101UL, 0x00ff000000ffff00UL, 0x00ff000000ff00ffUL,
    0x00ff000000ff0000UL, 0x00ff000000ff0001UL, 0x00ff000000ff0100UL, 0x00ff00000000ffffUL,
    0x00ff00000000ff00UL, 0x00ff0000000000ffUL, 0x00ff000000000000UL, 0x00ff000000000001UL,
    0x00ff0000000001ffUL, 0x00ff000000000100UL, 0x00ff00000001ff00UL, 0x00ff0000000100ffUL,
    0x00ff000000010000UL, 0x00ff000000010001UL, 0x00ff000000010100UL, 0x00ff000001ffff01UL,
    0x00ff000001ff00ffUL, 0x00ff000001ff0000UL, 0x00ff000001ff01ffUL, 0x00ff00000100ff00UL,
    0x00ff0000010000ffUL, 0x00ff000001000000UL, 0x00ff000001000001UL, 0x00ff000001000100UL,
    0x00ff000001000101UL, 0x00ff000001010000UL, 0x00ff0000010101ffUL, 0x00ff000001010101UL,
    0x00ff0001ffffff00UL, 0x00ff0001ffff0000UL, 0x00ff0001ffff0100UL, 0x00ff0001ff0000ffUL,
    0x00ff0001ff000000UL, 0x00ff0001ff0001ffUL, 0x00ff0001ff000101UL, 0x00ff0001ff01ff00UL,
    0x00ff0001ff0100ffUL, 0x00ff0001ff010100UL, 0x00ff000100ffffffUL, 0x00ff000100ffff01UL,
    0x00ff000100ff0000UL, 0x00ff000100ff01ffUL, 0x00ff00010000ffffUL, 0x00ff00010000ff00UL,
    0x00ff00010000ff01UL, 0x00ff000100000000UL, 0x00ff000100000001UL, 0x00ff000100000100UL,
    0x00ff00010001ff01UL, 0x00ff000100010000UL, 0x00ff0001000101ffUL, 0x00ff000101ffff00UL,
    0x00ff000101ff0000UL, 0x00ff000101ff0101UL, 0x00ff0001010000ffUL, 0x00ff000101000000UL,
    0x00ff00010101ff00UL, 0x00ff0001010100ffUL, 0x00ff000101010001UL, 0x00ff01ffffff0000UL,
    0x00ff01ffff00ff00UL, 0x00ff01ffff000000UL, 0x00ff01ffff000101UL, 0x00ff01ffff010000UL,
    0x00ff01ff00ffff01UL, 0x00ff01ff00ff0100UL, 0x00ff01ff0000ffffUL, 0x00ff01ff00000000UL,
    0x00ff01ff000001ffUL, 0x00ff01ff0001ff00UL, 0x00ff01ff000100ffUL, 0x00ff01ff00010001UL,
    0x00ff01ff00010100UL, 0x00ff01ff01ff0000UL, 0x00ff01ff0100ff00UL, 0x00ff01ff010000ffUL,
    0x00ff01ff01000001UL, 0x00ff01ff01000100UL, 0x00ff01ff01010000UL, 0x00ff0100ffffff00UL,
    0x00ff0100ffff0000UL, 0x00ff0100ffff0001UL, 0x00ff0100ffff0101UL, 0x00ff0100ff00ffffUL,
    0x00ff0100ff0000ffUL, 0x00ff0100ff000000UL, 0x00ff0100ff0001ffUL, 0x00ff0100ff01ff00UL,
    0x00ff0100ff0100ffUL, 0x00ff0100ff010001UL, 0x00ff010000ffffffUL, 0x00ff010000ff0000UL,
    0x00ff010000ff0101UL, 0x00ff01000000ff00UL, 0x00ff01000000ff01UL, 0x00ff0100000000ffUL,
    0x00ff010000000000UL, 0x00ff010000000001UL, 0x00ff010000000100UL, 0x00ff01000001ffffUL,
    0x00ff01000001ff01UL, 0x00ff010000010000UL, 0x00ff010000010001UL, 0x00ff010000010101UL,
    0x00ff010001ff0001UL, 0x00ff010001ff0100UL, 0x00ff01000100ff01UL, 0x00ff010001000000UL,
    0x00ff010001000001UL, 0x00ff0100010001ffUL, 0x00ff01000101ff00UL, 0x00ff0100010100ffUL,
    0x00ff010001010001UL, 0x00ff010001010100UL, 0x00ff0101ff000001UL, 0x00ff010100ff00ffUL,
    0x00ff010100ff0001UL, 0x00ff010100ff0100UL, 0x00ff010100000000UL, 0x00ff0101000001ffUL,
    0x00ff010100000101UL, 0x00ff0101000100ffUL, 0x00ff010100010100UL, 0x00ff0101010000ffUL,
    0x00ff010101010000UL, 0x0000ffffffffff00UL, 0x0000ffffffff00ffUL, 0x0000ffffffff0000UL,
    0x0000ffffffff0001UL, 0x0000ffffffff0100UL, 0x0000ffffff00ff01UL, 0x0000ffffff000000UL,
    0x0000ffffff000101UL, 0x0000ffffff01ff00UL, 0x0000ffffff0100ffUL, 0x0000ffffff010100UL,
    0x0000ffff00ffffffUL, 0x0000ffff00ff0000UL, 0x0000ffff00ff01ffUL, 0x0000ffff0000ff00UL,
    0x0000ffff000000ffUL, 0x0000ffff00000000UL, 0x0000ffff00000001UL, 0x0000ffff00000100UL,
    0x0000ffff00010000UL, 0x0000ffff000101ffUL, 0x0000ffff01ff0001UL, 0x0000ffff01ff0100UL,
    0x0000ffff01000000UL, 0x0000ffff010001ffUL, 0x0000ffff0101ffffUL, 0x0000ffff0101ff00UL,
    0x0000ffff01010001UL, 0x0000ffff01010100UL, 0x0000ff00ffff0000UL, 0x0000ff00ffff01ffUL,
    0x0000ff00ffff0100UL, 0x0000ff00ffff0101UL, 0x0000ff00ff00ff00UL, 0x0000ff00ff0000ffUL,
    0x0000ff00ff000000UL, 0x0000ff00ff000001UL, 0x0000ff00ff0001ffUL, 0x0000ff00ff000100UL,
    0x0000ff00ff01ffffUL, 0x0000ff00ff010000UL, 0x0000ff00ff010001UL, 0x0000ff00ff0101ffUL,
    0x0000ff00ff010101UL, 0x0000ff0000ffff00UL, 0x0000ff0000ff00ffUL, 0x0000ff0000ff0000UL,
    0x0000ff0000ff0001UL, 0x0000ff0000ff0100UL, 0x0000ff000000ffffUL, 0x0000ff000000ff00UL,
    0x0000ff000000ff01UL, 0x0000ff00000000ffUL, 0x0000ff0000000000UL, 0x0000ff0000000001UL,
    0x0000ff00000001ffUL, 0x0000ff0000000100UL, 0x0000ff0000000101UL, 0x0000ff000001ff00UL,
    0x0000ff00000100ffUL, 0x0000ff0000010000UL, 0x0000ff0000010001UL, 0x0000ff0000010100UL,
    0x0000ff0001ffff01UL, 0x0000ff0001ff0000UL, 0x0000ff000100ff00UL, 0x0000ff00010000ffUL,
    0x0000ff0001000000UL, 0x0000ff0001000001UL, 0x0000ff0001000100UL, 0x0000ff000101ffffUL,
    0x0000ff0001010000UL, 0x0000ff0001010101UL, 0x0000ff01ffffff00UL, 0x0000ff01ffff0001UL,
    0x0000ff01ff00ff01UL, 0x0000ff01ff000000UL, 0x0000ff01ff000101UL, 0x0000ff01ff01ff00UL,
    0x0000ff01ff0100ffUL, 0x0000ff0100ffff01UL, 0x0000ff0100ff0000UL, 0x0000ff0100ff0101UL,
    0x0000ff010000ff00UL, 0x0000ff01000000ffUL, 0x0000ff0100000000UL, 0x0000ff0100000001UL,
    0x0000ff0100000100UL, 0x0000ff010001ff01UL, 0x0000ff0100010000UL, 0x0000ff0101ff0000UL,
    0x0000ff010100ffffUL, 0x0000ff010100ff01UL, 0x0000ff0101000000UL, 0x0000ff0101000100UL,
    0x0000ff0101000101UL, 0x0000ff01010100ffUL, 0x000000ffffff00ffUL, 0x000000ffffff0000UL,
    0x000000ffff00ff00UL, 0x000000ffff0000ffUL, 0x000000ffff000000UL, 0x000000ffff000001UL,
    0x000000ffff0001ffUL, 0x000000ffff000100UL, 0x000000ffff01ff00UL, 0x000000ffff010000UL,
    0x000000ffff0101ffUL, 0x000000ffff010101UL, 0x000000ff00ffff00UL, 0x000000ff00ff00ffUL,
    0x000000ff00ff0000UL, 0x000000ff00ff0001UL, 0x000000ff00ff0100UL, 0x000000ff00ff0101UL,
    0x000000ff0000ffffUL, 0x000000ff0000ff00UL, 0x000000ff000000ffUL, 0x000000ff00000000UL,
    0x000000ff00000001UL, 0x000000ff000001ffUL, 0x000000ff00000100UL, 0x000000ff00000101UL,
    0x000000ff0001ff00UL, 0x000000ff0001ff01UL, 0x000000ff000100ffUL, 0x000000ff00010000UL,
    0x000000ff00010001UL, 0x000000ff00010100UL, 0x000000ff01ffffffUL, 0x000000ff01ff01ffUL,
    0x000000ff01ff0101UL, 0x000000ff0100ff00UL, 0x000000ff010000ffUL, 0x000000ff01000000UL,
    0x000000ff01000001UL, 0x000000ff01000100UL, 0x000000ff0101ff00UL, 0x000000ff010100ffUL,
    0x000000ff01010000UL, 0x000000ff01010101UL, 0x00000000ffffff00UL, 0x00000000ffffff01UL,
    0x00000000ffff00ffUL, 0x00000000ffff0000UL, 0x00000000ffff0001UL, 0x00000000ffff0100UL,
    0x00000000ff00ffffUL, 0x00000000ff00ff00UL, 0x00000000ff00ff01UL, 0x00000000ff0000ffUL,
    0x00000000ff000000UL, 0x00000000ff000001UL, 0x00000000ff000100UL, 0x00000000ff000101UL,
    0x00000000ff01ff00UL, 0x00000000ff0100ffUL, 0x00000000ff010000UL, 0x00000000ff010001UL,
    0x00000000ff010100UL, 0x0000000000ffffffUL, 0x0000000000ffff00UL, 0x0000000000ffff01UL,
    0x0000000000ff00ffUL, 0x0000000000ff0000UL, 0x0000000000ff0001UL, 0x0000000000ff01ffUL,
    0x0000000000ff0100UL, 0x000000000000ffffUL, 0x000000000000ff00UL, 0x000000000000ff01UL,
    0x00000000000000ffUL, 0x0000000000000000UL, 0x0000000000000001UL, 0x00000000000001ffUL,
    0x0000000000000100UL, 0x0000000000000101UL, 0x000000000001ffffUL, 0x000000000001ff00UL,
    0x00000000000100ffUL, 0x0000000000010000UL, 0x0000000000010001UL, 0x00000000000101ffUL,
    0x0000000000010100UL, 0x0000000000010101UL, 0x0000000001ffff00UL, 0x0000000001ff00ffUL,
    0x0000000001ff0000UL, 0x0000000001ff0100UL, 0x0000000001ff0101UL, 0x000000000100ffffUL,
    0x000000000100ff00UL, 0x00000000010000ffUL, 0x0000000001000000UL, 0x0000000001000001UL,
    0x00000000010001ffUL, 0x0000000001000100UL, 0x000000000101ff00UL, 0x00000000010100ffUL,
    0x0000000001010000UL, 0x0000000001010001UL, 0x0000000001010100UL, 0x00000001ffffffffUL,
    0x00000001ffffff00UL, 0x00000001ffffff01UL, 0x00000001ffff00ffUL, 0x00000001ffff0001UL,
    0x00000001ffff01ffUL, 0x00000001ffff0100UL, 0x00000001ff00ff00UL, 0x00000001ff0000ffUL,
    0x00000001ff000000UL, 0x00000001ff0001ffUL, 0x00000001ff000100UL, 0x00000001ff01ffffUL,
    0x00000001ff01ff00UL, 0x00000001ff01ff01UL, 0x00000001ff0100ffUL, 0x00000001ff010000UL,
    0x00000001ff010001UL, 0x00000001ff0101ffUL, 0x00000001ff010100UL, 0x0000000100ffff00UL,
    0x0000000100ff0000UL, 0x0000000100ff0001UL, 0x0000000100ff01ffUL, 0x0000000100ff0100UL,
    0x0000000100ff0101UL, 0x000000010000ffffUL, 0x000000010000ff00UL, 0x000000010000ff01UL,
    0x00000001000000ffUL, 0x0000000100000000UL, 0x0000000100000001UL, 0x00000001000001ffUL,
    0x0000000100000100UL, 0x0000000100000101UL, 0x000000010001ff00UL, 0x00000001000100ffUL,
    0x0000000100010000UL, 0x0000000100010100UL, 0x0000000101ffff01UL, 0x0000000101ff0000UL,
    0x0000000101ff0001UL, 0x0000000101ff01ffUL, 0x0000000101ff0100UL, 0x0000000101ff0101UL,
    0x000000010100ff00UL, 0x0000000101000000UL, 0x0000000101000101UL, 0x000000010101ff01UL,
    0x0000000101010000UL, 0x0000000101010001UL, 0x00000001010101ffUL, 0x0000000101010100UL,
    0x000001ffffff00ffUL, 0x000001ffffff0000UL, 0x000001ffffff0001UL, 0x000001ffffff0100UL,
    0x000001ffff00ffffUL, 0x000001ffff000000UL, 0x000001ffff0001ffUL, 0x000001ffff01ff00UL,
    0x000001ffff010101UL, 0x000001ff00ff0000UL, 0x000001ff00ff01ffUL, 0x000001ff00ff0101UL,
    0x000001ff0000ff00UL, 0x000001ff000000ffUL, 0x000001ff00000000UL, 0x000001ff00000001UL,
    0x000001ff000001ffUL, 0x000001ff00000100UL, 0x000001ff0001ffffUL, 0x000001ff0001ff01UL,
    0x000001ff000100ffUL, 0x000001ff00010000UL, 0x000001ff01ffff01UL, 0x000001ff01ff0100UL,
    0x000001ff0100ffffUL, 0x000001ff0100ff01UL, 0x000001ff01000000UL, 0x000001ff010001ffUL,
    0x000001ff0101ff00UL, 0x000001ff01010100UL, 0x00000100ffffff00UL, 0x00000100ffffff01UL,
    0x00000100ffff0000UL, 0x00000100ffff0101UL, 0x00000100ff00ff00UL, 0x00000100ff0000ffUL,
    0x00000100ff000000UL, 0x00000100ff000001UL, 0x00000100ff000100UL, 0x00000100ff010000UL,
    0x0000010000ffff00UL, 0x0000010000ff00ffUL, 0x0000010000ff0000UL, 0x0000010000ff0001UL,
    0x0000010000ff0100UL, 0x000001000000ffffUL, 0x000001000000ff00UL, 0x000001000000ff01UL,
    0x00000100000000ffUL, 0x0000010000000000UL, 0x0000010000000001UL, 0x00000100000001ffUL,
    0x0000010000000100UL, 0x0000010000000101UL, 0x000001000001ff00UL, 0x00000100000100ffUL,
    0x0000010000010000UL, 0x0000010000010001UL, 0x0000010000010100UL, 0x0000010001ffff00UL,
    0x0000010001ff0000UL, 0x0000010001ff0100UL, 0x000001000100ff00UL, 0x00000100010000ffUL,
    0x0000010001000000UL, 0x0000010001000001UL, 0x00000100010001ffUL, 0x0000010001000100UL,
    0x0000010001010000UL, 0x00000101ffff00ffUL, 0x00000101ffff01ffUL, 0x00000101ff000000UL,
    0x00000101ff000101UL, 0x00000101ff01ffffUL, 0x00000101ff010000UL, 0x00000101ff010001UL,
    0x00000101ff010100UL, 0x0000010100ff0000UL, 0x0000010100ff01ffUL, 0x0000010100ff0100UL,
    0x000001010000ff00UL, 0x0000010100000000UL, 0x0000010100000001UL, 0x00000101000001ffUL,
    0x0000010100000100UL, 0x000001010001ff01UL, 0x0000010100010000UL, 0x00000101000101ffUL,
    0x0000010100010101UL, 0x0000010101ffff00UL, 0x0000010101ff0101UL, 0x000001010100ff01UL,
    0x0000010101000000UL, 0x0000010101000001UL, 0x00000101010001ffUL, 0x0000010101000101UL,
    0x000001010101ff00UL, 0x0001ffffffff0000UL, 0x0001ffffff0000ffUL, 0x0001ffffff000001UL,
    0x0001ffffff000100UL, 0x0001ffffff010000UL, 0x0001ffff00ff00ffUL, 0x0001ffff0000ffffUL,
    0x0001ffff00000000UL, 0x0001ffff00000001UL, 0x0001ffff000001ffUL, 0x0001ffff00000101UL,
    0x0001ffff0001ff00UL, 0x0001ffff000100ffUL, 0x0001ffff00010001UL, 0x0001ffff00010100UL,
    0x0001ffff01ffff00UL, 0x0001ffff01000001UL, 0x0001ffff01010000UL, 0x0001ff00ffffff00UL,
    0x0001ff00ffff00ffUL, 0x0001ff00ffff0001UL, 0x0001ff00ffff0100UL, 0x0001ff00ff00ff01UL,
    0x0001ff00ff000000UL, 0x0001ff00ff01ff00UL, 0x0001ff00ff01ff01UL, 0x0001ff00ff010001UL,
    0x0001ff00ff010100UL, 0x0001ff0000ff0000UL, 0x0001ff0000ff0100UL, 0x0001ff000000ff00UL,
    0x0001ff0000000000UL, 0x0001ff0000000001UL, 0x0001ff0000000100UL, 0x0001ff0000010000UL,
    0x0001ff0000010001UL, 0x0001ff0000010101UL, 0x0001ff0001ff00ffUL, 0x0001ff0001ff0101UL,
    0x0001ff000100ff01UL, 0x0001ff0001000000UL, 0x0001ff000101ff00UL, 0x0001ff0001010001UL,
    0x0001ff0001010100UL, 0x0001ff01ff00ff00UL, 0x0001ff01ff000001UL, 0x0001ff01ff000100UL,
    0x0001ff0100ffffffUL, 0x0001ff0100ffff00UL, 0x0001ff0100ff0001UL, 0x0001ff0100000000UL,
    0x0001ff0100000001UL, 0x0001ff01000001ffUL, 0x0001ff010001ffffUL, 0x0001ff0101ff0000UL,
    0x0001ff010100ff00UL, 0x0001ff0101000001UL, 0x0001ff0101010000UL, 0x000100ffff00ff00UL,
    0x000100ffff00ff01UL, 0x000100ffff000000UL, 0x000100ffff000001UL, 0x000100ffff000101UL,
    0x000100ffff01ff00UL, 0x000100ffff010001UL, 0x000100ffff010100UL, 0x000100ff00ffffffUL,
    0x000100ff00ffff01UL, 0x000100ff00ff0000UL, 0x000100ff00ff01ffUL, 0x000100ff00ff0101UL,
    0x000100ff0000ff00UL, 0x000100ff000000ffUL, 0x000100ff00000000UL, 0x000100ff00000001UL,
    0x000100ff00000100UL, 0x000100ff00000101UL, 0x000100ff0001ffffUL, 0x000100ff0001ff01UL,
    0x000100ff00010000UL, 0x000100ff01ff00ffUL, 0x000100ff01ff0000UL, 0x000100ff01ff0100UL,
    0x000100ff0100ffffUL, 0x000100ff0100ff01UL, 0x000100ff010000ffUL, 0x000100ff01000000UL,
    0x000100ff01000001UL, 0x000100ff010001ffUL, 0x000100ff01000101UL, 0x000100ff0101ff00UL,
    0x000100ff010100ffUL, 0x000100ff01010100UL, 0x00010000ffff0000UL, 0x00010000ffff01ffUL,
    0x00010000ffff0101UL, 0x00010000ff00ff00UL, 0x00010000ff000000UL, 0x00010000ff000001UL,
    0x00010000ff000100UL, 0x0001000000ff00ffUL, 0x0001000000ff0000UL, 0x0001000000ff0001UL,
    0x0001000000ff0100UL, 0x000100000000ffffUL, 0x000100000000ff00UL, 0x00010000000000ffUL,
    0x0001000000000000UL, 0x0001000000000001UL, 0x0001000000000100UL, 0x000100000001ff00UL,
    0x00010000000100ffUL, 0x0001000000010000UL, 0x0001000000010001UL, 0x0001000000010100UL,
    0x0001000001ff0001UL, 0x0001000001ff0100UL, 0x0001000001ff0101UL, 0x000100000100ff00UL,
    0x0001000001000000UL, 0x0001000001000001UL, 0x0001000001000100UL, 0x0001000001000101UL,
    0x000100000101ff01UL, 0x0001000001010000UL, 0x0001000001010001UL, 0x00010000010101ffUL,
    0x00010001ffffff01UL, 0x00010001ffff0100UL, 0x00010001ff000000UL, 0x00010001ff01ffffUL,
    0x00010001ff010001UL, 0x00010001ff0101ffUL, 0x00010001ff010100UL, 0x0001000100ffffffUL,
    0x0001000100ff0000UL, 0x0001000100ff01ffUL, 0x0001000100ff0101UL, 0x000100010000ff00UL,
    0x00010001000000ffUL, 0x0001000100000000UL, 0x0001000100000001UL, 0x00010001000001ffUL,
    0x0001000100000101UL, 0x000100010001ffffUL, 0x0001000100010000UL, 0x00010001000101ffUL,
    0x0001000101ffffffUL, 0x0001000101ffff01UL, 0x0001000101ff0000UL, 0x0001000101ff0101UL,
    0x00010001010000ffUL, 0x0001000101000001UL, 0x00010001010001ffUL, 0x0001000101000100UL,
    0x000100010101ffffUL, 0x00010001010100ffUL, 0x0001000101010001UL, 0x0001000101010101UL,
    0x000101ffff000001UL, 0x000101ffff000100UL, 0x000101ffff010000UL, 0x000101ff00ffff00UL,
    0x000101ff0000ff01UL, 0x000101ff00000000UL, 0x000101ff00000101UL, 0x000101ff0001ff00UL,
    0x000101ff00010100UL, 0x000101ff01ff0000UL, 0x000101ff0100ff00UL, 0x000101ff010001ffUL,
    0x000101ff01010001UL, 0x00010100ffffff00UL, 0x00010100ffff00ffUL, 0x00010100ff00ffffUL,
    0x00010100ff000000UL, 0x00010100ff01ff00UL, 0x00010100ff0100ffUL, 0x00010100ff010001UL,
    0x00010100ff010100UL, 0x0001010000ffffffUL, 0x0001010000ffff00UL, 0x0001010000ff0000UL,
    0x0001010000ff0001UL, 0x0001010000ff01ffUL, 0x000101000000ff00UL, 0x00010100000000ffUL,
    0x0001010000000000UL, 0x0001010000000001UL, 0x0001010000000100UL, 0x000101000001ffffUL,
    0x0001010000010000UL, 0x0001010000010101UL, 0x0001010001ffff01UL, 0x0001010001ff00ffUL,
    0x0001010001ff0101UL, 0x0001010001000000UL, 0x000101000101ff00UL, 0x00010100010100ffUL,
    0x0001010001010000UL, 0x0001010001010100UL, 0x00010101ff00ff00UL, 0x00010101ff000001UL,
    0x00010101ff0001ffUL, 0x0001010100ffff00UL, 0x0001010100ff00ffUL, 0x0001010100ff0100UL,
    0x000101010000ffffUL, 0x0001010100000000UL, 0x00010101000001ffUL, 0x0001010100000101UL,
    0x00010101000100ffUL, 0x0001010100010000UL, 0x0001010100010100UL, 0x0001010101ff0001UL,
    0x00010101010000ffUL, 0x00010101010001ffUL, 0x0001010101000101UL, 0x0001010101010001UL,
    0x01ffffffffffffffUL, 0x01ffffffffffff01UL, 0x01ffffffffff01ffUL, 0x01ffffffffff0101UL,
    0x01ffffffff01ffffUL, 0x01ffffffff01ff01UL, 0x01ffffffff0101ffUL, 0x01ffffffff010101UL,
    0x01ffffff00ff0000UL, 0x01ffffff0000ffffUL, 0x01ffffff0000ff00UL, 0x01ffffff000000ffUL,
    0x01ffffff00000001UL, 0x01ffffff00000100UL, 0x01ffffff00010000UL, 0x01ffffff01ffffffUL,
    0x01ffffff01ffff01UL, 0x01ffffff01ff01ffUL, 0x01ffffff01ff0101UL, 0x01ffffff01000000UL,
    0x01ffffff0101ffffUL, 0x01ffffff0101ff01UL, 0x01ffffff010101ffUL, 0x01ffffff01010101UL,
    0x01ffff00ffff0000UL, 0x01ffff00ff00ff00UL, 0x01ffff00ff0000ffUL, 0x01ffff00ff000001UL,
    0x01ffff00ff000100UL, 0x01ffff00ff010000UL, 0x01ffff0000ffff00UL, 0x01ffff0000ff00ffUL,
    0x01ffff0000ff0100UL, 0x01ffff000000ffffUL, 0x01ffff000000ff01UL, 0x01ffff0000000000UL,
    0x01ffff0000000001UL, 0x01ffff00000001ffUL, 0x01ffff0000000100UL, 0x01ffff00000100ffUL,
    0x01ffff0000010001UL, 0x01ffff0000010100UL, 0x01ffff0001ff0000UL, 0x01ffff0001ff0100UL,
    0x01ffff00010000ffUL, 0x01ffff0001000001UL, 0x01ffff0001000100UL, 0x01ffff0001010000UL,
    0x01ffff01ffffffffUL, 0x01ffff01ffffff01UL, 0x01ffff01ffff01ffUL, 0x01ffff01ffff0101UL,
    0x01ffff01ff000000UL, 0x01ffff01ff01ffffUL, 0x01ffff01ff01ff01UL, 0x01ffff01ff0101ffUL,
    0x01ffff01ff010101UL, 0x01ffff010000ff00UL, 0x01ffff01000000ffUL, 0x01ffff0100000100UL,
    0x01ffff0100010000UL, 0x01ffff0101ffffffUL, 0x01ffff0101ffff01UL, 0x01ffff0101ff01ffUL,
    0x01ffff0101ff0101UL, 0x01ffff0101000000UL, 0x01ffff010101ffffUL, 0x01ffff010101ff01UL,
    0x01ffff01010101ffUL, 0x01ffff0101010101UL, 0x01ff00ffff0000ffUL, 0x01ff00ffff000100UL,
    0x01ff00ff00ffff00UL, 0x01ff00ff00ff00ffUL, 0x01ff00ff0000ff00UL, 0x01ff00ff00000000UL,
    0x01ff00ff00000101UL, 0x01ff00ff0001ff00UL, 0x01ff00ff000100ffUL, 0x01ff00ff00010100UL,
    0x01ff00ff010000ffUL, 0x01ff00ff01000100UL, 0x01ff0000ffffff00UL, 0x01ff0000ffff0100UL,
    0x01ff0000ff00ff01UL, 0x01ff0000ff000000UL, 0x01ff0000ff000101UL, 0x01ff0000ff010001UL,
    0x01ff0000ff010100UL, 0x01ff000000ffffffUL, 0x01ff000000ffff00UL, 0x01ff000000ff0000UL,
    0x01ff000000ff01ffUL, 0x01ff00000000ff00UL, 0x01ff0000000000ffUL, 0x01ff000000000000UL,
    0x01ff000000000001UL, 0x01ff000000000100UL, 0x01ff000000000101UL, 0x01ff000000010000UL,
    0x01ff000000010001UL, 0x01ff0000000101ffUL, 0x01ff000000010101UL, 0x01ff000001ffff00UL,
    0x01ff000001ff00ffUL, 0x01ff000001ff0001UL, 0x01ff000001ff0100UL, 0x01ff00000100ffffUL,
    0x01ff00000100ff01UL, 0x01ff000001000000UL, 0x01ff0000010001ffUL, 0x01ff000001010001UL,
    0x01ff0001ff00ff00UL, 0x01ff0001ff000001UL, 0x01ff0001ff000100UL, 0x01ff0001ff010000UL,
    0x01ff000100ffff00UL, 0x01ff000100ff00ffUL, 0x01ff000100ff0100UL, 0x01ff000100ff0101UL,
    0x01ff00010000ffffUL, 0x01ff000100000000UL, 0x01ff000100000100UL, 0x01ff000100000101UL,
    0x01ff00010001ff00UL, 0x01ff000100010001UL, 0x01ff000100010101UL, 0x01ff000101ff0000UL,
    0x01ff00010100ff00UL, 0x01ff000101000101UL, 0x01ff0001010100ffUL, 0x01ff01ffffffffffUL,
    0x01ff01ffffffff01UL, 0x01ff01ffffff01ffUL, 0x01ff01ffffff0101UL, 0x01ff01ffff000000UL,
    0x01ff01ffff01ffffUL, 0x01ff01ffff01ff01UL, 0x01ff01ffff0101ffUL, 0x01ff01ffff010101UL,
    0x01ff01ff00ffff00UL, 0x01ff01ff00ff0000UL, 0x01ff01ff0000ff00UL, 0x01ff01ff000000ffUL,
    0x01ff01ff00000100UL, 0x01ff01ff00010000UL, 0x01ff01ff00010100UL, 0x01ff01ff01ffffffUL,
    0x01ff01ff01ffff01UL, 0x01ff01ff01ff01ffUL, 0x01ff01ff01ff0101UL, 0x01ff01ff01000000UL,
    0x01ff01ff0101ffffUL, 0x01ff01ff0101ff01UL, 0x01ff01ff010101ffUL, 0x01ff01ff01010101UL,
    0x01ff0100ffff0000UL, 0x01ff0100ffff0001UL, 0x01ff0100ff00ff00UL, 0x01ff0100ff0000ffUL,
    0x01ff0100ff000001UL, 0x01ff0100ff010000UL, 0x01ff010000ffff00UL, 0x01ff010000ff00ffUL,
    0x01ff010000ff0001UL, 0x01ff010000ff0100UL, 0x01ff01000000ffffUL, 0x01ff01000000ff01UL,
    0x01ff010000000000UL, 0x01ff010000000101UL, 0x01ff01000001ff00UL, 0x01ff0100000100ffUL,
    0x01ff010001ff0000UL, 0x01ff010001000001UL, 0x01ff010001000100UL, 0x01ff010001010000UL,
    0x01ff0101ffffffffUL, 0x01ff0101ffffff01UL, 0x01ff0101ffff01ffUL, 0x01ff0101ffff0101UL,
    0x01ff0101ff000000UL, 0x01ff0101ff01ffffUL, 0x01ff0101ff01ff01UL, 0x01ff0101ff0101ffUL,
    0x01ff0101ff010101UL, 0x01ff010100ff0000UL, 0x01ff01010000ff00UL, 0x01ff0101000000ffUL,
    0x01ff010100000001UL, 0x01ff010101ffffffUL, 0x01ff010101ffff01UL, 0x01ff010101ff01ffUL,
    0x01ff010101ff0101UL, 0x01ff010101000000UL, 0x01ff01010101ffffUL, 0x01ff01010101ff01UL,
    0x01ff0101010101ffUL, 0x01ff010101010101UL, 0x0100ffffffff0000UL, 0x0100ffffff00ff00UL,
    0x0100ffffff000001UL, 0x0100ffffff0001ffUL, 0x0100ffffff000100UL, 0x0100ffffff010000UL,
    0x0100ffff00ffff00UL, 0x0100ffff00ff0001UL, 0x0100ffff00ff0100UL, 0x0100ffff00000000UL,
    0x0100ffff000001ffUL, 0x0100ffff00000101UL, 0x0100ffff00010100UL, 0x0100ffff00010101UL,
    0x0100ffff01ff0000UL, 0x0100ffff0100ff00UL, 0x0100ffff010000ffUL, 0x0100ffff01000001UL,
    0x0100ffff01000100UL, 0x0100ffff01010000UL, 0x0100ff00ffffff00UL, 0x0100ff00ffff00ffUL,
    0x0100ff00ffff0001UL, 0x0100ff00ffff0100UL, 0x0100ff00ff00ffffUL, 0x0100ff00ff000000UL,
    0x0100ff00ff0001ffUL, 0x0100ff00ff000101UL, 0x0100ff00ff01ff00UL, 0x0100ff00ff0100ffUL,
    0x0100ff00ff010001UL, 0x0100ff00ff010100UL, 0x0100ff0000ffffffUL, 0x0100ff0000ff0000UL,
    0x0100ff000000ffffUL, 0x0100ff000000ff00UL, 0x0100ff00000000ffUL, 0x0100ff0000000000UL,
    0x0100ff0000000001UL, 0x0100ff0000000100UL, 0x0100ff000001ff01UL, 0x0100ff0000010000UL,
    0x0100ff0001ff00ffUL, 0x0100ff0001ff0001UL, 0x0100ff000100ff01UL, 0x0100ff0001000000UL,
    0x0100ff00010001ffUL, 0x0100ff000101ff00UL, 0x0100ff00010100ffUL, 0x0100ff0001010001UL,
    0x0100ff0001010100UL, 0x0100ff01ffff0000UL, 0x0100ff01ff00ff00UL, 0x0100ff01ff0000ffUL,
    0x0100ff01ff000100UL, 0x0100ff01ff010000UL, 0x0100ff0100ff00ffUL, 0x0100ff0100ff0001UL,
    0x0100ff0100ff0100UL, 0x0100ff010000ffffUL, 0x0100ff010000ff01UL, 0x0100ff0100000000UL,
    0x0100ff01000001ffUL, 0x0100ff0100010001UL, 0x0100ff0100010100UL, 0x0100ff0101ff0000UL,
    0x0100ff01010000ffUL, 0x0100ff0101000001UL, 0x0100ff0101010100UL, 0x010000ffffffff00UL,
    0x010000ffffff00ffUL, 0x010000ffffff0001UL, 0x010000ffff00ffffUL, 0x010000ffff000000UL,
    0x010000ffff0001ffUL, 0x010000ffff010001UL, 0x010000ff00ffffffUL, 0x010000ff00ff0101UL,
    0x010000ff0000ff00UL, 0x010000ff000000ffUL, 0x010000ff00000000UL, 0x010000ff00000001UL,
    0x010000ff000001ffUL, 0x010000ff00000100UL, 0x010000ff0001ffffUL, 0x010000ff0001ff00UL,
    0x010000ff0001ff01UL, 0x010000ff00010000UL, 0x010000ff01ff00ffUL, 0x010000ff01ff0001UL,
    0x010000ff0100ff01UL, 0x010000ff010000ffUL, 0x010000ff01000000UL, 0x010000ff010001ffUL,
    0x010000ff0101ff00UL, 0x010000ff01010100UL, 0x01000000ffffffffUL, 0x01000000ffff0000UL,
    0x01000000ffff01ffUL, 0x01000000ffff0101UL, 0x01000000ff00ffffUL, 0x01000000ff00ff00UL,
    0x01000000ff0000ffUL, 0x01000000ff000000UL, 0x01000000ff000001UL, 0x01000000ff000100UL,
    0x01000000ff01ff00UL, 0x01000000ff010000UL, 0x01000000ff010100UL, 0x01000000ff010101UL,
    0x0100000000ffff00UL, 0x0100000000ff00ffUL, 0x0100000000ff0000UL, 0x0100000000ff0001UL,
    0x0100000000ff0100UL, 0x010000000000ffffUL, 0x010000000000ff00UL, 0x010000000000ff01UL,
    0x01000000000000ffUL, 0x0100000000000000UL, 0x0100000000000001UL, 0x01000000000001ffUL,
    0x0100000000000100UL, 0x0100000000000101UL, 0x010000000001ff00UL, 0x01000000000100ffUL,
    0x0100000000010000UL, 0x0100000000010001UL, 0x0100000000010100UL, 0x0100000001ffff00UL,
    0x0100000001ff0000UL, 0x0100000001ff01ffUL, 0x010000000100ff00UL, 0x010000000100ff01UL,
    0x01000000010000ffUL, 0x0100000001000000UL, 0x0100000001000001UL, 0x0100000001000100UL,
    0x0100000001000101UL, 0x010000000101ffffUL, 0x010000000101ff01UL, 0x0100000001010000UL,
    0x01000000010101ffUL, 0x0100000001010101UL, 0x01000001ffffff00UL, 0x01000001ffff00ffUL,
    0x01000001ff00ffffUL, 0x01000001ff000000UL, 0x01000001ff000100UL, 0x01000001ff01ffffUL,
    0x01000001ff010001UL, 0x01000001ff010100UL, 0x0100000100ff0000UL, 0x0100000100ff01ffUL,
    0x0100000100ff0100UL, 0x010000010000ff00UL, 0x010000010000ff01UL, 0x0100000100000000UL,
    0x0100000100000001UL, 0x0100000100000100UL, 0x0100000100010000UL, 0x01000001000101ffUL,
    0x0100000101ffff01UL, 0x0100000101ff00ffUL, 0x0100000101ff0100UL, 0x0100000101ff0101UL,
    0x010000010100ff01UL, 0x01000001010000ffUL, 0x0100000101000000UL, 0x01000001010100ffUL,
    0x0100000101010001UL, 0x0100000101010100UL, 0x010001ffffff0000UL, 0x010001ffff000001UL,
    0x010001ffff000100UL, 0x010001ffff010000UL, 0x010001ff00ffff00UL, 0x010001ff00ff0001UL,
    0x010001ff0000ffffUL, 0x010001ff0000ff01UL, 0x010001ff00000000UL, 0x010001ff00000001UL,
    0x010001ff00000101UL, 0x010001ff000100ffUL, 0x010001ff00010000UL, 0x010001ff01ff0000UL,
    0x010001ff0100ff00UL, 0x010001ff01000001UL, 0x010001ff01000100UL, 0x010001ff01010000UL,
    0x01000100ffff00ffUL, 0x01000100ffff0001UL, 0x01000100ffff0100UL, 0x01000100ff00ffffUL,
    0x01000100ff00ff01UL, 0x01000100ff000000UL, 0x01000100ff0001ffUL, 0x01000100ff000101UL,
    0x01000100ff01ffffUL, 0x01000100ff01ff00UL, 0x01000100ff0100ffUL, 0x01000100ff010001UL,
    0x0100010000ffffffUL, 0x0100010000ffff01UL, 0x0100010000ff0000UL, 0x0100010000ff01ffUL,
    0x0100010000ff0101UL, 0x010001000000ff00UL, 0x01000100000000ffUL, 0x0100010000000000UL,
    0x0100010000000001UL, 0x0100010000000100UL, 0x010001000001ff01UL, 0x0100010000010000UL,
    0x0100010000010001UL, 0x0100010000010101UL, 0x0100010001ffff00UL, 0x0100010001ff00ffUL,
    0x010001000100ffffUL, 0x010001000100ff01UL, 0x0100010001000000UL, 0x0100010001000101UL,
    0x010001000101ff00UL, 0x0100010001010001UL, 0x01000101ffff0000UL, 0x01000101ff000000UL,
    0x01000101ff010000UL, 0x0100010100ff00ffUL, 0x0100010100ff0001UL, 0x0100010100ff0100UL,
    0x010001010000ffffUL, 0x0100010100000000UL, 0x01000101000001ffUL, 0x010001010001ff00UL,
    0x0100010101ff0000UL, 0x010001010100ff00UL, 0x01000101010000ffUL, 0x0100010101000000UL,
    0x0100010101000001UL, 0x0101ffffffffffffUL, 0x0101ffffffffff01UL, 0x0101ffffffff01ffUL,
    0x0101ffffffff0101UL, 0x0101ffffff000000UL, 0x0101ffffff01ffffUL, 0x0101ffffff01ff01UL,
    0x0101ffffff0101ffUL, 0x0101ffffff010101UL, 0x0101ffff00ff0000UL, 0x0101ffff0000ff00UL,
    0x0101ffff000000ffUL, 0x0101ffff00000001UL, 0x0101ffff00000100UL, 0x0101ffff01ffffffUL,
    0x0101ffff01ffff01UL, 0x0101ffff01ff01ffUL, 0x0101ffff01ff0101UL, 0x0101ffff01000000UL,
    0x0101ffff0101ffffUL, 0x0101ffff0101ff01UL, 0x0101ffff010101ffUL, 0x0101ffff01010101UL,
    0x0101ff00ffff0000UL, 0x0101ff00ffff0100UL, 0x0101ff00ff00ff00UL, 0x0101ff00ff0000ffUL,
    0x0101ff00ff000001UL, 0x0101ff00ff000100UL, 0x0101ff00ff000101UL, 0x0101ff0000ff0001UL,
    0x0101ff0000ff0100UL, 0x0101ff000000ff00UL, 0x0101ff0000000000UL, 0x0101ff00000001ffUL,
    0x0101ff0000000101UL, 0x0101ff000001ff00UL, 0x0101ff00000100ffUL, 0x0101ff0001ff0000UL,
    0x0101ff000100ffffUL, 0x0101ff000100ff01UL, 0x0101ff0001000001UL, 0x0101ff0001000100UL,
    0x0101ff01ffffff01UL, 0x0101ff01ffff01ffUL, 0x0101ff01ffff0101UL, 0x0101ff01ff00ffffUL,
    0x0101ff01ff000100UL, 0x0101ff01ff01ff01UL, 0x0101ff01ff0101ffUL, 0x0101ff01ff010101UL,
    0x0101ff0100ff0000UL, 0x0101ff010000ff00UL, 0x0101ff0100000001UL, 0x0101ff0100000100UL,
    0x0101ff0100010000UL, 0x0101ff0101ffffffUL, 0x0101ff0101ffff01UL, 0x0101ff0101ff01ffUL,
    0x0101ff0101ff0101UL, 0x0101ff0101000000UL, 0x0101ff010101ffffUL, 0x0101ff010101ff01UL,
    0x0101ff01010101ffUL, 0x0101ff0101010101UL, 0x010100ffff000100UL, 0x010100ffff010000UL,
    0x010100ff00ffff00UL, 0x010100ff00ff00ffUL, 0x010100ff0000ffffUL, 0x010100ff000000ffUL,
    0x010100ff00000000UL, 0x010100ff000001ffUL, 0x010100ff00000101UL, 0x010100ff0001ff00UL,
    0x010100ff00010000UL, 0x010100ff00010001UL, 0x010100ff000101ffUL, 0x010100ff00010100UL,
    0x010100ff01ff0000UL, 0x01010000ffff0001UL, 0x01010000ffff0100UL, 0x01010000ff00ffffUL,
    0x01010000ff00ff01UL, 0x01010000ff000000UL, 0x01010000ff0001ffUL, 0x01010000ff010001UL,
    0x01010000ff010100UL, 0x0101000000ffff01UL, 0x0101000000ff0000UL, 0x010100000000ff00UL,
    0x01010000000000ffUL, 0x0101000000000000UL, 0x0101000000000001UL, 0x0101000000000100UL,
    0x0101000000010000UL, 0x0101000000010101UL, 0x0101000001ffff00UL, 0x0101000001ff00ffUL,
    0x0101000001ff0000UL, 0x0101000001ff0001UL, 0x0101000001ff0100UL, 0x010100000100ff01UL,
    0x0101000001000000UL, 0x01010000010001ffUL, 0x01010001ffff0000UL, 0x01010001ff00ff00UL,
    0x01010001ff000001UL, 0x01010001ff000101UL, 0x01010001ff01ff00UL, 0x01010001ff010000UL,
    0x0101000100ff00ffUL, 0x0101000100ff0001UL, 0x0101000100ff0101UL, 0x010100010000ff01UL,
    0x0101000100000000UL, 0x0101000100000001UL, 0x01010001000001ffUL, 0x010100010001ffffUL,
    0x010100010001ff01UL, 0x0101000101ff0001UL, 0x010100010100ffffUL, 0x0101000101000000UL,
    0x0101000101000001UL, 0x0101000101000100UL, 0x010100010101ff00UL, 0x01010001010100ffUL,
    0x0101000101010001UL, 0x010101ffffffffffUL, 0x010101ffffffff01UL, 0x010101ffffff01ffUL,
    0x010101ffffff0101UL, 0x010101ffff01ffffUL, 0x010101ffff01ff01UL, 0x010101ffff0101ffUL,
    0x010101ffff010101UL, 0x010101ff0000ff00UL, 0x010101ff000000ffUL, 0x010101ff00000001UL,
    0x010101ff00000100UL, 0x010101ff01ffffffUL, 0x010101ff01ffff01UL, 0x010101ff01ff01ffUL,
    0x010101ff01ff0101UL, 0x010101ff01000000UL, 0x010101ff0101ffffUL, 0x010101ff0101ff01UL,
    0x010101ff010101ffUL, 0x010101ff01010101UL, 0x01010100ffff0000UL, 0x01010100ff0000ffUL,
    0x01010100ff000100UL, 0x01010100ff01ff00UL, 0x01010100ff010000UL, 0x0101010000ffff00UL,
    0x010101000000ffffUL, 0x0101010000000000UL, 0x0101010000000101UL, 0x010101000001ff00UL,
    0x0101010000010001UL, 0x0101010000010100UL, 0x010101000100ffffUL, 0x0101010001000001UL,
    0x01010101ffffffffUL, 0x01010101ffffff01UL, 0x01010101ffff01ffUL, 0x01010101ffff0101UL,
    0x01010101ff01ffffUL, 0x01010101ff01ff01UL, 0x01010101ff0101ffUL, 0x01010101ff010101UL,
    0x010101010000ff00UL, 0x01010101000000ffUL, 0x0101010100000001UL, 0x0101010101ffffffUL,
    0x0101010101ffff01UL, 0x0101010101ff01ffUL, 0x0101010101ff0101UL, 0x0101010101000000UL,
    0x010101010101ffffUL, 0x010101010101ff01UL, 0x01010101010101ffUL, 0x0101010101010101UL,
};

constant uint iq3s_grid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu,
    0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u,
    0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u,
    0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u,
    0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u,
    0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu,
    0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u,
    0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u,
    0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u,
    0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu,
    0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u,
    0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu,
    0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u,
    0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u,
    0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u,
    0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u,
    0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u,
    0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u,
    0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u,
    0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu,
    0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u,
    0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u,
    0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u,
    0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u,
    0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u,
    0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u,
    0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u,
    0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u,
    0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u,
    0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u,
    0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u,
    0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u,
    0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u,
    0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u,
    0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu,
    0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u,
    0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u,
    0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u,
    0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u,
    0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u,
    0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u,
    0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u,
    0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u,
    0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u,
    0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u,
    0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu,
    0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u,
    0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u,
    0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu,
    0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u,
    0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u,
    0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u,
    0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u,
    0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u,
    0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u,
    0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u,
    0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u,
    0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu,
    0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u,
    0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u,
    0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u,
    0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u,
    0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u,
    0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u,
    0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

// ---- the low-bit types of the ISTA-DASLab GSQ-RCO files (patch 0022): kq type
// index 6 IQ2_XXS, 7 IQ2_S, 8 IQ1_M, 9 Q2_0, 10 IQ3_S, 11 IQ2_XS, blocks as stored.
// A 32-weight sub-block j32 of a row expands to its signed int8 values w[8]
// (weights 4i .. 4i+3 in w[i]) and two scales, s0 for weights 0..15 and s1
// for 16..31 (equal unless the type scales half sub-blocks: IQ2_S, IQ1_M).
// IQ1_M's values are 8 (g + delta) with delta = +-1/8, exact in int8, and
// its scales carry the 1/8. After ggml-quants.c's dequantize_row_*. ----

// bytes per block, 32-weight sub-blocks per block
#define LB_BYTES(T) ((T) == 6 ? 66 : (T) == 7 ? 82 : (T) == 8 ? 56 : (T) == 9 ? 18 : (T) == 11 ? 74 : 110)
#define LB_SUBS(T)  ((T) == 9 ? 2 : 8)
#define LB_ROW_BYTES(T, n32) ((ulong)((n32) / LB_SUBS(T)) * LB_BYTES(T))

// four sign bits -> 0xFF in the bytes to negate (32-bit only: Intel's GPUs
// emulate 64-bit integer arithmetic)
static inline uint lb_signmask4(uint b) {
    uint m = ((b & 0xFu) * 0x01010101u) & 0x08040201u;
    m = (m + 0x7F7F7F7Fu) & 0x80808080u;
    return (m >> 7) * 0xFFu;
}

// four unsigned grid bytes, negated where m's bytes are 0xFF. In 32-bit words
// (patch 0034): IGC splits char4 arithmetic per byte (moves to byte registers,
// widens, packs back: more than half of a low-bit matvec's instructions on the
// B60). Exact because no grid byte is 0 (the IQ2 grids hold 8, 25, 43; IQ3_XXS
// 4..62; IQ3_S odd 1..15): ~g + 1 never carries out of a byte.
// -DLB_CHAR4_ARITH keeps the char4 form (measurement).
static inline uint lb_sgn4(uint g, uint m) {
#ifdef LB_CHAR4_ARITH
    return as_uint(as_char4(g ^ m) - as_char4(m));
#else
    return (g ^ m) + (m & 0x01010101u);
#endif
}

// patch 0054: the IQ2_S / IQ3_S signs and Q2_0's fields spread to bytes by
// multiplies, as in mul_mv_iq_q8_1.cl (0053); -DLB_SIGN_MASK4 and
// -DQ2_SPREAD_SHIFT keep the old forms, -DLB_SIGN_RED and -DQ2_SPREAD_RED are
// the red cases
#if defined(LB_SIGN_RED) && (defined(LB_SIGN_MASK4) || defined(LB_CHAR4_ARITH))
#error "LB_SIGN_RED mutates the multiply form; LB_SIGN_MASK4 / LB_CHAR4_ARITH select the old one"
#endif
#if defined(Q2_SPREAD_RED) && defined(Q2_SPREAD_SHIFT)
#error "Q2_SPREAD_RED mutates the multiply form; Q2_SPREAD_SHIFT selects the old one"
#endif
static inline uint lb_sgnb4(uint g, uint b) {
#if defined(LB_SIGN_MASK4) || defined(LB_CHAR4_ARITH)
    return lb_sgn4(g, lb_signmask4(b));
#else
#ifndef LB_SIGN_RED
    const uint s = ((b & 0xFu) * 0x00204081u) & 0x01010101u;
#else
    const uint s = ((b & 0xFu) * 0x00204081u) & 0x01010100u;
#endif
    return (g ^ (s * 0xFFu)) + s;
#endif
}
#ifndef Q2_SPREAD_RED
#define Q2_SPREAD_HI 0x03000300u
#else
#define Q2_SPREAD_HI 0x00000300u
#endif

// a 64-bit table entry as two 32-bit halves
#define LB_TAB2(T, i) vload2((i), (constant const uint *)(T))

static inline uint lb_u32(global const uchar * p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

#define LB_LT_PARAM
#define LB_GRID_XXS(i) LB_TAB2(iq2xxs_grid, (i))
#define LB_SIGNS_XXS(i) LB_TAB2(ksigns64, (i))
#define LB_GRID_1M(i) LB_TAB2(iq1s_grid, (i))
#define LB_GRID_XS(i) LB_TAB2(iq2xs_grid, (i))
#define LB_SIGNS_XS(i) LB_TAB2(ksigns64, (i))
// A low-bit slice (one weight matrix, nblk blocks) is kept in planes on the
// card (ggml-opencl's ggml_cl_lb_planes): IQ2_XXS d | qs[64]; IQ2_XS d | qs[64] |
// scales[8]; IQ2_S d | qs[32]
// | signs[32] | qh[8] | scales[8]; Q2_0 d | qs[16]; IQ3_S d | qs[64] | qh[8] |
// signs[32] | scales[4], each field of every block back to back, so a lane's
// sub-block fields are aligned 32-bit words (the GGUF blocks of 18, 66, 82 or
// 110 bytes left them at 2-byte offsets: single-byte loads). IQ1_M stays as
// stored (its fields are aligned). Block bi of the slice, sub-block p (Q2_0: half).
static inline void lb_expand(const int TYPE, global const uchar * S, ulong nblk, ulong bi, int p,
                             uint * w, float * s0, float * s1 LB_LT_PARAM) {
    if (TYPE == 6) {          // IQ2_XXS
        const float d = vload_half(bi, (global const half *)S);
        const uint2 a = vload2(bi * 8 + p, (global const uint *)(S + nblk * 2));
        const uint a0 = a.s0, a1 = a.s1;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = LB_GRID_XXS((a0 >> (8 * l)) & 0xFF);
            const uint2 m = LB_SIGNS_XXS((a1 >> (7 * l)) & 127);
            w[2 * l]     = lb_sgn4(g.s0, m.s0);
            w[2 * l + 1] = lb_sgn4(g.s1, m.s1);
        }
        *s0 = *s1 = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
    } else if (TYPE == 11) {  // IQ2_XS: a code is a 9-bit grid index and a 7-bit sign index
        const float d = vload_half(bi, (global const half *)S);
        const uint2 a = vload2(bi * 8 + p, (global const uint *)(S + nblk * 2));
        const uint qv[4] = { a.s0 & 0xFFFF, a.s0 >> 16, a.s1 & 0xFFFF, a.s1 >> 16 };
        const uint sc = S[nblk * 66 + bi * 8 + p];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = LB_GRID_XS(qv[l] & 511);
            const uint2 m = LB_SIGNS_XS(qv[l] >> 9);
            w[2 * l]     = lb_sgn4(g.s0, m.s0);
            w[2 * l + 1] = lb_sgn4(g.s1, m.s1);
        }
        *s0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        *s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
    } else if (TYPE == 7) {   // IQ2_S
        const float d = vload_half(bi, (global const half *)S);
        const uint qw = ((global const uint *)(S + nblk * 2))[bi * 8 + p];
        const uint sw = ((global const uint *)(S + nblk * 34))[bi * 8 + p];
        const uint qh = S[nblk * 66 + bi * 8 + p], sc = S[nblk * 74 + bi * 8 + p];
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = LB_TAB2(iq2s_grid, ((qw >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300));
            const uint sb = (sw >> (8 * l)) & 0xFF;
            w[2 * l]     = lb_sgnb4(g.s0, sb);
            w[2 * l + 1] = lb_sgnb4(g.s1, sb >> 4);
        }
        *s0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        *s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
    } else if (TYPE == 8) {   // IQ1_M, as stored: qs[32], qh[16], scales[8]
        global const uchar * bp = S + bi * 56;
        global const ushort * sc = (global const ushort *)(bp + 48);
        const ushort u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = convert_float(as_half(u));
        const uint s = sc[p / 2] >> (6 * (p & 1));
        const uint qhw = ((global const ushort *)(bp + 32))[p];
        const uint qh0 = qhw & 0xFF, qh1 = qhw >> 8;
        const uint qsw = ((global const uint *)bp)[p];
        const uint idx[4] = { (qsw & 0xFF) | ((qh0 << 8) & 0x700), ((qsw >> 8) & 0xFF) | ((qh0 << 4) & 0x700),
                              ((qsw >> 16) & 0xFF) | ((qh1 << 8) & 0x700), (qsw >> 24) | ((qh1 << 4) & 0x700) };
        const int dl[4] = { (qh0 & 0x08) ? -1 : 1, (qh0 & 0x80) ? -1 : 1, (qh1 & 0x08) ? -1 : 1, (qh1 & 0x80) ? -1 : 1 };
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = LB_GRID_1M(idx[l]);
#ifdef LB_CHAR4_ARITH
            w[2 * l]     = as_uint(as_char4(g.s0) * (char4)(8) + (char4)(dl[l]));
            w[2 * l + 1] = as_uint(as_char4(g.s1) * (char4)(8) + (char4)(dl[l]));
#else
            // 8 g per byte (g in -1..1: the shift's spill masked off), then +-1 per byte with the
            // top bit flipped around it: no carry or borrow crosses a byte
            const uint t0 = ((g.s0 << 3) & 0xF8F8F8F8u) ^ 0x80808080u, t1 = ((g.s1 << 3) & 0xF8F8F8F8u) ^ 0x80808080u;
            w[2 * l]     = (dl[l] > 0 ? t0 + 0x01010101u : t0 - 0x01010101u) ^ 0x80808080u;
            w[2 * l + 1] = (dl[l] > 0 ? t1 + 0x01010101u : t1 - 0x01010101u) ^ 0x80808080u;
#endif
        }
        *s0 = d * (float)(2 * (s & 7) + 1) * 0.125f;
        *s1 = d * (float)(2 * ((s >> 3) & 7) + 1) * 0.125f;
    } else if (TYPE == 9) {   // Q2_0: p is the half of the 64-weight block
        const float d = vload_half(bi, (global const half *)S);
        const uint2 c = vload2(bi * 2 + p, (global const uint *)(S + nblk * 2));
        const uint cw[2] = { c.s0, c.s1 };
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint b = (cw[i >> 2] >> (8 * (i & 3))) & 0xFF;
#ifdef Q2_SPREAD_SHIFT
            const uint v = (b & 3) | (((b >> 2) & 3) << 8) | (((b >> 4) & 3) << 16) | (((b >> 6) & 3) << 24);
#else
            const uint v = (((b & 0x33u) * 0x1001u) & 0x00030003u) | (((b & 0xCCu) * 0x40040u) & Q2_SPREAD_HI);
#endif
#ifdef LB_CHAR4_ARITH
            w[i] = as_uint(as_char4(v) - (char4)(1));
#else
            w[i] = ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;   // v - 1 per byte (v in 0..3), no borrow across bytes
#endif
        }
        *s0 = *s1 = d;
    } else {                  // IQ3_S
        const float d = vload_half(bi, (global const half *)S);
        const uint2 q = vload2(bi * 8 + p, (global const uint *)(S + nblk * 2));
        const uint qv[4] = { q.s0 & 0xFFFF, q.s0 >> 16, q.s1 & 0xFFFF, q.s1 >> 16 };
        const uint qh = S[nblk * 66 + bi * 8 + p];
        const uint sw = ((global const uint *)(S + nblk * 74))[bi * 8 + p];
        const uint sc = (S[nblk * 106 + bi * 4 + p / 2] >> (4 * (p & 1))) & 0xF;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint g1 = iq3s_grid[(qv[l] & 0xFF) | ((qh << (8 - 2 * l)) & 256)];
            const uint g2 = iq3s_grid[(qv[l] >> 8)   | ((qh << (7 - 2 * l)) & 256)];
            const uint sb = (sw >> (8 * l)) & 0xFF;
            w[2 * l]     = lb_sgnb4(g1, sb);
            w[2 * l + 1] = lb_sgnb4(g2, sb >> 4);
        }
        *s0 = *s1 = d * (float)(1 + 2 * sc);
    }
}

// sub-block j32 of row r: its block in the slice, and p
#define LB_BI(T, r, nbr, j32) ((ulong)(r) * (nbr) + (ulong)((j32) / LB_SUBS(T)))
#define LB_P(T, j32)          ((j32) % LB_SUBS(T))

// the 32 weights of sub-block j32 of global row g (expert g / ne01's slice) as floats
static inline void lb_v32(const int TYPE, global const uchar * q, ulong g, int ne01, int n32, int j32, float * v) {
    const int   nbr  = n32 / LB_SUBS(TYPE);
    const ulong nblk = (ulong)ne01 * nbr;
    global const uchar * S = q + (g / ne01) * nblk * LB_BYTES(TYPE);
    uint w[8];
    float s0, s1;
    lb_expand(TYPE, S, nblk, LB_BI(TYPE, g % ne01, nbr, j32), LB_P(TYPE, j32), w, &s0, &s1);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const char4 c = as_char4(w[i]);
        const float sc = i < 4 ? s0 : s1;
        v[4 * i] = sc * c.s0; v[4 * i + 1] = sc * c.s1; v[4 * i + 2] = sc * c.s2; v[4 * i + 3] = sc * c.s3;
    }
}

// the low-bit types (kq type index 6..10, iq_low): unit (row r, 32-block j32),
// those past K skipped (Q2_0's K need only be a multiple of 64)
static inline void deq_lb(const int TYPE, local half * wt, global const uchar * q, ulong wrow0, int row0, int ne01, int n32, int ib) {
#if SG == 8
    for (int u = get_local_linear_id(); u < MT * 4; u += NTHR) {
        const int sb = ib >> 1, ch = ib & 1;
        const int rr = u >> 2, kb = 4 * ch + (u & 3);
        const int j32 = sb * 8 + kb;
        if (j32 >= n32) continue;
        const int r = min(row0 + rr, ne01 - 1);
        float v[32];
        lb_v32(TYPE, q, wrow0 + r, ne01, n32, j32, v);
        put16(wt, rr, 32 * (u & 3),      vload16(0, v));
        put16(wt, rr, 32 * (u & 3) + 16, vload16(1, v));
    }
#else
    for (int u = get_local_linear_id(); u < MT * 8; u += NTHR) {
        const int rr = u >> 3, kb = u & 7;
        const int j32 = ib * 8 + kb;
        if (j32 >= n32) continue;
        const int r = min(row0 + rr, ne01 - 1);
        float v[32];
        lb_v32(TYPE, q, wrow0 + r, ne01, n32, j32, v);
        put16(wt, rr, 32 * kb,      vload16(0, v));
        put16(wt, rr, 32 * kb + 16, vload16(1, v));
    }
#endif
}

// ---- the GEMM body. Per token group: the lane's column `tok` (an f16
// column of xh, any valid one when the lane has none) and its output byte
// offset in dst (`outp`, negative: none); `wrow0` is the weight matrix's
// first row (an expert's, for MUL_MAT_ID). ----

#if SG == 8
#define MM_GEMM(DEQ)                                                                            \
    local half wtbuf[2][MT * (QK_K / 2)];                                                       \
    local half *wt;                                                                             \
    const int sgid = get_sub_group_id();                                                        \
    const int sm = sgid % NSGM;                                                                 \
    /* super-blocks of 256, each in two 128-weight chunks */                                     \
    const int nb = (ne00 + QK_K - 1) / QK_K;                                                    \
    const int nch = 2 * nb;                                                                     \
    float8 acc[RG][TG];                                                                         \
    _Pragma("unroll")                                                                           \
    for (int rg = 0; rg < RG; ++rg)                                                             \
        _Pragma("unroll")                                                                       \
        for (int tg = 0; tg < TG; ++tg) acc[rg][tg] = (float8)(0.f);                            \
    wt = wtbuf[0];                                                                              \
    { const int ib = 0; DEQ; }                                                                  \
    for (int c = 0; c < nch; ++c) {                                                             \
        const int sb = c >> 1, ch = c & 1;                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        wt = wtbuf[c & 1];                                                                      \
        if (active) {                                                                           \
            const int kbase = ch * (QK_K / 2);                                                  \
            const int kmax = min(8, (ne00 - sb * QK_K - kbase) / 16);                           \
            _Pragma("unroll 4")                                                                 \
            for (int kk = 0; kk < kmax; ++kk) {                                                 \
                int8 bv[TG];                                                                    \
                _Pragma("unroll")                                                               \
                for (int tg = 0; tg < TG; ++tg)                                                 \
                    bv[tg] = as_int8(vload8(0, (global const uint *)(xh + (ulong)tok[tg] * ne00 + sb * QK_K + kbase + kk * 16))); \
                _Pragma("unroll")                                                               \
                for (int rg = 0; rg < RG; ++rg) {                                               \
                    const a8_t a = LOAD_A(wt + ((sm * RG + rg) * 8 + kk) * 128);                \
                    _Pragma("unroll")                                                           \
                    for (int tg = 0; tg < TG; ++tg) acc[rg][tg] = DPAS(a, bv[tg], acc[rg][tg]); \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        if (c + 1 < nch) { const int ib = c + 1; wt = wtbuf[(c + 1) & 1]; DEQ; }                \
    }                                                                                           \
    const int r_sg = row0 + sm * RG * 8;
#else
#define MM_GEMM(DEQ)                                                                            \
    local half wt[MT * QK_K];                                                                   \
    const int sgid = get_sub_group_id();                                                        \
    const int sm = sgid % NSGM;                                                                 \
    /* super-blocks of 256; IQ4_NL's K need only be a multiple of 32, its last */            \
    /* super-block partial (kmax k16 steps) */                                                  \
    const int nb = (ne00 + QK_K - 1) / QK_K;                                                    \
    float8 acc[RG][TG];                                                                         \
    _Pragma("unroll")                                                                           \
    for (int rg = 0; rg < RG; ++rg)                                                             \
        _Pragma("unroll")                                                                       \
        for (int tg = 0; tg < TG; ++tg) acc[rg][tg] = (float8)(0.f);                            \
    for (int ib = 0; ib < nb; ++ib) {                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        DEQ;                                                                                    \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        if (!active) continue;                                                                  \
        const int kmax = min(16, (ne00 - ib * QK_K) / 16);                                      \
        _Pragma("unroll 4")                                                                     \
        for (int kk = 0; kk < kmax; ++kk) {                                                     \
            int8 bv[TG];                                                                        \
            _Pragma("unroll")                                                                   \
            for (int tg = 0; tg < TG; ++tg)                                                     \
                bv[tg] = as_int8(vload8(0, (global const uint *)(xh + (ulong)tok[tg] * ne00 + ib * QK_K + kk * 16))); \
            _Pragma("unroll")                                                                   \
            for (int rg = 0; rg < RG; ++rg) {                                                   \
                const a8_t a = LOAD_A(wt + ((sm * RG + rg) * 16 + kk) * 128);                   \
                _Pragma("unroll")                                                               \
                for (int tg = 0; tg < TG; ++tg) acc[rg][tg] = DPAS(a, bv[tg], acc[rg][tg]);     \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    const int r_sg = row0 + sm * RG * 8;
#endif
#define MM_BODY(DEQ)                                                                            \
    MM_GEMM(DEQ)                                                                                \
    _Pragma("unroll")                                                                           \
    for (int tg = 0; tg < TG; ++tg) {                                                           \
        if (outp[tg] < 0) continue;                                                             \
        global float * out = (global float *)(dst + outp[tg]);                                  \
        const float csc = xscale[tok[tg]];                                                      \
        _Pragma("unroll")                                                                       \
        for (int rg = 0; rg < RG; ++rg) {                                                       \
            const int r = r_sg + rg * 8;                                                        \
            const float8 v = acc[rg][tg] * csc;                                                 \
            if (r + 8 <= ne01) {                                                                \
                vstore8(v, 0, out + r);                                                         \
            } else {                                                                            \
                const float vv[8] = { v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7 };         \
                for (int m = 0; m < 8; ++m) if (r + m < ne01) out[r + m] = vv[m];               \
            }                                                                                   \
        }                                                                                       \
    }

// ---- MUL_MAT: dst [ne01, ne11], column t at t * nb1 ----

#define PLAIN_COLS                                                                    \
    const int row0 = get_group_id(1) * MT;                                            \
    const int sn = get_sub_group_id() / NSGM;                                         \
    const int tok0 = get_group_id(0) * NT + sn * TG * SG;                             \
    int  tok[TG];                                                                     \
    long outp[TG];                                                                    \
    _Pragma("unroll")                                                                 \
    for (int tg = 0; tg < TG; ++tg) {                                                 \
        const int t = tok0 + tg * SG + get_sub_group_local_id();                      \
        tok[tg]  = min(t, ne11 - 1);                                                  \
        outp[tg] = t < ne11 ? (long)(offd + (ulong)t * nb1) : -1;                     \
    }                                                                                 \
    const ulong wrow0 = 0;                                                            \
    const bool active = tok0 < ne11;  /* a sub-group with no column only dequantizes */

#define PLAIN_ARGS global const half * xh, global const float * xscale, global char * dst, ulong offd, ulong nb1, int ne00, int ne01, int ne11

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_q4_K_f16(global const uchar * q, global const uchar * s, global const half * d,
                                   global const half * dm, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_q45_K(wt, q, q, s, d, dm, 0, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_q5_K_f16(global const uchar * q, global const uchar * qh, global const uchar * s,
                                   global const half * d, global const half * dm, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_q45_K(wt, q, qh, s, d, dm, 1, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_q6_K_f16(global const uchar * ql, global const uchar * qh, global const char * s,
                                   global const half * d, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_q6_K(wt, ql, qh, s, d, wrow0, row0, ne01 , nb, ib))
}

// ---- F16 weights (BF16 is stored as F16 on the device) on the same XMX tiles:
// a prompt's hyper-connection products, routers and injects went through the
// generic tiled GEMM without XMX (~0.9 s a 2,048-token Flash-Next chunk, B60).
// Unit (row r, 16-value chunk of super-block ib); K past ne00 reads zero
// (ne00 a multiple of 16). ----
#if SG != 8
static inline void deq_f16w(local half * wt, global const half * w, ulong wrow0, int row0, int ne01, int ne00, int ib) {
    for (int u = get_local_linear_id(); u < MT * 16; u += NTHR) {
        const int rr = u >> 4, c = u & 15;
        const int r = min(row0 + rr, ne01 - 1);
        const int k = ib * QK_K + c * 16;
        const float16 v = k + 16 <= ne00 ? vload_half16(0, w + (wrow0 + r) * (ulong)ne00 + k) : (float16)(0.f);
        put16(wt, rr, c * 16, v);
    }
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_f16w_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_f16w(wt, (global const half *)(q + offq), wrow0, row0, ne01, ne00, ib))
}

// ---- Q8_0: int8 code + f16 block scale dequantized to f16, then fp16 DPAS.
// The two device buffers are the tree's q8_0 split (codes per 32, scale per
// 32). Same tile as the f16-weights GEMM. Chunks past ne00 are not written:
// the GEMM's kmax never reads them. kzero > 0 writes the last kzero weights
// of every row as zero (the red case: those K terms left out). ----
static inline void deq_q8_0(local half * wt, global const uchar * q, ulong offq, global const half * dp,
                            ulong wrow0, int row0, int ne01, int n32, int ib, int kzero) {
    for (int u = get_local_linear_id(); u < MT * 16; u += NTHR) {
        const int rr = u >> 4, c = u & 15;
        const int r = min(row0 + rr, ne01 - 1);
        const int kb = ib * 8 + (c >> 1);
        if (kb >= n32) continue;
        const ulong blk = (wrow0 + r) * (ulong)n32 + (ulong)kb;
        const global uchar * qq = q + offq + blk * 32 + (ulong)(c & 1) * 16;
        const float sc = (float) vload_half(blk, dp);
        const char16 cc = as_char16(vload4(0, (global const uint *) qq));
        const bool zero = kzero > 0 && ib * QK_K + c * 16 >= n32 * 32 - kzero;
        put16(wt, rr, c * 16, zero ? (float16)(0.f) : convert_float16(cc) * sc);   // k within the super-block
    }
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_q8_0_f16(global const uchar * q, ulong offq, global const half * dp, int kzero, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_q8_0(wt, q, offq, dp, wrow0, row0, ne01, ne00 / 32, ib, kzero))
}
#endif

// ---- MUL_MAT_ID routing: the (token, slot) pairs sorted by expert and cut
// into tiles of `tile` pairs of one expert (expert, first pair, count); one
// work-group. ----

#ifndef MAX_EXPERT
#define MAX_EXPERT 512
#endif

// one work-group
kernel void kernel_moe_route_tiles(global const char * ids, ulong offi, ulong nbi1, int n_used, int n_tok, int n_expert,
                                   int tile, global int * pairs, global int4 * tiles, global int * n_tiles) {
    local int cnt[MAX_EXPERT], base[MAX_EXPERT];
    const int lid = get_local_id(0), lsz = get_local_size(0);
    const int n_pairs = n_used * n_tok;
    for (int e = lid; e < n_expert; e += lsz) cnt[e] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = lid; p < n_pairs; p += lsz) {
        const int r = ((global const int *)(ids + offi + (p / n_used) * nbi1))[p % n_used];
        if (r < 0) continue;   // an expert cache's other branch: no tile, the row stays zero (filled)
        const int e = min(r, n_expert - 1);
        atomic_inc(&cnt[e]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int acc = 0, nt = 0;
        for (int e = 0; e < n_expert; ++e) {
            base[e] = acc;
            for (int s0 = 0; s0 < cnt[e]; s0 += tile) tiles[nt++] = (int4)(e, acc + s0, min(tile, cnt[e] - s0), 0);
            acc += cnt[e];
            cnt[e] = 0;
        }
        *n_tiles = nt;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = lid; p < n_pairs; p += lsz) {
        const int r = ((global const int *)(ids + offi + (p / n_used) * nbi1))[p % n_used];
        if (r < 0) continue;
        const int e = min(r, n_expert - 1);
        pairs[base[e] + atomic_inc(&cnt[e])] = p;
    }
}

// ---- an expert cache's bank in host memory: every expert some pair routes to,
// copied into a card-side mirror at the same offsets with 16-byte loads (the
// link's rate; the MoE kernels' own loads reach a tenth of it over the link).
// Group (c, e): expert e, chunk c of its bytes. ----
// n_split: an expert cache's single branch routes bank entry e as n_split + e
// (0: the ids are the bank's own)
kernel void kernel_moe_gather_experts(global const uint4 * src, global uint4 * dst, ulong n16,
                                      global const char * ids, ulong offi, ulong nbi1, int n_used, int n_tok,
                                      int n_split) {
    local int used;
    const int e = get_group_id(1);
    if (get_local_id(0) == 0) used = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int n_pairs = n_used * n_tok;
    for (int p = get_local_id(0); p < n_pairs; p += get_local_size(0)) {
        if (((global const int *)(ids + offi + (p / n_used) * nbi1))[p % n_used] == n_split + e) used = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!used) return;
    // chunk c: a contiguous slice, streamed by the group (page by page)
    const ulong base = (ulong) e * n16;
    const ulong len  = (n16 + get_num_groups(0) - 1) / get_num_groups(0);
    const ulong lo   = get_group_id(0) * len, hi = min(lo + len, n16);
    for (ulong i = lo + get_local_id(0); i < hi; i += get_local_size(0)) {
        dst[base + i] = src[base + i];
    }
}

// one token's routed pairs are distinct experts: group (c, p) copies chunk c of
// pair p's expert (a decode step; no scan of the experts)
kernel void kernel_moe_gather_pairs(global const uint4 * src, global uint4 * dst, ulong n16,
                                    global const char * ids, ulong offi, ulong nbi1, int n_used, int n_split) {
    // ids below n_split are not the bank's (slots, or -1); the bank's at id - n_split
    const int p = get_group_id(1);
    int e = ((global const int *)(ids + offi + (p / n_used) * nbi1))[p % n_used];
    if (e < n_split) return;
    // several tokens (a verify window): an expert an earlier pair routes to is
    // that pair's copy
    for (int q = 0; q < p; ++q)
        if (((global const int *)(ids + offi + (q / n_used) * nbi1))[q % n_used] == e) return;
    e -= n_split;
    // chunk c: a contiguous slice, streamed by the group (page by page)
    const ulong base = (ulong) e * n16;
    const ulong len  = (n16 + get_num_groups(0) - 1) / get_num_groups(0);
    const ulong lo   = get_group_id(0) * len, hi = min(lo + len, n16);
    for (ulong i = lo + get_local_id(0); i < hi; i += get_local_size(0)) {
        dst[base + i] = src[base + i];
    }
}

// ---- MUL_MAT_ID: experts [ne00, ne01, n_expert]; activation columns
// [ne00, ne11, n_tok] in f16; dst [ne01, n_used, n_tok]. Tiles of NT pairs
// of one expert from kernel_moe_route_tiles; pair
// p = t * n_used + i has column (i % ne11) + t * ne11 and output
// dst + t * nb2 + i * nb1. ----

#define ID_COLS                                                                       \
    const int row0 = get_group_id(1) * MT;                                            \
    const int ti = get_group_id(0);                                                   \
    if (ti >= *n_tiles) return;                                                       \
    const int4 te = tiles[ti];                                                        \
    const int sn = get_sub_group_id() / NSGM;                                         \
    int  tok[TG];                                                                     \
    long outp[TG];                                                                    \
    _Pragma("unroll")                                                                 \
    for (int tg = 0; tg < TG; ++tg) {                                                 \
        const int k = sn * TG * SG + tg * SG + get_sub_group_local_id();              \
        const int p = pairs[te.y + min(k, te.z - 1)];                                 \
        const int t = p / n_used, i = p % n_used;                                     \
        tok[tg]  = (i % ne11) + t * ne11;                                             \
        outp[tg] = k < te.z ? (long)(offd + (ulong)t * nb2 + (ulong)i * nb1) : -1;    \
    }                                                                                 \
    const ulong wrow0 = (ulong)te.x * ne01;                                           \
    const bool active = sn * TG * SG < te.z;

#define ID_ARGS global const half * xh, global const float * xscale, global const int * pairs, global const int4 * tiles, global const int * n_tiles, \
    int n_used, int ne11, global char * dst, ulong offd, ulong nb1, ulong nb2, int ne00, int ne01

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_q4_K_f16(global const uchar * q, global const uchar * s, global const half * d,
                                      global const half * dm, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_q45_K(wt, q, q, s, d, dm, 0, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_q5_K_f16(global const uchar * q, global const uchar * qh, global const uchar * s,
                                      global const half * d, global const half * dm, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_q45_K(wt, q, qh, s, d, dm, 1, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_q6_K_f16(global const uchar * ql, global const uchar * qh, global const char * s,
                                      global const half * d, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_q6_K(wt, ql, qh, s, d, wrow0, row0, ne01 , nb, ib))
}

// ---- IQ3_XXS, IQ4_XS (blocks as stored, at q + offq) and IQ4_NL (flat
// planes q, dp); plain and MUL_MAT_ID as above ----

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq3_xxs_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_iq3_xxs(wt, q + offq, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq4_xs_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_iq4_xs(wt, q + offq, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq4_nl_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_iq4_nl(wt, q + offq, dp, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_q4_0_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_q4_0(wt, q + offq, dp, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq3_xxs_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_iq3_xxs(wt, q + offq, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq4_xs_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_iq4_xs(wt, q + offq, wrow0, row0, ne01 , nb, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq4_nl_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_iq4_nl(wt, q + offq, dp, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq2_xxs_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_lb(6, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq2_xxs_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_lb(6, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq2_s_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_lb(7, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq2_s_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_lb(7, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq1_m_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_lb(8, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq1_m_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_lb(8, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_q2_0_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_lb(9, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_q2_0_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_lb(9, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq3_s_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_lb(10, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq3_s_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_lb(10, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_iq2_xs_f16(global const uchar * q, ulong offq, global const half * dp, PLAIN_ARGS) {
    PLAIN_COLS
    MM_BODY(deq_lb(11, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id_iq2_xs_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    MM_BODY(deq_lb(11, wt, q + offq, wrow0, row0, ne01, ne00 / 32, ib))
}

// ---- MUL_MAT_ID with the tokens as the a operand (the shape of arcint's
// OpenVINO patch 0064): a work-group takes up to XT pairs of one expert (a
// tile of kernel_moe_route_tiles) and XSG * SG weight rows, one row per lane.
// Per 256-weight super-block the tile's activation columns are staged in
// local memory in the a layout (LOFF, rows = pairs) once for all sub-groups;
// each lane dequantizes its row's 16 weights of a k16 step into registers as
// the b operand. The weights are read once per tile and never pass through
// local memory; the result is lane = row, component m = pair. ----

#ifndef XT
#define XT 16
#endif
#ifndef XSG
#define XSG 8
#endif
#define XGR (XT / 8)
#define XTHR (XSG * SG)
#define XSTAGE (XT * 32 / XTHR)   /* 8-half units of the staged block per work-item */
#if XSTAGE * XTHR != XT * 32
#error "mul_mm_kq_f16.cl: XT * 32 must be a multiple of the work-group size"
#endif

static inline int8 b16_pack(float16 v) {
    return as_int8(convert_half16(v));
}

static inline int8 b16_iq4(uint4 qw, int hi, float a) {
    const uchar16 c = as_uchar16(qw);
    float v[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = a * kvalues_iq4nl[hi ? c[j] >> 4 : c[j] & 0xF];
    return b16_pack(vload16(0, v));
}

static inline int8 b16_iq4_xs(global const uchar * q, ulong b, int sl) {
    global const uchar * bp = q + b * 136;
    const int kb = sl >> 1;
    const uint sh = ((global const ushort *)bp)[1];
    const int ls = ((bp[4 + kb / 2] >> (4 * (kb & 1))) & 0xF) | (((sh >> (2 * kb)) & 3) << 4);
    return b16_iq4(vload4(0, (global const uint *)(bp + 8 + 16 * kb)), sl & 1,
                   vload_half(0, (global const half *)bp) * (float)(ls - 32));
}

// IQ4_NL: blk = the row's 32-block (flat planes)
static inline int8 b16_iq4_nl(global const uchar * q, global const half * dp, ulong blk, int sl) {
    return b16_iq4(vload4(0, (global const uint *)(q + blk * 16)), sl & 1, vload_half(blk, dp));
}

static inline int8 b16_iq3_xxs(global const uchar * q, ulong b, int sl) {
    global const uchar * bp = q + b * 98;
    global const ushort * h = (global const ushort *)bp;
    const int kb = sl >> 1, l0 = 2 * (sl & 1);   /* 8-weight groups l0, l0 + 1 of 32-block kb */
    const float d = vload_half(0, (global const half *)bp);
    const uint aux = (uint)h[1 + 32 + 2 * kb] | ((uint)h[1 + 32 + 2 * kb + 1] << 16);
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    float v[16];
    #pragma unroll
    for (int ll = 0; ll < 2; ++ll) {
        const int l = l0 + ll;
        const ulong m = ksigns64[(aux >> (7 * l)) & 127];
        const uchar8 mb = as_uchar8(m);
        const ushort gw = h[1 + 4 * kb + l];
        const uchar4 g1 = as_uchar4(iq3xxs_grid[gw & 0xFF]), g2 = as_uchar4(iq3xxs_grid[gw >> 8]);
        v[8 * ll + 0] = (mb.s0 ? -db : db) * g1.s0; v[8 * ll + 1] = (mb.s1 ? -db : db) * g1.s1;
        v[8 * ll + 2] = (mb.s2 ? -db : db) * g1.s2; v[8 * ll + 3] = (mb.s3 ? -db : db) * g1.s3;
        v[8 * ll + 4] = (mb.s4 ? -db : db) * g2.s0; v[8 * ll + 5] = (mb.s5 ? -db : db) * g2.s1;
        v[8 * ll + 6] = (mb.s6 ? -db : db) * g2.s2; v[8 * ll + 7] = (mb.s7 ? -db : db) * g2.s3;
    }
    return b16_pack(vload16(0, v));
}

// Q4_K / Q5_K without dequantizing: the b operand is the codes themselves as
// halves 1024 + q, built from the code words by integer operations (0x6400
// | q, two halves an instruction), byte pairs (0, 2) and (1, 3) of each word
// (the activations are staged with the middle two of every four halves
// swapped to match, ID2_PERM). Each 32-weight sub-block j is its own DPAS
// chain, applied in f32: sum_k a_k w_k = D sc_j (chain - 1024 S_j) - M m_j S_j,
// with S_j the sub-block's activation sum (staged with the activations).
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

// staged 8 halves: as read (PERM 0), or the middle two of every four swapped
static inline uint4 id2_perm(uint4 v, const int perm) {
    if (!perm) return v;
    return (uint4)((v.s0 & 0xFFFFu) | (v.s1 << 16), (v.s0 >> 16) | (v.s1 & 0xFFFF0000u),
                   (v.s2 & 0xFFFFu) | (v.s3 << 16), (v.s2 >> 16) | (v.s3 & 0xFFFF0000u));
}

// PREP: per super-block ib, per lane (row `wr`, the expert's row of this lane,
// clamped); B16: the lane's b operand for step sl; SCALE: the factor of the
// super-block's chain; PERM: the activations' order (id2_perm)
#define ID2_BODY(PERM, PREP, B16, SCALE)                                                        \
    local half xs[XT * QK_K];                                                                   \
    const int ti = get_group_id(0);                                                             \
    if (ti >= *n_tiles) return;                                                                 \
    const int4 te = tiles[ti];                                                                  \
    const int lane = get_sub_group_local_id(), lid = get_local_linear_id();                     \
    const int r = get_group_id(1) * XTHR + get_sub_group_id() * SG + lane;                      \
    const ulong wr = (ulong)te.x * ne01 + min(r, ne01 - 1);                                     \
    const int nb = (ne00 + QK_K - 1) / QK_K;                                                    \
    long src[XSTAGE];                                                                           \
    int  dsto[XSTAGE];                                                                          \
    _Pragma("unroll")                                                                           \
    for (int j = 0; j < XSTAGE; ++j) {                                                          \
        const int u = lid + j * XTHR, t = u >> 5, o8 = u & 31;                                  \
        const int p = pairs[te.y + min(t, te.z - 1)];                                           \
        src[j]  = (long)((p % n_used) % ne11 + (p / n_used) * ne11) * ne00 + 8 * o8;            \
        dsto[j] = LOFF(t, 8 * o8);                                                              \
    }                                                                                           \
    float8 acc[XGR];                                                                            \
    _Pragma("unroll")                                                                           \
    for (int g = 0; g < XGR; ++g) acc[g] = (float8)(0.f);                                       \
    for (int ib = 0; ib < nb; ++ib) {                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        const int kleft = ne00 - ib * QK_K;                                                     \
        _Pragma("unroll")                                                                       \
        for (int j = 0; j < XSTAGE; ++j) {                                                      \
            const int o = (lid + j * XTHR) & 31;                                                \
            const uint4 v = 8 * o < kleft ? vload4(0, (global const uint *)(xh + src[j] + ib * QK_K)) : (uint4)(0); \
            vstore4(id2_perm(v, PERM), 0, (local uint *)(xs + dsto[j]));                        \
        }                                                                                       \
        PREP;                                                                                   \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        const int kmax = min(16, kleft / 16);                                                   \
        float8 tg[XGR];                                                                         \
        _Pragma("unroll")                                                                       \
        for (int g = 0; g < XGR; ++g) tg[g] = (float8)(0.f);                                    \
        for (int sl = 0; sl < kmax; ++sl) {                                                     \
            const int8 bw = B16;                                                                \
            _Pragma("unroll")                                                                   \
            for (int g = 0; g < XGR; ++g) tg[g] = DPAS(LOAD_A(xs + (g * 16 + sl) * 128), bw, tg[g]); \
        }                                                                                       \
        _Pragma("unroll")                                                                       \
        for (int g = 0; g < XGR; ++g) acc[g] = fma(tg[g], (float8)(SCALE), acc[g]);            \
    }                                                                                           \
    if (r >= ne01) return;                                                                      \
    _Pragma("unroll")                                                                           \
    for (int g = 0; g < XGR; ++g) {                                                             \
        const float vv[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3,                       \
                              acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };                     \
        _Pragma("unroll")                                                                       \
        for (int m = 0; m < 8; ++m) {                                                           \
            const int k = g * 8 + m;                                                            \
            if (k >= te.z) continue;                                                            \
            const int p = pairs[te.y + k], t = p / n_used, i = p % n_used;                      \
            global float * out = (global float *)(dst + offd + (ulong)t * nb2 + (ulong)i * nb1); \
            out[r] = vv[m] * xscale[(i % ne11) + t * ne11];                                     \
        }                                                                                       \
    }

// the Q4_K / Q5_K form of ID2_BODY (b16_q45_codes): the activations staged
// permuted, with their 32-sums xsum[j][pair]; one DPAS chain per sub-block
#define ID2_Q45_BODY(QH, Q5)                                                                    \
    local half  xs[XT * QK_K];                                                                  \
    local float xsum[8 * XT];                                                                   \
    const int ti = get_group_id(0);                                                             \
    if (ti >= *n_tiles) return;                                                                 \
    const int4 te = tiles[ti];                                                                  \
    const int lane = get_sub_group_local_id(), lid = get_local_linear_id();                     \
    const int r = get_group_id(1) * XTHR + get_sub_group_id() * SG + lane;                      \
    const ulong wr = (ulong)te.x * ne01 + min(r, ne01 - 1);                                     \
    const int nb = ne00 / QK_K;                                                                 \
    long src[XSTAGE];                                                                           \
    int  dsto[XSTAGE];                                                                          \
    _Pragma("unroll")                                                                           \
    for (int j = 0; j < XSTAGE; ++j) {                                                          \
        const int u = lid + j * XTHR, t = u >> 5, o8 = u & 31;                                  \
        const int p = pairs[te.y + min(t, te.z - 1)];                                           \
        src[j]  = (long)((p % n_used) % ne11 + (p / n_used) * ne11) * ne00 + 8 * o8;            \
        dsto[j] = LOFF(t, 8 * o8);                                                              \
    }                                                                                           \
    float8 acc[XGR];                                                                            \
    _Pragma("unroll")                                                                           \
    for (int g = 0; g < XGR; ++g) acc[g] = (float8)(0.f);                                       \
    for (int ib = 0; ib < nb; ++ib) {                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        _Pragma("unroll")                                                                       \
        for (int j = 0; j < XSTAGE; ++j) {                                                      \
            const int u = lid + j * XTHR;                                                       \
            const uint4 v = vload4(0, (global const uint *)(xh + src[j] + ib * QK_K));          \
            vstore4(id2_perm(v, 1), 0, (local uint *)(xs + dsto[j]));                           \
            const half8 hv = as_half8(v);                                                       \
            float sm = (float)hv.s0 + (float)hv.s1 + (float)hv.s2 + (float)hv.s3 +              \
                       (float)hv.s4 + (float)hv.s5 + (float)hv.s6 + (float)hv.s7;               \
            sm += intel_sub_group_shuffle_xor(sm, 1);                                           \
            sm += intel_sub_group_shuffle_xor(sm, 2);                                           \
            if ((u & 3) == 0) xsum[((u & 31) >> 2) * XT + (u >> 5)] = sm;                       \
        }                                                                                       \
        const ulong b = wr * nb + ib;                                                           \
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));                          \
        const float D = vload_half(0, d + b), M = vload_half(0, dm + b);                        \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        _Pragma("unroll")                                                                       \
        for (int sb = 0; sb < 8; ++sb) {                                                        \
            float sc, m;                                                                        \
            scale_min(sw, sb, &sc, &m);                                                         \
            const float ka = D * sc, kb = 1024.f * ka + M * m;                                  \
            const int c4 = sb >> 1, hi = sb & 1;                                                \
            float8 tg[XGR];                                                                     \
            _Pragma("unroll")                                                                   \
            for (int g = 0; g < XGR; ++g) tg[g] = (float8)(0.f);                                \
            _Pragma("unroll")                                                                   \
            for (int hh = 0; hh < 2; ++hh) {                                                    \
                const uint4 w = vload4(0, (global const uint *)(q + b * 128 + 32 * c4 + 16 * hh)); \
                const uint4 h = Q5 ? vload4(0, (global const uint *)(QH + b * 32 + 16 * hh)) : (uint4)(0); \
                const int8 bw = b16_q45_codes(w, h, 4 * hi, 2 * c4 + hi, Q5);                   \
                _Pragma("unroll")                                                               \
                for (int g = 0; g < XGR; ++g)                                                   \
                    tg[g] = DPAS(LOAD_A(xs + (g * 16 + 2 * sb + hh) * 128), bw, tg[g]);         \
            }                                                                                   \
            _Pragma("unroll")                                                                   \
            for (int g = 0; g < XGR; ++g) {                                                     \
                const float8 sx = vload8(0, xsum + sb * XT + g * 8);                            \
                acc[g] = fma(tg[g], (float8)(ka), acc[g]);                                      \
                acc[g] = fma(sx, (float8)(-kb), acc[g]);                                        \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    if (r >= ne01) return;                                                                      \
    _Pragma("unroll")                                                                           \
    for (int g = 0; g < XGR; ++g) {                                                             \
        const float vv[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3,                       \
                              acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };                     \
        _Pragma("unroll")                                                                       \
        for (int m = 0; m < 8; ++m) {                                                           \
            const int k = g * 8 + m;                                                            \
            if (k >= te.z) continue;                                                            \
            const int p = pairs[te.y + k], t = p / n_used, i = p % n_used;                      \
            global float * out = (global float *)(dst + offd + (ulong)t * nb2 + (ulong)i * nb1); \
            out[r] = vv[m] * xscale[(i % ne11) + t * ne11];                                     \
        }                                                                                       \
    }

// Q6_K the same way: the codes 1024 + q (6 bits from ql and qh), one DPAS
// chain per k16 step (its own int8 scale s): D s (chain - 1056 S) with S the
// step's activation sum, xsum[step][pair]
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

#define ID2_Q6_BODY                                                                             \
    local half  xs[XT * QK_K];                                                                  \
    local float xsum[16 * XT];                                                                  \
    const int ti = get_group_id(0);                                                             \
    if (ti >= *n_tiles) return;                                                                 \
    const int4 te = tiles[ti];                                                                  \
    const int lane = get_sub_group_local_id(), lid = get_local_linear_id();                     \
    const int r = get_group_id(1) * XTHR + get_sub_group_id() * SG + lane;                      \
    const ulong wr = (ulong)te.x * ne01 + min(r, ne01 - 1);                                     \
    const int nb = ne00 / QK_K;                                                                 \
    long src[XSTAGE];                                                                           \
    int  dsto[XSTAGE];                                                                          \
    _Pragma("unroll")                                                                           \
    for (int j = 0; j < XSTAGE; ++j) {                                                          \
        const int u = lid + j * XTHR, t = u >> 5, o8 = u & 31;                                  \
        const int p = pairs[te.y + min(t, te.z - 1)];                                           \
        src[j]  = (long)((p % n_used) % ne11 + (p / n_used) * ne11) * ne00 + 8 * o8;            \
        dsto[j] = LOFF(t, 8 * o8);                                                              \
    }                                                                                           \
    float8 acc[XGR];                                                                            \
    _Pragma("unroll")                                                                           \
    for (int g = 0; g < XGR; ++g) acc[g] = (float8)(0.f);                                       \
    for (int ib = 0; ib < nb; ++ib) {                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        _Pragma("unroll")                                                                       \
        for (int j = 0; j < XSTAGE; ++j) {                                                      \
            const int u = lid + j * XTHR;                                                       \
            const uint4 v = vload4(0, (global const uint *)(xh + src[j] + ib * QK_K));          \
            vstore4(id2_perm(v, 1), 0, (local uint *)(xs + dsto[j]));                           \
            const half8 hv = as_half8(v);                                                       \
            float sm = (float)hv.s0 + (float)hv.s1 + (float)hv.s2 + (float)hv.s3 +              \
                       (float)hv.s4 + (float)hv.s5 + (float)hv.s6 + (float)hv.s7;               \
            sm += intel_sub_group_shuffle_xor(sm, 1);                                           \
            if ((u & 1) == 0) xsum[((u & 31) >> 1) * XT + (u >> 5)] = sm;                       \
        }                                                                                       \
        const ulong b = wr * nb + ib;                                                           \
        const float D = vload_half(0, d + b);                                                   \
        barrier(CLK_LOCAL_MEM_FENCE);                                                           \
        _Pragma("unroll")                                                                       \
        for (int sl = 0; sl < 16; ++sl) {                                                       \
            const int kb = sl >> 1, hh = sl & 1, n = kb >> 2, p = kb & 3;                       \
            const float ka = D * (float)s[b * 16 + 8 * n + 2 * p + hh], kc = 1056.f * ka;       \
            const uint4 w = vload4(0, (global const uint *)(ql + b * 128 + 64 * n + 32 * (p & 1) + 16 * hh)); \
            const uint4 h = vload4(0, (global const uint *)(qh + b * 64 + 32 * n + 16 * hh));   \
            const int8 bw = b16_q6_codes(w, h, 4 * (p >> 1), 2 * p);                            \
            _Pragma("unroll")                                                                   \
            for (int g = 0; g < XGR; ++g) {                                                     \
                const float8 tg = DPAS(LOAD_A(xs + (g * 16 + sl) * 128), bw, (float8)(0.f));    \
                const float8 sx = vload8(0, xsum + sl * XT + g * 8);                            \
                acc[g] = fma(tg, (float8)(ka), acc[g]);                                         \
                acc[g] = fma(sx, (float8)(-kc), acc[g]);                                        \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    if (r >= ne01) return;                                                                      \
    _Pragma("unroll")                                                                           \
    for (int g = 0; g < XGR; ++g) {                                                             \
        const float vv[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3,                       \
                              acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };                     \
        _Pragma("unroll")                                                                       \
        for (int m = 0; m < 8; ++m) {                                                           \
            const int k = g * 8 + m;                                                            \
            if (k >= te.z) continue;                                                            \
            const int pp = pairs[te.y + k], t = pp / n_used, i = pp % n_used;                   \
            global float * out = (global float *)(dst + offd + (ulong)t * nb2 + (ulong)i * nb1); \
            out[r] = vv[m] * xscale[(i % ne11) + t * ne11];                                     \
        }                                                                                       \
    }

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id2_q4_K_f16(global const uchar * q, global const uchar * s, global const half * d,
                                       global const half * dm, ID_ARGS) {
    ID2_Q45_BODY(q, 0)
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id2_q5_K_f16(global const uchar * q, global const uchar * qh, global const uchar * s,
                                       global const half * d, global const half * dm, ID_ARGS) {
    ID2_Q45_BODY(qh, 1)
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id2_q6_K_f16(global const uchar * ql, global const uchar * qh, global const char * s,
                                       global const half * d, ID_ARGS) {
    ID2_Q6_BODY
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id2_iq3_xxs_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID2_BODY(0, (void)0, b16_iq3_xxs(q + offq, wr * nb + ib, sl), 1.f)
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id2_iq4_xs_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID2_BODY(0, (void)0, b16_iq4_xs(q + offq, wr * nb + ib, sl), 1.f)
}

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_mul_mm_id2_iq4_nl_f16(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID2_BODY(0, (void)0, b16_iq4_nl(q + offq, dp, wr * (ne00 / 32) + ib * 8 + (sl >> 1), sl), 1.f)
}

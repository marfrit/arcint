// 0071: MUL_MAT_ID on int8 DPAS (the s8 8x16x32, sub-group 16). Its own
// program: the helpers come from kq_i8_ops.inc (prepended to this text by
// ggml-opencl.cpp), and the weight tables it reads (iq3s_grid,
// kvalues_iq4nl) from mul_mm_kq_f16.cl, which is prepended too. A failure in
// this text costs the int8 arm only; the XMX GEMM program does not contain it.
//
// One work-group takes I8XT pairs of one expert (the route's tile), stages
// their int8 activations a 256-weight super-block in local memory once for
// its I8XSG sub-groups (one barrier pair a super-block), and every lane
// decodes its own weight row's 32 codes a block into the int8 b operand in
// registers. One int8 DPAS a 32-block per 8-pair group, converted to float
// and scaled by (weight d * (1 + 2 sc)) x (activation scale), K ascending.
// This is Strata's ESIMD form (esimd_moe.dp.cpp gate_up/down: decode once a
// 32-block, apply to every row tile) in OpenCL; the differences are named in
// the patch README.
//
// The staged block layout is [block][pair][32 bytes], so a 8-pair group of a
// block is 256 contiguous bytes for the sub-group block read (the a operand).
// Scales are [block][pair], so one vload8 serves a 8-pair group a block. A
// staged row is the pair's activation column (kq_i8_col), the fp16 kernel's
// form: the int8 codes are held once a column, not once a route pair.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

#ifndef SG
#define SG 16
#endif
#if SG == 16
#ifndef I8XT
#define I8XT 32
#endif
#ifndef I8XSG
#define I8XSG 8
#endif
#define I8XTHR (I8XSG * 16)

// q8_1-style int8 activations for src1's columns: one work-item a (column, 32
// of K). The column is the fp16 tile kernel's activation column, kq_i8_col
// (i % ne11) + t * ne11; with a broadcast b (ne11 = 1) a token's n_used pairs
// share one column, so the codes are stored once a column (n_cols = ne11 *
// ne12, the fp16 path's size) and not once a route pair (n_used times more at
// the served gate/up). The grid is flat and 64 wide, so a group reads and
// writes contiguous blocks of a column.
kernel void kernel_kq_mm_id_quant_i8(global const char * xsrc, ulong offx,
                                     int n_cols, int ne00,
                                     global char * xq, global float * xd) {
    const int n32 = ne00 / 32;
    const int gid = get_global_id(0);
    const int col = gid / n32, kb = gid - col * n32;
    if (col >= n_cols) return;
    global const float *x = (global const float *)(xsrc + offx) + (ulong)col * ne00 + (ulong)kb * 32;
    float xv[32];
    _Pragma("unroll")
    for (int j = 0; j < 8; ++j) {
        const float4 v = vload4(0, x + 4 * j);
        xv[4 * j] = v.s0; xv[4 * j + 1] = v.s1; xv[4 * j + 2] = v.s2; xv[4 * j + 3] = v.s3;
    }
    kq_s8 q[32]; float sc;
    kq_i8_quant32(xv, q, &sc);
    global char *dst = xq + (ulong)col * ne00 + (ulong)kb * 32;
    _Pragma("unroll")
    for (int j = 0; j < 8; ++j)
        ((global uint *)dst)[j] = (uint)(uchar)q[4 * j] | ((uint)(uchar)q[4 * j + 1] << 8) |
                                  ((uint)(uchar)q[4 * j + 2] << 16) | ((uint)(uchar)q[4 * j + 3] << 24);
    xd[(ulong)col * n32 + kb] = sc;
}

// the staged activation super-block and its scales: one place for both formats.
// Two barriers a 256-weight group (fill, then use); the last group of a K that
// is not a multiple of 256 is staged to its remaining nb blocks.
#define I8_STAGE                                                                                       \
    local uchar xa[I8XT * 256];                                                                        \
    local float xs[I8XT * 8];                                                                          \
    for (int k0 = 0; k0 < ne00; k0 += 256) {                                                           \
        const int nb = min((ne00 - k0) / 32, 8);                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                  \
        for (int u = get_local_linear_id(); u < I8XT * nb * 2; u += I8XTHR) {                            \
            const int row = u / (nb * 2), unit = u % (nb * 2);                      \
            const int j = unit >> 1, hpart = unit & 1;                                  \
            uchar16 v = (uchar16)(0);                                                  \
            /* the staged row is the pair's activation column (kq_i8_col), as the */    \
            /* fp16 kernel stages xh by the pair's column; a broadcast b shares */     \
            /* one column across a token's pairs. */                                    \
            if (te.y + row < te.y + te.z)                                              \
                v = vload16(0, (global const uchar *)(xq +                                 \
                    (ulong)kq_i8_col(pairs[te.y + row], n_used, ne11) * ne00 + (ulong)k0 + \
                    (ulong)j * 32 + (ulong)hpart * 16));                                   \
            vstore16(v, 0, xa + (j * I8XT + row) * 32 + hpart * 16);                   \
        }                                                                              \
        for (int u = get_local_linear_id(); u < I8XT * nb; u += I8XTHR) {               \
            const int j = u / I8XT, row = u % I8XT;                                     \
            xs[j * I8XT + row] = te.y + row < te.y + te.z                              \
                ? xd[(ulong)kq_i8_col(pairs[te.y + row], n_used, ne11) * (ne00 / 32) + (k0 / 32) + j] : 0.f; \
        }                                                                                              \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                  \
        for (int kb = k0; kb < k0 + nb * 32; kb += 32) {                                               \
            const int j = (kb - k0) / 32;

// the lane's accumulator loop over the block loop's body, shared by both formats
#define I8_DPAS(WS, BQ)                                                                              \
        if (active) {                                                                                \
            _Pragma("unroll")                                                                        \
            for (int g = 0; g < I8XT / 8; ++g) {                                                     \
                const short8 av = as_short8(intel_sub_group_block_read_us8((local const ushort *)(xa + (j * I8XT + g * 8) * 32))); \
                const int8 tf = intel_sub_group_i8_i8_matrix_mad_k32(av, BQ, (int8)(0));             \
                acc[g] += convert_float8(tf) * (float8)(WS) * vload8(0, xs + j * I8XT + g * 8);      \
            }                                                                                        \
        }                                                                                            \
        }                                                                                            \
    }

// the stores: component m of the g-th accumulator is the route's pair slot
// te.y + 8g + m, column r of dst
#define I8_STORE                                                                                     \
    if (active) {                                                                                    \
        _Pragma("unroll")                                                                            \
        for (int g = 0; g < I8XT / 8; ++g) {                                                         \
            const float vv[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3,                        \
                                  acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };                      \
            _Pragma("unroll")                                                                        \
            for (int m = 0; m < 8; ++m) {                                                            \
                const int slot = te.y + 8 * g + m;                                                   \
                if (slot >= te.y + te.z) continue;                                                   \
                const int pp = pairs[slot], t = pp / n_used, i = pp % n_used;                        \
                ((global float *)(dst + offd + (ulong)t * nb2 + (ulong)i * nb1))[r] = vv[m];     \
            }                                                                                        \
        }                                                                                            \
    }

#define I8_COLS                                                                                      \
    const int ti = get_group_id(0);                                                                  \
    if (ti >= *n_tiles) return;                                                                      \
    const int4 te = tiles[ti];                                                                       \
    const int lane = get_sub_group_local_id(), sgi = get_sub_group_id();                             \
    const int r = get_group_id(1) * I8XTHR + sgi * 16 + lane;                                        \
    const bool active = r < ne01;                                                                    \
    float8 acc[I8XT / 8];                                                                            \
    _Pragma("unroll")                                                                                \
    for (int g = 0; g < I8XT / 8; ++g) acc[g] = (float8)(0.f);

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_id_iq3_s_i8(global const uchar * q, ulong offq, global const half * dp,
                                      global const char * xq, global const float * xd,
                                      global const int * pairs, global const int4 * tiles, global const int * n_tiles,
                                      int n_used, int ne11, global char * dst, ulong offd, ulong nb1, ulong nb2,
                                      int ne00, int ne01) {
    I8_COLS
    const int nbr = ne00 / 256;
    const ulong nblk = (ulong)ne01 * nbr;
    const ulong rr = active ? (ulong)r : 0;
    global const uchar *B = q + offq + (ulong)te.x * nblk * 110;
    I8_STAGE
        const kq_u32 zero = 0u;
        kq_u32 w[8] = { zero, zero, zero, zero, zero, zero, zero, zero };
        float ws = 0.f;
        if (active) {
            const int sb = kq_i8_iq3s_sb(kb), p = kq_i8_iq3s_p(kb);
            const ulong bi = rr * nbr + sb;
            const uchar d0 = B[bi * 2], d1 = B[bi * 2 + 1];
            const float d = (float)as_half((ushort)((uint)d0 | ((uint)d1 << 8)));
            const int sc = kq_i8_iq3s_sc(B[nblk * 106 + bi * 4 + (p >> 1)], p);
            ws = d * (float)(1 + 2 * sc);
            // the plane fields are 4-aligned in the plane form (patch 0022), as
            // lb_expand reads them: two aligned words for the 8 qs bytes, one for
            // the 4 sign bytes; qh and the scale byte stay single-byte fields
            const uint2 qw = vload2(bi * 8 + p, (global const uint *)(B + nblk * 2));
            const kq_u32 q0 = qw.s0, q1 = qw.s1;
            const kq_u32 s0 = ((global const uint *)(B + nblk * 74))[bi * 8 + p];
            const kq_u32 qh = B[nblk * 66 + bi * 8 + p];
            kq_i8_decode_iq3s(q0, q1, s0, qh, iq3s_grid, w);
        }
        const int8 bq = (int8)(as_int(w[0]), as_int(w[1]), as_int(w[2]), as_int(w[3]),
                               as_int(w[4]), as_int(w[5]), as_int(w[6]), as_int(w[7]));
        I8_DPAS(ws, bq)
    I8_STORE
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_id_iq4_nl_i8(global const uchar * q, ulong offq, global const half * dp,
                                       global const char * xq, global const float * xd,
                                       global const int * pairs, global const int4 * tiles, global const int * n_tiles,
                                       int n_used, int ne11, global char * dst, ulong offd, ulong nb1, ulong nb2,
                                       int ne00, int ne01) {
    I8_COLS
    const int n32 = ne00 / 32;
    const ulong rr = active ? (ulong)r : 0;
    const ulong wrow0 = (ulong)te.x * ne01;
    I8_STAGE
        const kq_u32 zero = 0u;
        kq_u32 w[8] = { zero, zero, zero, zero, zero, zero, zero, zero };
        float ws = 0.f;
        if (active) {
            const ulong blk = (wrow0 + rr) * n32 + (ulong)kq_i8_iq4nl_blk(kb);
            const uint4 qd = vload4(0, (global const uint *)(q + offq + blk * 16));
            ws = (float)dp[blk];
            kq_i8_decode_iq4nl(qd.s0, qd.s1, qd.s2, qd.s3, kvalues_iq4nl, w);
        }
        const int8 bq = (int8)(as_int(w[0]), as_int(w[1]), as_int(w[2]), as_int(w[3]),
                               as_int(w[4]), as_int(w[5]), as_int(w[6]), as_int(w[7]));
        I8_DPAS(ws, bq)
    I8_STORE
}
#endif  // SG == 16

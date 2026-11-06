#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

// K-quant x q8_1 matrix-vector products for Intel Arc: the activation is
// quantized once to int8 blocks of 32 (llama.cpp's q8_1: codes, the block's
// scale and the sum of its float values), and the weights are multiplied with
// 4x8-bit integer dot products (cl_khr_integer_dot_product), the way
// llama.cpp's CUDA MMVQ path does it (vec_dot_q4_K_q8_1 and its siblings).
// The weights are ggml's flat planes (kernel_convert_block_q{4,5,6}_K).
//
// A subgroup of 16 takes N_DST rows; 8 lanes cover one 256-weight block, so
// the subgroup walks two blocks of each row per step. The *_id kernels are
// MUL_MAT_ID: work-group dimension 2 is one (expert slot, token) pair, whose
// expert is read from `ids` on the device.

#define QK_K 256
#ifndef N_DST
#define N_DST 4
#endif
#ifndef N_SIMDGROUP
#define N_SIMDGROUP 4
#endif

// One work-item per block of 32; `n` values (any number of contiguous
// columns, each a multiple of 32 long). A block is two int8 terms: codes q
// at scale d, and the codes q2 of what q leaves (at scale d2 <= d / 254),
// at xq + n and xd + n / 32 -- x = d q + d2 q2, so the integer dot products
// carry the activation to about 1/32000 of its block maximum instead of
// 1/254 (q8_1 alone moved the dense 27B's decode KL by 0.0026 nats).
kernel void kernel_quantize_q8_1(global const char * src, ulong offs, global char * xq, global float * xd,
                                 global float * xs, int n) {
    const int b = get_global_id(0);
    if (b * 32 >= n) return;
    global const float * v = (global const float *)(src + offs) + b * 32;
    float amax = 0.f, sum = 0.f;
    for (int i = 0; i < 32; ++i) { amax = fmax(amax, fabs(v[i])); sum += v[i]; }
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    float rmax = 0.f;
    for (int i = 0; i < 32; ++i) {
        const char q = convert_char_sat_rte(v[i] * id);
        xq[b * 32 + i] = q;
        rmax = fmax(rmax, fabs(v[i] - d * (float)q));
    }
    const float d2 = rmax / 127.f;
    const float id2 = d2 > 0.f ? 1.f / d2 : 0.f;
    for (int i = 0; i < 32; ++i) xq[n + b * 32 + i] = convert_char_sat_rte((v[i] - d * (float)xq[b * 32 + i]) * id2);
    xd[b] = d;
    xd[n / 32 + b] = d2;
    xs[b] = sum;
}

// The same, a sub-group of QSG lanes per block (32 / QSG values a lane, read
// coalesced), the block's maxima and sum by sub-group reductions: one
// work-item per block left 64 work-items for a 2048-wide row (7.4 us on a
// B60, 10 on an A770, measured-here). The sum is added in another order.
#ifdef QSG
__attribute__((intel_reqd_sub_group_size(QSG)))
kernel void kernel_quantize_q8_1_sg(global const char * src, ulong offs, global char * xq, global float * xd,
                                    global float * xs, int n) {
    const int b = get_group_id(0) * (get_local_size(0) / QSG) + get_sub_group_id();
    if (b * 32 >= n) return;   // uniform over the sub-group
    const int lane = get_sub_group_local_id();
    global const float * p = (global const float *)(src + offs) + b * 32;
    float v[32 / QSG];
    float amax = 0.f, sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 32 / QSG; ++i) {
        v[i] = p[i * QSG + lane];
        amax = fmax(amax, fabs(v[i]));
        sum += v[i];
    }
    amax = sub_group_reduce_max(amax);
    sum  = sub_group_reduce_add(sum);
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    char q[32 / QSG];
    float rmax = 0.f;
    #pragma unroll
    for (int i = 0; i < 32 / QSG; ++i) {
        q[i] = convert_char_sat_rte(v[i] * id);
        xq[b * 32 + i * QSG + lane] = q[i];
        rmax = fmax(rmax, fabs(v[i] - d * (float)q[i]));
    }
    rmax = sub_group_reduce_max(rmax);
    const float d2 = rmax / 127.f;
    const float id2 = d2 > 0.f ? 1.f / d2 : 0.f;
    #pragma unroll
    for (int i = 0; i < 32 / QSG; ++i) xq[n + b * 32 + i * QSG + lane] = convert_char_sat_rte((v[i] - d * (float)q[i]) * id2);
    if (lane == 0) {
        xd[b] = d;
        xd[n / 32 + b] = d2;
        xs[b] = sum;
    }
}
#endif

// The scales and mins of sub-blocks 2j and 2j+1 from the block's 12 scale
// bytes, read as three words (get_scale_min_k4 in ggml).
static inline uint kbyte(uint3 w, int k) {
    const uint v = k < 4 ? w.s0 : (k < 8 ? w.s1 : w.s2);
    return (v >> (8 * (k & 3))) & 0xFFu;
}

static inline void scale_min_pair(global const uchar * s, int j, float * sc0, float * m0, float * sc1, float * m1) {
    const uint3 w = vload3(0, (global const uint *)s);
    const int k0 = 2 * j, k1 = 2 * j + 1;
    if (j < 2) {
        *sc0 = kbyte(w, k0) & 63; *m0 = kbyte(w, k0 + 4) & 63;
        *sc1 = kbyte(w, k1) & 63; *m1 = kbyte(w, k1 + 4) & 63;
    } else {
        *sc0 = (kbyte(w, k0 + 4) & 0xF) | ((kbyte(w, k0 - 4) >> 6) << 4);  *m0 = (kbyte(w, k0 + 4) >> 4) | ((kbyte(w, k0) >> 6) << 4);
        *sc1 = (kbyte(w, k1 + 4) & 0xF) | ((kbyte(w, k1 - 4) >> 6) << 4);  *m1 = (kbyte(w, k1 + 4) >> 4) | ((kbyte(w, k1) >> 6) << 4);
    }
}

static inline int dot4_us(uint4 w, uint4 x) {
    int a = 0;
    a = dot_acc_sat_4x8packed_us_int(w.s0, x.s0, a);
    a = dot_acc_sat_4x8packed_us_int(w.s1, x.s1, a);
    a = dot_acc_sat_4x8packed_us_int(w.s2, x.s2, a);
    a = dot_acc_sat_4x8packed_us_int(w.s3, x.s3, a);
    return a;
}

// NCOLS columns per pass (a compile-time constant: the per-column values
// must stay in registers), so a short batch (a speculative verify, a few
// lanes) reads each weight once for all of them, as llama.cpp's CUDA MMVQ
// does for up to 8. Column c's activation is at c * ne00 codes (c * ne00/32
// scales and sums), its output at out + c * ostride. The MUL_MAT_ID kernels
// are used from the NCOLS = 1 build.
#ifndef NCOLS
#define NCOLS 1
#endif
#define MAXC NCOLS

static inline void reduce_store(float acc[N_DST][MAXC], global float * out, ulong ostride, int first_row, int ne01) {
    const int lane = get_sub_group_local_id();
    #pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        #pragma unroll
        for (int r = 0; r < N_DST; ++r) {
            const float t = sub_group_reduce_add(acc[r][c]);
            if (lane == 0 && first_row + r < ne01) out[c * ostride + first_row + r] = t;
        }
    }
}

// Q4_K (q5 = 0) and Q5_K (q5 = 1). Lane l covers bytes [16 (l%8), +16) of
// its block: the low nibbles of word i are the weights 16h + 4i .. +3 of
// sub-block 2j, the high ones the same positions of sub-block 2j+1
// (j = (l%8)/2, h = l%2). Q5_K's fifth bits are bit 2j (low) and 2j+1 (high)
// of the qh bytes 16h .. 16h+15. `blk0`: the first row's first block.
static inline void mv_q45_K(
    global const uchar * q, global const uchar * qhp, global const uchar * sp, global const half * dp, global const half * dmp,
    const int q5, ulong blk0, int nb, int first_row, int ne01,
    global const char * xq, global const float * xd, global const float * xs, global float * out,
    ulong ostride, ulong res) {
    const int lane = get_sub_group_local_id();
    const int p = lane & 7, j = p >> 1, h = p & 1;
    global const char  * xq2 = xq + res;       // the residual codes and scales
    global const float * xd2 = xd + res / 32;
    const int ne00 = nb * QK_K, n32 = nb * 8;

    float acc[N_DST][MAXC];
    #pragma unroll
    for (int r = 0; r < N_DST; ++r) for (int c = 0; c < MAXC; ++c) acc[r][c] = 0.f;

    for (int ib = lane >> 3; ib < nb; ib += 2) {
        const int b32lo = ib * 8 + 2 * j, b32hi = b32lo + 1;
        uint4 xl[MAXC], xh[MAXC], xl2[MAXC], xh2[MAXC];
        float dl[MAXC], dh[MAXC], sl[MAXC], sh[MAXC], dl2[MAXC], dh2[MAXC];
        #pragma unroll
        for (int c = 0; c < MAXC; ++c) {
                xl[c] = vload4(0, (global const uint *)(xq + c * ne00 + b32lo * 32 + 16 * h));
            xh[c] = vload4(0, (global const uint *)(xq + c * ne00 + b32hi * 32 + 16 * h));
            xl2[c] = vload4(0, (global const uint *)(xq2 + c * ne00 + b32lo * 32 + 16 * h));
            xh2[c] = vload4(0, (global const uint *)(xq2 + c * ne00 + b32hi * 32 + 16 * h));
            dl[c] = xd[c * n32 + b32lo]; dh[c] = xd[c * n32 + b32hi];
            dl2[c] = xd2[c * n32 + b32lo]; dh2[c] = xd2[c * n32 + b32hi];
            sl[c] = xs[c * n32 + b32lo]; sh[c] = xs[c * n32 + b32hi];
        }
        #pragma unroll
        for (int r = 0; r < N_DST; ++r) {
            if (first_row + r >= ne01) break;
            const ulong b = blk0 + (ulong)r * nb + ib;
            const uint4 w = vload4(0, (global const uint *)(q + b * 128 + 16 * p));
            uint4 wl = w & 0x0F0F0F0Fu;
            uint4 wh = (w >> 4) & 0x0F0F0F0Fu;
            if (q5) {
                const uint4 qh = vload4(0, (global const uint *)(qhp + b * 32 + 16 * h));
                wl |= ((qh >> (2 * j)) & 0x01010101u) << 4;
                wh |= ((qh >> (2 * j + 1)) & 0x01010101u) << 4;
            }
            float sc0, m0, sc1, m1;
            scale_min_pair(sp + b * 12, j, &sc0, &m0, &sc1, &m1);
            const float d  = vload_half(0, dp + b);
            const float dm = vload_half(0, dmp + b);
            #pragma unroll
            for (int c = 0; c < MAXC; ++c) {
                        const int il = dot4_us(wl, xl[c]),  ih = dot4_us(wh, xh[c]);
                const int il2 = dot4_us(wl, xl2[c]), ih2 = dot4_us(wh, xh2[c]);
                // each sub-block's sum is split over two lanes (h = 0, 1): half each
                acc[r][c] += d * (sc0 * (dl[c] * (float)il + dl2[c] * (float)il2) + sc1 * (dh[c] * (float)ih + dh2[c] * (float)ih2))
                           - dm * 0.5f * (m0 * sl[c] + m1 * sh[c]);
            }
        }
    }
    reduce_store(acc, out, ostride, first_row, ne01);
}

// Q6_K. Lane l: half n = (l%8)/4 of its block, positions t0 .. t0+7 with
// t0 = 8 (l%4). ql[64n + t] holds weights 128n + t (low nibble) and +64
// (high), ql[64n + 32 + t] weights +32 and +96, qh[32n + t] the two top bits
// of all four. The codes are 6-bit unsigned with an offset of 32, taken off
// per byte so the dot product is signed x signed.
static inline int dot2_ss(uint2 w, uint2 x) {
    const uint2 q = as_uint2(as_char8(w) - (char8)(32));
    int a = 0;
    a = dot_acc_sat_4x8packed_ss_int(q.s0, x.s0, a);
    a = dot_acc_sat_4x8packed_ss_int(q.s1, x.s1, a);
    return a;
}

static inline void mv_q6_K(
    global const uchar * ql, global const uchar * qhp, global const char * sp, global const half * dp,
    ulong blk0, int nb, int first_row, int ne01,
    global const char * xq, global const float * xd, global float * out, ulong ostride, ulong res) {
    global const char  * xq2 = xq + res;       // the residual codes and scales
    global const float * xd2 = xd + res / 32;
    const int lane = get_sub_group_local_id();
    const int p = lane & 7, n = p >> 2, t0 = (p & 3) * 8;
    const int is = 8 * n + (t0 >> 4);   // scale of weight 128n + t0; +2 per 32 further
    const int ne00 = nb * QK_K, n32 = nb * 8;

    float acc[N_DST][MAXC];
    #pragma unroll
    for (int r = 0; r < N_DST; ++r) for (int c = 0; c < MAXC; ++c) acc[r][c] = 0.f;

    for (int ib = lane >> 3; ib < nb; ib += 2) {
        const int b32 = ib * 8 + 4 * n;   // the activation's 32-blocks b32 .. b32+3
        uint2 x0[MAXC], x1[MAXC], x2[MAXC], x3[MAXC], y0[MAXC], y1[MAXC], y2[MAXC], y3[MAXC];
        float d0[MAXC], d1[MAXC], d2[MAXC], d3[MAXC], e0[MAXC], e1[MAXC], e2[MAXC], e3[MAXC];
        #pragma unroll
        for (int c = 0; c < MAXC; ++c) {
                global const char * xc = xq + c * ne00;
            x0[c] = vload2(0, (global const uint *)(xc + (b32 + 0) * 32 + t0));
            x1[c] = vload2(0, (global const uint *)(xc + (b32 + 1) * 32 + t0));
            x2[c] = vload2(0, (global const uint *)(xc + (b32 + 2) * 32 + t0));
            x3[c] = vload2(0, (global const uint *)(xc + (b32 + 3) * 32 + t0));
            d0[c] = xd[c * n32 + b32]; d1[c] = xd[c * n32 + b32 + 1]; d2[c] = xd[c * n32 + b32 + 2]; d3[c] = xd[c * n32 + b32 + 3];
            global const char * yc = xq2 + c * ne00;
            y0[c] = vload2(0, (global const uint *)(yc + (b32 + 0) * 32 + t0));
            y1[c] = vload2(0, (global const uint *)(yc + (b32 + 1) * 32 + t0));
            y2[c] = vload2(0, (global const uint *)(yc + (b32 + 2) * 32 + t0));
            y3[c] = vload2(0, (global const uint *)(yc + (b32 + 3) * 32 + t0));
            e0[c] = xd2[c * n32 + b32]; e1[c] = xd2[c * n32 + b32 + 1]; e2[c] = xd2[c * n32 + b32 + 2]; e3[c] = xd2[c * n32 + b32 + 3];
        }
        #pragma unroll
        for (int r = 0; r < N_DST; ++r) {
            if (first_row + r >= ne01) break;
            const ulong b = blk0 + (ulong)r * nb + ib;
            const uint2 a  = vload2(0, (global const uint *)(ql + b * 128 + 64 * n + t0));
            const uint2 c  = vload2(0, (global const uint *)(ql + b * 128 + 64 * n + 32 + t0));
            const uint2 hb = vload2(0, (global const uint *)(qhp + b * 64 + 32 * n + t0));
            const uint2 w0 = (a & 0x0F0F0F0Fu)        | ((hb & 0x03030303u) << 4);
            const uint2 w1 = (c & 0x0F0F0F0Fu)        | (((hb >> 2) & 0x03030303u) << 4);
            const uint2 w2 = ((a >> 4) & 0x0F0F0F0Fu) | (((hb >> 4) & 0x03030303u) << 4);
            const uint2 w3 = ((c >> 4) & 0x0F0F0F0Fu) | (((hb >> 6) & 0x03030303u) << 4);
            // the four scales: bytes is, is+2, is+4, is+6 of the 8 at 8n
            const uint2 sw = vload2(0, (global const uint *)(sp + b * 16 + 8 * n));
            const int sh = 8 * (is & 1);
            const float s0 = (float)(char)((sw.s0 >> sh) & 0xFF), s1 = (float)(char)((sw.s0 >> (sh + 16)) & 0xFF);
            const float s2 = (float)(char)((sw.s1 >> sh) & 0xFF), s3 = (float)(char)((sw.s1 >> (sh + 16)) & 0xFF);
            const float d = vload_half(0, dp + b);
            #pragma unroll
            for (int cc = 0; cc < MAXC; ++cc) {
                const int i0 = dot2_ss(w0, x0[cc]), i1 = dot2_ss(w1, x1[cc]);
                const int i2 = dot2_ss(w2, x2[cc]), i3 = dot2_ss(w3, x3[cc]);
                const int k0 = dot2_ss(w0, y0[cc]), k1 = dot2_ss(w1, y1[cc]);
                const int k2 = dot2_ss(w2, y2[cc]), k3 = dot2_ss(w3, y3[cc]);
                acc[r][cc] += d * (s0 * (d0[cc] * (float)i0 + e0[cc] * (float)k0) + s1 * (d1[cc] * (float)i1 + e1[cc] * (float)k1) +
                                   s2 * (d2[cc] * (float)i2 + e2[cc] * (float)k2) + s3 * (d3[cc] * (float)i3 + e3[cc] * (float)k3));
            }
        }
    }
    reduce_store(acc, out, ostride, first_row, ne01);
}

#define FIRST_ROW (int)((get_group_id(0) * N_SIMDGROUP + get_sub_group_id()) * N_DST)

// ---- one weight matrix times NCOLS packed columns; dst column c at
// offd + c * nb1 ----

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mv_q4_K_q8_1(
    global const uchar * q, global const uchar * s, global const half * d, global const half * dm,
    global const char * xq, global const float * xd, global const float * xs,
    global char * dst, ulong offd, ulong nb1, int ne00, int ne01) {
    const int first_row = FIRST_ROW;
    if (first_row >= ne01) return;
    const int nb = ne00 / QK_K;
    mv_q45_K(q, q, s, d, dm, 0, (ulong)first_row * nb, nb, first_row, ne01, xq, xd, xs, (global float *)(dst + offd),
             nb1 / sizeof(float), (ulong)ne00 * NCOLS);
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mv_q5_K_q8_1(
    global const uchar * q, global const uchar * qh, global const uchar * s, global const half * d, global const half * dm,
    global const char * xq, global const float * xd, global const float * xs,
    global char * dst, ulong offd, ulong nb1, int ne00, int ne01) {
    const int first_row = FIRST_ROW;
    if (first_row >= ne01) return;
    const int nb = ne00 / QK_K;
    mv_q45_K(q, qh, s, d, dm, 1, (ulong)first_row * nb, nb, first_row, ne01, xq, xd, xs, (global float *)(dst + offd),
             nb1 / sizeof(float), (ulong)ne00 * NCOLS);
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mv_q6_K_q8_1(
    global const uchar * ql, global const uchar * qh, global const char * s, global const half * d,
    global const char * xq, global const float * xd,
    global char * dst, ulong offd, ulong nb1, int ne00, int ne01) {
    const int first_row = FIRST_ROW;
    if (first_row >= ne01) return;
    const int nb = ne00 / QK_K;
    mv_q6_K(ql, qh, s, d, (ulong)first_row * nb, nb, first_row, ne01, xq, xd, (global float *)(dst + offd),
            nb1 / sizeof(float), (ulong)ne00 * NCOLS);
}

// ---- MUL_MAT_ID: expert matrices [ne00, ne01, n_expert], activations
// [ne00, ne11, n_tokens] quantized contiguously, ids [n_used, n_tokens] ----
// Pair k = get_group_id(2): slot i = k % n_used, token t = k / n_used; its
// expert is ids[t][i], its column (i % ne11) + t * ne11, its output
// dst + t * nb2 + i * nb1.

#define ID_PROLOGUE                                                                         \
    const int first_row = FIRST_ROW;                                                        \
    if (first_row >= ne01) return;                                                          \
    const int nb = ne00 / QK_K;                                                             \
    const int k = get_group_id(2);                                                          \
    const int i = k % n_used, t = k / n_used;                                               \
    const int e = ((global const int *)((global const char *)ids + offi + t * nbi1))[i];    \
    if (e < 0) return;   /* defensive: an expert cache never routes -1 to these types */     \
    const int col = (i % ne11) + t * ne11;                                                  \
    const ulong blk0 = ((ulong)e * ne01 + first_row) * nb;                                  \
    global const char * cxq = xq + (ulong)col * ne00;                                       \
    global const float * cxd = xd + (ulong)col * (ne00 / 32);                               \
    global float * out = (global float *)(dst + offd + t * nb2 + i * nb1);

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mv_id_q4_K_q8_1(
    global const uchar * q, global const uchar * s, global const half * d, global const half * dm,
    global const char * xq, global const float * xd, global const float * xs,
    global const char * ids, ulong offi, ulong nbi1, int n_used, int ne11,
    global char * dst, ulong offd, ulong nb1, ulong nb2, int ne00, int ne01, ulong res) {
    ID_PROLOGUE
    mv_q45_K(q, q, s, d, dm, 0, blk0, nb, first_row, ne01, cxq, cxd, xs + (ulong)col * (ne00 / 32), out, 0, res);
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mv_id_q5_K_q8_1(
    global const uchar * q, global const uchar * qh, global const uchar * s, global const half * d, global const half * dm,
    global const char * xq, global const float * xd, global const float * xs,
    global const char * ids, ulong offi, ulong nbi1, int n_used, int ne11,
    global char * dst, ulong offd, ulong nb1, ulong nb2, int ne00, int ne01, ulong res) {
    ID_PROLOGUE
    mv_q45_K(q, qh, s, d, dm, 1, blk0, nb, first_row, ne01, cxq, cxd, xs + (ulong)col * (ne00 / 32), out, 0, res);
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mv_id_q6_K_q8_1(
    global const uchar * ql, global const uchar * qh, global const char * s, global const half * d,
    global const char * xq, global const float * xd,
    global const char * ids, ulong offi, ulong nbi1, int n_used, int ne11,
    global char * dst, ulong offd, ulong nb1, ulong nb2, int ne00, int ne01, ulong res) {
    ID_PROLOGUE
    mv_q6_K(ql, qh, s, d, blk0, nb, first_row, ne01, cxq, cxd, out, 0, res);
}

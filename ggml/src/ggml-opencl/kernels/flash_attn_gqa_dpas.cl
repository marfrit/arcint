#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
// 0088: the local-memory strided block read for the int8 a operand
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

// Decode and verify attention (1-8 query rows) on the XMX units, Xe2
// (sub-group 16), after NInfer's small-T kernel (src/ops/attention/
// causal_softmax/small_t_bf16.cuh): one work-group per (KV head, KV split)
// takes every query head of that KV head and every query row, so K and V are
// read once per work-group instead of once per (query head, row).
//
// Query "rows" are (head, token) pairs: row m is query head hk * G + m % G,
// token m / G, G = n_head / n_head_kv. GQA_SIDES sub-groups own 8 rows (the
// DPAS M), each a range of the output columns; with GQA_S_SPLIT they split S
// by 16-key group and exchange it through local memory. Per BK keys: S =
// Q K^T by DPAS (Q the a operand, kept in registers; K the b operand read
// straight from the cache by its key's lane, or staged with GQA_K_STAGED),
// then V staged in local memory as f16, transposed with the key fastest and
// converted from its cache type once (flash_attn_dpas.cl's staging), one
// barrier covering V and the S exchange; the mask added, the online softmax
// in f32, and O += P V by DPAS (P is S's layout read as the a operand). The
// result is flash_attn_f32_merge's partial record (m, l and the unnormalized
// O of each (head, token) and split).
//
// Measured on the B60 (one attention layer, dense 27B geometry, f16, 131,072
// keys, 6 rows): f16 K staged 5.27 ms, direct 4.83; 2 sides against 1 (10.6) or
// 4 (6.96); BK 32 against 16 (4.91) or 64 (9.67). GQA_SIDES, GQA_K_STAGED and
// BK are the switches a structural search varies (docs/campaigns/
// kernel-autotune-ga.md).
//
// 0088 (GGML_OPENCL_FA_GQA_I8=1, q8_0 K only): Q K^T as s8 x s8 DPAS on the
// cache's raw bytes, Q quantized to int8 once a call (one RNE scale per row
// per 32-dim block), the integer result scaled by dq[row] * dk[key]; P V stays
// f16 DPAS. 0089 (GQA_ST, the 1-3 row form): the raw q8_0 key rows staged
// through local memory first. Both are rounding-only (Q's int8 rounding);
// off, this source compiles to the f16 form it was.
//
// K and V as f16 (0), q8_0 (1) or q4_0 (2) (KT, VT). DK = DV, 256 or 128. No ALiBi
// or softcap (the host keeps those on flash_attn_f32_f16_q1_split_intel).

#ifndef DK
#define DK 256
#endif
#define DV DK
#define KC (DK / 16)            // k chunks of the Q K^T product
#ifndef GQA_SIDES
#define GQA_SIDES 2             // sub-groups sharing 8 rows, each a column range
#endif
// The sides of a tile split Q K^T by 16-key group and exchange S through
// local memory (GQA_S_SPLIT, default with two or more sides; GQA_NO_S_SPLIT
// turns it off). Found by the structural search (tools/kq_tune/fagqa.py,
// docs/campaigns/kernel-autotune-ga.md): one attention layer at the dense
// 27B's geometry, 4-8 rows, 32k and 131k keys, f16, 0.78x 0.5.9's time with
// the host's split of 512 keys, measured-here.
#if GQA_SIDES > 1 && !defined(GQA_NO_S_SPLIT) && !defined(GQA_S_SPLIT)
#define GQA_S_SPLIT
#endif
#if GQA_SIDES < 1 || DV % (16 * GQA_SIDES) != 0
#error "GQA_SIDES must split DV into 16-column tiles"
#endif
#define NDH (DV / GQA_SIDES / 16)   // 16-column output tiles a sub-group
#ifndef BK
#define BK 32                   // keys a staged tile
#endif
#define NKB (BK / 16)
#if DK % 64 != 0 || DV % 64 != 0
#error "flash_attn_gqa_dpas.cl stages K and V rows in 64-value units"
#endif
#define M_INIT (-3.0e38f)
#define PARTIAL (2 + DV)
#ifndef KT
#define KT 0
#endif
#ifndef VT
#define VT 0
#endif
// K read straight from the cache when it is f16, staged (converted once a
// work-group) when it is quantized: direct quantized K has every sub-group
// convert the same blocks (served, q8_0 KV, 62,597 tokens deep: 10.8 t/s
// decode against 15.6 staged, measured-here); GQA_K_DIRECT / GQA_K_STAGED
// override
#if !defined(GQA_K_DIRECT) && !defined(GQA_K_STAGED) && KT != 0
#define GQA_K_STAGED
#endif

#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(a, b, c)

// 0088: the int8 Q K^T of the fork's DPAS decode attention (arc-b580
// fattn-dec.cpp:688-747): Q rows int8 once a call, the key's raw q8_0 codes
// as the b operand, the integer result scaled by dq[row][block] *
// dk[key][block]. Built only when GGML_OPENCL_FA_GQA_I8 is on and K is q8_0;
// otherwise the file is the f16 code it was.
#if GQA_I8 && KT == 1
#define QB (DK / 32)            // 32-dim q8_0 blocks a key row
#define KROW (QB * 34 / 4)      // a raw q8_0 row in dwords
#define KSTR (KROW + 1)         // staged stride: one dword more (bank spread)
#define DPAS8(a, b) intel_sub_group_i8_i8_matrix_mad_k32(a, b, (int8)(0))
#endif

// values 8 c8 .. 8 c8 + 7 of a row (flash_attn_dpas.cl's load8)
static inline half8 load8(const int t, const global char * row, const int c8) {
    if (t == 0) return vload8(c8, (const global half *) row);
    const global char * blk = row + (c8 >> 2) * (t == 1 ? 34 : 18);
    const float d = vload_half(0, (const global half *) blk);
    if (t == 1) return convert_half8(convert_float8(vload8(0, blk + 2 + (c8 & 3) * 8)) * d);
    // q4_0: values 0..15 the low nibbles of qs[0..15], 16..31 the high ones
    const uchar8 q = vload8(0, (const global uchar *) blk + 2 + (c8 & 1) * 8);
    const uchar8 n = (c8 & 2) ? (q >> (uchar8)(4)) : (q & (uchar8)(15));
    return convert_half8((convert_float8(n) - 8.f) * d);
}

#ifndef GQA_K_STAGED
// the b operand of Q K^T straight from the cache: values 16c .. 16c + 15 of
// this lane's key row (16 consecutive values being the packed pair layout)
static inline int8 kb16(const int t, const global char * row, const int c) {
    if (t == 0) return as_int8(vload8(0, (const global uint *)(row + c * 32)));
    const global char * blk = row + (c >> 1) * (t == 1 ? 34 : 18);
    const float d = vload_half(0, (const global half *) blk);
    if (t == 1) return as_int8(convert_half16(convert_float16(vload16(0, blk + 2 + (c & 1) * 16)) * d));
    const uchar16 q = vload16(0, (const global uchar *) blk + 2);
    const uchar16 n = (c & 1) ? (q >> (uchar16)(4)) : (q & (uchar16)(15));
    return as_int8(convert_half16((convert_float16(n) - 8.f) * d));
}
#endif

#if GQA_I8 && KT == 1
// the b operand of the int8 Q K^T: a key's 32 raw code bytes as 8 ints (lane =
// key; s8 bits, so the byte sign is the code's), no conversion pass
#define FA_GQA_I8_B32_BODY(P)                                     \
    const uchar16 a16 = vload16(0, (P));                          \
    const uchar16 b16 = vload16(0, (P) + 16);                     \
    int8 r;                                                       \
    r.s0 = (int)a16.s0 | ((int)a16.s1 << 8) | ((int)a16.s2 << 16) | ((int)a16.s3 << 24); \
    r.s1 = (int)a16.s4 | ((int)a16.s5 << 8) | ((int)a16.s6 << 16) | ((int)a16.s7 << 24); \
    r.s2 = (int)a16.s8 | ((int)a16.s9 << 8) | ((int)a16.sa << 16) | ((int)a16.sb << 24); \
    r.s3 = (int)a16.sc | ((int)a16.sd << 8) | ((int)a16.se << 16) | ((int)a16.sf << 24); \
    r.s4 = (int)b16.s0 | ((int)b16.s1 << 8) | ((int)b16.s2 << 16) | ((int)b16.s3 << 24); \
    r.s5 = (int)b16.s4 | ((int)b16.s5 << 8) | ((int)b16.s6 << 16) | ((int)b16.s7 << 24); \
    r.s6 = (int)b16.s8 | ((int)b16.s9 << 8) | ((int)b16.sa << 16) | ((int)b16.sb << 24); \
    r.s7 = (int)b16.sc | ((int)b16.sd << 8) | ((int)b16.se << 16) | ((int)b16.sf << 24)
static inline int8 fa_gqa_i8_b32_g(const global uchar * p) { FA_GQA_I8_B32_BODY(p); return r; }
static inline int8 fa_gqa_i8_b32_l(const local uchar * p) { FA_GQA_I8_B32_BODY(p); return r; }
#undef FA_GQA_I8_B32_BODY
#endif

// 16 consecutive values (two 8-value units); for f16 one 32-byte load, so the
// staging load uses a whole 32-byte sector rather than half of one
static inline half16 load16(const int t, const global char * row, const int c16) {
    if (t == 0) return vload16(c16, (const global half *) row);
    const half8 a = load8(t, row, 2 * c16);
    const half8 b = load8(t, row, 2 * c16 + 1);
    return (half16)(a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7,
                    b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7);
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void flash_attn_gqa_dpas(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias, const float m0, const float m1, const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1, const ulong mask_nb2, const ulong mask_nb3,
    const int mask_ne2, const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
#if defined(GQA_K_STAGED) && !(GQA_I8 && KT == 1)
    local half ks[BK * DK];     // [key][DK]
#endif
#if GQA_I8 && KT == 1
    local uint q8s[8 * QB * 8 * 8];     // [tile][block][row][8 dwords]: the DPAS a operand
    local float q8d[8 * QB * 8];       // [tile][block][row]: dq = amax / 127
#if GQA_ST
    local uint ks8[BK * KSTR];         // [key][dword]: the raw q8_0 rows
#endif
#endif
    local half vt[DV * BK];     // [column][key]
#ifdef GQA_S_SPLIT
    // the sides of a tile split Q K^T by 16-key group and exchange S here,
    // up to 8 tiles: [tile][group][lane][row], rows contiguous per lane, so
    // the exchange is one vstore8 / vload8
    local float sx[8 * NKB * 16 * 8];
#endif

    const int lane  = get_sub_group_local_id();
    const int sg    = get_sub_group_id();
    const int lid   = get_local_id(0), nthr = get_local_size(0);
    const int tile  = sg / GQA_SIDES, side = sg % GQA_SIDES;
    const int hk    = get_group_id(1) % n_head_kv;
    const int batch = get_group_id(1) / n_head_kv;
    const int split = get_group_id(2);
    const int G     = n_head / n_head_kv;
    const int n_rows = G * n_q;

    const int kv_start = split * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);

    // this sub-group's rows: (query head, token), valid when m < n_rows
    int  rh[8], rt[8];
    bool rv[8];
    for (int r = 0; r < 8; ++r) {
        const int m = tile * 8 + r;
        rv[r] = m < n_rows;
        rh[r] = hk * G + (rv[r] ? m % G : 0);
        rt[r] = rv[r] ? m / G : 0;
    }

    float m[8];
    float8 lv = (float8)(0.f);
    for (int r = 0; r < 8; ++r) m[r] = M_INIT;
    float8 o[NDH];
    for (int i = 0; i < NDH; ++i) o[i] = (float8)(0.f);

    // Q (scaled, half) as the a operand: component r = row r, lane = column
#if GQA_I8 && KT == 1
    // Q -> int8 once a call, one RNE scale per row per 32-dim block; the
    // sub-group's QK^T then reads the a operand from local memory (the rows
    // are shared by the sides, so the quantization is done by the work-group)
    {
        // every row of every row tile the batch needs, not the tile count:
        // (row, block) pairs a call (see fa_gqa_i8_stage_rows)
        for (int e = lid; e < fa_gqa_i8_stage_rows(n_rows) * QB; e += nthr) {
            const int r = e / QB, b = e % QB;
            float x[32];
            if (r < n_rows) {
                const int rhq = hk * G + (r % G), rtq = r / G;
                const global float * qrow = (const global float *)((const global char *) q_void + q_offset +
                    batch * q_nb3 + rhq * q_nb2 + rtq * q_nb1) + b * 32;
                for (int i = 0; i < 32; ++i) x[i] = qrow[i] * scale;
            } else {
                for (int i = 0; i < 32; ++i) x[i] = 0.f;
            }
            float dq; int q[32];
            fa_gqa_i8_quant32(x, &dq, q);
            const int t = r / 8, rr = r % 8;
            q8d[(t * QB + b) * 8 + rr] = dq;
            local uint * qrow = q8s + ((t * QB + b) * 8 + rr) * 8;
            for (int j = 0; j < 8; ++j) {
                qrow[j] = (uint)(q[4 * j] & 0xFF) | ((uint)(q[4 * j + 1] & 0xFF) << 8) |
                          ((uint)(q[4 * j + 2] & 0xFF) << 16) | ((uint)(q[4 * j + 3] & 0xFF) << 24);
            }
        }
    }
#else
    short8 qa[KC];
    for (int c = 0; c < KC; ++c) {
        half qv[8];
        for (int r = 0; r < 8; ++r) {
            const global float * qrow = (const global float *)((const global char *) q_void + q_offset +
                batch * q_nb3 + rh[r] * q_nb2 + rt[r] * q_nb1);
            qv[r] = rv[r] ? (half)(qrow[c * 16 + lane] * scale) : 0.0h;
        }
        qa[c] = as_short8((half8)(qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6], qv[7]));
    }
#endif
    const global char * kb = (const global char *) k_void + k_offset + batch * k_nb3 + hk * k_nb2;
    const global char * vb = (const global char *) v_void + v_offset + batch * v_nb3 + hk * v_nb2;
    const global half * mrow[8];
    for (int r = 0; r < 8; ++r) {
        mrow[r] = mask_void != 0
            ? (const global half *)((const global char *) mask_void + mask_offset + (batch % mask_ne3) * mask_nb3 +
                                    (rh[r] % mask_ne2) * mask_nb2 + rt[r] * mask_nb1)
            : 0;
    }

    for (int k0 = kv_start; k0 < kv_end; k0 += BK) {
        barrier(CLK_LOCAL_MEM_FENCE);
#if defined(GQA_K_STAGED) && !(GQA_I8 && KT == 1)
        for (int u = lid; u < BK * DK / 64; u += nthr) {
            const int key = u / (DK / 64), c64 = u % (DK / 64), kk = min(k0 + key, kv_end - 1);
            const global char * krow = kb + kk * k_nb1;
            vstore16(load16(KT, krow, 4 * c64), 0, ks + key * DK + 64 * c64);
            vstore16(load16(KT, krow, 4 * c64 + 1), 0, ks + key * DK + 64 * c64 + 16);
            vstore16(load16(KT, krow, 4 * c64 + 2), 0, ks + key * DK + 64 * c64 + 32);
            vstore16(load16(KT, krow, 4 * c64 + 3), 0, ks + key * DK + 64 * c64 + 48);
        }
        // staged K is read through local memory, so it needs a barrier before
        // QK^T; K direct (f16) reads global and needs none
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if GQA_ST && GQA_I8 && KT == 1
        // the raw q8_0 rows staged once a work-group: coalesced dword reads,
        // then lane = key reads its own block from local memory (the 1-3 row
        // form; the fork's ST mode with nq == 1, fattn-dec.cpp:703-731)
        for (int u = lid; u < BK * KROW; u += nthr) {
            const int key = u / KROW, d = u % KROW, kk = min(k0 + key, kv_end - 1);
            ks8[key * KSTR + d] = ((const global uint *)(kb + kk * k_nb1))[d];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        // QK^T
        float8 s[NKB];
        for (int g = 0; g < NKB; ++g) {
#ifdef GQA_S_SPLIT
            if (g % GQA_SIDES != side) continue;
#endif
            float8 acc = (float8)(0.f);
#if GQA_I8 && KT == 1
            for (int b = 0; b < QB; ++b) {
                // the s8 x s8 a operand: 8 rows x (dims 2l, 2l+1), the strided
                // block read over a row-major [row][32 codes] tile (0071's form)
                const short8 av = as_short8(intel_sub_group_block_read_us8(
                    (const local ushort *)(q8s + ((tile * QB + b) * 8) * 8)));
                int8 bv; float dk;
#if GQA_ST
                const local uchar * bp = (const local uchar *)(ks8 + (g * 16 + lane) * KSTR) + fa_gqa_i8_blk(b);
                dk = (float) as_half((ushort)(bp[0] | (bp[1] << 8)));
                bv = fa_gqa_i8_b32_l(bp + 2);
#else
                const global uchar * bp = (const global uchar *)(kb + min(k0 + g * 16 + lane, kv_end - 1) * k_nb1) + fa_gqa_i8_blk(b);
                dk = (float) vload_half(0, (const global half *) bp);
                bv = fa_gqa_i8_b32_g(bp + 2);
#endif
                acc += convert_float8(DPAS8(av, bv)) * vload8(0, q8d + (tile * QB + b) * 8) *
                       (float8)(dk, dk, dk, dk, dk, dk, dk, dk);
            }
#else
            for (int c = 0; c < KC; ++c) {
#ifndef GQA_K_STAGED
                const int8 b = kb16(KT, kb + min(k0 + g * 16 + lane, kv_end - 1) * k_nb1, c);
#else
                const int8 b = as_int8(vload8(0, (local const uint *)(ks + (g * 16 + lane) * DK + c * 16)));
#endif
                acc = DPAS(qa[c], b, acc);
            }
#endif
            s[g] = acc;
#ifdef GQA_S_SPLIT
            vstore8(acc, 0, sx + ((tile * NKB + g) * 16 + lane) * 8);
#endif
        }
        // QK^T ran before the V staging, so the one barrier below covers both
        // the V tile and the S exchange
        // V transposed with the key fastest: neighbouring lanes store
        // neighbouring halves
        for (int u = lid; u < BK * DV / 64; u += nthr) {
            const int key = u % BK, c64 = u / BK, kk = min(k0 + key, kv_end - 1);
            const global char * vrow = vb + kk * v_nb1;
            const half16 v0 = load16(VT, vrow, 4 * c64);
            const half16 v1 = load16(VT, vrow, 4 * c64 + 1);
            const half16 v2 = load16(VT, vrow, 4 * c64 + 2);
            const half16 v3 = load16(VT, vrow, 4 * c64 + 3);
            const half vv[64] = { v0.s0, v0.s1, v0.s2, v0.s3, v0.s4, v0.s5, v0.s6, v0.s7,
                                  v0.s8, v0.s9, v0.sa, v0.sb, v0.sc, v0.sd, v0.se, v0.sf,
                                  v1.s0, v1.s1, v1.s2, v1.s3, v1.s4, v1.s5, v1.s6, v1.s7,
                                  v1.s8, v1.s9, v1.sa, v1.sb, v1.sc, v1.sd, v1.se, v1.sf,
                                  v2.s0, v2.s1, v2.s2, v2.s3, v2.s4, v2.s5, v2.s6, v2.s7,
                                  v2.s8, v2.s9, v2.sa, v2.sb, v2.sc, v2.sd, v2.se, v2.sf,
                                  v3.s0, v3.s1, v3.s2, v3.s3, v3.s4, v3.s5, v3.s6, v3.s7,
                                  v3.s8, v3.s9, v3.sa, v3.sb, v3.sc, v3.sd, v3.se, v3.sf };
            for (int j = 0; j < 64; ++j) vt[(64 * c64 + j) * BK + key] = vv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef GQA_S_SPLIT
        for (int g = 0; g < NKB; ++g) {
            if (g % GQA_SIDES == side) continue;
            s[g] = vload8(0, sx + ((tile * NKB + g) * 16 + lane) * 8);
        }
#endif
        float mnew[8];
        for (int r = 0; r < 8; ++r) mnew[r] = m[r];
        for (int g = 0; g < NKB; ++g) {
            const int key = k0 + g * 16 + lane;
            const bool kvis = key < kv_end;
            float mk[8];
            for (int r = 0; r < 8; ++r) {
                if (kvis && rv[r]) mk[r] = mrow[r] != 0 ? (float) mrow[r][key] : 0.f;
                else mk[r] = M_INIT;
            }
            // -inf (masked) gives s + -inf = -inf, clamped below to M_INIT; no
            // per-element -inf test needed
            const float8 m8 = (float8)(mk[0], mk[1], mk[2], mk[3], mk[4], mk[5], mk[6], mk[7]);
            s[g] = fmax(s[g] + m8, (float8)(M_INIT));
        }
        // one reduction per row over all groups instead of one per (row, group)
        float8 mx8 = s[0];
        for (int g = 1; g < NKB; ++g) mx8 = fmax(mx8, s[g]);
        {
            const float mv[8] = { mx8.s0, mx8.s1, mx8.s2, mx8.s3, mx8.s4, mx8.s5, mx8.s6, mx8.s7 };
            for (int r = 0; r < 8; ++r) mnew[r] = fmax(mnew[r], sub_group_reduce_max(mv[r]));
        }
        float corr[8];
        for (int r = 0; r < 8; ++r) {
            corr[r] = m[r] > 0.5f * M_INIT ? exp(m[r] - mnew[r]) : 0.f;
            m[r]  = mnew[r];
        }
        const float8 cv = (float8)(corr[0], corr[1], corr[2], corr[3], corr[4], corr[5], corr[6], corr[7]);
        for (int i = 0; i < NDH; ++i) o[i] *= cv;
        lv *= cv;
        short8 pa[NKB];
        {
            // P for both key groups at once, as float8 vectors
            const float8 m8 = (float8)(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
            const float8 mcut = (float8)(0.5f * M_INIT);
            for (int g = 0; g < NKB; ++g) {
                float8 e = native_exp(s[g] - m8);
                e = select((float8)(0.f), e, s[g] > mcut);   // M_INIT entries -> 0
                const half8 ph = convert_half8(e);
                lv += e;
                pa[g] = as_short8(ph);
            }
        }
        for (int i = 0; i < NDH; ++i) {
            const int col = side * (DV / GQA_SIDES) + i * 16 + lane;
            float8 acc = o[i];
            for (int g = 0; g < NKB; ++g) {
                const int8 b = as_int8(vload8(0, (local const uint *)(vt + col * BK + g * 16)));
                acc = DPAS(pa[g], b, acc);
            }
            o[i] = acc;
        }
    }

    // the partial records: m and l by the left half's lane 0, O by column
    const float larr[8] = { lv.s0, lv.s1, lv.s2, lv.s3, lv.s4, lv.s5, lv.s6, lv.s7 };
    for (int r = 0; r < 8; ++r) {
        const float lr = sub_group_reduce_add(larr[r]);
        if (!rv[r]) continue;
        const ulong rec_idx = (((ulong) batch * n_head + rh[r]) * n_q + rt[r]) * n_splits + split;
        global float * rec = partial_void + rec_idx * (ulong) PARTIAL;
        if (side == 0 && lane == 0) { rec[0] = m[r]; rec[1] = lr; }
        for (int i = 0; i < NDH; ++i) {
            const float ov[8] = { o[i].s0, o[i].s1, o[i].s2, o[i].s3, o[i].s4, o[i].s5, o[i].s6, o[i].s7 };
            rec[2 + side * (DV / GQA_SIDES) + i * 16 + lane] = ov[r];
        }
    }
}

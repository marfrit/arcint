#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

// Flash attention for a prompt on the XMX units, Intel Xe2 (sub-group 16),
// Q as f16 (flash_attn_q_to_f16, scaled), f16 K/V, f16 mask, after vLLM's head-256 policy for Xe2
// (vllm-xpu-kernels csrc/xpu/attn/fmha_utils.hpp chunk_policy_head256: 8
// query rows a sub-group with all output columns, NSG sub-groups a
// work-group, 32-key tiles) and sycl-tla's mainloop (xe_fmha_fwd_mainloop.hpp:
// K and V straight from global memory into registers by 2D block reads, the
// next tile prefetched into the cache, no local memory, no barrier).
// - S = Q K^T: the a operands 2D reads of 8 rows x 32 dims of Q, two 16-dim
//   chunks a read (16 dims a read under -DFA2_Q1; re-read
//   each tile as sycl-tla does: Q in registers left no room at head size
//   256), the b operand a transposed 2D read of 16 keys x 16 dims (lane =
//   key, 8 dwords = dims);
// - P V: the b operand a transforming 2D read of 16 keys x 16 dims (lane =
//   dim, 8 dwords = pairs of keys, VNNI), so V needs no transpose;
// - the online softmax and the mask as flash_attn_dpas.cl (f32 S, O, m, l).
// The K/V rows must be f16, 64-byte aligned at the head's base, the row
// pitch a multiple of 16 bytes (the host checks and keeps flash_attn_dpas.cl
// otherwise). FA_CARRY: one block of keys a call, the softmax state carried
// as in flash_attn_dpas.cl; and the keys split across work-groups (n_split >
// 1, llama.cpp CUDA's parallel_blocks, fattn-common.cuh launch_fattn): the
// work-groups of split i take keys [i * split_len, (i + 1) * split_len), each
// with its own state, merged by flash_attn_split_combine.

#ifndef DK
#define DK 256
#endif
#define DV DK
#ifndef NSG
#define NSG 32
#endif
#define BK 32
#define SG 16
#define BQ (NSG * 8)
#define KC (DK / 16)
#define NKB (BK / 16)
#define NDC (DV / 16)
#define NTHR (NSG * SG)

#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(a, b, c)
#define M_INIT (-3.0e38f)

// The reads use IGC's builtins under the cl_intel_subgroup_2d_block_io
// functions: those write a private array, and IGC kept such arrays in
// memory (a store and a load a read). Width, height and pitch minus one.
ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long base, int w, int h, int p, int2 c);
ushort16 __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(long base, int w, int h, int p, int2 c);
uint8   __builtin_IB_subgroup_block_read_flat_transpose_u32_k8(long base, int w, int h, int p, int2 c);
uint8   __builtin_IB_subgroup_block_read_flat_transform_u16_k16(long base, int w, int h, int p, int2 c);
// Q rows y .. y + 7, dims 16 c .. 16 c + 15: lane = dim, component = row (the a operand)
#define QREAD(c, y) as_short8(__builtin_IB_subgroup_block_read_flat_u16_m8k16v1((long) qb, DK * 2 - 1, n_q - 1, qp - 1, (int2)(16 * (c), (y))))
// Q rows y .. y + 7, dims 32 j .. 32 j + 31: two a operands from one read
#define QREAD2(j, y) __builtin_IB_subgroup_block_read_flat_u16_m8k16v2((long) qb, DK * 2 - 1, n_q - 1, qp - 1, (int2)(32 * (j), (y)))
// 16 keys from row y, dims 16 c .. 16 c + 15: lane = key (the b operand of S)
#define KREAD(c, y) as_int8(__builtin_IB_subgroup_block_read_flat_transpose_u32_k8((long) kb, kw - 1, n_kv - 1, kp - 1, (int2)(8 * (c), (y) + FA2_RED_SHIFT)))
// the red case (-DFA2_RED): every key read one row off
#ifdef FA2_RED
#define FA2_RED_SHIFT 1
#else
#define FA2_RED_SHIFT 0
#endif
// 16 keys from row y, dims 16 c .. 16 c + 15: lane = dim, key pairs packed (the b operand of P V)
#define VREAD(c, y) as_int8(__builtin_IB_subgroup_block_read_flat_transform_u16_k16((long) vb, vw - 1, n_kv - 1, vp - 1, (int2)(16 * (c), (y))))

// Q to f16, scaled, [batch][head][row][DK]: a work-item per 8 values
kernel void flash_attn_q_to_f16(global const char * q, ulong q_offset, ulong q_nb1, ulong q_nb2, ulong q_nb3,
                                float scale, int n_q, int n_head, global half * dst) {
    const int c8 = get_global_id(0), row = get_global_id(1);
    const int head = get_global_id(2) % n_head, batch = get_global_id(2) / n_head;
    if (c8 >= DK / 8 || row >= n_q) return;
    const float8 x = vload8(c8, (global const float *)(q + q_offset + batch * q_nb3 + head * q_nb2 + row * q_nb1));
    vstore8(convert_half8(x * scale), 0, dst + (((ulong) batch * n_head + head) * n_q + row) * DK + 8 * c8);
}

// The mask by blocks, after llama.cpp's Vulkan flash_attn_mask_opt.comp: a
// byte per 8 rows x 32 keys, 0 = mixed (read the mask), 1 = every visible
// cell 0 (no mask reads), 2 = nothing visible (the sub-group skips the tile).
// Rows from n_q on and keys from n_kv on do not count. A sub-group a block,
// a lane two keys of 8 rows; grid (n_kv / 32 tiles, n_q / 8 groups, mask
// batches x heads as the attention kernel indexes them).
#define MCLS_MIXED 0
#define MCLS_ZERO  1
#define MCLS_NONE  2
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void flash_attn_mask_cls(global const char * mask_void, ulong mask_offset, ulong mask_nb1, ulong mask_nb2, ulong mask_nb3,
                                int n_q, int n_kv, int n_tiles, int mask_ne2, global uchar * cls) {
    const int lane = get_sub_group_local_id();
    const int tile = get_group_id(0), grp = get_group_id(1), mz = get_group_id(2);
    const int i2 = mz % mask_ne2, i3 = mz / mask_ne2;
    global const char * mb = mask_void + mask_offset + i3 * mask_nb3 + i2 * mask_nb2;
    int nz = 0, vis = 0;
    for (int r = 0; r < 8; ++r) {
        const int qr = grp * 8 + r;
        if (qr >= n_q) break;
        global const half * mrow = (global const half *)(mb + qr * mask_nb1);
        for (int j = 0; j < 2; ++j) {
            const int key = tile * 32 + 2 * lane + j;
            if (key < n_kv) {
                const ushort h = as_ushort(mrow[key]);
                vis |= h != 0xFC00;
                nz  |= (h & 0x7FFF) != 0;   // neither +0 nor -0
            }
        }
    }
    vis = sub_group_reduce_max(vis);
    nz  = sub_group_reduce_max(nz);
    if (lane == 0) cls[((ulong) mz * ((n_q + 7) / 8) + grp) * n_tiles + tile] = !vis ? MCLS_NONE : !nz ? MCLS_ZERO : MCLS_MIXED;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(NTHR, 1, 1)))
kernel void flash_attn_f32_f16_dpas2(
        global const char * q_void, ulong q_offset,
        global const char * k_void, ulong k_offset,
        global const char * v_void, ulong v_offset,
        global const char * mask_void, ulong mask_offset,
        global char * o_void, ulong o_offset,
        float scale, int n_q, int n_kv, int n_head, int n_head_kv,
        ulong q_nb1, ulong q_nb2, ulong q_nb3,
        ulong k_nb1, ulong k_nb2, ulong k_nb3,
        ulong v_nb1, ulong v_nb2, ulong v_nb3,
        ulong o_nb1, ulong o_nb2, ulong o_nb3,
        ulong mask_nb1, ulong mask_nb2, ulong mask_nb3, int mask_ne2, int mask_ne3,
        int n_batch, int k_ne3, int v_ne3,
        global const uchar * mcls, int mcls_tiles
#ifdef FA_CARRY
        , global float * carry, int kv0, int carry_in, int carry_out
        , global float * parts, int n_split, int split_len
#endif
        ) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), lid = get_local_id(0);
    const int head = get_group_id(1), batch = get_group_id(2);
    const int hk   = head / (n_head / n_head_kv);
#ifdef FA_CARRY
    // the split of the keys: dimension 0 is n_split x the row tiles
    const int n_tq = (n_q + BQ - 1) / BQ;
    const int split = get_group_id(0) / n_tq;
    const int q0    = (get_group_id(0) % n_tq) * BQ;
#else
    const int q0   = get_group_id(0) * BQ;
#endif

    global const char * qb = q_void + q_offset + batch * q_nb3 + head * q_nb2;
    global const char * kb = k_void + k_offset + (batch / (n_batch / k_ne3)) * k_nb3 + hk * k_nb2;
    global const char * vb = v_void + v_offset + (batch / (n_batch / v_ne3)) * v_nb3 + hk * v_nb2;
    global const half * mb = mask_void != 0
        ? (global const half *)(mask_void + mask_offset + (batch % mask_ne3) * mask_nb3 + (head % mask_ne2) * mask_nb2)
        : 0;
#ifndef FA_CARRY
    const int kv0 = 0;
#endif
    if (mb != 0) mb += kv0;
    // this sub-group's row of the mask classes (null: no map, read the mask);
    // a sub-group past the last row has nothing visible
    const bool sg_rows = q0 + 8 * sg < n_q;
    global const uchar * mc = mcls != 0 && mb != 0 && sg_rows
        ? mcls + ((ulong) ((batch % mask_ne3) * mask_ne2 + head % mask_ne2) * ((n_q + 7) / 8) + (q0 / 8 + sg)) * mcls_tiles + kv0 / BK
        : 0;
    // the 2D surfaces: a row a key, DK halves wide
    const int kw = DK * 2, vw = DV * 2, kp = (int) k_nb1, vp = (int) v_nb1, qp = (int) q_nb1;

    // the last key any row of the work-group sees
    int kv_end = n_kv;
    if (mb != 0) {
        int last = -1;
        if (lid < BQ && q0 + lid < n_q) {
            global const half * mrow = (global const half *)((global const char *) mb + (q0 + lid) * mask_nb1);
            last = n_kv - 1;
            while (last >= 0 && as_ushort(mrow[last]) == 0xFC00) --last;   // -inf
        }
        kv_end = work_group_reduce_max(last) + 1;
    }

    float8 o[NDC];
    _Pragma("unroll")
    for (int c = 0; c < NDC; ++c) o[c] = (float8)(0.f);
    float m[8], l[8];
    _Pragma("unroll")
    for (int r = 0; r < 8; ++r) { m[r] = M_INIT; l[r] = 0.f; }
    const int q_sg = q0 + sg * 8;

#ifdef FA_CARRY
    const int n_g = (n_q + BQ - 1) / BQ * (BQ / 8);
    // split 0's state is the carry, split i's the parts' slot i - 1
    const ulong cst = ((ulong) batch * n_head + head) * n_g + q_sg / 8;
    global float * cb = (split == 0 ? carry : parts + (ulong) (split - 1) * n_batch * n_head * n_g * ((NDC + 2) * 128))
                      + cst * (ulong) ((NDC + 2) * 128) + lane * 8;
    const int t_beg = split * split_len, t_end = min(kv_end, t_beg + split_len);
    if (carry_in && split == 0) {
        _Pragma("unroll")
        for (int c = 0; c < NDC; ++c) o[c] = vload8(0, cb + c * 128);
        const float8 mv = vload8(0, cb + NDC * 128);
#ifndef FA_CARRY_RED
        const float8 lv8 = vload8(0, cb + (NDC + 1) * 128);
#else
        const float8 lv8 = (float8)(0.f);   // the red case: the earlier blocks' sums dropped
#endif
        m[0] = mv.s0; m[1] = mv.s1; m[2] = mv.s2; m[3] = mv.s3; m[4] = mv.s4; m[5] = mv.s5; m[6] = mv.s6; m[7] = mv.s7;
        l[0] = lv8.s0; l[1] = lv8.s1; l[2] = lv8.s2; l[3] = lv8.s3; l[4] = lv8.s4; l[5] = lv8.s5; l[6] = lv8.s6; l[7] = lv8.s7;
    }
#else
    const int t_beg = 0, t_end = kv_end;
#endif

    for (int t0 = t_beg; t0 < t_end; t0 += BK) {
#ifndef FA2_NOPF
        // the next tile into the cache, split across the sub-groups: K's and
        // V's BK rows of DK halves as blocks of 8 rows x 32 halves
        if (t0 + BK < t_end) {
            for (int i = sg; i < (BK / 8) * (DK / 32); i += NSG) {
                const int2 at = (int2)(32 * (i % (DK / 32)), t0 + BK + 8 * (i / (DK / 32)));
                intel_sub_group_2d_block_prefetch_16b_8r16x2c((global void *) kb, kw, n_kv, kp, at);
                intel_sub_group_2d_block_prefetch_16b_8r16x2c((global void *) vb, vw, n_kv, vp, at);
            }
        }
#endif
        const int mclass = !sg_rows ? MCLS_NONE : mc != 0 ? mc[t0 / BK] : MCLS_MIXED;
#ifdef FA2_MCLS_RED
        // the red case: a mixed block taken for an all-zero one (the diagonal's -inf cells let through)
        const int mclass_red = mclass == MCLS_MIXED && mc != 0 ? MCLS_ZERO : mclass;
#define mclass mclass_red
#endif
        if (mclass == MCLS_NONE) continue;
        // S: chunk pairs outer, the reduction split in two chains a key group
        float8 s0[NKB], s1[NKB];
        _Pragma("unroll")
        for (int g = 0; g < NKB; ++g) { s0[g] = (float8)(0.f); s1[g] = (float8)(0.f); }
#ifndef FA2_Q1
        // Q two chunks a read, each chain keeping its chunks in the one-chunk
        // order (s0 over 0 .. KC/2 - 1, s1 over KC/2 .. KC - 1): 8 reads of two
        // blocks a tile at head size 256, where 32 one-block reads cost ~9 of
        // 31 ms at 65k keys. (Interleaving the chains instead, even chunks into
        // s0 and odd into s1, reordered the sums: KL alike, but the dense
        // agent's greedy acceptance answer flipped from 10/10 to 2/10.)
        _Pragma("unroll")
        for (int j = 0; j < KC / 4; ++j) {
            const ushort16 q0 = QREAD2(j, q_sg), q1 = QREAD2(j + KC / 4, q_sg);
            _Pragma("unroll")
            for (int g = 0; g < NKB; ++g) {
                s0[g] = DPAS(as_short8(q0.lo), KREAD(2 * j,              t0 + 16 * g), s0[g]);
                s1[g] = DPAS(as_short8(q1.lo), KREAD(2 * j + KC / 2,     t0 + 16 * g), s1[g]);
                s0[g] = DPAS(as_short8(q0.hi), KREAD(2 * j + 1,          t0 + 16 * g), s0[g]);
                s1[g] = DPAS(as_short8(q1.hi), KREAD(2 * j + 1 + KC / 2, t0 + 16 * g), s1[g]);
            }
        }
#else
        _Pragma("unroll")
        for (int c = 0; c < KC / 2; ++c) {
            _Pragma("unroll")
            for (int g = 0; g < NKB; ++g) {
                s0[g] = DPAS(QREAD(c, q_sg),          KREAD(c, t0 + 16 * g),          s0[g]);
                s1[g] = DPAS(QREAD(c + KC / 2, q_sg), KREAD(c + KC / 2, t0 + 16 * g), s1[g]);
            }
        }
#endif
        float8 s[NKB];
        _Pragma("unroll")
        for (int g = 0; g < NKB; ++g) s[g] = s0[g] + s1[g];
        if (mclass == MCLS_MIXED) {
            _Pragma("unroll")
            for (int g = 0; g < NKB; ++g) {
                const int key = t0 + g * 16 + lane;
                float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
                _Pragma("unroll")
                for (int r = 0; r < 8; ++r) {
                    const int qr = min(q_sg + r, n_q - 1);
                    half mh = 0.0h;
                    if (mb != 0 && key < n_kv) mh = ((global const half *)((global const char *) mb + qr * mask_nb1))[key];
                    const float mk  = (float) mh;
                    const bool  vis = key < n_kv && q_sg + r < n_q && as_ushort(mh) != 0xFC00;
                    sv[r] = vis ? sv[r] + mk : M_INIT;
                }
                s[g] = (float8)(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]);
            }
        } else if (t0 + BK > n_kv || q_sg + 8 > n_q) {
            // a zero-mask tile at the end of the keys or the rows: bounds only
            _Pragma("unroll")
            for (int g = 0; g < NKB; ++g) {
                const int key = t0 + g * 16 + lane;
                float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
                _Pragma("unroll")
                for (int r = 0; r < 8; ++r) sv[r] = key < n_kv && q_sg + r < n_q ? sv[r] : M_INIT;
                s[g] = (float8)(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]);
            }
        }
        float smax[8];
        {
            const float sv[8] = { s[0].s0, s[0].s1, s[0].s2, s[0].s3, s[0].s4, s[0].s5, s[0].s6, s[0].s7 };
            _Pragma("unroll")
            for (int r = 0; r < 8; ++r) smax[r] = sv[r];
        }
        _Pragma("unroll")
        for (int g = 1; g < NKB; ++g) {
            const float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            _Pragma("unroll")
            for (int r = 0; r < 8; ++r) smax[r] = fmax(smax[r], sv[r]);
        }
        float dmax = -3.0e38f;
        _Pragma("unroll")
        for (int r = 0; r < 8; ++r) dmax = fmax(dmax, smax[r] - m[r]);
        int anyg = (dmax > 0.f) ? 1 : 0;
        anyg |= intel_sub_group_shuffle_xor(anyg, 8);
        anyg |= intel_sub_group_shuffle_xor(anyg, 4);
        anyg |= intel_sub_group_shuffle_xor(anyg, 2);
        anyg |= intel_sub_group_shuffle_xor(anyg, 1);
        if (anyg) {
            float corr[8];
            _Pragma("unroll")
            for (int r = 0; r < 8; ++r) {
                const float mrow = fmax(m[r], sub_group_reduce_max(smax[r]));
                if (mrow > m[r]) {
                    corr[r] = native_exp(m[r] - mrow);
                    l[r]   *= corr[r];
                } else {
                    corr[r] = 1.0f;
                }
                m[r] = mrow;
            }
            const float8 cv = (float8)(corr[0], corr[1], corr[2], corr[3], corr[4], corr[5], corr[6], corr[7]);
            _Pragma("unroll")
            for (int c = 0; c < NDC; ++c) o[c] *= cv;
        }
        short8 pa[NKB];
        _Pragma("unroll")
        for (int g = 0; g < NKB; ++g) {
            float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            half  ph[8];
            _Pragma("unroll")
            for (int r = 0; r < 8; ++r) {
                ph[r] = (half)(sv[r] > 0.5f * M_INIT ? native_exp(sv[r] - m[r]) : 0.f);
                l[r] += (float) ph[r];
            }
            pa[g] = as_short8((half8)(ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6], ph[7]));
        }
        _Pragma("unroll")
        for (int c = 0; c < NDC; ++c) {
            float8 acc = o[c];
            _Pragma("unroll")
            for (int g = 0; g < NKB; ++g) acc = DPAS(pa[g], VREAD(c, t0 + 16 * g), acc);
            o[c] = acc;
        }
    }

#ifdef FA_CARRY
    if (carry_out) {
        _Pragma("unroll")
        for (int c = 0; c < NDC; ++c) vstore8(o[c], 0, cb + c * 128);
        vstore8((float8)(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]), 0, cb + NDC * 128);
        vstore8((float8)(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]), 0, cb + (NDC + 1) * 128);
        return;
    }
#endif
    float lv[8];
    _Pragma("unroll")
    for (int r = 0; r < 8; ++r) lv[r] = sub_group_reduce_add(l[r]);
    _Pragma("unroll")
    for (int c = 0; c < NDC; ++c) {
        const float ov[8] = { o[c].s0, o[c].s1, o[c].s2, o[c].s3, o[c].s4, o[c].s5, o[c].s6, o[c].s7 };
        _Pragma("unroll")
        for (int r = 0; r < 8; ++r) {
            const int qr = q_sg + r;
            if (qr < n_q) {
                global float * orow = (global float *)(o_void + o_offset + batch * o_nb3 + qr * o_nb2 + head * o_nb1);
                orow[c * 16 + lane] = lv[r] > 0.f ? ov[r] / lv[r] : 0.f;
            }
        }
    }
}

#ifdef FA_CARRY
// The split states merged, after llama.cpp CUDA's flash_attn_combine_results
// (fattn-common.cuh): state 0 the carry (split 0, which took the earlier
// blocks' state in), state i the parts' slot i - 1, each O, m and l in the
// attention kernel's carry layout (8 rows of an 8-row group a lane). Into the
// carry (carry_out: more blocks of keys follow) or O, as the attention
// kernel's epilogue writes it. Grid: (n_g sub-groups, n_head, n_batch).
__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, 1, 1)))
kernel void flash_attn_split_combine(global float * carry, global const float * parts, int n_split,
                                     global char * o_void, ulong o_offset, ulong o_nb1, ulong o_nb2, ulong o_nb3,
                                     int n_q, int n_head, int n_batch, int carry_out) {
    const int lane = get_sub_group_local_id();
    const int grp = get_group_id(0), head = get_group_id(1), batch = get_group_id(2);
    const int n_g = (n_q + BQ - 1) / BQ * (BQ / 8);
    const ulong st   = (NDC + 2) * 128;
    const ulong slot = (ulong) n_batch * n_head * n_g * st;
    const ulong off  = (((ulong) batch * n_head + head) * n_g + grp) * st + lane * 8;
    global float * cb = carry + off;

    float8 mx = vload8(0, cb + NDC * 128);
    for (int i = 1; i < n_split; ++i) mx = fmax(mx, vload8(0, parts + (ulong) (i - 1) * slot + off + NDC * 128));
    float8 o[NDC];
    _Pragma("unroll")
    for (int c = 0; c < NDC; ++c) o[c] = (float8)(0.f);
    float8 l = (float8)(0.f);
    for (int i = 0; i < n_split; ++i) {
        global const float * pb = i == 0 ? cb : parts + (ulong) (i - 1) * slot + off;
#ifndef FA_SPLIT_RED
        const float8 w = exp(vload8(0, pb + NDC * 128) - mx);
#else
        const float8 w = (float8)(1.f);   // the red case: the states added without their max
#endif
        _Pragma("unroll")
        for (int c = 0; c < NDC; ++c) o[c] += w * vload8(0, pb + c * 128);
        l += w * vload8(0, pb + (NDC + 1) * 128);
    }
    if (carry_out) {
        _Pragma("unroll")
        for (int c = 0; c < NDC; ++c) vstore8(o[c], 0, cb + c * 128);
        vstore8(mx, 0, cb + NDC * 128);
        vstore8(l,  0, cb + (NDC + 1) * 128);
        return;
    }
    const float lv[8] = { sub_group_reduce_add(l.s0), sub_group_reduce_add(l.s1), sub_group_reduce_add(l.s2), sub_group_reduce_add(l.s3),
                          sub_group_reduce_add(l.s4), sub_group_reduce_add(l.s5), sub_group_reduce_add(l.s6), sub_group_reduce_add(l.s7) };
    _Pragma("unroll")
    for (int c = 0; c < NDC; ++c) {
        const float ov[8] = { o[c].s0, o[c].s1, o[c].s2, o[c].s3, o[c].s4, o[c].s5, o[c].s6, o[c].s7 };
        _Pragma("unroll")
        for (int r = 0; r < 8; ++r) {
            const int qr = grp * 8 + r;
            if (qr < n_q) {
                global float * orow = (global float *)(o_void + o_offset + batch * o_nb3 + qr * o_nb2 + head * o_nb1);
                orow[c * 16 + lane] = lv[r] > 0.f ? ov[r] / lv[r] : 0.f;
            }
        }
    }
}
#endif

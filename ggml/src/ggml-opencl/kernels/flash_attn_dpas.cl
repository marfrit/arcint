#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

// Flash attention for a prompt on the XMX units (sub-group 16 for Xe2,
// sub-group 8 for Xe-HPG below):
// f32 Q, f16 K/V, f16 mask. A work-group takes NSG*8 query rows of one head;
// sub-group s rows 8s..8s+7 (the DPAS M). Per tile of BK keys, staged in
// local memory for all sub-groups (V transposed): S = Q K^T by DPAS (lane =
// key, component = row), the mask added, the online softmax in f32 (row max
// and sum by sub-group reductions), and O += P V by DPAS, P being S's layout
// read as the a operand. The work-group stops after the last key any of its
// rows sees (from the mask; llama.cpp's causal mask leaves the upper
// triangle out). Q and P are f16 in the products, S, O and the softmax f32:
// GGML_PREC_F32 (which llama.cpp sets on every attention) asks for f32
// accumulation, and the CPU reference rounds Q to f16 and accumulates P V
// in f16. K/V broadcast over dimension 3 as ggml-cpu does
// (k index batch / (n_batch / k_ne3)). No ALiBi, softcap or sinks (the
// host keeps those on flash_attn_f32_f16.cl).

#ifndef DK
#define DK 256
#endif
#define DV DK
// DVH: the output columns a work-group computes, DV unless set (0106; the SG 8
// kernel only). At head size 512 the pair's halves of 512 columns would be 256
// GRF a sub-group, so the host builds DVH 256: two work-groups a head, one a
// half of V's columns, each computing S over all DK itself
#ifndef DVH
#define DVH DV
#endif
#define NVH (DV / DVH)
#ifndef NSG
#define NSG 16
#endif
#ifndef BK
#define BK 32
#endif
#ifndef FA_SG
#define FA_SG 16
#endif
#define SG FA_SG
#if SG == 16
#define BQ (NSG * 8)
#define KC (DK / 16)
#define NKB (BK / 16)
#define NDC (DV / 16)
#define NTHR (NSG * SG)
// the SG 16 kernel stages a K or V tile with at most four 8-value units a
// work-item (unrolled; the host builds NSG 16 with BK 32)
#if BK * DK / 8 > 4 * NTHR
#error "flash_attn_dpas.cl (SG 16): a K / V tile needs more than four units a work-item"
#endif
#if KC % 2 != 0
#error "flash_attn_dpas.cl (SG 16): Q K^T runs as two chains over halves of DK / 16"
#endif
#if DVH != DV
#error "flash_attn_dpas.cl (SG 16): one work-group a head, all DV columns"
#endif

#else
// SG 8 (Xe-HPG): a pair of sub-groups shares 8 query rows, each owning half
// of the output columns (its DV / 2 columns of O, DV / 16 float8, in registers) and
// computing S itself; the a operand packs two keys a lane, so P is re-laid
// from S (lane = key) by shuffles (without FA_VBLK; with it, keys j and
// j + 8 are a lane's pair and need none)
#define BQ (NSG / 2 * 8)
#define KC (DK / 16)
#define NKB (BK / 16)
#define NDC (DVH / 2 / 8)
#if DV % DVH != 0 || DVH % 32 != 0
#error "flash_attn_dpas.cl (SG 8): DVH must divide DV in multiples of 32"
#endif
#define NTHR (NSG * SG)
#endif

#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(a, b, c)

// K and V as f16 (0), q8_0 (1) or q4_0 (2), ggml's block layouts (34 / 18
// bytes for 32 values). The tiles are staged in local memory as f16 either
// way, so only the staging load changes: values 8 c8 .. 8 c8 + 7 of a row.
#ifndef KT
#define KT 0
#endif
#ifndef VT
#define VT 0
#endif
static inline half8 load8(const int t, const global char * row, const int c8) {
    if (t == 0) return vload8(c8, (const global half *) row);
    const global char * blk = row + (c8 >> 2) * (t == 1 ? 34 : 18);
    const float d = vload_half(0, (const global half *) blk);
#if SG == 8 && !defined(FA_LOAD8_BYTES)
    // the codes as halves 1024 + u built by integer operations on whole words
    // (0006's form): u in the low byte of each half, 0x6400 the exponent;
    // x - 1152 (q8_0, after ^ 0x80) or x - 1032 (q4_0) is the exact signed
    // code, then one rounding by d, as the byte-by-byte conversion gave.
    // The blocks are 2-byte aligned (34 / 18 bytes): four ushort loads.
    const ushort4 us = vload4(0, (const global ushort *)(blk + 2 + (t == 1 ? (c8 & 3) : (c8 & 1)) * 8));
    uint w0 = (uint) us.s0 | ((uint) us.s1 << 16), w1 = (uint) us.s2 | ((uint) us.s3 << 16);
    if (t == 1) {
        w0 ^= 0x80808080u; w1 ^= 0x80808080u;
    } else {
        // q4_0: values 0..15 the low nibbles of qs[0..15], 16..31 the high ones
        const uint sh = (c8 & 2) ? 4 : 0;
        w0 = (w0 >> sh) & 0x0F0F0F0Fu; w1 = (w1 >> sh) & 0x0F0F0F0Fu;
    }
    const uint4 h = (uint4)((w0 & 0xFFu) | ((w0 << 8) & 0xFF0000u), ((w0 >> 16) & 0xFFu) | ((w0 >> 8) & 0xFF0000u),
                            (w1 & 0xFFu) | ((w1 << 8) & 0xFF0000u), ((w1 >> 16) & 0xFFu) | ((w1 >> 8) & 0xFF0000u)) | 0x64006400u;
    return (as_half8(h) - (half8)(t == 1 ? 1152.0h : 1032.0h)) * (half8)((half) d);
#elif SG == 8
    if (t == 1) return convert_half8(convert_int8(vload8(0, blk + 2 + (c8 & 3) * 8))) * (half8)(d);
#else
    if (t == 1) return convert_half8(convert_float8(vload8(0, blk + 2 + (c8 & 3) * 8)) * d);
#endif
    // q4_0: values 0..15 the low nibbles of qs[0..15], 16..31 the high ones
    const uchar8 q = vload8(0, (const global uchar *) blk + 2 + (c8 & 1) * 8);
    const uchar8 n = (c8 & 2) ? (q >> (uchar8)(4)) : (q & (uchar8)(15));
    return convert_half8((convert_float8(n) - 8.f) * d);
}
#ifdef FA_CARRY
// keys [0, nkeys) of a quantized K or V view (type t, head size d) to f16,
// [dim][key][head][batch]: a work-item per 8 values, the attention kernel's
// own conversion (load8), one 16-byte store
kernel void flash_attn_kv_to_f16(global const char * src, ulong off, ulong nb1, ulong nb2, ulong nb3,
                                 int t, int d, int nkeys, int ne2, global half * dst) {
    const int c8 = get_global_id(0), key = get_global_id(1);
    const int h = get_global_id(2) % ne2, b = get_global_id(2) / ne2;
    if (c8 >= d / 8 || key >= nkeys) return;
    const half8 x = load8(t, src + off + (ulong) key * nb1 + (ulong) h * nb2 + (ulong) b * nb3, c8);
    vstore8(x, 0, dst + (((ulong) b * ne2 + h) * nkeys + key) * d + 8 * c8);
}
#endif
#if SG == 8
#define LOAD_A(p) as_int8(intel_sub_group_block_read8((local const uint *)(p)))
#endif
#define M_INIT (-3.0e38f)

#if SG == 16
__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(NTHR, 1, 1)))
kernel void flash_attn_f32_f16_dpas(
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
        int n_batch, int k_ne3, int v_ne3
#ifdef FA_CARRY
        // one block of keys a call (kv0..kv0 + n_kv of the mask): the online
        // softmax state (m, l, O unnormalised, f32) per row carried in
        // `carry` from the previous block (carry_in) to the next (carry_out)
        , global float * carry, int kv0, int carry_in, int carry_out
#endif
        ) {
    local half kv[BK * DK];     // one buffer (16 KB at DK 256): K, then rewritten as V

    const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), lid = get_local_id(0);
    const int head = get_group_id(1), batch = get_group_id(2);
    const int hk   = head / (n_head / n_head_kv);
    const int q0   = get_group_id(0) * BQ;

    global const char * qb = q_void + q_offset + batch * q_nb3 + head * q_nb2;
    global const char * kb = k_void + k_offset + (batch / (n_batch / k_ne3)) * k_nb3 + hk * k_nb2;
    global const char * vb = v_void + v_offset + (batch / (n_batch / v_ne3)) * v_nb3 + hk * v_nb2;
    global const half * mb = mask_void != 0
        ? (global const half *)(mask_void + mask_offset + (batch % mask_ne3) * mask_nb3 + (head % mask_ne2) * mask_nb2)
        : 0;
#ifdef FA_CARRY
    if (mb != 0) mb += kv0;
#endif

    // the last key any row of the work-group sees: a row a work-item, its
    // masked tail scanned backwards (short for a causal mask)
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
    for (int c = 0; c < NDC; ++c) o[c] = (float8)(0.f);
    // l: this lane's share of each row's sum, reduced once at the end
    float m[8], l[8];
    for (int r = 0; r < 8; ++r) { m[r] = M_INIT; l[r] = 0.f; }
    const int q_sg = q0 + sg * 8;
    // prefetch the first K tile while the Q registers are built, so the Q
    // loop hides the K global-load latency
    half8 kp0, kp1, kp2, kp3;
    {
        int u = lid;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); kp0 = load8(KT, kb + min(key, n_kv - 1) * k_nb1, c8); }
        u += NTHR;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); kp1 = load8(KT, kb + min(key, n_kv - 1) * k_nb1, c8); }
        u += NTHR;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); kp2 = load8(KT, kb + min(key, n_kv - 1) * k_nb1, c8); }
        u += NTHR;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); kp3 = load8(KT, kb + min(key, n_kv - 1) * k_nb1, c8); }
    }
    // Q is constant across every key tile, so the DPAS a-operand registers
    // are built once straight from global memory: no local staging of Q, no
    // qs buffer, no barrier. The a operand (M=8 rows, K=16) is distributed
    // lane = K-element (column), component = M row, so lane i holds column
    // c*16+i of all 8 rows.
    short8 qreg[KC];
    {
        const int d0 = lane;
        for (int c = 0; c < KC; ++c) {
            half hv[8];
            for (int j = 0; j < 8; ++j) {
                const int qr = q_sg + j;
                const float xv = qr < n_q ? ((global const float *)(qb + qr * q_nb1))[c * 16 + d0] * scale : 0.f;
                hv[j] = (half) xv;
            }
            qreg[c] = as_short8((half8)(hv[0], hv[1], hv[2], hv[3], hv[4], hv[5], hv[6], hv[7]));
        }
    }

    // K is staged in the single buffer, S runs from the K layout, the buffer
    // is then rewritten transposed as V and PV runs. Halving local memory is
    // what buys resident work-groups on this card, so the extra barriers are
    // paid for on purpose.
    {
        int u = lid;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); vstore8(kp0, 0, kv + key * DK + 8 * c8); }
        u += NTHR;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); vstore8(kp1, 0, kv + key * DK + 8 * c8); }
        u += NTHR;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); vstore8(kp2, 0, kv + key * DK + 8 * c8); }
        u += NTHR;
        if (u < BK * DK / 8) { const int key = u / (DK / 8), c8 = u % (DK / 8); vstore8(kp3, 0, kv + key * DK + 8 * c8); }
    }

#ifdef FA_CARRY
    // the carried state in the registers' own layout, per 8-row group:
    // [NDC + 2][lane][8] floats, a work-item's float8 of O per c, then its
    // m and l shares (l unreduced: the sum runs as in one pass). Loaded only
    // here: gathering O row by row spilled the build past 256 GRF.
    const int n_g = (n_q + BQ - 1) / BQ * (BQ / 8);
    global float * cb = carry + (((ulong) batch * n_head + head) * n_g + q_sg / 8) * (ulong) ((NDC + 2) * 128) + lane * 8;
    if (carry_in) {
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
#endif
    for (int t0 = 0; t0 < kv_end; t0 += BK) {
        // prefetch this tile's V into registers before the barrier, so the
        // loads are already in flight while the work-group syncs
        half8 vp0, vp1, vp2, vp3;
        {
            int u = lid;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; vp0 = load8(VT, vb + min(t0 + key, n_kv - 1) * v_nb1, c8); }
            u += NTHR;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; vp1 = load8(VT, vb + min(t0 + key, n_kv - 1) * v_nb1, c8); }
            u += NTHR;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; vp2 = load8(VT, vb + min(t0 + key, n_kv - 1) * v_nb1, c8); }
            u += NTHR;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; vp3 = load8(VT, vb + min(t0 + key, n_kv - 1) * v_nb1, c8); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // c outer, g inner: Q a-operand reused across key groups; the K
        // reduction is split in two so four independent DPAS chains overlap
        float8 s0[NKB], s1[NKB];
        for (int g = 0; g < NKB; ++g) { s0[g] = (float8)(0.f); s1[g] = (float8)(0.f); }
        for (int c = 0; c < KC / 2; ++c) {
            const short8 a0 = qreg[c];
            const short8 a1 = qreg[c + KC / 2];
            for (int g = 0; g < NKB; ++g) {
                const int8 b0 = as_int8(vload8(0, (local const uint *)(kv + (g * 16 + lane) * DK + c * 16)));
                s0[g] = DPAS(a0, b0, s0[g]);
                const int8 b1 = as_int8(vload8(0, (local const uint *)(kv + (g * 16 + lane) * DK + (c + KC / 2) * 16)));
                s1[g] = DPAS(a1, b1, s1[g]);
            }
        }
        float8 s[NKB];
        for (int g = 0; g < NKB; ++g) s[g] = s0[g] + s1[g];
        barrier(CLK_LOCAL_MEM_FENCE);   // S reads done: kv may be overwritten
        // V transposed with the key fastest: neighbouring lanes store
        // neighbouring halves (key-major, they all hit one bank)
        {
            int u = lid;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; const half vv[8] = { vp0.s0, vp0.s1, vp0.s2, vp0.s3, vp0.s4, vp0.s5, vp0.s6, vp0.s7 }; for (int j = 0; j < 8; ++j) kv[(8 * c8 + j) * BK + key] = vv[j]; }
            u += NTHR;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; const half vv[8] = { vp1.s0, vp1.s1, vp1.s2, vp1.s3, vp1.s4, vp1.s5, vp1.s6, vp1.s7 }; for (int j = 0; j < 8; ++j) kv[(8 * c8 + j) * BK + key] = vv[j]; }
            u += NTHR;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; const half vv[8] = { vp2.s0, vp2.s1, vp2.s2, vp2.s3, vp2.s4, vp2.s5, vp2.s6, vp2.s7 }; for (int j = 0; j < 8; ++j) kv[(8 * c8 + j) * BK + key] = vv[j]; }
            u += NTHR;
            if (u < BK * DK / 8) { const int key = u % BK, c8 = u / BK; const half vv[8] = { vp3.s0, vp3.s1, vp3.s2, vp3.s3, vp3.s4, vp3.s5, vp3.s6, vp3.s7 }; for (int j = 0; j < 8; ++j) kv[(8 * c8 + j) * BK + key] = vv[j]; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // V ready
        // mask all groups first, then one row-max reduction per row: the
        // elementwise max over the NKB key groups is taken before the
        // sub-group reduction, so 8 reductions replace NKB * 8
        for (int g = 0; g < NKB; ++g) {
            const int key = t0 + g * 16 + lane;
            float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            for (int r = 0; r < 8; ++r) {
                const int qr = min(q_sg + r, n_q - 1);
                half mh = 0.0h;
                if (mb != 0 && key < n_kv) mh = ((global const half *)((global const char *) mb + qr * mask_nb1))[key];
                const float mk  = (float) mh;
                const bool  vis = key < n_kv && q_sg + r < n_q && as_ushort(mh) != 0xFC00;   // -inf: masked
                sv[r] = vis ? sv[r] + mk : M_INIT;
            }
            s[g] = (float8)(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]);
        }
        float smax[8];
        {
            const float sv[8] = { s[0].s0, s[0].s1, s[0].s2, s[0].s3, s[0].s4, s[0].s5, s[0].s6, s[0].s7 };
            for (int r = 0; r < 8; ++r) smax[r] = sv[r];
        }
        for (int g = 1; g < NKB; ++g) {
            const float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            for (int r = 0; r < 8; ++r) smax[r] = fmax(smax[r], sv[r]);
        }
        // ONE reduction decides whether any row's running max can move: when
        // it cannot (the common case past the diagonal) the 8 row reductions
        // and the whole rescale are skipped
        float dmax = -3.0e38f;
        for (int r = 0; r < 8; ++r) dmax = fmax(dmax, smax[r] - m[r]);
        int anyg = (dmax > 0.f) ? 1 : 0;
        anyg |= intel_sub_group_shuffle_xor(anyg, 8);
        anyg |= intel_sub_group_shuffle_xor(anyg, 4);
        anyg |= intel_sub_group_shuffle_xor(anyg, 2);
        anyg |= intel_sub_group_shuffle_xor(anyg, 1);
        float corr[8];
        if (anyg) {
            for (int r = 0; r < 8; ++r) {
                const float mrow = fmax(m[r], sub_group_reduce_max(smax[r]));
                if (mrow > m[r]) {
                    // no M_INIT guard: on a row's first visible tile m - mrow
                    // is about -3e38, which native_exp takes to 0; the guard's
                    // select costs the q8_0 build 896 bytes of spill and 15 %
                    // of pp512 (measured-here)
                    corr[r] = native_exp(m[r] - mrow);
                    l[r]   *= corr[r];
                } else {
                    corr[r] = 1.0f;
                }
                m[r] = mrow;
            }
            const float8 cv = (float8)(corr[0], corr[1], corr[2], corr[3], corr[4], corr[5], corr[6], corr[7]);
            for (int c = 0; c < NDC; ++c) o[c] *= cv;
        } else {
            for (int r = 0; r < 8; ++r) corr[r] = 1.0f;
        }
        short8 pa[NKB];
        for (int g = 0; g < NKB; ++g) {
            float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            half  ph[8];
            for (int r = 0; r < 8; ++r) {
                // the sum of what P V multiplies: P rounded to half
                ph[r] = (half)(sv[r] > 0.5f * M_INIT ? native_exp(sv[r] - m[r]) : 0.f);
                l[r] += (float) ph[r];
            }
            pa[g] = as_short8((half8)(ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6], ph[7]));
        }
        for (int c = 0; c < NDC; ++c) {
            const local half * vrow = kv + (c * 16 + lane) * BK;
            float8 acc = o[c];
            for (int g = 0; g < NKB; ++g) {
                const int8 b = as_int8(vload8(0, (local const uint *)(vrow + g * 16)));
                acc = DPAS(pa[g], b, acc);
            }
            o[c] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // PV reads done: kv may be overwritten
        const int t1 = t0 + BK;
        if (t1 < kv_end) {
            for (int u = lid; u < BK * DK / 8; u += NTHR) {
                const int key = u / (DK / 8), c8 = u % (DK / 8), kk = min(t1 + key, n_kv - 1);
                vstore8(load8(KT, kb + kk * k_nb1, c8), 0, kv + key * DK + 8 * c8);
            }
        }
    }

#ifdef FA_CARRY
    if (carry_out) {
        for (int c = 0; c < NDC; ++c) vstore8(o[c], 0, cb + c * 128);
        vstore8((float8)(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]), 0, cb + NDC * 128);
        vstore8((float8)(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]), 0, cb + (NDC + 1) * 128);
        return;
    }
#endif
    float lv[8];
    for (int r = 0; r < 8; ++r) lv[r] = sub_group_reduce_add(l[r]);
    for (int c = 0; c < NDC; ++c) {
        const float ov[8] = { o[c].s0, o[c].s1, o[c].s2, o[c].s3, o[c].s4, o[c].s5, o[c].s6, o[c].s7 };
        for (int r = 0; r < 8; ++r) {
            const int qr = q_sg + r;
            if (qr < n_q) {
                global float * orow = (global float *)(o_void + o_offset + batch * o_nb3 + qr * o_nb2 + head * o_nb1);
                orow[c * 16 + lane] = lv[r] > 0.f ? ov[r] / lv[r] : 0.f;
            }
        }
    }
}
#endif  // SG == 16

#if SG == 8
#ifdef FA_VBLK
#if VT != 0
#error "FA_VBLK: f16 V only, a quantized V decode is exposed in this staging, see patch 0048"
#endif
// V a key pair a work-item (patch 0048): two keys' dims 8 c8v .. 8 c8v + 7
// packed into 8 uints (the P V b operand's pairs) and stored as 8 dwords
// into the transposed tile, one a dim, instead of 16 scalar halves
// (c8o: the first 8-value unit of this work-group's columns of V, 0106)
static inline void stage_v_pair(local half * vt, const global char * vb, ulong v_nb1, int t0, int n_kv, int kp, int c8v, int c8o) {
    // pair kp of a 16-key group holds keys j and j + 8 (j = kp % 8): P's a
    // operand then comes from S's two 8-key halves without shuffles, and a
    // sub-group's loads read 8 consecutive rows
    const int k0 = t0 + (kp >> 3) * 16 + (kp & 7), k1 = k0 + 8;
    half8 a = vload8(c8o + c8v, (const global half *)(vb + min(k0, n_kv - 1) * v_nb1));
    half8 b = vload8(c8o + c8v, (const global half *)(vb + min(k1, n_kv - 1) * v_nb1));
#ifdef FA_VBLK_RED
    { const half8 x = a; a = b; b = x; }   // the red case: the pair's keys swapped
#endif
    const uint w[8] = { as_uint((half2)(a.s0, b.s0)), as_uint((half2)(a.s1, b.s1)), as_uint((half2)(a.s2, b.s2)), as_uint((half2)(a.s3, b.s3)),
                        as_uint((half2)(a.s4, b.s4)), as_uint((half2)(a.s5, b.s5)), as_uint((half2)(a.s6, b.s6)), as_uint((half2)(a.s7, b.s7)) };
    local uint * vd = (local uint *) vt + (8 * c8v) * ((BK + 16) / 2) + kp;
    for (int j = 0; j < 8; ++j) vd[j * ((BK + 16) / 2)] = w[j];
}
#endif
#endif
#if SG == 8
#ifdef FA_QG
// FA_QG: Q to f16 once a call, scaled, in the a-operand blocks the kernel
// read from local memory before ([batch][head][8-row group, padded to BQ][KC]
// [8 rows][16 halves]), so local memory holds K and V only and a work-group
// can have more sub-groups sharing them (the A770 ran one 8-thread
// work-group per core with Q in local memory: 58 KB of its 64)
__attribute__((intel_reqd_sub_group_size(SG)))
kernel void flash_attn_q_to_f16_blk(global const char * q, ulong q_offset, ulong q_nb1, ulong q_nb2, ulong q_nb3,
                                    float scale, int n_q, int n_head, int n_g, global half * dst) {
    const int c8 = get_global_id(0), r = get_global_id(1);
    const int head = get_global_id(2) % n_head, batch = get_global_id(2) / n_head;
    if (c8 >= DK / 8 || r >= n_g * 8) return;
    const float8 x = r < n_q ? vload8(c8, (global const float *)(q + q_offset + batch * q_nb3 + head * q_nb2 + r * q_nb1)) : (float8)(0.f);
    vstore8(convert_half8(x * scale), 0,
            dst + (((ulong) batch * n_head + head) * n_g + (r >> 3)) * (KC * 128) + (c8 >> 1) * 128 + (r & 7) * 16 + (c8 & 1) * 8);
}
#endif
__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(NTHR, 1, 1)))
kernel void flash_attn_f32_f16_dpas(
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
        int n_batch, int k_ne3, int v_ne3) {
#ifndef FA_QG
    local half qs[BQ * DK];
#endif
    local half ks[BK * (DK + 16)];
    // V transposed, [dim][key] halves: a dim's uint p holds a pair of keys,
    // the P V b operand's (lane = dim reads its 16 keys as 8 uints): keys
    // 2p, 2p + 1, or with FA_VBLK keys j, j + 8 of a 16-key group
    local half vt[DVH * (BK + 16)];

    const int lane = get_sub_group_local_id(), sg = get_sub_group_id(), lid = get_local_id(0);
    const int rg = sg >> 1, half_dv = sg & 1;          // the pair's row group, this one's columns
    // dimension 1: a head slot, the head times NVH plus this work-group's part of V
    const int hs = get_group_id(1), batch = get_group_id(2);
    const int head = hs / NVH, vh = hs % NVH;
#ifdef FA_DVH_RED
    const int vc8 = 0;                                 // the red case: every part reads V's first columns
#else
    const int vc8 = vh * (DVH / 8);                    // the part's first 8-value unit of a V row
#endif
    const int hk   = head / (n_head / n_head_kv);
    const int q0   = get_group_id(0) * BQ;

    global const char * qb = q_void + q_offset + batch * q_nb3 + head * q_nb2;
    global const char * kb = k_void + k_offset + (batch / (n_batch / k_ne3)) * k_nb3 + hk * k_nb2;
    global const char * vb = v_void + v_offset + (batch / (n_batch / v_ne3)) * v_nb3 + hk * v_nb2;
    global const half * mb = mask_void != 0
        ? (global const half *)(mask_void + mask_offset + (batch % mask_ne3) * mask_nb3 + (head % mask_ne2) * mask_nb2)
        : 0;

#ifdef FA_QG
    // q_void: flash_attn_q_to_f16_blk's blocks, n_g 8-row groups a head
    const int n_g = (n_q + BQ - 1) / BQ * (BQ / 8);
    global const uint * qg = (global const uint *)(q_void + q_offset) + (((ulong) batch * n_head + head) * n_g + q0 / 8) * (KC * 64);
#ifndef FA_QG_RED
#define QA(p) as_int8(intel_sub_group_block_read8(qg + (p) / 2))
#else
#define QA(p) as_int8(intel_sub_group_block_read8(qg + (p) / 2 + 64))   // the red case: the next 16 dims
#endif
#else
#define QA(p) LOAD_A(qs + (p))
    for (int u = lid; u < BQ * (DK / 8); u += NTHR) {
        const int r = u / (DK / 8), c8 = u % (DK / 8), qr = q0 + r;
        const float8 x = qr < n_q ? vload8(0, (const global float *)(qb + qr * q_nb1) + c8 * 8) : (float8)(0.f);
        const half8 h = convert_half8(x * scale);
        // block layout: ((r>>3)*KC + (c8>>1))*128 + (r&7)*16 + (c8&1)*8
        vstore8(h, 0, qs + ((r >> 3) * KC + (c8 >> 1)) * 128 + (r & 7) * 16 + (c8 & 1) * 8);
    }
#endif
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
    for (int c = 0; c < NDC; ++c) o[c] = (float8)(0.f);
    // l: this lane's share of each row's sum, reduced once at the end
    float m[8], l[8];
    for (int r = 0; r < 8; ++r) { m[r] = M_INIT; l[r] = 0.f; }
    const int q_sg = q0 + rg * 8;
    ulong qroff[8];
    for (int r = 0; r < 8; ++r) qroff[r] = (ulong)min(q_sg + r, n_q - 1) * mask_nb1;

    for (int t0 = 0; t0 < kv_end; t0 += BK) {
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int u = lid; u < BK * DK / 8; u += NTHR) {
            const int key = u / (DK / 8), c8 = u % (DK / 8), kk = min(t0 + key, n_kv - 1);
            vstore8(load8(KT, kb + kk * k_nb1, c8), 0, ks + key * (DK + 16) + 8 * c8);
#ifndef FA_VBLK
            // V transposed with the key fastest: neighbouring lanes store
            // neighbouring halves (key-major, they all hit one bank)
            const int keyv = u % BK, c8v = u / BK, kkv = min(t0 + keyv, n_kv - 1);
            if (c8v < DVH / 8) {
                const half8 vv8 = load8(VT, vb + kkv * v_nb1, vc8 + c8v);
                const half vv[8] = { vv8.s0, vv8.s1, vv8.s2, vv8.s3, vv8.s4, vv8.s5, vv8.s6, vv8.s7 };
                for (int j = 0; j < 8; ++j) vt[(8 * c8v + j) * (BK + 16) + keyv] = vv[j];
            }
#endif
        }
#ifdef FA_VBLK
        // the pair fastest: a sub-group's stores of one dim are 8 consecutive dwords
#pragma unroll
        for (int u = lid; u < (BK / 2) * (DVH / 8); u += NTHR) {
            stage_v_pair(vt, vb, v_nb1, t0, n_kv, u % (BK / 2), u / (BK / 2), vc8);
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // S in 8-key groups: lane = key, component = row
        float8 s[2 * NKB];
#if NKB == 2
        // c outer with explicit accumulators: the Q a-operand block read
        // happens once per c (16x) instead of once per (g,c) (64x)
        {
            float8 s0 = (float8)(0.f), s1 = (float8)(0.f), s2 = (float8)(0.f), s3 = (float8)(0.f);
#pragma unroll
            for (int c = 0; c < KC; ++c) {
                const int8 a = QA(((rg * KC + c) * 8) * 16);
                s0 = DPAS(a, as_int8(vload8(0, (local const uint *)(ks + lane * (DK + 16) + c * 16))), s0);
                s1 = DPAS(a, as_int8(vload8(0, (local const uint *)(ks + (8 + lane) * (DK + 16) + c * 16))), s1);
                s2 = DPAS(a, as_int8(vload8(0, (local const uint *)(ks + (16 + lane) * (DK + 16) + c * 16))), s2);
                s3 = DPAS(a, as_int8(vload8(0, (local const uint *)(ks + (24 + lane) * (DK + 16) + c * 16))), s3);
            }
            s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
        }
#else
        for (int g = 0; g < 2 * NKB; ++g) {
            float8 acc = (float8)(0.f);
            for (int c = 0; c < KC; ++c) {
                const int8 a = QA(((rg * KC + c) * 8) * 16);
                const int8 b = as_int8(vload8(0, (local const uint *)(ks + (g * 8 + lane) * (DK + 16) + c * 16)));
                acc = DPAS(a, b, acc);
            }
            s[g] = acc;
        }
#endif
        // mask every group, then elementwise max across the groups, then
        // ONE sub-group vote decides whether any row's running max rises;
        // the common case past the diagonal skips the reductions and rescale
        if (mb != 0) {
#pragma unroll
            for (int g = 0; g < 2 * NKB; ++g) {
                const int key = t0 + g * 8 + lane;
                float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
                for (int r = 0; r < 8; ++r) {
                    half mh = ((global const half *)((global const char *) mb + qroff[r]))[min(key, n_kv - 1)];
                    const float mk  = (float) mh;
                    const bool  vis = key < n_kv && q_sg + r < n_q && as_ushort(mh) != 0xFC00;   // -inf: masked
                    sv[r] = vis ? sv[r] + mk : M_INIT;
                }
                s[g] = (float8)(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]);
            }
        } else {
#pragma unroll
            for (int g = 0; g < 2 * NKB; ++g) {
                const int key = t0 + g * 8 + lane;
                float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
                for (int r = 0; r < 8; ++r) {
                    const bool vis = key < n_kv && q_sg + r < n_q;
                    sv[r] = vis ? sv[r] : M_INIT;
                }
                s[g] = (float8)(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]);
            }
        }
        float smax[8];
        {
            const float sv[8] = { s[0].s0, s[0].s1, s[0].s2, s[0].s3, s[0].s4, s[0].s5, s[0].s6, s[0].s7 };
            for (int r = 0; r < 8; ++r) smax[r] = sv[r];
        }
        for (int g = 1; g < 2 * NKB; ++g) {
            const float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            for (int r = 0; r < 8; ++r) smax[r] = fmax(smax[r], sv[r]);
        }
        float dmax = -3.0e38f;
        for (int r = 0; r < 8; ++r) dmax = fmax(dmax, smax[r] - m[r]);
        const int anyg = (sub_group_reduce_max(dmax) > 0.f) ? 1 : 0;
        float corr[8];
        if (anyg) {
            for (int r = 0; r < 8; ++r) {
                const float mrow = fmax(m[r], sub_group_reduce_max(smax[r]));
                corr[r] = m[r] > 0.5f * M_INIT ? native_exp(m[r] - mrow) : 0.f;
                m[r]    = mrow;
                l[r]   *= corr[r];
            }
            const float8 cv = (float8)(corr[0], corr[1], corr[2], corr[3], corr[4], corr[5], corr[6], corr[7]);
            for (int c = 0; c < NDC; ++c) o[c] *= cv;
        } else {
            for (int r = 0; r < 8; ++r) corr[r] = 1.0f;
        }
        // P (half), row sums; then the a operand of 16 keys: lane i holds keys
        // 2i, 2i+1 of each row (keys 0..7 in the low group, 8..15 in the high;
        // with FA_VBLK keys i and i + 8)
        half8 ph[2 * NKB];
#pragma unroll
        for (int g = 0; g < 2 * NKB; ++g) {
            float sv[8] = { s[g].s0, s[g].s1, s[g].s2, s[g].s3, s[g].s4, s[g].s5, s[g].s6, s[g].s7 };
            half  hv[8];
            for (int r = 0; r < 8; ++r) {
                hv[r] = (half)(sv[r] > 0.5f * M_INIT ? native_exp(sv[r] - m[r]) : 0.f);
                l[r] += (float) hv[r];
            }
            ph[g] = (half8)(hv[0], hv[1], hv[2], hv[3], hv[4], hv[5], hv[6], hv[7]);
        }
        int8 pa[NKB];
#ifdef FA_VBLK
        // lane i: keys i and i + 8 of each 16-key group, V's pairs (stage_v_pair)
#pragma unroll
        for (int g = 0; g < NKB; ++g) {
            const half8 a0 = ph[2 * g], a1 = ph[2 * g + 1];
            pa[g] = (int8)(as_int((half2)(a0.s0, a1.s0)), as_int((half2)(a0.s1, a1.s1)),
                           as_int((half2)(a0.s2, a1.s2)), as_int((half2)(a0.s3, a1.s3)),
                           as_int((half2)(a0.s4, a1.s4)), as_int((half2)(a0.s5, a1.s5)),
                           as_int((half2)(a0.s6, a1.s6)), as_int((half2)(a0.s7, a1.s7)));
        }
#else
#pragma unroll
        for (int g = 0; g < NKB; ++g) {
            const int  src0 = (2 * lane) & 7, src1 = (2 * lane + 1) & 7;
            const bool hi   = lane >= 4;
            // the shuffles move the 8 halves as a uint4 (no half8 shuffle),
            // all four in every lane: a shuffle in a branch reads lanes the
            // branch left inactive
            const uint4 lo = as_uint4(ph[2 * g]), hi4 = as_uint4(ph[2 * g + 1]);
            const uint4 l0 = intel_sub_group_shuffle(lo, src0), h0 = intel_sub_group_shuffle(hi4, src0);
            const uint4 l1 = intel_sub_group_shuffle(lo, src1), h1 = intel_sub_group_shuffle(hi4, src1);
            const half8 a0 = as_half8(hi ? h0 : l0);
            const half8 a1 = as_half8(hi ? h1 : l1);
            pa[g] = (int8)(as_int((half2)(a0.s0, a1.s0)), as_int((half2)(a0.s1, a1.s1)),
                           as_int((half2)(a0.s2, a1.s2)), as_int((half2)(a0.s3, a1.s3)),
                           as_int((half2)(a0.s4, a1.s4)), as_int((half2)(a0.s5, a1.s5)),
                           as_int((half2)(a0.s6, a1.s6)), as_int((half2)(a0.s7, a1.s7)));
        }
#endif
#pragma unroll
        for (int c = 0; c < NDC; ++c) {
            const int col = half_dv * (DVH / 2) + c * 8 + lane;
            float8 acc = o[c];
            for (int g = 0; g < NKB; ++g) {
                const int8 b = as_int8(vload8(0, (local const uint *)(vt + col * (BK + 16) + g * 16)));
                acc = DPAS(pa[g], b, acc);
            }
            o[c] = acc;
        }
    }

    float lv[8], linv[8];
    for (int r = 0; r < 8; ++r) { lv[r] = sub_group_reduce_add(l[r]); linv[r] = lv[r] > 0.f ? 1.f / lv[r] : 0.f; }
    for (int c = 0; c < NDC; ++c) {
        const float ov[8] = { o[c].s0, o[c].s1, o[c].s2, o[c].s3, o[c].s4, o[c].s5, o[c].s6, o[c].s7 };
        for (int r = 0; r < 8; ++r) {
            const int qr = q_sg + r;
            if (qr < n_q) {
                global float * orow = (global float *)(o_void + o_offset + batch * o_nb3 + qr * o_nb2 + head * o_nb1);
                orow[vh * DVH + half_dv * (DVH / 2) + c * 8 + lane] = ov[r] * linv[r];
            }
        }
    }
}
#endif  // SG == 8

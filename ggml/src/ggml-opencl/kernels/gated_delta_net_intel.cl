#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable

// Gated delta net (GGML_OP_GATED_DELTA_NET, scalar gate) for Intel, in the
// shape of OpenVINO's paged_gated_delta_net_opt.cl: lanes along the key
// dimension, each sub-group holding R state rows in registers (lane l keeps
// keys m*SG + l, m = 0..KL-1, the order sub-group block reads give);
// reductions by sub-group shuffles. Same arguments and output as
// kernel_gated_delta_net (gated_delta_net.cl).
//
// State row j (a value index) holds keys i = 0..S_V-1 contiguously. Per token
// (ggml-cpu ops.cpp): S *= exp(g); delta_j = (v_j - S_j.k) beta;
// S_j += k delta_j; out_j = S_j.q / sqrt(S_V).
//
// GDN_PP (prompt): a work-group of NSG sub-groups per (head, sequence,
// NSG*R rows); q/k/v/g/beta come in chunks of TC tokens staged in local
// memory, the next chunk loaded into registers while the current one is
// computed, so the global latency is paid once per chunk and q/k are read
// once per work-group. Otherwise (generation): one sub-group per (head,
// sequence, R rows), the next token's inputs fetched while the current one is
// computed (Strata's gdn_rec_cols_pipe_kernel).

#ifndef S_V
#define S_V 128
#endif
#ifndef SG
#define SG 16
#endif
#ifndef R
#define R 2
#endif
#define KL (S_V / SG)

#define BLOCK_READ_F(p)     as_float(intel_sub_group_block_read((global const uint *)(p)))
#define BLOCK_WRITE_F(p, v) intel_sub_group_block_write((global uint *)(p), as_uint(v))
#define LBLOCK_READ_F(p)    as_float(intel_sub_group_block_read((local const uint *)(p)))

// the gating folded in (patch 0051): g = softplus(alpha + dt) * A and
// beta = sigmoid(b) from the raw projections, with kernel_add_row's,
// kernel_softplus_f32's, kernel_mul_row's and kernel_sigmoid_f32's
// expressions; the XOR with an always-0 argument after each step keeps the
// separate ops' roundings under -cl-fast-relaxed-math (as 0049's norm)
static inline float gdn_gate(float a, float dt, float A, uint opaque) {
    float t = as_float(as_uint(a + dt) ^ opaque);
    float sp = (t > 20.0f) ? t : log(1.0f + exp(t));
    sp = as_float(as_uint(sp) ^ opaque);
    return as_float(as_uint(sp * A) ^ opaque);
}
static inline float gdn_beta(float b, uint opaque) {
    return as_float(as_uint(1.0f / (1.0f + exp(-b))) ^ opaque);
}


#ifdef GDN_RPL

// GATED_DELTA_NET_REPLAY (rollback by replay, see ggml_gated_delta_net_replay):
// the generation kernel's per-token step, operation for operation, with
//   - a replay prologue that re-applies n_rec stored records to the loaded base
//     state with the step's own update (g*S + k*delta, fma), and
//   - an epilogue that stores the base state (after token t_base) and one
//     record per later token: delta per value row, k per key, g per head.
// n_tokens == 0 replays only and stores the replayed state as the base.
kernel void kernel_gated_delta_net_rpl(
        global const char * q_buf,     ulong off_q,
        global const char * k_buf,     ulong off_k,
        global const char * v_buf,     ulong off_v,
        global const char * g_buf,     ulong off_g,
        global const char * beta_buf,  ulong off_beta,
        global const char * state_buf, ulong off_state,
        global const char * rec_buf,   ulong off_rec,
        global const char * ctrl_buf,  ulong off_ctrl,
        global       char * dst_buf,   ulong off_dst,
        uint  H_v,
        uint  n_tokens,
        uint  n_seqs,
        uint  sq1, uint sq2, uint sq3,
        uint  sv1, uint sv2, uint sv3,
        uint  sb1, uint sb2, uint sb3,
        uint  H_k,
        uint  rq3,
        float scale,
        uint  K,
        uint  RK,
        uint  rec_bank_stride, uint rec_seq_stride) {
    global const float * data_q     = (global const float *)(q_buf     + off_q);
    global const float * data_k     = (global const float *)(k_buf     + off_k);
    global const float * data_v     = (global const float *)(v_buf     + off_v);
    global const float * data_g     = (global const float *)(g_buf     + off_g);
    global const float * data_beta  = (global const float *)(beta_buf  + off_beta);
    global const float * data_state = (global const float *)(state_buf + off_state);
    global const float * data_rec   = (global const float *)(rec_buf   + off_rec);
    global const int   * data_ctrl  = (global const int   *)(ctrl_buf  + off_ctrl);
    global       float * data_dst   = (global       float *)(dst_buf   + off_dst);

    const uint head = get_group_id(0);
    const uint seq  = get_group_id(1);
    const uint row0 = get_group_id(2) * R;
    const uint lane = get_sub_group_local_id();

    const uint iq1 = head % H_k;
    const uint iq3 = seq / rq3;
    const ulong state_size = (ulong)S_V * S_V;
    const ulong state_base = ((ulong)seq * H_v + head) * state_size;
    const ulong q_base  = (ulong)iq3 * sq3 + (ulong)iq1 * sq1;
    const ulong v_base  = (ulong)seq * sv3 + (ulong)head * sv1;
    const ulong gb_base = (ulong)seq * sb3 + (ulong)head * sb1;

    float S[R][KL];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int m = 0; m < KL; ++m)
            S[r][m] = BLOCK_READ_F(data_state + state_base + (ulong)(row0 + r) * S_V + m * SG);

    // replay prologue
    const int n_rec = data_ctrl[2 * seq + 0];
    const int bank  = data_ctrl[2 * seq + 1];
    global const float * rb = data_rec + (ulong)seq * rec_seq_stride + (ulong)bank * rec_bank_stride;
    for (int j = 0; j < n_rec; ++j) {
        global const float * r  = rb + (ulong)j * RK;
        const float gv = r[2 * H_v * S_V + head];
        #pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const float dp = r[head * S_V + row0 + rr];
            #pragma unroll
            for (int m = 0; m < KL; ++m)
                S[rr][m] = fma(r[H_v * S_V + head * S_V + m * SG + lane], dp, gv * S[rr][m]);
        }
    }

    const ulong attn_base = 0;
    const ulong base_off  = (ulong)S_V * H_v * n_tokens * n_seqs;
    const ulong rec_off   = base_off + state_size * H_v * n_seqs;
    global float * base_out = data_dst + base_off + ((ulong)seq * H_v + head) * state_size;
    const int t_base = (int)n_tokens - 1 - min((int)n_tokens - 1, (int)K - 1);

    if (n_tokens == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < KL; ++m)
                BLOCK_WRITE_F(base_out + (ulong)(row0 + r) * S_V + m * SG, S[r][m]);
        return;
    }

    // token 0's inputs; then each step fetches the next token's first
    float kn[KL], qn[KL], vn[R], gn = 0.f, bn = 0.f;
    if (n_tokens > 0) {
        #pragma unroll
        for (int m = 0; m < KL; ++m) {
            kn[m] = BLOCK_READ_F(data_k + q_base + m * SG);
            qn[m] = BLOCK_READ_F(data_q + q_base + m * SG);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) vn[r] = data_v[v_base + row0 + r];
        gn = data_g[gb_base];
        bn = data_beta[gb_base];
    }

    ulong attn_off = attn_base + ((ulong)seq * n_tokens * H_v + head) * S_V;
    for (uint t = 0; t < n_tokens; ++t) {
        float k[KL], q[KL], v[R];
        #pragma unroll
        for (int m = 0; m < KL; ++m) { k[m] = kn[m]; q[m] = qn[m]; }
        #pragma unroll
        for (int r = 0; r < R; ++r) v[r] = vn[r];
        const float g = exp(gn), beta = bn;
        if (t + 1 < n_tokens) {
            const ulong qo = q_base + (ulong)(t + 1) * sq2;
            #pragma unroll
            for (int m = 0; m < KL; ++m) {
                kn[m] = BLOCK_READ_F(data_k + qo + m * SG);
                qn[m] = BLOCK_READ_F(data_q + qo + m * SG);
            }
            const ulong vo = v_base + (ulong)(t + 1) * sv2;
            #pragma unroll
            for (int r = 0; r < R; ++r) vn[r] = data_v[vo + row0 + r];
            const ulong go = gb_base + (ulong)(t + 1) * sb2;
            gn = data_g[go];
            bn = data_beta[go];
        }

        float kv[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            float p = 0.f;
            #pragma unroll
            for (int m = 0; m < KL; ++m) p = fma(S[r][m], k[m], p);
            kv[r] = sub_group_reduce_add(p) * g;
        }
        float out[R], dl[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float d = (v[r] - kv[r]) * beta;
            dl[r] = d;
            float p = 0.f;
            #pragma unroll
            for (int m = 0; m < KL; ++m) {
                S[r][m] = fma(k[m], d, g * S[r][m]);
                p = fma(S[r][m], q[m], p);
            }
            out[r] = sub_group_reduce_add(p);
        }
        float mine = 0.f;
        #pragma unroll
        for (int r = 0; r < R; ++r) mine = (lane == r) ? out[r] : mine;
        if (lane < R) data_dst[attn_off + row0 + lane] = mine * scale;
        attn_off += (ulong)S_V * H_v;

        if ((int)t == t_base) {
            #pragma unroll
            for (int r = 0; r < R; ++r)
                #pragma unroll
                for (int m = 0; m < KL; ++m)
                    BLOCK_WRITE_F(base_out + (ulong)(row0 + r) * S_V + m * SG, S[r][m]);
        } else if ((int)t > t_base) {
            global float * rt = data_dst + rec_off + (ulong)seq * rec_seq_stride +
                                (ulong)((int)t - t_base - 1) * RK;
            if (lane < R) rt[head * S_V + row0 + lane] = dl[lane];
            #pragma unroll
            for (int m = 0; m < KL; ++m) rt[H_v * S_V + head * S_V + m * SG + lane] = k[m];
            if (lane == 0) rt[2 * H_v * S_V + head] = g;
        }
    }
}

#endif // GDN_RPL

#ifdef GDN_PP

#ifndef NSG
#define NSG 32
#endif
#ifndef TC
#define TC 16
#endif
#define ROWS (NSG * R)
#define NTHR (NSG * SG)
// one chunk: q[TC][S_V], k[TC][S_V], v[TC][ROWS], g[TC], beta[TC]
#define CH_FLOATS (TC * (2 * S_V + ROWS + 2))
#define PER_THR ((CH_FLOATS + NTHR - 1) / NTHR)

// element e of a chunk (0 <= e < CH_FLOATS): where it comes from in global
// memory. Layout in local memory: q, k, v, g, beta, each token-major.
static inline float chunk_load(int e, uint t0, uint n_tokens,
        global const float * dq, global const float * dk, global const float * dv,
        global const float * dg, global const float * db,
        ulong q_base, ulong v_base, ulong gb_base, uint sq2, uint sv2, uint sb2, uint row0,
        float dt_h, float a_h, uint gating, uint opaque) {
    int t, x;
    if (e < TC * S_V)                 { t = e / S_V; x = e % S_V;
        return t0 + t < n_tokens ? dq[q_base + (ulong)(t0 + t) * sq2 + x] : 0.f; }
    e -= TC * S_V;
    if (e < TC * S_V)                 { t = e / S_V; x = e % S_V;
        return t0 + t < n_tokens ? dk[q_base + (ulong)(t0 + t) * sq2 + x] : 0.f; }
    e -= TC * S_V;
    if (e < TC * ROWS)                { t = e / ROWS; x = e % ROWS;
        return t0 + t < n_tokens ? dv[v_base + (ulong)(t0 + t) * sv2 + row0 + x] : 0.f; }
    e -= TC * ROWS;
    if (e < TC)                       { if (t0 + e >= n_tokens) return 0.f;
                                        const float x = dg[gb_base + (ulong)(t0 + e) * sb2];
                                        return (gating & 1) ? gdn_gate(x, dt_h, a_h, opaque) : x; }
    e -= TC;
    if (e < TC)                       { if (t0 + e >= n_tokens) return 0.f;
                                        const float x = db[gb_base + (ulong)(t0 + e) * sb2];
                                        return (gating & 2) ? gdn_beta(x, opaque) : x; }
    return 0.f;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, NSG, 1)))
kernel void kernel_gated_delta_net_pp(
        global const char * q_buf,     ulong off_q,
        global const char * k_buf,     ulong off_k,
        global const char * v_buf,     ulong off_v,
        global const char * g_buf,     ulong off_g,
        global const char * beta_buf,  ulong off_beta,
        global const char * state_buf, ulong off_state,
        global       char * dst_buf,   ulong off_dst,
        uint  H_v,
        uint  n_tokens,
        uint  n_seqs,
        uint  s_off,
        uint  sq1, uint sq2, uint sq3,
        uint  sv1, uint sv2, uint sv3,
        uint  sb1, uint sb2, uint sb3,
        uint  H_k,
        uint  rq3,
        float scale,
        uint K,
        // the K snapshots' destination (patch 0050): dst after the attention
        // output (snap_buf = dst_buf), or the recurrent cache's rows when the
        // cpy into them is fused; snap_stride = floats from one slot to the next
        global char * snap_buf, ulong off_snap, ulong snap_stride,
        // gating (patch 0051): bit 1, g_buf holds the raw alpha; bit 2, beta_buf the raw beta
        // projections, dt and A per head: g and beta are computed here
        global const char * dt_buf, ulong off_dt, global const char * a_buf, ulong off_a, uint gating, uint opaque,
        // rows mode (patch 0081): the state is the 2D recurrent cache, read at row
        // rows[seq] (state_row_stride floats from one row to the next) instead of the
        // gathered scratch; off_rows is in bytes. use_rows 0 keeps the gather's layout.
        global const char * rows_buf, ulong off_rows, ulong state_row_stride, uint use_rows) {
    global const float * data_q     = (global const float *)(q_buf     + off_q);
    global const float * data_k     = (global const float *)(k_buf     + off_k);
    global const float * data_v     = (global const float *)(v_buf     + off_v);
    global const float * data_g     = (global const float *)(g_buf     + off_g);
    global const float * data_beta  = (global const float *)(beta_buf  + off_beta);
    global const float * data_state = (global const float *)(state_buf + off_state);
    global       float * data_dst   = (global       float *)(dst_buf   + off_dst);

    local float buf[2][CH_FLOATS];

    const uint head = get_group_id(0);
    const float dt_h = (gating & 1) ? ((global const float *)(dt_buf + off_dt))[head] : 0.f;
    const float a_h  = (gating & 1) ? ((global const float *)(a_buf  + off_a ))[head] : 0.f;
    const uint seq  = get_group_id(1);
    const uint wrow = get_group_id(2) * ROWS;         // the work-group's first row
    const uint sg   = get_sub_group_id();
    const uint row0 = wrow + sg * R;                  // this sub-group's first row
    const uint lane = get_sub_group_local_id();
    const uint tid  = get_local_linear_id();

    const uint iq1 = head % H_k;
    const uint iq3 = seq / rq3;
    const ulong state_size = (ulong)S_V * S_V;
    // 0081 fix: the rows form READS the state from its cache row, but the snapshot and dst writes keep the
    // per-sequence layout the CPU reference writes (ops.cpp: s_out = state_out_base + (seq * H + head) * S_v * S_v).
    const ulong state_base = ((ulong)seq * H_v + head) * state_size;
    const ulong state_base_rd = use_rows
        ? ((ulong) ((global const int *)(rows_buf + off_rows))[seq] * state_row_stride + (ulong)head * state_size)
        : state_base;
    const ulong q_base  = (ulong)iq3 * sq3 + (ulong)iq1 * sq1;
    const ulong v_base  = (ulong)seq * sv3 + (ulong)head * sv1;
    const ulong gb_base = (ulong)seq * sb3 + (ulong)head * sb1;

    float S[R][KL];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int m = 0; m < KL; ++m)
            S[r][m] = BLOCK_READ_F(data_state + state_base_rd + (ulong)(row0 + r) * S_V + m * SG);

    // chunk 0 straight into buffer 0
    #pragma unroll
    for (int i = 0; i < PER_THR; ++i) {
        const int e = tid + i * NTHR;
        if (e < CH_FLOATS) buf[0][e] = chunk_load(e, 0, n_tokens, data_q, data_k, data_v, data_g, data_beta,
                                                  q_base, v_base, gb_base, sq2, sv2, sb2, wrow, dt_h, a_h, gating, opaque);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint n_chunks = (n_tokens + TC - 1) / TC;
    ulong attn_off = ((ulong)seq * n_tokens * H_v + head) * S_V;
    for (uint c = 0; c < n_chunks; ++c) {
        local const float * cb = buf[c & 1];
        // the next chunk's share, into registers (its loads overlap this chunk)
        float nxt[PER_THR];
        const uint t1 = (c + 1) * TC;
        #pragma unroll
        for (int i = 0; i < PER_THR; ++i) {
            const int e = tid + i * NTHR;
            nxt[i] = (c + 1 < n_chunks && e < CH_FLOATS)
                ? chunk_load(e, t1, n_tokens, data_q, data_k, data_v, data_g, data_beta, q_base, v_base, gb_base, sq2, sv2, sb2, wrow, dt_h, a_h, gating, opaque)
                : 0.f;
        }

        const uint tn = min((uint)TC, n_tokens - c * TC);
        for (uint tt = 0; tt < tn; ++tt) {
            float k[KL], q[KL];
            #pragma unroll
            for (int m = 0; m < KL; ++m) {
                q[m] = LBLOCK_READ_F(cb + tt * S_V + m * SG);
                k[m] = LBLOCK_READ_F(cb + TC * S_V + tt * S_V + m * SG);
            }
            const float g    = exp(cb[2 * TC * S_V + TC * ROWS + tt]);
            const float beta = cb[2 * TC * S_V + TC * ROWS + TC + tt];

            float kv[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float p = 0.f;
                #pragma unroll
                for (int m = 0; m < KL; ++m) p = fma(S[r][m], k[m], p);
                kv[r] = sub_group_reduce_add(p) * g;
            }
            float out[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float d = (cb[2 * TC * S_V + tt * ROWS + sg * R + r] - kv[r]) * beta;
                float p = 0.f;
                #pragma unroll
                for (int m = 0; m < KL; ++m) {
                    S[r][m] = fma(k[m], d, g * S[r][m]);
                    p = fma(S[r][m], q[m], p);
                }
                out[r] = sub_group_reduce_add(p);
            }
            float mine = 0.f;
            #pragma unroll
            for (int r = 0; r < R; ++r) mine = (lane == r) ? out[r] : mine;
            if (lane < R) data_dst[attn_off + row0 + lane] = mine * scale;
            attn_off += (ulong)S_V * H_v;

            if (K > 1u) {
                const int slot = (int)n_tokens - 1 - (int)(c * TC + tt);
                if (slot >= 0 && slot < (int)K) {
                    global float * sb = (global float *)(snap_buf + off_snap) + (ulong)slot * snap_stride + state_base;
                    #pragma unroll
                    for (int r = 0; r < R; ++r)
                        #pragma unroll
                        for (int m = 0; m < KL; ++m) BLOCK_WRITE_F(sb + (ulong)(row0 + r) * S_V + m * SG, S[r][m]);
                }
            }
        }

        if (c + 1 < n_chunks) {
            #pragma unroll
            for (int i = 0; i < PER_THR; ++i) {
                const int e = tid + i * NTHR;
                if (e < CH_FLOATS) buf[(c + 1) & 1][e] = nxt[i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (K == 1u) {
        global float * sb = data_dst + s_off + state_base;
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < KL; ++m) BLOCK_WRITE_F(sb + (ulong)(row0 + r) * S_V + m * SG, S[r][m]);
    }
}

#else // generation

__attribute__((intel_reqd_sub_group_size(SG)))
kernel void kernel_gated_delta_net_tg(
        global const char * q_buf,     ulong off_q,
        global const char * k_buf,     ulong off_k,
        global const char * v_buf,     ulong off_v,
        global const char * g_buf,     ulong off_g,
        global const char * beta_buf,  ulong off_beta,
        global const char * state_buf, ulong off_state,
        global       char * dst_buf,   ulong off_dst,
        uint  H_v,
        uint  n_tokens,
        uint  n_seqs,
        uint  s_off,
        uint  sq1, uint sq2, uint sq3,
        uint  sv1, uint sv2, uint sv3,
        uint  sb1, uint sb2, uint sb3,
        uint  H_k,
        uint  rq3,
        float scale,
        uint K,
        // the K snapshots' destination (patch 0050): dst after the attention
        // output (snap_buf = dst_buf), or the recurrent cache's rows when the
        // cpy into them is fused; snap_stride = floats from one slot to the next
        global char * snap_buf, ulong off_snap, ulong snap_stride,
        // gating (patch 0051): bit 1, g_buf holds the raw alpha; bit 2, beta_buf the raw beta
        // projections, dt and A per head: g and beta are computed here
        global const char * dt_buf, ulong off_dt, global const char * a_buf, ulong off_a, uint gating, uint opaque,
        // rows mode (patch 0081): the state is the 2D recurrent cache, read at row
        // rows[seq] (state_row_stride floats from one row to the next) instead of the
        // gathered scratch; off_rows is in bytes. use_rows 0 keeps the gather's layout.
        global const char * rows_buf, ulong off_rows, ulong state_row_stride, uint use_rows) {
    global const float * data_q     = (global const float *)(q_buf     + off_q);
    global const float * data_k     = (global const float *)(k_buf     + off_k);
    global const float * data_v     = (global const float *)(v_buf     + off_v);
    global const float * data_g     = (global const float *)(g_buf     + off_g);
    global const float * data_beta  = (global const float *)(beta_buf  + off_beta);
    global const float * data_state = (global const float *)(state_buf + off_state);
    global       float * data_dst   = (global       float *)(dst_buf   + off_dst);

    const uint head = get_group_id(0);
    const float dt_h = (gating & 1) ? ((global const float *)(dt_buf + off_dt))[head] : 0.f;
    const float a_h  = (gating & 1) ? ((global const float *)(a_buf  + off_a ))[head] : 0.f;
    const uint seq  = get_group_id(1);
    const uint row0 = get_group_id(2) * R;
    const uint lane = get_sub_group_local_id();

    const uint iq1 = head % H_k;
    const uint iq3 = seq / rq3;
    const ulong state_size = (ulong)S_V * S_V;
    // 0081 fix: the rows form READS the state from its cache row, but the snapshot and dst writes keep the
    // per-sequence layout the CPU reference writes (ops.cpp: s_out = state_out_base + (seq * H + head) * S_v * S_v).
    const ulong state_base = ((ulong)seq * H_v + head) * state_size;
    const ulong state_base_rd = use_rows
        ? ((ulong) ((global const int *)(rows_buf + off_rows))[seq] * state_row_stride + (ulong)head * state_size)
        : state_base;
    const ulong q_base  = (ulong)iq3 * sq3 + (ulong)iq1 * sq1;   // q and k share strides
    const ulong v_base  = (ulong)seq * sv3 + (ulong)head * sv1;
    const ulong gb_base = (ulong)seq * sb3 + (ulong)head * sb1;

    float S[R][KL];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int m = 0; m < KL; ++m)
            S[r][m] = BLOCK_READ_F(data_state + state_base_rd + (ulong)(row0 + r) * S_V + m * SG);

    // token 0's inputs; then each step fetches the next token's first
    float kn[KL], qn[KL], vn[R], gn = 0.f, bn = 0.f;
    if (n_tokens > 0) {
        #pragma unroll
        for (int m = 0; m < KL; ++m) {
            kn[m] = BLOCK_READ_F(data_k + q_base + m * SG);
            qn[m] = BLOCK_READ_F(data_q + q_base + m * SG);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) vn[r] = data_v[v_base + row0 + r];
        gn = (gating & 1) ? gdn_gate(data_g[gb_base], dt_h, a_h, opaque) : data_g[gb_base];
        bn = (gating & 2) ? gdn_beta(data_beta[gb_base], opaque) : data_beta[gb_base];
    }

    ulong attn_off = ((ulong)seq * n_tokens * H_v + head) * S_V;
    for (uint t = 0; t < n_tokens; ++t) {
        float k[KL], q[KL], v[R];
        #pragma unroll
        for (int m = 0; m < KL; ++m) { k[m] = kn[m]; q[m] = qn[m]; }
        #pragma unroll
        for (int r = 0; r < R; ++r) v[r] = vn[r];
        const float g = exp(gn), beta = bn;
        if (t + 1 < n_tokens) {
            const ulong qo = q_base + (ulong)(t + 1) * sq2;
            #pragma unroll
            for (int m = 0; m < KL; ++m) {
                kn[m] = BLOCK_READ_F(data_k + qo + m * SG);
                qn[m] = BLOCK_READ_F(data_q + qo + m * SG);
            }
            const ulong vo = v_base + (ulong)(t + 1) * sv2;
            #pragma unroll
            for (int r = 0; r < R; ++r) vn[r] = data_v[vo + row0 + r];
            const ulong go = gb_base + (ulong)(t + 1) * sb2;
            gn = (gating & 1) ? gdn_gate(data_g[go], dt_h, a_h, opaque) : data_g[go];
            bn = (gating & 2) ? gdn_beta(data_beta[go], opaque) : data_beta[go];
        }

        // kv_r = (g S_r).k: the R reductions are independent and interleave
        float kv[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            float p = 0.f;
            #pragma unroll
            for (int m = 0; m < KL; ++m) p = fma(S[r][m], k[m], p);
            kv[r] = sub_group_reduce_add(p) * g;
        }
        float out[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float d = (v[r] - kv[r]) * beta;
            float p = 0.f;
            #pragma unroll
            for (int m = 0; m < KL; ++m) {
                S[r][m] = fma(k[m], d, g * S[r][m]);
                p = fma(S[r][m], q[m], p);
            }
            out[r] = sub_group_reduce_add(p);
        }
        // lane r < R writes row r's output
        float mine = 0.f;
        #pragma unroll
        for (int r = 0; r < R; ++r) mine = (lane == r) ? out[r] : mine;
        if (lane < R) data_dst[attn_off + row0 + lane] = mine * scale;
        attn_off += (ulong)S_V * H_v;

        if (K > 1u) {
            const int slot = (int)n_tokens - 1 - (int)t;
            if (slot >= 0 && slot < (int)K) {
                global float * sb = (global float *)(snap_buf + off_snap) + (ulong)slot * snap_stride + state_base;
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    #pragma unroll
                    for (int m = 0; m < KL; ++m) BLOCK_WRITE_F(sb + (ulong)(row0 + r) * S_V + m * SG, S[r][m]);
            }
        }
    }

    if (K == 1u) {
        global float * sb = data_dst + s_off + state_base;
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int m = 0; m < KL; ++m) BLOCK_WRITE_F(sb + (ulong)(row0 + r) * S_V + m * SG, S[r][m]);
    }
}

#endif // GDN_PP

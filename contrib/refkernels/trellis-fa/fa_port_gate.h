// fa_port_gate.h — the port arm's eligibility, as one pure function (no OpenCL call, no
// backend state), so a CPU test can feed it a graph export.
//
// The rules exist because of what flash_attn_port.cl indexes, and each is named with the
// kernel line it protects:
//   q: read by flash_attn_q_to_f16 (the prepass), head*nb[2] + row*nb[1], 8 floats a vload8
//      - nb[1] the row pitch, nb[2] the head stride, both byte strides;
//   k/v: the 2D block reads, surface (D*2-1, n_kv-1, nb[1]-1), base + kvh*nb[2];
//   mask: row * nb[1] + key;
//   table: ((row*n_head + head) * n_bins + bin) - nb[1] == n_bins*4, nb[2] == n_head*n_bins*4;
//   sink: row*n_head + head (sinks_q);
//   positions and flags: element i.
// 0198's first form asked for contiguity and declined every real call: the model's
// FLASH_ATTN_EXT inputs are permuted views (q nb1 12288 / nb2 512, K/V nb1 2048 / nb2 256),
// so the rules are the strides the kernel indexes, not a contiguous order.
#ifndef FA_PORT_GATE_H
#define FA_PORT_GATE_H

#include <ggml.h>
#include <stdint.h>
#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif

// 1 accept, 0 refuse with *why set. Pure: the tensors' type/ne/nb and the op's scalars only.
static inline int fa_port_gate_why(
        const struct ggml_tensor * q, const struct ggml_tensor * k, const struct ggml_tensor * v,
        const struct ggml_tensor * mask, const struct ggml_tensor * dst,
        const struct ggml_tensor * table, const struct ggml_tensor * indices,
        const struct ggml_tensor * sinks_q,
        const struct ggml_tensor * kpos, const struct ggml_tensor * qpos,
        const struct ggml_tensor * cflag, const struct ggml_tensor * rflag,
        int64_t n_bias_sources, int is_intel, int paged, int has_bias,
        float max_bias, float logit_softcap, int32_t window,
        const char ** why_out) {
#define REFUSE(r) do { *why_out = (r); return 0; } while (0)
    if (!has_bias || table == NULL || indices == NULL) REFUSE("no-bias");
    if (!is_intel) REFUSE("device");
    if (paged) REFUSE("paged");
    if (kpos == NULL || qpos == NULL || cflag == NULL || rflag == NULL) REFUSE("no-positions");
    if (sinks_q == NULL) REFUSE("no-sinks");
    if (q->ne[3] != 1 || k->ne[3] != 1) REFUSE("streams");
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) REFUSE("kv-type");
    if (q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128) REFUSE("head");
    if (q->ne[1] < 9) REFUSE("decode");          // n_q <= 8 keeps 0143's split kernels
    if (mask == NULL || mask->type != GGML_TYPE_F16 || mask->ne[3] != 1) REFUSE("mask");
    if (max_bias != 0.0f) REFUSE("alibi");
    if (logit_softcap != 0.0f) REFUSE("softcap");
    if (window < 0) REFUSE("window");

    const int64_t rows = q->ne[1], n_head = q->ne[2], n_kv = k->ne[1];
    if (rows < 1 || n_head % k->ne[2] != 0) REFUSE("shape");
    if (mask->ne[0] < n_kv || mask->ne[1] < rows) REFUSE("mask");
    if (dst == NULL || dst->type != GGML_TYPE_F32 || dst->ne[0] != v->ne[0] ||
        dst->ne[1] != n_head || dst->ne[2] < rows) REFUSE("dst");

    // 0198's original rule: a contiguous order. Red-proven by port_gate_test.cpp on the
    // export's prefill lines - the model's tensors are permuted views.
    const int64_t q_elem = (int64_t) q->nb[0];
    const int64_t k_elem = (int64_t) k->nb[0];
    const int64_t v_elem = (int64_t) v->nb[0];
    if (q->nb[1] != q->ne[0]*q_elem || q->nb[2] != q->ne[1]*q->nb[1] ||
        k->nb[1] != k->ne[0]*k_elem || k->nb[2] != k->ne[1]*k->nb[1] ||
        v->nb[1] != v->ne[0]*v_elem || v->nb[2] != v->ne[1]*v->nb[1]) REFUSE("layout");

    if (sinks_q->type != GGML_TYPE_F32 || sinks_q->ne[0] != n_head || sinks_q->ne[1] < rows ||
        sinks_q->ne[2] != 1) REFUSE("sinks");
    if (table->type != GGML_TYPE_F32 || table->ne[0] <= 0 || table->ne[1] != n_head ||
        table->ne[2] < rows || table->ne[3] != 1) REFUSE("bias-table");
    if (indices->type != GGML_TYPE_I32 || indices->ne[0] < n_kv || indices->ne[1] < rows) REFUSE("bias-indices");
    if (kpos->type != GGML_TYPE_I32 || qpos->type != GGML_TYPE_I32 ||
        kpos->ne[0] < n_kv || qpos->ne[0] < rows) REFUSE("positions");
    if (cflag->type != GGML_TYPE_I8 || rflag->type != GGML_TYPE_I8 ||
        cflag->ne[0] < (n_kv + 15) / 16 || rflag->ne[0] < (rows + 7) / 8) REFUSE("flags");
    (void) n_bias_sources;
#undef REFUSE
    return 1;
}
#if defined(__cplusplus)
}
#endif
#endif

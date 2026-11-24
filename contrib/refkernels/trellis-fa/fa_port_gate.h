// fa_port_gate.h - the port arm's eligibility, as one pure function (no OpenCL call, no
// backend state), so a CPU test can feed it a graph export's own lines.
//
// Each rule exists because of what flash_attn_port.cl indexes:
//   q:     kernel_flash_attn_port_q16 (the prepass) reads 8 floats at
//          head*nb[2] + row*nb[1] + 32*c; any row and head stride, D contiguous (nb[0] 4);
//   k/v:   the 2D block reads, surface (D*2-1, n_kv-1, nb[1]-1) at base + kvh*nb[2]: f16,
//          the head's base 64-byte aligned, the row pitch a multiple of 16 bytes and
//          at least 64 and D*2, below 2^24 (the rules flash_attn_dpas2.cl's reads have);
//   mask:  f16, row*nb[1]/2 + key;
//   table: f32 [n_bins, n_head, rows], ((row*n_head + head)*n_bins + bin), a run of 16 bins;
//   sink:  f32 [n_head, rows], row*n_head + head;
//   positions and flags: element i;
//   dst:   f32 [D, n_head, rows], head*nb[1] + row*nb[2] + 4*col.
// 0198's first form asked for ggml_is_contiguous(q/k/v) and declined every call of the
// model (the first card window, reason=layout): its FLASH_ATTN_EXT inputs are permuted views, q
// [128, Lq, 24] nb [4, 12288, 512] and K/V [128, n_kv, 8] nb [2, 2048, 256]. That rule is
// kept under FA_PORT_GATE_RED for the unit test's red run (port_gate_test.cpp) only.
#ifndef FA_PORT_GATE_H
#define FA_PORT_GATE_H

#include <ggml.h>
#include <stdint.h>
#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif

// 1 accept, 0 refuse with *why set. Pure: the tensors' type/ne/nb/view_offs and scalars.
// buf_align is the backend's buffer alignment (a tensor's base offset is a multiple of it).
static inline int fa_port_gate_why(
        const struct ggml_tensor * q, const struct ggml_tensor * k, const struct ggml_tensor * v,
        const struct ggml_tensor * mask, const struct ggml_tensor * dst,
        const struct ggml_tensor * table, const struct ggml_tensor * indices,
        const struct ggml_tensor * sinks_q,
        const struct ggml_tensor * kpos, const struct ggml_tensor * qpos,
        const struct ggml_tensor * cflag, const struct ggml_tensor * rflag,
        size_t buf_align, int is_intel, int paged, int has_bias,
        float max_bias, float logit_softcap, int32_t window,
        const char ** why_out) {
#define REFUSE(r) do { *why_out = (r); return 0; } while (0)
    if (!has_bias || table == NULL || indices == NULL) REFUSE("no-bias");
    if (!is_intel) REFUSE("device");
    if (paged) REFUSE("paged");
    if (kpos == NULL || qpos == NULL || cflag == NULL || rflag == NULL) REFUSE("no-positions");
    if (sinks_q == NULL) REFUSE("no-sinks");
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) REFUSE("streams");
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) REFUSE("kv-type");
    if (q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128) REFUSE("head");
    if (q->ne[1] < 9) REFUSE("decode");          // n_q <= 8: the model keeps today's graph and kernels
    if (mask == NULL || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1) REFUSE("mask");
    if (max_bias != 0.0f) REFUSE("alibi");
    if (logit_softcap != 0.0f) REFUSE("softcap");
    if (window < 0) REFUSE("window");

    const int64_t D = 128, rows = q->ne[1], n_head = q->ne[2], n_kv = k->ne[1], n_head_kv = k->ne[2];
    if (n_kv < 1 || n_kv >= (1 << 24) || n_head_kv < 1 || n_head % n_head_kv != 0 ||
        v->ne[1] != n_kv || v->ne[2] != n_head_kv) REFUSE("shape");
    if (mask->ne[0] < n_kv || mask->ne[1] < rows || mask->nb[0] != 2 || mask->nb[1] % 2 != 0 ||
        mask->nb[1] < (size_t) (n_kv * 2)) REFUSE("mask");
    if (dst == NULL || dst->type != GGML_TYPE_F32 || dst->ne[0] != D || dst->ne[1] != n_head ||
        dst->ne[2] < rows || dst->nb[0] != 4 || dst->nb[1] != (size_t) (D * 4) ||
        dst->nb[2] != (size_t) (n_head * D * 4)) REFUSE("dst");

#ifdef FA_PORT_GATE_RED
    // 0198's original rule, a contiguous order: refuses the model's permuted views
    if (q->nb[1] != q->ne[0]*q->nb[0] || q->nb[2] != q->ne[1]*q->nb[1] ||
        k->nb[1] != k->ne[0]*k->nb[0] || k->nb[2] != k->ne[1]*k->nb[1] ||
        v->nb[1] != v->ne[0]*v->nb[0] || v->nb[2] != v->ne[1]*v->nb[1]) REFUSE("layout");
#else
    // q: the prepass's vload8 at head*nb[2] + row*nb[1]: D contiguous floats, float-aligned strides
    if (q->nb[0] != 4 || q->nb[1] < 4 || q->nb[2] < 4 || q->nb[1] % 4 != 0 || q->nb[2] % 4 != 0 ||
        q->view_offs % 4 != 0) REFUSE("layout-q");
    // k/v: the 2D block reads' surface
    if (buf_align % 64 != 0) REFUSE("layout-align");
    if (k->nb[0] != 2 || k->nb[1] % 16 != 0 || k->nb[1] < 64 || k->nb[1] < (size_t) (D * 2) ||
        k->nb[1] >= ((size_t) 1 << 24) || k->nb[2] % 64 != 0 || k->view_offs % 64 != 0) REFUSE("layout-k");
    if (v->nb[0] != 2 || v->nb[1] % 16 != 0 || v->nb[1] < 64 || v->nb[1] < (size_t) (D * 2) ||
        v->nb[1] >= ((size_t) 1 << 24) || v->nb[2] % 64 != 0 || v->view_offs % 64 != 0) REFUSE("layout-v");
#endif

    if (sinks_q->type != GGML_TYPE_F32 || sinks_q->ne[0] != n_head || sinks_q->ne[1] < rows ||
        sinks_q->ne[2] != 1 || sinks_q->nb[0] != 4 || sinks_q->nb[1] != (size_t) (n_head * 4)) REFUSE("sinks");
    // the table: [n_bins, n_head, rows], bin fastest; a run read takes 16 bins
    if (table->type != GGML_TYPE_F32 || table->ne[0] < 17 || table->ne[1] != n_head ||
        table->ne[2] < rows || table->ne[3] != 1) REFUSE("bias-table");
    if (table->nb[0] != 4 || table->nb[1] != (size_t) (table->ne[0] * 4) ||
        table->nb[2] != (size_t) (n_head * table->ne[0] * 4)) REFUSE("bias-table-layout");
    if (indices->type != GGML_TYPE_I32 || indices->ne[0] < n_kv || indices->ne[1] < rows) REFUSE("bias-indices");
    if (kpos->type != GGML_TYPE_I32 || qpos->type != GGML_TYPE_I32 || kpos->nb[0] != 4 || qpos->nb[0] != 4 ||
        kpos->ne[0] < n_kv || qpos->ne[0] < rows) REFUSE("positions");
    if (cflag->type != GGML_TYPE_I8 || rflag->type != GGML_TYPE_I8 || cflag->nb[0] != 1 || rflag->nb[0] != 1 ||
        cflag->ne[0] < (n_kv + 15) / 16 || rflag->ne[0] < (rows + 7) / 8) REFUSE("flags");
#undef REFUSE
    return 1;
}
#if defined(__cplusplus)
}
#endif
#endif

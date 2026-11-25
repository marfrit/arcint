// fa_port_gate.h - the port arm's eligibility, as one pure function (no OpenCL call, no
// backend state), so a CPU test can feed it a graph export's own lines.
//
// Each rule exists because of what flash_attn_port.cl indexes:
//   q:     kernel_flash_attn_port_q16 (the prepass) reads 8 floats at
//          head*nb[2] + row*nb[1] + 32*c; any row and head stride, D contiguous (nb[0] 4);
//   k/v:   the 2D block reads, surface (D*2-1, n_kv-1, nb[1]-1) at base + kvh*nb[2]: f16,
//          the head's base 64-byte aligned, the row pitch a multiple of 16 bytes and
//          at least 64 and D*2, below 2^24 (the rules flash_attn_dpas2.cl's reads have);
//   k/v, a q8_0 cache (0199q1, the kernel's PORT_KV8 build): ggml's blocks, 34 bytes for 32
//          values, so a (key, kv head) is 136 bytes and nb[0] is 34. The kernel rounds a head's
//          first byte (base + kvh*nb[2]) down to 64 for the surface base and puts the
//          remainder into the x coordinate, in dwords for K: the head's first byte must be
//          a multiple of 4 (view offset and head stride), NOT of 64 - the model's head stride
//          is 136. The surface row then holds up to 60 + 136 bytes, so the row pitch is at
//          least 196 (and a multiple of 16, below 2^24, as for f16). K and V are both f16 or
//          both q8_0; any other pair is refused (kv-type).
//   mask:  f16, row*nb[1]/2 + key;
//   table: f32 [n_bins, n_head, rows], ((row*n_head + head)*n_bins + bin), a run of 16 bins;
//   sink:  f32 [n_head, rows], row*n_head + head;
//   positions and flags: element i;
//   dst:   f32 [D, n_head, rows], head*nb[1] + row*nb[2] + 4*col.
// 0198's first form asked for ggml_is_contiguous(q/k/v) and declined every call of the
// model on the card (reason=layout): its FLASH_ATTN_EXT inputs are permuted views, q
// [128, Lq, 24] nb [4, 12288, 512] and K/V [128, n_kv, 8] nb [2, 2048, 256]. That rule is
// kept under FA_PORT_GATE_RED for the unit test's red run (port_gate_test.cpp) only.
#ifndef FA_PORT_GATE_H
#define FA_PORT_GATE_H

#include <ggml.h>
#include <ggml-backend.h>
#include <stdint.h>
#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif

// 1 accept, 0 refuse with *why set. Pure: the tensors' type/ne/nb/view_offs and scalars.
// buf_align is the backend's buffer alignment: a tensor's placed offset is its buffer's
// (a multiple of buf_align, by the allocator) plus its view offset, so "the K/V base is
// 64-byte aligned" is decided here, from the view and the alignment, before any tensor is
// placed - the launch only asserts it. wg_max is the work-group the device grants a
// kernel: one sub-group (16 lanes) a head of a kv head's group must fit it.
static inline int fa_port_gate_why(
        const struct ggml_tensor * q, const struct ggml_tensor * k, const struct ggml_tensor * v,
        const struct ggml_tensor * mask, const struct ggml_tensor * dst,
        const struct ggml_tensor * table, const struct ggml_tensor * indices,
        const struct ggml_tensor * sinks_q,
        const struct ggml_tensor * kpos, const struct ggml_tensor * qpos,
        const struct ggml_tensor * cflag, const struct ggml_tensor * rflag,
        size_t buf_align, size_t wg_max, int is_intel, int paged, int has_bias,
        float max_bias, float logit_softcap, int32_t window,
        const char ** why_out) {
#define REFUSE(r) do { *why_out = (r); return 0; } while (0)
    if (!has_bias || table == NULL || indices == NULL) REFUSE("no-bias");
    if (!is_intel) REFUSE("device");
    if (paged) REFUSE("paged");
    if (kpos == NULL || qpos == NULL || cflag == NULL || rflag == NULL) REFUSE("no-positions");
    if (sinks_q == NULL) REFUSE("no-sinks");
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) REFUSE("streams");
#ifdef FA_PORT_GATE_RED_KV8_TYPE
    // the rule before 0199q1: an f16 cache only - refuses the q8_0 lines
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) REFUSE("kv-type");
#endif
    const int kv8 = k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0;
    if (q->type != GGML_TYPE_F32 || !(kv8 || (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16))) REFUSE("kv-type");
    if (q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128) REFUSE("head");
    if (q->ne[1] < 9) REFUSE("decode");          // n_q <= 8: the model keeps today's graph and kernels
    if (mask == NULL || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1) REFUSE("mask");
    if (max_bias != 0.0f) REFUSE("alibi");
    if (logit_softcap != 0.0f) REFUSE("softcap");
    if (window < 0) REFUSE("window");

    const int64_t D = 128, rows = q->ne[1], n_head = q->ne[2], n_kv = k->ne[1], n_head_kv = k->ne[2];
    if (n_kv < 1 || n_kv >= (1 << 24) || n_head_kv < 1 || n_head % n_head_kv != 0 ||
        v->ne[1] != n_kv || v->ne[2] != n_head_kv) REFUSE("shape");
    if ((size_t) (16 * (n_head / n_head_kv)) > wg_max) REFUSE("work-group");
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
    if (kv8) {
#ifdef FA_PORT_GATE_RED_KV8_STRIDE
        // a wrong rule, the f16 build's head stride asked of a q8_0 cache: refuses the model (136)
        if (k->nb[2] % 64 != 0 || v->nb[2] % 64 != 0) REFUSE("layout-k");
#endif
        const size_t hb = (size_t) (D / 32) * 34;      // 136 bytes a (key, kv head)
        if (k->nb[0] != 34 || k->nb[1] % 16 != 0 || k->nb[1] < 60 + hb || k->nb[1] >= ((size_t) 1 << 24) ||
            k->nb[2] % 4 != 0 || k->view_offs % 4 != 0) REFUSE("layout-k");
        if (v->nb[0] != 34 || v->nb[1] % 16 != 0 || v->nb[1] < 60 + hb || v->nb[1] >= ((size_t) 1 << 24) ||
            v->nb[2] % 4 != 0 || v->view_offs % 4 != 0) REFUSE("layout-v");
    } else {
    if (k->nb[0] != 2 || k->nb[1] % 16 != 0 || k->nb[1] < 64 || k->nb[1] < (size_t) (D * 2) ||
        k->nb[1] >= ((size_t) 1 << 24) || k->nb[2] % 64 != 0 || k->view_offs % 64 != 0) REFUSE("layout-k");
    if (v->nb[0] != 2 || v->nb[1] % 16 != 0 || v->nb[1] < 64 || v->nb[1] < (size_t) (D * 2) ||
        v->nb[1] >= ((size_t) 1 << 24) || v->nb[2] % 64 != 0 || v->view_offs % 64 != 0) REFUSE("layout-v");
    }
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

// ---- the op-level forms: everything below reads the op itself, so supports_op, the launch
// ---- and a CPU test ask the same question of the same tensor ----

// the op's slots and params the port reads (ggml-impl.h's GGML_FA_SRC_* / GGML_FA_PARAM_*;
// repeated here so this header needs ggml.h only, and checked against them where both are seen)
enum {
    FA_PORT_SRC_PAGES = 5, FA_PORT_SRC_RING_K = 6, FA_PORT_SRC_RING_V = 7, FA_PORT_SRC_TAG = 8,
    FA_PORT_SRC_HOST_K = 9, FA_PORT_SRC_HOST_V = 10, FA_PORT_SRC_PLAN = 11,
    FA_PORT_SRC_TABLE = 12, FA_PORT_SRC_IDX = 13, FA_PORT_SRC_KPOS = 14, FA_PORT_SRC_QPOS = 15,
    FA_PORT_SRC_CFLAG = 16, FA_PORT_SRC_RFLAG = 17, FA_PORT_SRC_SINK_Q = 18,
    FA_PORT_PARAM_BIAS_FLAG = 15, FA_PORT_PARAM_WINDOW = 16,
};
#ifdef GGML_FA_SRC_SINK_Q
#if GGML_FA_SRC_PAGES != 5 || GGML_FA_SRC_RING_K != 6 || GGML_FA_SRC_RING_V != 7 || GGML_FA_SRC_TAG != 8 || \
    GGML_FA_SRC_HOST_K != 9 || GGML_FA_SRC_HOST_V != 10 || GGML_FA_SRC_PLAN != 11 || GGML_FA_SRC_BIAS_TABLE != 12 || \
    GGML_FA_SRC_BIAS_IDX != 13 || GGML_FA_SRC_KPOS != 14 || GGML_FA_SRC_QPOS != 15 || GGML_FA_SRC_CFLAG != 16 || \
    GGML_FA_SRC_RFLAG != 17 || GGML_FA_SRC_SINK_Q != 18 || GGML_FA_PARAM_BIAS_FLAG != 15 || GGML_FA_PARAM_WINDOW != 16
#error "fa_port_gate.h's slots are not ggml-impl.h's"
#endif
#endif

// any source of the paged K/V forms: the page table, its plan and tag, and the ring and
// host stores they read from. The port reads K/V in place from src[1] / src[2] only, so an
// op that names any of them is not the port's. (FA_PORT_GATE_RED_PAGED: the first form,
// which looked at three of the seven slots - the unit test's red run.)
static inline int fa_port_op_paged(const struct ggml_tensor * op) {
#ifdef FA_PORT_GATE_RED_PAGED
    return op->src[FA_PORT_SRC_PAGES] != NULL || op->src[FA_PORT_SRC_PLAN] != NULL || op->src[FA_PORT_SRC_TAG] != NULL;
#else
    for (int i = FA_PORT_SRC_PAGES; i <= FA_PORT_SRC_PLAN; ++i) {
        if (op->src[i] != NULL) return 1;
    }
    return 0;
#endif
}

// a call carries the port's sources: an indexed bias and the per-query sinks. Such a call is
// the port's or nobody's - no other kernel reads its table or its sink.
static inline int fa_port_carries(const struct ggml_tensor * op) {
    int32_t flag;
    memcpy(&flag, (const char *) op->op_params + FA_PORT_PARAM_BIAS_FLAG * sizeof(int32_t), sizeof flag);
    return flag != 0 && op->src[FA_PORT_SRC_SINK_Q] != NULL;
}

// 0199q1: the op's K/V are a q8_0 cache (the kernel's PORT_KV8 build serves it). Asked only
// of an op the gate accepted, so K and V are then both q8_0 or both f16.
static inline int fa_port_kv_q8(const struct ggml_tensor * op) {
    return op->src[1]->type == GGML_TYPE_Q8_0 && op->src[2]->type == GGML_TYPE_Q8_0;
}

// THE gate, on the op: supports_op and the launch both call this and nothing else of their own
static inline int fa_port_gate_op(const struct ggml_tensor * op, size_t buf_align, size_t wg_max, int is_intel,
                                  const char ** why_out) {
    int32_t flag, window;
    float max_bias, softcap;
    memcpy(&flag,     (const char *) op->op_params + FA_PORT_PARAM_BIAS_FLAG * sizeof(int32_t), sizeof flag);
    memcpy(&window,   (const char *) op->op_params + FA_PORT_PARAM_WINDOW * sizeof(int32_t), sizeof window);
    memcpy(&max_bias, (const char *) op->op_params + 1 * sizeof(float), sizeof max_bias);
    memcpy(&softcap,  (const char *) op->op_params + 2 * sizeof(float), sizeof softcap);
    return fa_port_gate_why(op->src[0], op->src[1], op->src[2], op->src[3], op,
                            op->src[FA_PORT_SRC_TABLE], op->src[FA_PORT_SRC_IDX], op->src[FA_PORT_SRC_SINK_Q],
                            op->src[FA_PORT_SRC_KPOS], op->src[FA_PORT_SRC_QPOS],
                            op->src[FA_PORT_SRC_CFLAG], op->src[FA_PORT_SRC_RFLAG],
                            buf_align, wg_max, is_intel, fa_port_op_paged(op), flag != 0,
                            max_bias, softcap, window, why_out);
}

// Which layer gets the port's graph. An op with the port's sources has a table of another
// layout ([n_bins, n_head, rows], the bias only) and its sink at src[18]; a backend that
// knows the indexed bias but not these sources would answer supports_op "yes" and read the
// table its own way - asking supports_op cannot tell that backend from one that serves the
// port, because the unaware "yes" is the hazard. So a backend DECLARES that it reads the
// port's sources, by answering this name through its registry's get_proc_address (any
// non-null pointer); supports_op then decides each op as before. No backend is named here,
// and absent means refused: a backend without the name (another GPU backend, a remote one,
// an older build of this one), a layer on the CPU, a layer with no device - each gets
// today's graph. So a CPU-only run with the knob set builds the graph it builds without it.
// guard = 0 is the form before this rule (every layer gets the sources): the tests' red.
#define FA_PORT_CAPABILITY_PROC "ggml_backend_flash_attn_port_sources"
static inline int fa_port_layer_attaches(ggml_backend_dev_t dev, int guard) {
    if (!guard) return 1;
    if (dev == NULL) return 0;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != NULL && ggml_backend_reg_get_proc_address(reg, FA_PORT_CAPABILITY_PROC) != NULL;
}

// What a graph builder does with an op it has just given the port's sources: the op keeps
// them only if `backend` is the layer's own backend (backend_is_layers: the scheduler has a
// backend for the layer's device - otherwise the op would run somewhere else) and that
// backend's supports_op accepts exactly this op. Else they come off and the op is today's,
// with `table_today` ([n_head, n_bins, rows], the form ggml_flash_attn_ext_add_indexed_bias
// takes). Returns 1 when the op keeps the port's sources. The one place this is decided:
// the model's build_attn_mha calls it, and the unit test calls it with a mock backend.
static inline int fa_port_settle(struct ggml_tensor * op, ggml_backend_t backend, int backend_is_layers,
                                 struct ggml_tensor * table_today, struct ggml_tensor * indices) {
    if (backend != NULL && backend_is_layers && ggml_backend_supports_op(backend, op)) {
        return 1;
    }
    ggml_flash_attn_ext_drop_port_sources(op, table_today, indices);
    return 0;
}

// The dispatch's block split of a 9 to 64-row call (rows are independent for the kernels it
// feeds: it re-enters the dispatch with stack copies of the op, 8 rows each). The predicate
// is here so that the dispatch and the routing test below share it.
static inline int fa_row_blocks_split(int64_t n_q, int64_t n_batch, int is_intel, int q_is_f32, int mask_ok,
                                      int paged, int gqa_wide, int rb_on) {
    return rb_on && !paged && !gqa_wide && is_intel && n_q > 8 && n_q <= 64 && n_batch == 1 && q_is_f32 && mask_ok;
}

// The query rows of the op the port's launch is handed. port_first = the dispatch takes a
// call that carries the port's sources before any other branch (what it does). port_first
// = 0 is the order it had before: the block split ran first, and a 9 to 64-row port call
// reached the port as blocks of at most 8 rows that still carried its sources - the launch's
// gate then said "decode" for a call supports_op had accepted, and aborted (the card: the
// 1088 x 64 case; 20 and 12 rows the same way).
static inline int64_t fa_port_launch_rows(const struct ggml_tensor * op, int port_first, int is_intel, int gqa_wide, int rb_on) {
    const struct ggml_tensor * q = op->src[0], * mask = op->src[3];
    const int64_t n_q = q->ne[1];
    if (port_first) return n_q;
    return fa_row_blocks_split(n_q, q->ne[3], is_intel, q->type == GGML_TYPE_F32, mask != NULL && mask->ne[2] == 1,
                               fa_port_op_paged(op), gqa_wide, rb_on) ? (n_q < 8 ? n_q : 8) : n_q;
}

// The launch-time decision for an op supports_op was asked about: the same gate, on the op as
// the launch receives it. With port_first it is the op itself, so the two cannot differ.
static inline int fa_port_gate_at_launch(const struct ggml_tensor * op, int port_first, int gqa_wide, int rb_on,
                                         size_t buf_align, size_t wg_max, int is_intel, const char ** why_out) {
    const int64_t rows = fa_port_launch_rows(op, port_first, is_intel, gqa_wide, rb_on);
    if (rows == op->src[0]->ne[1]) return fa_port_gate_op(op, buf_align, wg_max, is_intel, why_out);
    struct ggml_tensor qv = *op->src[0], dv = *op;      // a block, as the split builds it
    qv.ne[1] = rows; dv.ne[2] = rows; dv.src[0] = &qv;
    return fa_port_gate_op(&dv, buf_align, wg_max, is_intel, why_out);
}

// ---- 0223: the key split of a global call (0045's rule, on the port's wave) ----
//
// A call's work-groups are n_head_kv x ngroups (ngroups = ceil(n_q / 8 / qtw)); at ub1024 that is
// 1,024, 5.12 waves on the B60 with the last 12 % full. A split gives each work-group 1/p of the key
// blocks, with its own online-softmax state, and a merge kernel joins the p states. The rule is
// llama.cpp CUDA's parallel_blocks as 0045 has it: the fewest p whose waves are at least 95 % full,
// else the fullest. Two port-specific limits: every split walks at least FA_PORT_SPLIT_MIN_BLOCKS
// 32-key blocks (4,096 keys: a split repeats the tile's setup, and the merge costs a launch), which
// also means no call below 2 x 128 blocks (8,192 keys) splits; and a windowed call (the SWA ring)
// never splits (its visited keys lie in one part of the ring, and the seam moves).

// The port's wave: the work-groups the device holds at once. A work-group is G x qtw SIMD16
// threads, one an (head, 8-row tile); a Xe core holds 32 threads at 256 GRF and 64 at 128 (8
// vector engines x 4 or 8), and CL_DEVICE_MAX_COMPUTE_UNITS counts the engines. B60 (160) at 256
// GRF, G 3, qtw 1: 20 x 10 = 200; B70 (256): 320.
static inline int fa_port_split_wave(int n_cu, int grf, int threads_per_wg) {
    if (n_cu < 8 || threads_per_wg < 1) return 0;
    const int per_core = (grf == 256 ? 32 : 64) / threads_per_wg;
    return per_core < 1 ? 0 : (n_cu / 8) * per_core;
}

#define FA_PORT_SPLIT_MAX        16
#define FA_PORT_SPLIT_MIN_BLOCKS 128

// p for one call: nwg work-groups, the wave, the call's keys and window. cap (GGML_OPENCL_FA_PORT_SPLIT_MAX,
// 0 = none) bounds p; force (GGML_OPENCL_FA_PORT_SPLIT_FORCE, the tests only) sets it, windowed calls too,
// bounded by 16 and the call's blocks. min_blocks (GGML_OPENCL_FA_PORT_SPLIT_MINBLK, 0 = the default
// FA_PORT_SPLIT_MIN_BLOCKS) is the floor of blocks a split walks: a knob so that a window can price it
// (1 = no floor, 0045's and CUDA's form). FA_PORT_SPLIT_RED_NO_FLOOR: the rule without the floor whatever
// min_blocks says (the unit test's red: it splits a 4k call).
static inline int fa_port_splits(int64_t nwg, int64_t wave, int64_t n_kv, int32_t window, int cap, int force, int min_blocks) {
    const int64_t blocks = (n_kv + 31) / 32;
    if (force > 0) {
        int64_t p = force < FA_PORT_SPLIT_MAX ? force : FA_PORT_SPLIT_MAX;
        if (p > blocks) p = blocks;
        return p < 1 ? 1 : (int) p;
    }
    if (window != 0 || wave <= 0 || nwg <= 0) return 1;
#ifdef FA_PORT_SPLIT_RED_NO_FLOOR
    int64_t smax = blocks;
#else
    int64_t smax = blocks / (min_blocks > 0 ? min_blocks : FA_PORT_SPLIT_MIN_BLOCKS);
#endif
    if (smax > FA_PORT_SPLIT_MAX) smax = FA_PORT_SPLIT_MAX;
    if (cap > 0 && smax > cap) smax = cap;
    if (smax <= 1) return 1;
    int best = 1;
    int64_t eff_best = 0, nw_best = 0;
    for (int p = 1; p <= smax; ++p) {
        const int64_t nb = nwg * p, nw = (nb + wave - 1) / wave, eff = 100 * nb / (nw * wave);
        if (eff_best >= 95 && nw > nw_best) break;
        if (eff > eff_best) { eff_best = eff; nw_best = nw; best = p; }
    }
    return best;
}

// the 32-key blocks a split walks: split s takes [s * b, min(B, (s + 1) * b)) of the call's B blocks.
// A split past B (p does not divide B evenly) walks none: its state is (M_INIT, 0) and weighs 0.
static inline int fa_port_split_blocks(int64_t n_kv, int p) {
    const int64_t blocks = (n_kv + 31) / 32;
    return (int) ((blocks + p - 1) / (p < 1 ? 1 : p));
}

// a split state's floats: O as a sub-group holds it (8 chunks x 8 rows x 16 lanes), then m (8) and l (8)
#define FA_PORT_SPLIT_STATE_FLOATS (8 * 128 + 16)

#if defined(__cplusplus)
}
#endif
#endif

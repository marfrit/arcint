// 0088: the int8 Q K^T of `flash_attn_gqa_dpas.cl` -- the once-a-call Q
// quantization, the q8_0 block arithmetic and the staged row layout, one text
// for the kernel (prepended by the backend at build time, as 0071 does with
// kq_i8_ops.inc) and the device-free test (tests/test-opencl-fa-gqa-i8.cpp).
//
// Mechanism from the fork's DPAS decode attention (arc-b580,
// ggml/src/ggml-sycl/fattn-dec.cpp:688-747 and its inline xe2-dpas.hpp): the
// key cache's raw q8_0 bytes are the B operand, Q rows are int8 with one RNE
// scale per row per 32-dim block, the integer result is scaled by
// dq[row][block] * dk[key][block]. Here B is s8 x s8 (our cache is q8_0); the
// fork's q4_0 K is s8 x s4, which OpenCL's public builtins do not offer (the
// fork's own s4 form is inline Xe2 vISA, an option this patch does not take).
//
// Evidence: the arithmetic is the fork's (`code`, fattn-dec.cpp:688-747,
// xe2-dpas.hpp:15-42); the q8_0 block layout is ggml's (ggml-common.h block_q8_0,
// ggml-quants.c). A q8_0 block is 34 bytes: half d (2 bytes), then 32 signed
// codes: dims 0..15 in bytes 0..15, dims 16..31 in bytes 16..31.
//
// Operand form: for s8 x s8 k32 the a operand's per-lane 16 bytes are 8 rows
// x (dims 2l, 2l+1), and `intel_sub_group_block_read_us8` reads them strided
// (p[l], p[l + 16], ...) from a row-major [row][32 codes] tile; that is the
// form 0071's int8 expert GEMM stages and tests. The fork's s8 x s4 is
// different: its B packs the nibbles as dims i and i + 16 in one byte, so its
// a short is (code[l], code[l+16]) per row (fattn-dec.cpp:721-731). So Q's
// codes are stored ascending here, not interleaved.

#ifndef FA_GQA_I8_OPS_INC
#define FA_GQA_I8_OPS_INC

#ifdef __cplusplus
#include <cmath>
#define FA_GQA_I8_RINT(x) std::rint(x)
#else
#define FA_GQA_I8_RINT(x) rint(x)
#endif

// ---- q8_0 block offsets (bytes) ----
// the block's fp16 scale d and its 32 signed codes
static inline int fa_gqa_i8_blk(int b) { return 34 * b; }
static inline int fa_gqa_i8_qs(int b) { return 34 * b + 2; }
// the second half of a block: dims 16..31 live behind byte 16 of the codes
static inline int fa_gqa_i8_half(int lane) { return 16 + lane; }
// a q8_0 key row in dwords (head size dk, a multiple of 32) and the staged row
// stride: one dword more than the row, so lane = key reads land on different
// banks
static inline int fa_gqa_i8_row_dw(int dk) { return (dk / 32) * 34 / 4; }
static inline int fa_gqa_i8_stride_dw(int dk) { return fa_gqa_i8_row_dw(dk) + 1; }

// ---- the rows the Q staging loop covers ----
// The a operand holds 8 row tiles of 8 (head, token) rows; vertices of a tile
// are staged here once a call. Every row of every tile the batch needs must be
// written: a row left unwritten is whatever the local allocation held, and its
// S is then that garbage (the card's red: 135 of 135 FLASH_ATTN_EXT cases at
// ERR ~0.10). Rows past n_rows are staged as zeros with dq 0, which is the f16
// path's zero Q row.
static inline int fa_gqa_i8_stage_rows(int n_rows) {
    const int tiles = (n_rows + 7) / 8;
    return (tiles > 8 ? 8 : tiles) * 8;
}

// ---- Q quantization: one RNE scale per row per 32-dim block ----
static inline float fa_gqa_i8_amax32(const float x[32]) {
    float a = 0.f;
    for (int i = 0; i < 32; ++i) {
        const float v = x[i] < 0.f ? -x[i] : x[i];
        a = v > a ? v : a;
    }
    return a;
}

// code = round-to-nearest-even(x / d), d = amax / 127; x here already carries
// the softmax scale, so dq carries it too (the fork's form)
static inline int fa_gqa_i8_code(float x, float id) {
    return (int) FA_GQA_I8_RINT(x * id);
}

// one (row, block): dq and the 32 codes in dim order (zero rows give dq 0
// and codes 0)
static inline void fa_gqa_i8_quant32(const float x[32], float *dq, int q[32]) {
    const float amax = fa_gqa_i8_amax32(x);
    const float d    = amax / 127.f;
    const float id   = amax > 0.f ? 127.f / amax : 0.f;
    *dq = d;
    for (int i = 0; i < 32; ++i) q[i] = fa_gqa_i8_code(x[i], id);
}

#endif

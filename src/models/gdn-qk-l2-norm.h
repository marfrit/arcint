#pragma once

// Patch 0082: one l2_norm over the joint q/k head groups of the GDN conv output.
//
// The gated delta-net layers' q and k norm is rms_norm(x, eps/n) * (1/sqrt(n))
// per row (build_gdn_l2_norm, src/models/models.h), run once for q and once for
// k. In the conv output q and k are adjacent head groups of the same width, so
// one l2_norm over the joint view [head_k_dim, 2*num_k_heads, n_seq_tokens,
// n_seqs] normalises both, and q and k are views into its dense result -- the
// same normalization per row, one pair of nodes fewer a layer.
//
// Device-free by construction (ggml.h only), so tests/test_gdn_qk_l2_norm.cpp
// runs it against the two separate norms on the CPU backend and compares bytes.
//
// This header is NOT a general API: it assumes q then k adjacent in the conv
// output at the same width, which is what the qwen35-family builders lay out.

#include "ggml.h"

#include <cmath>

static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

// q_out/k_out: the normalised [head_k_dim, num_k_heads, n_seq_tokens, n_seqs] slices.
// conv_qkv_mix: the strided conv output; nb1_qkv: its row stride in bytes.
static inline void build_gdn_qk_l2_norm(
        ggml_context * ctx,
        ggml_tensor  * conv_qkv_mix,
              int64_t   head_k_dim,
              int64_t   num_k_heads,
              int64_t   n_seq_tokens,
              int64_t   n_seqs,
              size_t    nb1_qkv,
              float     eps,
        ggml_tensor ** q_out,
        ggml_tensor ** k_out) {
    // q and k are the first 2*num_k_heads head groups of the conv output; the joint
    // view keeps the conv output's row stride (nb[1] = head_k_dim contiguous, the
    // group stride nb[2] = nb1_qkv), so every dim-0 row is contiguous
    ggml_tensor * qk = ggml_view_4d(ctx, conv_qkv_mix, head_k_dim, 2 * num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);
    qk = build_gdn_l2_norm(ctx, qk, eps);

    // the norm's result is dense: q and k are its first and second head groups
    const size_t nb1 = qk->nb[1];
    *q_out = ggml_view_4d(ctx, qk, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            nb1, qk->nb[2], qk->nb[3], 0);
    *k_out = ggml_view_4d(ctx, qk, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            nb1, qk->nb[2], qk->nb[3], num_k_heads * nb1);
}

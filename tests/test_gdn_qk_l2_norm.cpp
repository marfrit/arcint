// Patch 0082's device-free cell: one l2_norm over the joint q/k head groups.
//
// The gated delta-net layer normed q and k separately: two rms_norm+scale pairs
// over strided views of the conv output. The joint form normalises the adjacent
// q|k head groups in one pair and returns q and k as views of its dense result.
// This cell builds both forms on the CPU backend from the same conv mix and
// requires q_conv's and k_conv's bytes identical at four shapes.
//
// The helper is src/models/gdn-qk-l2-norm.h, the code the qwen35-family builders
// call under LLAMA_FUSE_QK_L2_NORM=1; this test does not read the switch (the
// switch-off arm is byte-identical by construction: it is the old code).
//
// Red cases (one-line, in the patch README section): the k view's offset drops
// its head-group term (`num_k_heads * nb1` -> `nb1`), the joint view's group
// stride taken as contiguous (`nb1_qkv` -> `ggml_row_size(type, head_k_dim)`),
// and the norm applied after the split (two norms again). Each fails the
// mismatch cells.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/models/gdn-qk-l2-norm.h"

namespace {

int g_failures = 0;

#define CHECK(cond, what) do { \
        if (!(cond)) { printf("FAIL: %s\n", what); ++g_failures; } \
    } while (0)

// the joint form returns q and k as views into its dense result: their token
// stride carries both head groups (nb[2] = 2*num_k_heads rows), so the logical
// elements are walked with the tensor's strides, not copied linearly
std::vector<float> extract(ggml_tensor * t) {
    std::vector<float> r((size_t) ggml_nelements(t));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    r[k++] = *(const float *) ((const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                }
            }
        }
    }
    return r;
}

float pseudo(int i) {
    return std::sin(0.41f * (float) i) * 2.0f;
}

struct shape {
    int64_t head_k_dim;
    int64_t num_k_heads;
    int64_t n_seq_tokens;
    int64_t n_seqs;
};

struct out {
    std::vector<float> q;
    std::vector<float> k;
};

ggml_tensor * q_view(ggml_context * ctx, ggml_tensor * mix, const shape & s) {
    return ggml_view_4d(ctx, mix, s.head_k_dim, s.num_k_heads, s.n_seq_tokens, s.n_seqs,
            ggml_row_size(mix->type, s.head_k_dim),
            mix->nb[1],
            mix->nb[1] * s.n_seq_tokens,
            0);
}

ggml_tensor * k_view(ggml_context * ctx, ggml_tensor * mix, const shape & s) {
    return ggml_view_4d(ctx, mix, s.head_k_dim, s.num_k_heads, s.n_seq_tokens, s.n_seqs,
            ggml_row_size(mix->type, s.head_k_dim),
            mix->nb[1],
            mix->nb[1] * s.n_seq_tokens,
            s.head_k_dim * s.num_k_heads * ggml_element_size(mix));
}

out run(const shape & s, bool joint) {
    ggml_init_params ip = { /*.mem_size =*/ (size_t) 64 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t qk_dim = 2 * s.num_k_heads * s.head_k_dim;
    const int64_t v_dim  = 8;   // a non-empty v tail, so the mix row is wider than q|k
    ggml_tensor * mix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, qk_dim + v_dim, s.n_seq_tokens, s.n_seqs);
    float * p = (float *) mix->data;
    for (int i = 0; i < (int) ggml_nelements(mix); ++i) p[i] = pseudo(i);

    ggml_tensor * q;
    ggml_tensor * k;
    const float eps = 1e-6f;
    if (joint) {
        build_gdn_qk_l2_norm(ctx, mix, s.head_k_dim, s.num_k_heads, s.n_seq_tokens, s.n_seqs,
                mix->nb[1], eps, &q, &k);
    } else {
        q = build_gdn_l2_norm(ctx, q_view(ctx, mix, s), eps);
        k = build_gdn_l2_norm(ctx, k_view(ctx, mix, s), eps);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, k);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    out r;
    r.q = extract(q);
    r.k = extract(k);
    ggml_free(ctx);
    return r;
}

void compare(const shape & s, const char * name) {
    out a = run(s, false);
    out b = run(s, true);
    CHECK(a.q.size() == b.q.size() && a.k.size() == b.k.size(), name);
    if (a.q.size() != b.q.size() || a.k.size() != b.k.size()) return;
    CHECK(memcmp(a.q.data(), b.q.data(), a.q.size() * sizeof(float)) == 0, name);
    CHECK(memcmp(a.k.data(), b.k.data(), a.k.size() * sizeof(float)) == 0, name);
    if (memcmp(a.k.data(), b.k.data(), a.k.size() * sizeof(float)) != 0) {
        for (size_t i = 0; i < a.k.size(); ++i) {
            if (memcmp(&a.k[i], &b.k[i], sizeof(float)) != 0) {
                printf("  first differing k element %zu: separate %a joint %a\n", i, (double) a.k[i], (double) b.k[i]);
                break;
            }
        }
    }
}

} // namespace

int main() {
    compare({ 4, 2, 1, 1 }, "head 4, 2 k heads, one token, one sequence");
    compare({ 4, 2, 3, 1 }, "head 4, 2 k heads, three tokens");
    compare({ 4, 2, 1, 2 }, "head 4, 2 k heads, two sequences");
    compare({ 8, 3, 2, 2 }, "head 8, 3 k heads, two tokens, two sequences");

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the joint q/k norm's bytes equal the two separate norms'\n");
    return 0;
}

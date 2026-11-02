// Patch 0081's device-free cell: the gated delta-net's rows form reads the same
// bytes the gather hands over.
//
// The rows form (ggml_gated_delta_net_rows, src[6] = I32 row indices over the 2D
// recurrent cache) exists so the per-layer GET_ROWS gather of the live state rows
// disappears. This cell runs both forms on the CPU backend, the row indices
// pointing at cache rows that are neither 0 nor their sequence's ordinal, and
// requires the packed outputs -- the attention scores and the K state snapshots
// -- to be byte-identical. It is device-free: ggml's CPU kernels take the op.
//
// Cases: one sequence (ids {2}), two sequences (ids {1, 3}, K = 1 and K = 3),
// three tokens a sequence. The multi-sequence case is the one the build gate
// declines (llama_gdn_state_rows_enabled: s_copy rows may alias the extra
// relocation), but the CPU op itself must read each sequence's own row right.
//
// Red cases (one-line, each run before the patch was accepted; the commands are
// in the patch README section): the CPU op reads the sequence ordinal instead of
// state_rows[seq] (`state_rows ? ... : iv3 * state_seq_stride` -> `iv3 *
// state_seq_stride`), and the row stride taken from nb[3] instead of nb[1]
// (`src_state->nb[1]` -> `src_state->nb[3]`). Each fails the mismatch cells.
//
// Not covered here: the OpenCL kernel's row read (the cards' arm: the same
// comparison in a test-backend-ops-style run on the card, README 0081) and the
// graph's build gate (tests/python/test_gdn_state_rows.py, source text).

#define GGML_USE_CPU 1

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, what) do { \
        if (!(cond)) { printf("FAIL: %s\n", what); ++g_failures; } \
    } while (0)

struct case_cfg {
    int64_t S_v;
    int64_t H;
    int64_t n_tokens;
    int      n_seqs;
    int32_t  K;
    std::vector<int32_t> ids;
};

float pseudo(int i) {
    // deterministic, well in the f32 range
    return std::sin(0.37f * (float) i) * 1.5f;
}

// one form's packed output, computed on the CPU
std::vector<float> run(const case_cfg & c, bool rows) {
    ggml_init_params ip = { /*.mem_size =*/ (size_t) 256 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t D = c.S_v * c.S_v * c.H;
    const int64_t n_rows = 8;   // the cache is wider than the used rows: row strides do not add up

    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);
    ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seqs);
    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, c.H, c.n_tokens, c.n_seqs);

    float * pc = (float *) cache->data;
    for (int i = 0; i < (int) ggml_nelements(cache); ++i) pc[i] = pseudo(i + 17);
    int32_t * pi = (int32_t *) ids->data;
    for (int s = 0; s < c.n_seqs; ++s) pi[s] = c.ids[s];
    float * pq = (float *) q->data;
    for (int i = 0; i < (int) ggml_nelements(q); ++i)       pq[i] = pseudo(i);
    float * pk = (float *) k->data;
    for (int i = 0; i < (int) ggml_nelements(k); ++i)       pk[i] = pseudo(i + 101);
    float * pv = (float *) v->data;
    for (int i = 0; i < (int) ggml_nelements(v); ++i)       pv[i] = pseudo(i + 202);
    float * pgg = (float *) g->data;
    for (int i = 0; i < (int) ggml_nelements(g); ++i)       pgg[i] = -0.2f - 0.05f * (float) (i % 7);
    float * pb = (float *) beta->data;
    for (int i = 0; i < (int) ggml_nelements(beta); ++i)    pb[i] = 0.5f;

    ggml_tensor * out;
    if (rows) {
        out = ggml_gated_delta_net_rows(ctx, q, k, v, g, beta, cache, ids, c.K);
    } else {
        ggml_tensor * gathered = ggml_get_rows(ctx, cache, ids);
        gathered = ggml_reshape_4d(ctx, gathered, c.S_v, c.S_v, c.H, c.n_seqs);
        out = ggml_gated_delta_net(ctx, q, k, v, g, beta, gathered, c.K);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::vector<float> res((size_t) ggml_nelements(out));
    memcpy(res.data(), out->data, res.size() * sizeof(float));
    ggml_free(ctx);
    return res;
}

void compare(const case_cfg & c, const char * name) {
    std::vector<float> a = run(c, false);
    std::vector<float> b = run(c, true);
    CHECK(a.size() == b.size(), name);
    if (a.size() != b.size()) return;
    const size_t bytes = a.size() * sizeof(float);
    CHECK(memcmp(a.data(), b.data(), bytes) == 0, name);
    if (memcmp(a.data(), b.data(), bytes) != 0) {
        for (size_t i = 0; i < a.size(); ++i) {
            if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                printf("  first differing element %zu: gather %a rows %a\n", i, (double) a[i], (double) b[i]);
                break;
            }
        }
    }
    // the row read must not be the sequence ordinal: ids 0 is not 0 and ids 1 is not 1
    for (int s = 0; s < c.n_seqs; ++s) {
        CHECK(c.ids[s] != s, name);
    }
}

} // namespace

int main() {
    // one sequence, one token, ids[0] = 2 (neither 0 nor the ordinal)
    compare({ 4, 2, 1, 1, 1, { 2 } }, "one sequence K=1");
    // two sequences, K=1: each reads its own row (2 != 0, 3 != 1)
    compare({ 4, 2, 1, 2, 1, { 2, 3 } }, "two sequences K=1");
    // two sequences, three tokens, K=3: the packed output carries 3 snapshots
    compare({ 4, 2, 3, 2, 3, { 1, 3 } }, "two sequences K=3, three tokens");
    // the row stride comes from nb[1] (the cache's row stride), not the plane stride
    compare({ 4, 3, 2, 1, 3, { 5 } }, "one sequence K=3, head 3");

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the rows form's bytes equal the gathered form's\n");
    return 0;
}

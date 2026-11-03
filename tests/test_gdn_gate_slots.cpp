// Patch 0081's device-free cell for the GDN family's source slots: the served
// graph's gate chain with 0051's raw projections attached, all new switches off
// (and then rows on, and replay on). The three features used to take src[6]:
// 0051's alpha, the rows form's row index, the replay's records. Served with
// 0051's fold on, src[6] held the F32 alpha, the OpenCL supports_op declined the
// node, it fell to the CPU, and the CPU kernel asserted
// GGML_ASSERT(src_rows->type == GGML_TYPE_I32) -- all three units crashed.
//
// The cell builds the node exactly as the served graph does:
//   beta  = sigmoid(reshape(1, H, T, B) of the beta projection)
//   gate  = reshape(1, H, T, B) of mul(softplus(add(alpha, dt_bias)), A)
// and attaches the raw projections through gdn_keep_gating_inputs -- the same
// header the builders use (src/models/gdn-gate-inputs.h), so a mutation of the
// slots is a mutation of this cell. Then it computes on the CPU backend:
//  - plain (gathered state), no extras              -> the baseline bytes
//  - plain + extras (0051 fold active, switches off) -> must equal the baseline
//  - rows form + extras                             -> must equal the baseline
//  - replay + extras                                -> attention region equals the
//    baseline's, and the whole output equals the same replay node without extras
//
// Red cases (one line each, run before the fix was accepted; commands in the
// patch README section): gdn_keep_gating_inputs writing src[6]/src[7] again (the
// served pre-fix form) makes the plain+extras case abort in the CPU op's
// GGML_ASSERT; the plain builder's mode word set to ROWS makes it abort as well
// (a rows node without a row index).
//
// Not covered here: the OpenCL fold's match and the kernels' row read on the
// cards (the card arms in README 0081/0086/0087).

#define GGML_USE_CPU 1

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gdn-gate-inputs.h"

namespace {

int g_failures = 0;

#define CHECK(cond, what) do { \
        if (!(cond)) { printf("FAIL: %s\n", what); ++g_failures; } \
    } while (0)

const int64_t S_v = 16;   // the GDN head dimension
const int64_t H   = 3;    // value heads
const int64_t T   = 3;    // tokens (a verify-sized batch)
const int     K   = 3;    // snapshot slots
const int64_t D   = S_v * S_v * H;

float pseudo(int i) {
    return std::sin(0.37f * (float) i) * 1.5f;
}

// one arm's packed output, computed on the CPU backend
//   mode: 0 plain gathered, 1 rows form, 2 replay
//   extras: attach the raw gating projections the way the served graph does
std::vector<float> run(int mode, bool extras) {
    ggml_init_params ip = { /*.mem_size =*/ (size_t) 256 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t n_rows = 8;   // the cache is wider than the used rows
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);
    ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, 1);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, 1);
    ggml_tensor * alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, T, 1);   // the alpha projection, pre-bias
    ggml_tensor * dt    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);         // ssm_dt
    ggml_tensor * A     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);         // -A_log.exp()
    ggml_tensor * braw  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, 1); // the beta projection, pre-sigmoid

    float * pc = (float *) cache->data;
    for (int i = 0; i < (int) ggml_nelements(cache); ++i) pc[i] = pseudo(i + 17);
    ((int32_t *) ids->data)[0] = 2;   // neither 0 nor the sequence's ordinal
    float * pq = (float *) q->data;
    for (int i = 0; i < (int) ggml_nelements(q); ++i) pq[i] = pseudo(i);
    float * pk = (float *) k->data;
    for (int i = 0; i < (int) ggml_nelements(k); ++i) pk[i] = pseudo(i + 101);
    float * pv = (float *) v->data;
    for (int i = 0; i < (int) ggml_nelements(v); ++i) pv[i] = pseudo(i + 202);
    float * pa = (float *) alpha->data;
    for (int i = 0; i < (int) ggml_nelements(alpha); ++i) pa[i] = pseudo(i + 303);
    float * pd = (float *) dt->data;
    for (int i = 0; i < (int) ggml_nelements(dt); ++i) pd[i] = 0.4f + 0.1f * (float) i;
    float * pA = (float *) A->data;
    for (int i = 0; i < (int) ggml_nelements(A); ++i) pA[i] = -0.5f - 0.2f * (float) i;
    float * pb = (float *) braw->data;
    for (int i = 0; i < (int) ggml_nelements(braw); ++i) pb[i] = pseudo(i + 404);

    // the served gate chain: softplus(alpha + dt_bias) * A, then sigmoid(b)
    ggml_tensor * alpha_biased   = ggml_add      (ctx, alpha, dt);
    ggml_tensor * alpha_softplus = ggml_softplus (ctx, alpha_biased);
    ggml_tensor * gate           = ggml_mul      (ctx, alpha_softplus, A);
    ggml_tensor * g              = ggml_reshape_4d(ctx, gate, 1, H, T, 1);
    ggml_tensor * beta_sig       = ggml_sigmoid  (ctx, braw);   // [1,H,T,1], as the builder reshapes it first

    ggml_tensor * gathered = ggml_get_rows(ctx, cache, ids);       // the plain path's gather
    gathered = ggml_reshape_4d(ctx, gathered, S_v, S_v, H, 1);

    ggml_tensor * out;
    if (mode == 2) {
        // the replay's records and control: two banks, no pending record
        const int64_t R = H * (2 * S_v + 1);
        ggml_tensor * rec  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, (K - 1) * R, 2, 1);
        ggml_tensor * ctrl = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
        float * pr = (float *) rec->data;
        for (int i = 0; i < (int) ggml_nelements(rec); ++i) pr[i] = 0.0f;
        ((int32_t *) ctrl->data)[0] = 0;   // n_rec
        ((int32_t *) ctrl->data)[1] = 0;   // bank
        out = ggml_gated_delta_net_replay(ctx, q, k, v, g, beta_sig, gathered, rec, ctrl, K);
    } else if (mode == 1) {
        out = ggml_gated_delta_net_rows(ctx, q, k, v, g, beta_sig, cache, ids, K);
    } else {
        out = ggml_gated_delta_net(ctx, q, k, v, g, beta_sig, gathered, K);
    }
    if (extras) {
        gdn_keep_gating_inputs(out, g, beta_sig);
        CHECK(out->src[8] != nullptr, "the alpha projection rides src[8]");
        CHECK(out->src[9] != nullptr, "the b projection rides src[9]");
        // the family's mode word (ggml.h): read here as the builders write it
        const int32_t mode = ((const int32_t *) out->op_params)[1];
        CHECK(out->src[6] == nullptr || mode != GGML_GDN_MODE_GATHERED,
              "src[6] is not a gating input of a gathered node");
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    std::vector<float> res((size_t) ggml_nelements(out));
    memcpy(res.data(), out->data, res.size() * sizeof(float));
    ggml_free(ctx);
    return res;
}

int g_cmp_failures = 0;

// n == SIZE_MAX: the whole outputs, which must be the same size; otherwise the
// first n elements only (the replay's attention region against the plain path)
void compare(const std::vector<float> & a, const std::vector<float> & b, const char * name, size_t n = SIZE_MAX) {
    const bool whole = n == SIZE_MAX;
    if (whole && a.size() != b.size()) {
        printf("FAIL: %s (sizes %zu / %zu)\n", name, a.size(), b.size());
        ++g_cmp_failures;
        return;
    }
    const size_t len = whole ? a.size() : (n < a.size() && n < b.size() ? n : (a.size() < b.size() ? a.size() : b.size()));
    for (size_t i = 0; i < len; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            printf("FAIL: %s, first differing element %zu (%a / %a)\n", name, i, (double) a[i], (double) b[i]);
            ++g_cmp_failures;
            return;
        }
    }
    printf("ok %s\n", name);
}

const size_t attn = (size_t) (S_v * H * T);

} // namespace

int main() {
    // the served graph with 0051's fold active and every new switch off: the raw
    // projections attached to the gathered node. Today this asserts in the CPU
    // op (src[6] = alpha), so it is the red arm.
    const std::vector<float> plain  = run(0, false);
    const std::vector<float> served = run(0, true);
    compare(plain, served, "plain + 0051's raw projections equals the plain path");

    // rows on, with the same raw projections attached
    const std::vector<float> rows = run(1, true);
    compare(plain, rows, "rows form + the raw projections equals the plain path");

    // replay on, with the same raw projections attached: the attention region is
    // the plain path's, and the extras change nothing at all
    const std::vector<float> replay_x = run(2, true);
    const std::vector<float> replay_0 = run(2, false);
    compare(replay_x, replay_0, "replay: the raw projections change nothing");
    compare(plain, replay_x, "replay: the attention region equals the plain path", attn);

    if (g_failures || g_cmp_failures) {
        printf("FAILED: %d checks, %d comparisons\n", g_failures, g_cmp_failures);
        return 1;
    }
    printf("OK: the GDN features' source slots are disjoint\n");
    return 0;
}

// Patch 0086/0087's device-free cell: GDN rollback by replay reproduces the
// snapshot path's bytes.
//
// A chain of prompt steps runs twice on the CPU backend: the snapshot form
// (ggml_gated_delta_net, one launch per step, rollback by reading the snapshot
// plane) and the replay form (ggml_gated_delta_net_replay, a base state plus
// per-token records replayed after a rollback). The attention outputs of every
// step and the final base state must be byte-identical. It is device-free: the
// CPU kernels take both ops; the cards' arm is the replay-chain case in
// test-backend-ops (README 0086/0087).
//
// Steps: n_tokens 5 (no rollback), 3 with rollback 2 (the prompt is cut back to
// the state before the split point), then 1. K = 5: five snapshot slots, four
// records per bank. H = 4 heads over 1 v-repeat (4 value heads).
//
// Red cases, each run before the patch was accepted (commands in the README):
//   (a) the replay prologue's mad dropped -> the final base differs;
//   (b) the record's gate written as 0 -> the final base differs;
//   (c) the base state read from plane 1 instead of plane 0 -> the final base differs.

#define GGML_USE_CPU 1

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, what) do { \
        if (!(cond)) { printf("FAIL: %s\n", what); ++g_failures; } \
    } while (0)

int64_t S = 32;
int64_t H = 4;
int64_t K = 5;
int64_t D = S * S * H;
int64_t R = H * (2 * S + 1);
int64_t RK = R * (K - 1);

float pseudo(int i) {
    return std::sin(0.37f * (float) i) * 1.5f;
}

struct run_result {
    std::vector<float> attn;
    std::vector<float> base;
};

// the same chain, one launch per step; uses_snap = the snapshot form
run_result run_chain(bool uses_snap) {
    // one context for the whole chain: the views carry the dependency graph
    ggml_init_params ip = { /*.mem_size =*/ (size_t) 256 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, uses_snap ? K : 2);
    std::vector<float> s0((size_t) D);
    for (size_t i = 0; i < s0.size(); ++i) s0[i] = pseudo((int) i);
    std::vector<float> z((size_t) ggml_nelements(cache), 0.0f);
    std::copy(s0.begin(), s0.end(), z.begin());
    memcpy(cache->data, z.data(), z.size() * sizeof(float));

    const std::vector<std::pair<int64_t, int64_t>> steps = { { 5, 0 }, { 3, 2 }, { 1, 0 } };

    ggml_tensor * src = cache;   // rows the next launch reads
    ggml_tensor * rec = nullptr;
    if (!uses_snap) {
        // no records before the first step: both banks zero
        rec = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, RK, 2, 1);
        memset(rec->data, 0, ggml_nbytes(rec));
    }
    int64_t snap_plane = 0, n_rec = 0, bank = 0;

    std::vector<ggml_tensor *> outs;
    int64_t seed = 0;
    for (const auto & st : steps) {
        const int64_t n = st.first, r = st.second;
        const int64_t m = std::min(n - 1, K - 1);
        GGML_ASSERT(r <= m);

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, n, 1);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
        for (ggml_tensor * t : { q, k, v, g, b }) {
            for (int i = 0; i < (int) ggml_nelements(t); ++i) {
                float x = pseudo((int) (i + seed));
                if (t == g)  x = -0.2f - 0.05f * (float) (i % 7);
                if (t == b)  x = 0.5f;
                if (t == v)  x = pseudo((int) (i + 202));
                ((float *) t->data)[i] = x;
            }
        }
        seed += 1000;

        ggml_tensor * st_view;
        if (uses_snap) {
            ggml_tensor * sidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            ((int32_t *) sidx->data)[0] = (int32_t) snap_plane;
            st_view = ggml_reshape_4d(ctx, ggml_get_rows(ctx, src, sidx), S, S, H, 1);
            ggml_tensor * o = ggml_gated_delta_net(ctx, q, k, v, g, b, st_view, (int) K);
            outs.push_back(ggml_view_1d(ctx, o, S * H * n, 0));
            const int64_t n_written = std::min(n, K);
            ggml_tensor * cs = ggml_cpy(ctx,
                ggml_view_3d(ctx, o, D, 1, n_written, D * sizeof(float), D * sizeof(float), S * H * n * sizeof(float)),
                ggml_view_3d(ctx, cache, D, 1, n_written, cache->nb[1], cache->nb[1], 0));
            ggml_build_forward_expand(gf, cs);
            src = ggml_reshape_2d(ctx, cs, D, n_written);
            snap_plane = r;
        } else {
            ggml_tensor * ctrl = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            ((int32_t *) ctrl->data)[0] = (int32_t) n_rec;
            ((int32_t *) ctrl->data)[1] = (int32_t) bank;
            ggml_tensor * wrow = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
            ((int64_t *) wrow->data)[0] = bank ^ 1;
            ggml_tensor * zero = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            ((int32_t *) zero->data)[0] = 0;
            ggml_tensor * bs = ggml_reshape_4d(ctx, ggml_get_rows(ctx, cache, zero), S, S, H, 1);
            ggml_tensor * o = ggml_gated_delta_net_replay(ctx, q, k, v, g, b, bs, rec, ctrl, (int) K);
            outs.push_back(ggml_view_1d(ctx, o, S * H * n, 0));
            const size_t base_off = S * H * n * sizeof(float);
            ggml_tensor * cb = ggml_cpy(ctx, ggml_view_2d(ctx, o, D, 1, D * sizeof(float), base_off),
                                             ggml_view_2d(ctx, cache, D, 1, cache->nb[1], 0));
            ggml_tensor * dstv = ggml_view_2d(ctx, cache, RK, 2, RK * sizeof(float), D * sizeof(float));
            ggml_tensor * srcv = ggml_view_2d(ctx, o, RK, 1, RK * sizeof(float), base_off + D * sizeof(float));
            ggml_tensor * sr = ggml_set_rows(ctx, dstv, srcv, wrow);
            ggml_build_forward_expand(gf, cb);
            ggml_build_forward_expand(gf, sr);
            rec = ggml_reshape_3d(ctx, sr, RK, 2, 1);
            n_rec = m - r;
            bank ^= 1;
        }
        (void) m;
    }

    // the replay base plane is the final state: gather it as its row 0
    if (!uses_snap) {
        ggml_tensor * zero = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *) zero->data)[0] = 0;
        ggml_tensor * out = ggml_get_rows(ctx, cache, zero);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        run_result res;
        for (ggml_tensor * t : outs) {
            const size_t off = res.attn.size();
            res.attn.resize(off + (size_t) ggml_nelements(t));
            memcpy(res.attn.data() + off, t->data, ggml_nbytes(t));
        }
        res.base.assign((float *) out->data, (float *) out->data + D);
        ggml_free(ctx);
        return res;
    }

    ggml_build_forward_expand(gf, src);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    run_result res;
    for (ggml_tensor * t : outs) {
        const size_t off = res.attn.size();
        res.attn.resize(off + (size_t) ggml_nelements(t));
        memcpy(res.attn.data() + off, t->data, ggml_nbytes(t));
    }
    // sync: read the plane the last rollback points at (already computed)
    res.base.assign((float *) src->data, (float *) src->data + D);
    ggml_free(ctx);
    return res;
}

void compare(const char * what) {
    run_result a = run_chain(true);
    run_result b = run_chain(false);
    CHECK(a.attn.size() == b.attn.size(), what);
    CHECK(a.base.size() == b.base.size(), what);
    if (a.attn.size() != b.attn.size() || a.base.size() != b.base.size()) return;

    const bool attn_ok = memcmp(a.attn.data(), b.attn.data(), a.attn.size() * sizeof(float)) == 0;
    const bool base_ok = memcmp(a.base.data(), b.base.data(), a.base.size() * sizeof(float)) == 0;
    CHECK(attn_ok, what);
    CHECK(base_ok, what);
    if (!attn_ok || !base_ok) {
        const std::vector<float> & x = attn_ok ? a.base : a.attn;
        const std::vector<float> & y = attn_ok ? b.base : b.attn;
        for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
            if (memcmp(&x[i], &y[i], sizeof(float)) != 0) {
                printf("  first differing element %zu: snapshot %a replay %a\n", i, (double) x[i], (double) y[i]);
                break;
            }
        }
    }
}

} // namespace

int main() {
    compare("rollback by replay equals the snapshot path");

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the replay chain's bytes equal the snapshot chain's\n");
    return 0;
}

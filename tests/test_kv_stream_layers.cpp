// Patch 0123zzzzz3's device-free cell: the KV stream's page moves are per attention
// layer. Several layers share two ring slots (the stream's slots[ikv % 2]); each
// attention call must read its OWN host-store pages, never the pages the previous
// layer put in the slot. Design: docs/campaigns/kv-streaming-design.md, increment
// I5.
//
// Three f16 attention layers over one 768-cell cache, a pool of ONE page (256
// cells) and a ring of two pages. Two ubatches write cells 0..255 and 256..511;
// the second one makes page 1 resident and evicts page 0, which then streams with
// the bytes the first ubatch wrote. Every layer reuses the same single slot. The
// three FLASH_ATTN_EXT nodes are one graph: each reads the pool through the page
// table, its streamed page through the slot, and its own host store and plan.
// Each output is compared byte for byte with a contiguous attention over the same
// layer's first 512 cells (rows an ubatch did not write are zero, as the stream's
// zero_host leaves them).
//
// Red case, compiled in (branch kv-stream-layers-red): the per-layer copy in
// ggml_compute_forward_flash_attn_ext_paged is skipped after its first call
// (`params->ith == 0` masked by a `first` flag), which is the old bug's shape --
// every layer reading whatever the last copy left in the slot. Layers 2 and 3
// then read layer 1's pages and their outputs differ; layer 1 still passes.
// One-line mutation: add `static bool first = true; if (!first) ...` around the
// copy loop, or guard the whole block with `if (false && plan)` to simulate a
// prepare-all-up-front copy.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-kv-stream.h"
#include "ggml-impl.h"   // GGML_FA_SRC_* and GGML_MAX_SRC: the sources must fit src[]

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, name, what)                                                        \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            printf("FAIL %s: %s\n", name, what);                                       \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

float pseudo(int64_t i) {
    return 0.5f*std::sin((double) i*0.017) + 0.25f*std::cos((double) i*0.031);
}

const int64_t hs = 256, nh = 4, n_layers = 3, n_kv = 768;
const int64_t n_q = 6;
const size_t row = (size_t) hs*nh*sizeof(ggml_fp16_t);

// the bytes of a layer's logical cache: cells 0..255 written by the ubatch, the
// rest never written, so zero (the host store's zero_host)
std::vector<char> layer_bytes(int64_t l, bool is_v) {
    std::vector<char> res((size_t) n_kv*row, 0);
    std::vector<ggml_fp16_t> r((size_t) hs*nh);
    for (int64_t t = 0; t < 256; ++t) {
        for (int64_t i = 0; i < hs*nh; ++i)
            r[(size_t) i] = ggml_fp32_to_fp16(pseudo(i + t*131 + l*10007 + (is_v ? 7 : 0)));
        memcpy(res.data() + (size_t) t*row, r.data(), row);
    }
    return res;
}

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> k, v;   // [hs*nh, 256] pool tensors, one page

    fixture() {
        ggml_init_params ip = { /*.mem_size =*/ (size_t) 8*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ctx.reset(ggml_init(ip));
        for (int64_t l = 0; l < n_layers; ++l) {
            k.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, hs*nh, 256));
            v.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, hs*nh, 256));
            ggml_set_name(k[(size_t) l], "k_pool");
            ggml_set_name(v[(size_t) l], "v_pool");
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    }

    std::vector<llama_kv_stream::layer> layers() const {
        std::vector<llama_kv_stream::layer> res;
        for (int64_t l = 0; l < n_layers; ++l) res.push_back({ k[(size_t) l], v[(size_t) l] });
        return res;
    }

    ggml_tensor * view(ggml_context * ctx, ggml_tensor * t, int64_t rows, const char * name) {
        ggml_tensor * base = t->view_src ? t->view_src : t;
        ggml_tensor * r = ggml_view_4d(ctx, base, hs, rows, nh, 1,
                base->nb[1], ggml_row_size(t->type, hs), ggml_nbytes(base), t->view_offs);
        ggml_set_name(r, name);
        return r;
    }
};

} // namespace

int main() {
    fixture f;
    llama_kv_stream s(f.layers(), /*n_slots =*/ 1, /*n_ring =*/ 2, n_kv, ggml_backend_cpu_buffer_type());

    CHECK(s.n_pool() == 1 && s.n_context_pages() == 3, "layers construct", "pool or page count wrong");

    // two ubatches: prepare, then write each one's cells (the graph's cache store)
    std::vector<std::vector<char>> kb(n_layers), vb(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        kb[(size_t) l] = layer_bytes(l, false);
        vb[(size_t) l] = layer_bytes(l, true);
    }
    for (uint32_t ub = 0; ub < 2; ++ub) {
        const uint32_t c0 = ub*256, c1 = c0 + 256;
        std::vector<uint32_t> cells;
        for (uint32_t c = c0; c < c1; ++c) cells.push_back(c);
        s.prepare(cells, c1);
        for (int64_t l = 0; l < n_layers; ++l) {
            s.set_rows((int32_t) l, false, c0, c1, kb[(size_t) l].data() + (size_t) c0*row);
            s.set_rows((int32_t) l, true,  c0, c1, vb[(size_t) l].data() + (size_t) c0*row);
        }
    }

    // the plan is the stream's own tensor now: page 0 streamed (the ring's row 0), page 1
    // resident (the pool)
    {
        const int32_t * pl = (const int32_t *) s.plan()->data;
        CHECK(pl[0] >= 0 && pl[1] < 0, "layers plan", "the plan does not hold one streamed page and one pool page");
        CHECK(!s.resident(0) && s.resident(1), "layers plan", "the residency disagrees with the plan");
        CHECK(s.n_streamed_pages() == 1, "layers plan", "the ubatch streams the wrong number of pages");
    }

    // the page table input the nodes share
    ggml_init_params ip = { /*.mem_size =*/ (size_t) 4*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context_ptr tctx(ggml_init(ip));
    const int64_t n_att = 2*256;   // the ubatches' cells
    ggml_tensor * pages = ggml_new_tensor_2d(tctx.get(), GGML_TYPE_I32, n_att/llama_kv_stream::page_size, n_layers);
    ggml_backend_buffer_ptr tbuf(ggml_backend_alloc_ctx_tensors_from_buft(tctx.get(), ggml_backend_cpu_buffer_type()));
    s.set_input_pages(pages);

    // the graph: one q and mask, three attention layers, one ring slot shared by all
    ggml_init_params gp = { /*.mem_size =*/ (size_t) 128*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context_ptr ctx(ggml_init(gp));

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hs, n_q, nh*n_q, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[(size_t) i] = pseudo(i);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_att, n_q, 1, 1);
    for (int64_t i = 0; i < n_q; ++i)
        for (int64_t j = 0; j < n_att; ++j)
            ((ggml_fp16_t *) mask->data)[(size_t) (i*n_att + j)] =
                j <= n_att - n_q + i ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
    const float scale = 1.0f/std::sqrt((float) hs);

    std::vector<ggml_tensor *> out(n_layers), ref(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        ggml_tensor * kp = f.view(ctx.get(), f.k[(size_t) l], 256, "op_k");
        ggml_tensor * vp = f.view(ctx.get(), f.v[(size_t) l], 256, "op_v");
        ggml_tensor * rk = f.view(ctx.get(), s.slot_k(0), s.slot_k(0)->ne[1], "op_ring_k");
        ggml_tensor * rv = f.view(ctx.get(), s.slot_v(0), s.slot_v(0)->ne[1], "op_ring_v");
        out[(size_t) l] = ggml_flash_attn_ext(ctx.get(), q, kp, vp, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_pages(out[(size_t) l], pages, (int32_t) l, (int32_t) llama_kv_stream::page_size, rk, rv);
        ggml_flash_attn_ext_set_host(out[(size_t) l], s.host_k((int32_t) l), s.host_v((int32_t) l), s.plan());
        // 0126 (B60 window 266): a source index past GGML_MAX_SRC wrote into the
        // tensor's following fields and the node read garbage as its plan. The
        // node's sources are readable at their indices and the last one is the plan
        CHECK(GGML_FA_SRC_PLAN < GGML_MAX_SRC, "layers sources", "the plan's index fits src[]");
        CHECK(out[(size_t) l]->src[GGML_FA_SRC_HOST_K]  == s.host_k((int32_t) l) &&
              out[(size_t) l]->src[GGML_FA_SRC_HOST_V]  == s.host_v((int32_t) l) &&
              out[(size_t) l]->src[GGML_FA_SRC_PLAN]    == s.plan() &&
              out[(size_t) l]->src[GGML_FA_SRC_TAG]     == nullptr,
              "layers sources", "the host store and the plan landed inside src[]");
        ggml_set_name(out[(size_t) l], "op_out");

        // the reference: this layer's contiguous cache cut at n_kv
        ggml_tensor * kc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, hs*nh, n_att);
        ggml_tensor * vc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, hs*nh, n_att);
        memcpy(kc->data, kb[(size_t) l].data(), (size_t) n_att*row);
        memcpy(vc->data, vb[(size_t) l].data(), (size_t) n_att*row);
        ggml_tensor * rkc = f.view(ctx.get(), kc, n_att, "ref_k");
        ggml_tensor * rvc = f.view(ctx.get(), vc, n_att, "ref_v");
        ref[(size_t) l] = ggml_flash_attn_ext(ctx.get(), q, rkc, rvc, mask, scale, 0.0f, 0.0f);
        ggml_set_name(ref[(size_t) l], "ref_out");
    }

    {
        ggml_cgraph * gf = ggml_new_graph(ctx.get());
        for (int64_t l = 0; l < n_layers; ++l) ggml_build_forward_expand(gf, out[(size_t) l]);
        ggml_graph_compute_with_ctx(ctx.get(), gf, /*n_threads =*/ 1);
    }
    {
        ggml_cgraph * gf = ggml_new_graph(ctx.get());
        for (int64_t l = 0; l < n_layers; ++l) ggml_build_forward_expand(gf, ref[(size_t) l]);
        ggml_graph_compute_with_ctx(ctx.get(), gf, /*n_threads =*/ 1);
    }

    for (int64_t l = 0; l < n_layers; ++l) {
        char name[64];
        snprintf(name, sizeof(name), "layers attention %lld", (long long) l);
        CHECK(ggml_nbytes(out[(size_t) l]) == ggml_nbytes(ref[(size_t) l]), name, "output shapes differ");
        CHECK(memcmp(out[(size_t) l]->data, ref[(size_t) l]->data, ggml_nbytes(ref[(size_t) l])) == 0,
              name, "a layer reads pages the stream did not copy for it");
    }

    // after the graph the single slot holds the LAST layer's streamed pages
    {
        const int32_t * pl = (const int32_t *) s.plan()->data;
        const char * rk = (const char *) s.slot_k(0)->data;
        const char * hk = (const char *) s.host_k(n_layers - 1)->data;
        for (int64_t p = 0; p < n_att/llama_kv_stream::page_size; ++p) {
            if (pl[p] < 0) continue;
            const size_t off = (size_t) pl[p]*256*row;
            CHECK(memcmp(rk + off, hk + (size_t) p*256*row, 256*row) == 0,
                  "layers slot", "the slot does not end with the last layer's pages");
        }
    }

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: every attention layer reads its own streamed pages\n");
    return 0;
}

// 0129b: the KV stream beyond its pool reads right -- at least TWO attention
// layers over a context whose pages STREAM (the pool smaller than the context),
// at the agent's shapes, q8_0 K / q4_0 V and f16. The streamed run's attention
// output is compared byte for byte with an all-resident run over the same bytes.
// Window 273 (B60): at 261,000 tokens a served chain answered garbage where
// every depth that fits the pool answered right -- the depth where pages moved.
// This is the device-free arm of that finding: the CPU paged path moves each
// layer's pages into the shared ring before its attention (0123zzzzz3), and the
// page table is read through the same -1-ring convention as the card's kernels.
//
// Red case: skip the per-layer copy in ggml_compute_forward_flash_attn_ext_paged
// (one line: `if (false && plan)` around the copy block); the streamed run then
// reads whatever the previous layer/zero filled the ring with, and its outputs
// differ from the resident run.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-kv-stream.h"
#include "ggml-impl.h"

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
    return 0.4f*std::sin((double) i*0.013) + 0.3f*std::cos((double) i*0.029);
}

const int64_t hs = 256, nh = 4, n_layers = 2, n_q = 6;
const int64_t n_written = 512;              // cells the ubatches write (two pages)
const int64_t n_kv = 6*256;                 // six pages: two resident, four stream

// one layer's logical cache bytes for cells [0, n_written): valid GGUF blocks of
// the type (d small, codes pseudo), so the same bytes feed both runs
std::vector<char> layer_bytes(int64_t l, bool is_v, ggml_type t) {
    const size_t row = ggml_row_size(t, hs*nh);
    std::vector<char> res(row*(size_t) n_written, 0);
    const int blk = ggml_blck_size(t), ts = ggml_type_size(t);
    const size_t nb = (size_t) hs*nh/blk;
    for (int64_t cell = 0; cell < n_written; ++cell) {
        char * r = res.data() + (size_t) cell*row;
        for (size_t b = 0; b < nb; ++b) {
            char * p = r + b*ts;
            const float d = 0.02f + 0.01f*(float) ((b + cell + l + (is_v ? 3 : 0)) % 7);
            memcpy(p, &d, 2);   // a small f16 scale is the exact two bytes either way
            const int qb = ts - 2;
            for (int j = 0; j < qb; ++j) {
                p[2 + j] = (char) ((int) (pseudo((int64_t) cell*131 + (int64_t) b*29 + j + l*10007 + (is_v ? 71 : 0)) * 97) & 0xFF);
            }
        }
    }
    return res;
}

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> k, v;

    fixture(ggml_type tk, ggml_type tv, int64_t pool_rows) {
        ggml_init_params ip = { /*.mem_size =*/ (size_t) 8*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ctx.reset(ggml_init(ip));
        for (int64_t l = 0; l < n_layers; ++l) {
            k.push_back(ggml_new_tensor_2d(ctx.get(), tk, hs*nh, pool_rows));
            v.push_back(ggml_new_tensor_2d(ctx.get(), tv, hs*nh, pool_rows));
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    }

    std::vector<llama_kv_stream::layer> layers() const {
        std::vector<llama_kv_stream::layer> res;
        for (int64_t l = 0; l < n_layers; ++l) res.push_back({ k[(size_t) l], v[(size_t) l] });
        return res;
    }

    ggml_tensor * view(ggml_context * gc, ggml_tensor * t, int64_t rows, const char * name) {
        ggml_tensor * base = t->view_src ? t->view_src : t;
        ggml_tensor * r = ggml_view_4d(gc, base, hs, rows, nh, 1,
                base->nb[1], ggml_row_size(t->type, hs), ggml_nbytes(base), t->view_offs);
        ggml_set_name(r, name);
        return r;
    }
};

// the two FA nodes over a stream's own pool, ring, host store, plan and table;
// the reference arm is the same stream with every page resident (the suite's
// all-resident run), so the comparison is byte-exact
std::vector<ggml_tensor *> build(ggml_context * gc, fixture & f, llama_kv_stream & s,
                                 std::vector<ggml_tensor *> & outs, ggml_tensor * q, ggml_tensor * mask,
                                 ggml_tensor * pages, const float scale) {
    for (int64_t l = 0; l < n_layers; ++l) {
        ggml_tensor * kp = f.view(gc, f.k[(size_t) l], s.n_pool()*llama_kv_stream::page_size, "op_k");
        ggml_tensor * vp = f.view(gc, f.v[(size_t) l], s.n_pool()*llama_kv_stream::page_size, "op_v");
        ggml_tensor * rk = f.view(gc, s.slot_k((int32_t) l), s.slot_k((int32_t) l)->ne[1], "op_ring_k");
        ggml_tensor * rv = f.view(gc, s.slot_v((int32_t) l), s.slot_v((int32_t) l)->ne[1], "op_ring_v");
        ggml_tensor * o = ggml_flash_attn_ext(gc, q, kp, vp, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_pages(o, pages, (int32_t) l, (int32_t) llama_kv_stream::page_size, rk, rv);
        ggml_flash_attn_ext_set_host(o, s.host_k((int32_t) l), s.host_v((int32_t) l), s.plan());
        outs.push_back(o);
    }
    return outs;
}

void arm(ggml_type tk, ggml_type tv) {
    const size_t row_k = ggml_row_size(tk, hs*nh), row_v = ggml_row_size(tv, hs*nh);

    // the pool: two pages streamed, six pages resident (the suite's all-resident run)
    fixture f1(tk, tv, 2*llama_kv_stream::page_size), f2(tk, tv, 6*llama_kv_stream::page_size);
    llama_kv_stream s1(f1.layers(), /*n_slots =*/ 2, /*n_ring =*/ 4, n_kv, ggml_backend_cpu_buffer_type());
    llama_kv_stream s2(f2.layers(), /*n_slots =*/ 6, /*n_ring =*/ 1, n_kv, ggml_backend_cpu_buffer_type());

    std::vector<std::vector<char>> kb(n_layers), vb(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        kb[(size_t) l] = layer_bytes(l, false, tk);
        vb[(size_t) l] = layer_bytes(l, true, tv);
    }
    std::vector<uint32_t> cells;
    for (int64_t c = 0; c < n_written; ++c) cells.push_back((uint32_t) c);
    for (llama_kv_stream * s : { &s1, &s2 }) {
        s->prepare(cells, n_kv);
        for (int64_t l = 0; l < n_layers; ++l) {
            s->set_rows((int32_t) l, false, 0, n_written, kb[(size_t) l].data());
            s->set_rows((int32_t) l, true,  0, n_written, vb[(size_t) l].data());
        }
    }

    {
        const int32_t * pl = (const int32_t *) s1.plan()->data;
        int streamed = 0;
        for (int64_t p = 0; p < n_kv/llama_kv_stream::page_size; ++p) if (pl[p] >= 0) streamed++;
        CHECK(streamed == 4 && s1.n_streamed_pages() == 4, "deep plan",
              "the streamed arm does not stream four pages (pool smaller than the context)");
        CHECK(s2.n_streamed_pages() == 0, "deep plan", "the all-resident arm streams");
    }

    ggml_init_params ip = { (size_t) 128*1024*1024, nullptr, false };   // q, mask and the nodes' data
    ggml_context_ptr gc(ggml_init(ip));
    ggml_tensor * q = ggml_new_tensor_4d(gc.get(), GGML_TYPE_F32, hs, n_q, nh*n_q, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[(size_t) i] = pseudo(i*7);
    ggml_tensor * mask = ggml_new_tensor_4d(gc.get(), GGML_TYPE_F16, n_kv, n_q, 1, 1);
    for (int64_t i = 0; i < n_q; ++i)
        for (int64_t j = 0; j < n_kv; ++j)
            ((ggml_fp16_t *) mask->data)[(size_t) (i*n_kv + j)] =
                j <= n_kv - n_q + i ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
    const float scale = 1.0f/std::sqrt((float) hs);

    ggml_init_params tp = { (size_t) 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr tc(ggml_init(tp));
    ggml_tensor * pages1 = ggml_new_tensor_2d(tc.get(), GGML_TYPE_I32, n_kv/llama_kv_stream::page_size, n_layers);
    ggml_tensor * pages2 = ggml_new_tensor_2d(tc.get(), GGML_TYPE_I32, n_kv/llama_kv_stream::page_size, n_layers);
    ggml_backend_buffer_ptr tbuf2(ggml_backend_alloc_ctx_tensors_from_buft(tc.get(), ggml_backend_cpu_buffer_type()));
    s1.set_input_pages(pages1);
    s2.set_input_pages(pages2);

    std::vector<ggml_tensor *> o1, o2;
    build(gc.get(), f1, s1, o1, q, mask, pages1, scale);
    build(gc.get(), f2, s2, o2, q, mask, pages2, scale);

    {
        ggml_cgraph * gf = ggml_new_graph(gc.get());
        for (ggml_tensor * t : o1) ggml_build_forward_expand(gf, t);
        for (ggml_tensor * t : o2) ggml_build_forward_expand(gf, t);
        ggml_graph_compute_with_ctx(gc.get(), gf, /*n_threads =*/ 4);
    }

    for (int64_t l = 0; l < n_layers; ++l) {
        char name[96];
        snprintf(name, sizeof(name), "deep attention %lld (%zu/%zu)", (long long) l, (size_t) row_k, (size_t) row_v);
        CHECK(memcmp(o1[(size_t) l]->data, o2[(size_t) l]->data, ggml_nbytes(o2[(size_t) l])) == 0,
              name, "a streamed page's attention differs from the all-resident run");
    }
}

} // namespace

int main() {
    arm(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);   // the agent's cache pair
    arm(GGML_TYPE_F16, GGML_TYPE_F16);

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: streamed pages beyond the pool match the all-resident run\n");
    return 0;
}

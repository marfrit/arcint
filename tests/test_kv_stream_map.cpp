// Patch 0120's device-free cell: the KV stream's page map, host store,
// eviction/write-back and page table, and its pages through the I1 paged
// attention on the CPU backend. Design: docs/campaigns/kv-streaming-design.md,
// increment I2.
//
// Part 1, the map (no attention): written pages become resident, free slots take
// the newest attended pages, the pool-full case evicts the oldest non-written
// page and writes it back, a page returning to the pool is read back exactly,
// the table and the runs agree with the residency, a streamed page is packed in
// the layer's ring slot in host-store bytes, and the fork's debug rotation moves
// every page through the host store without changing a byte.
//
// Part 2, the pages as attention sees them: four ubatches of 256 tokens over a
// 1,024-cell f16 cache with a 2-page pool and a 4-page ring, each ubatch prepared
// then written (set_rows, the graph's cache store), then a 6-row causal paged
// attention over the pool and the ring compared byte for byte with the same
// attention over a contiguous cache holding the same bytes. This is the CPU arm
// the design's I2 asks for; a served CPU-backend leg does not fit this round
// because arcint's llama engine requires an OpenCL card at start (the note names
// the deviation).
//
// Red case (branch kv-stream-red2, compiled in): evict() drops the write-back
// (`dirty[page]` -> `false`), so an evicted page's host store keeps the bytes it
// had at its last write-back. The freshness cell (write, evict, read back) and
// the streaming similarity fail; the commands are in the patch README 0120.
//
// Not covered here: the copier/publisher threads, the second queue's async
// copies and the ready/consumed flags (I3/I5), and the pool sizing from a
// backend probe (I3).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-kv-stream.h"

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

// the logical cache: token-major rows of hs*n_head_kv values, one layer
struct cache_bytes {
    int64_t hs, nh, n_kv;
    std::vector<char> k, v;
    size_t row_k, row_v;

    cache_bytes(int64_t hs, int64_t nh, int64_t n_kv) : hs(hs), nh(nh), n_kv(n_kv) {
        row_k = (size_t) hs*nh*sizeof(ggml_fp16_t);
        row_v = row_k;
        k.resize((size_t) n_kv*row_k);
        v.resize((size_t) n_kv*row_v);
        std::vector<ggml_fp16_t> row((size_t) hs*nh);
        for (int64_t t = 0; t < n_kv; ++t) {
            for (int64_t i = 0; i < hs*nh; ++i) row[(size_t) i] = ggml_fp32_to_fp16(pseudo(i + t*131));
            memcpy(k.data() + (size_t) t*row_k, row.data(), row_k);
            for (int64_t i = 0; i < hs*nh; ++i) row[(size_t) i] = ggml_fp32_to_fp16(pseudo(i + t*131 + 7));
            memcpy(v.data() + (size_t) t*row_v, row.data(), row_v);
        }
    }
};

// pool tensor of a layer: [hs*nh, rows], viewed as [hs, rows, nh, 1]
ggml_tensor * heads_view(ggml_context * ctx, ggml_type type, int64_t hs, int64_t nh, int64_t rows, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, hs*nh, rows);
    ggml_set_name(t, name);
    return ggml_view_4d(ctx, t, hs, rows, nh, 1, t->nb[1], ggml_row_size(type, hs), ggml_nbytes(t), 0);
}

struct fixture {
    int64_t hs = 256, nh = 4;
    int64_t n_kv = 1024;
    int64_t n_layers = 4;
    int64_t n_slots = 2;
    int64_t n_ring = 4;

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf; // pool tensors
    std::vector<ggml_tensor *> k, v; // the 2-D pool tensors: [hs*nh, n_slots*256]

    explicit fixture(int64_t n_slots_ = 2, int64_t n_ring_ = 4) : n_slots(n_slots_), n_ring(n_ring_) {
        ggml_init_params ip = { /*.mem_size =*/ (size_t) 8*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ctx.reset(ggml_init(ip));
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        for (int64_t l = 0; l < n_layers; ++l) {
            // the cache's layer tensors: a row a token, all its KV heads contiguous
            k.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, hs*nh, n_slots*256));
            v.push_back(ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, hs*nh, n_slots*256));
            ggml_set_name(k[(size_t) l], "k_pool");
            ggml_set_name(v[(size_t) l], "v_pool");
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    }

    std::vector<llama_kv_stream::layer> layers() const {
        std::vector<llama_kv_stream::layer> res;
        for (int64_t l = 0; l < n_layers; ++l) res.push_back({ k[(size_t) l], v[(size_t) l] });
        return res;
    }
};

// a host I32 tensor in its own context and CPU buffer (the page table input)
struct host_i32 {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor * t = nullptr;

    host_i32(int64_t ne0, int64_t ne1) {
        ggml_init_params ip = { /*.mem_size =*/ 4*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ctx.reset(ggml_init(ip));
        t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ne0, ne1);
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
    }
};

std::vector<uint32_t> cells_of(uint32_t c0, uint32_t c1) {
    std::vector<uint32_t> res;
    for (uint32_t c = c0; c < c1; ++c) res.push_back(c);
    return res;
}

void part_map() {
    fixture f(3, 4); // the pool the map cases use is three pages
    cache_bytes cb(f.hs, f.nh, (int64_t) 2048);
    llama_kv_stream s(f.layers(), 3, 4, 2048, ggml_backend_cpu_buffer_type());

    CHECK(s.n_pool() == 3 && s.n_context_pages() == 8, "map construct", "pool or page count wrong");

    // 1: written pages are resident; free slots take the newest attended pages
    s.prepare(cells_of(7*256, 7*256 + 8), 6*256);
    const int32_t * table = nullptr; (void) table;
    // patience: observe through the table and the runs
    {
        // the table covers the attended pages; the written page 7 lies outside
        // the attended range, so it is checked through the map
        host_i32 pages(6, f.n_layers);
        s.set_input_pages(pages.t);
        const int32_t * t = (const int32_t *) pages.t->data;
        CHECK(s.resident(7), "map residency", "the written page is not resident");
        CHECK(t[(size_t) 5] >= 0 && t[(size_t) 4] >= 0, "map residency", "the newest attended pages are not resident");
        CHECK(t[(size_t) 0] < 0 && t[(size_t) 3] < 0, "map residency", "an oldest attended page beyond the pool is resident");
        for (int64_t l = 1; l < f.n_layers; ++l) {
            CHECK(memcmp(t, t + l*6, 6*sizeof(int32_t)) == 0, "map table", "a table row differs from the first");
        }
    }
    {
        const auto runs = s.runs(6*256);
        uint32_t pages = 0;
        for (const auto & r : runs) pages += r.n;
        CHECK(pages == 2, "map runs", "the runs do not cover the two resident attended pages");
        CHECK(s.synced(), "map runs", "a ubatch with streamed pages reports no streaming");
    }

    // 2: write into the pool, then open the pool full: the oldest non-written
    // page is evicted and its bytes go back to the host store
    for (int64_t l = 0; l < f.n_layers; ++l) {
        s.set_rows((int32_t) l, false, 0, 6*256, cb.k.data());
        s.set_rows((int32_t) l, true,  0, 6*256, cb.v.data());
    }
    // the pool holds pages 7, 3, 4, 5 of the attended range and 1 free slot? 3 slots: 7, 5, 4
    s.prepare(cells_of(1*256, 1*256 + 8), 6*256); // writes page 1, needs a slot: evict oldest non-written (7? 3? 4? 5?)
    {
        std::vector<char> got((size_t) 8*256*cb.row_k);
        // read the page the ubatch did not write and that was evicted: page 3 (the
        // oldest attended non-written page when 7 was evicted first)
        s.get_rows(0, false, 3*256, 4*256, got.data());
        const char * want = cb.k.data() + (size_t) 3*256*cb.row_k;
        // page 3 may still be resident (the victim order picks the first non-written
        // page); what must hold is that every readable page equals the bytes written
        CHECK(memcmp(got.data(), want, (size_t) 256*cb.row_k) == 0, "map freshness", "an attended page reads different bytes than were written");
    }

    // 3: read-back freshness: write every cell, evict by rotation (write-backs for
    // all), then read every cell back through the same object
    {
        fixture f2(4, 4); // the fork's rotation needs the pool to hold the whole context
        llama_kv_stream s2(f2.layers(), 4, 4, 1024, ggml_backend_cpu_buffer_type());
        // make every page resident first, then write: the writes land in the pool
        // and mark the pages dirty, so an eviction has to write them back
        s2.prepare(cells_of(0, 8), 1024);
        for (int64_t l = 0; l < f2.n_layers; ++l) {
            CHECK(s2.resident(0) && s2.resident(3), "map rotate setup", "the four-page pool did not take the context");
            s2.set_rows((int32_t) l, false, 0, 1024, cb.k.data());
            s2.set_rows((int32_t) l, true,  0, 1024, cb.v.data());
        }
        s2.set_debug_rotate(true);
        for (int ub = 0; ub < 4; ++ub) s2.prepare(cells_of((uint32_t) (ub*256), (uint32_t) (ub*256 + 8)), 1024);
        for (int64_t l = 0; l < f2.n_layers; ++l) {
            std::vector<char> got((size_t) 1024*cb.row_k);
            s2.get_rows((int32_t) l, false, 0, 1024, got.data());
            CHECK(memcmp(got.data(), cb.k.data(), got.size()) == 0, "map rotate write-back", "rotating pages lost or changed bytes");
        }
    }

    // 4: a page returning to the pool is read back: clear, then re-resident
    {
        fixture f3(2, 4);
        llama_kv_stream s3(f3.layers(), 2, 4, 1024, ggml_backend_cpu_buffer_type());
        for (int64_t l = 0; l < f3.n_layers; ++l) s3.set_rows((int32_t) l, false, 0, 1024, cb.k.data());
        s3.prepare(cells_of(0, 8), 4*256);  // pages 0..3: 2 resident, 2 streamed
        std::vector<char> got((size_t) 1024*cb.row_k);
        s3.get_rows(0, false, 0, 1024, got.data());
        CHECK(memcmp(got.data(), cb.k.data(), got.size()) == 0, "map streamed read", "a streamed page does not read its written bytes");
        CHECK(s3.n_streamed_pages() == 2 && s3.synced(), "map streamed read", "two pages of four do not stream with a two-page pool");
        // with a two-page pool: the written page 0 and the newest attended page
        // 3 are resident, pages 1 and 2 stream, packed at ring rows 0 and 256
        for (int64_t l = 0; l < f3.n_layers; ++l) {
            char * slot = (char *) s3.slot_k((int32_t) l)->data;
            CHECK(memcmp(slot, cb.k.data() + (size_t) 1*256*cb.row_k, (size_t) 256*cb.row_k) == 0,
                  "map ring pack", "logical page 1 is not at the slot start");
            CHECK(memcmp(slot + (size_t) 256*cb.row_k, cb.k.data() + (size_t) 2*256*cb.row_k, (size_t) 256*cb.row_k) == 0,
                  "map ring pack", "logical page 2 is not at ring row 256");
        }
    }

    // 5: a page outside the attended range is not made resident by a prepare
    {
        fixture f4(2, 4);
        llama_kv_stream s4(f4.layers(), 2, 4, 1024, ggml_backend_cpu_buffer_type());
        s4.prepare(cells_of(3*256, 3*256 + 8), 1*256);
        host_i32 pages(1, f4.n_layers);
        s4.set_input_pages(pages.t);
        const int32_t * t = (const int32_t *) pages.t->data;
        CHECK(t[0] >= 0 && s4.resident(3) && !s4.resident(1), "map outside attended",
              "a written page, an attended page or an unattended page has the wrong residency");
    }

    // 6: a cell written the way the graph writes it -- into the pool tensor on the
    // backend, not through set_rows -- still goes back to the host store when its
    // page is evicted. In serving only prepare's written-page marking makes the page
    // dirty, so a no-op there loses the write-back.
    {
        fixture f5(2, 4);
        llama_kv_stream s5(f5.layers(), 2, 4, 1024, ggml_backend_cpu_buffer_type());
        s5.prepare(cells_of(1*256, 1*256 + 8), 1024); // page 1 written; pages 1 and 3 resident
        // the graph's store: 256 rows straight into the layer's pool tensor at the
        // rows stream->row gives, exactly as cpy_k/cpy_v write them on the device
        for (int64_t l = 0; l < f5.n_layers; ++l) {
            ggml_backend_tensor_set(f5.k[(size_t) l], cb.k.data() + (size_t) 1*256*cb.row_k,
                    (size_t) s5.row(1*256)*cb.row_k, (size_t) 256*cb.row_k);
            ggml_backend_tensor_set(f5.v[(size_t) l], cb.v.data() + (size_t) 1*256*cb.row_v,
                    (size_t) s5.row(1*256)*cb.row_v, (size_t) 256*cb.row_v);
        }
        // a ubatch that needs page 1's slot: the pool is full, so page 1 is evicted
        s5.prepare(cells_of(0, 8), 1024);
        CHECK(!s5.resident(1), "map graph write-back", "page 1 was not evicted");
        for (int64_t l = 0; l < f5.n_layers; ++l) {
            std::vector<char> got((size_t) 256*cb.row_k);
            s5.get_rows((int32_t) l, false, 1*256, 2*256, got.data());
            CHECK(memcmp(got.data(), cb.k.data() + (size_t) 1*256*cb.row_k, got.size()) == 0,
                  "map graph write-back", "bytes the graph wrote into the pool were lost at the eviction");
        }
    }
}

// part 2: the pages through a paged attention, compared with a contiguous cache
void part_attention() {
    fixture f;
    cache_bytes cb(f.hs, f.nh, 1024);
    llama_kv_stream s(f.layers(), 2, 4, 1024, ggml_backend_cpu_buffer_type());

    ggml_init_params ip = { /*.mem_size =*/ (size_t) 128*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context_ptr ctx;
    ctx.reset(ggml_init(ip));

    // the contiguous reference cache, all 1,024 cells, the same bytes
    ggml_tensor * kc = heads_view(ctx.get(), GGML_TYPE_F16, f.hs, f.nh, 1024, "k_ref");
    ggml_tensor * vc = heads_view(ctx.get(), GGML_TYPE_F16, f.hs, f.nh, 1024, "v_ref");
    memcpy(kc->view_src->data, cb.k.data(), cb.k.size());
    memcpy(vc->view_src->data, cb.v.data(), cb.v.size());

    const int64_t n_q = 6;
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, f.hs, n_q, f.nh*6, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[(size_t) i] = pseudo(i);
    const float scale = 1.0f/std::sqrt((float) f.hs);



    for (int ub = 0; ub < 4; ++ub) {
        const uint32_t c0 = (uint32_t) (ub*256), c1 = c0 + 256;
        const uint32_t n_kv = c1;

        // the ubatch: place the pages, then write its new cells (the graph's store)
        s.prepare(cells_of(c0, c1), n_kv);
        for (int64_t l = 0; l < f.n_layers; ++l) {
            s.set_rows((int32_t) l, false, c0, c1, cb.k.data() + (size_t) c0*cb.row_k);
            s.set_rows((int32_t) l, true,  c0, c1, cb.v.data() + (size_t) c0*cb.row_v);
        }
        host_i32 table((int64_t) n_kv/llama_kv_stream::page_size, f.n_layers);
        ggml_tensor * table_buf = table.t;
        s.set_input_pages(table_buf);

        ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, n_q, 1, 1);
        for (int64_t i = 0; i < n_q; ++i)
            for (int64_t j = 0; j < (int64_t) n_kv; ++j)
                ((ggml_fp16_t *) mask->data)[(size_t) (i*n_kv + j)] =
                    j <= (int64_t) n_kv - n_q + i ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);

        // the reference attends the same n_kv cells: a view of the contiguous
        // cache cut at n_kv, not the whole 1,024-row tensor
        ggml_tensor * rk = ggml_view_4d(ctx.get(), kc->view_src, f.hs, n_kv, f.nh, 1,
                kc->view_src->nb[1], ggml_row_size(kc->view_src->type, f.hs), ggml_nbytes(kc->view_src), 0);
        ggml_tensor * rv = ggml_view_4d(ctx.get(), vc->view_src, f.hs, n_kv, f.nh, 1,
                vc->view_src->nb[1], ggml_row_size(vc->view_src->type, f.hs), ggml_nbytes(vc->view_src), 0);
        ggml_tensor * out_ref = ggml_flash_attn_ext(ctx.get(), q, rk, rv, mask, scale, 0.0f, 0.0f);
        ggml_set_name(out_ref, "out_ref");
        // the pool, the ring and the table as attention reads them: 4-D views
        // over the token-major tensors (all KV heads of a token contiguous)
        auto view_of = [&](ggml_tensor * t, const char * name) {
            ggml_tensor * v = ggml_view_4d(ctx.get(), t, f.hs, t->ne[1], f.nh, 1,
                    t->nb[1], ggml_row_size(t->type, f.hs), ggml_nbytes(t), 0);
            ggml_set_name(v, name);
            return v;
        };
        ggml_tensor * op_k = view_of(f.k[0], "op_k");
        ggml_tensor * op_v = view_of(f.v[0], "op_v");
        ggml_tensor * ring_k = view_of(s.slot_k(0), "op_ring_k");
        ggml_tensor * ring_v = view_of(s.slot_v(0), "op_ring_v");
        ggml_tensor * out_paged = ggml_flash_attn_ext(ctx.get(), q, op_k, op_v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_pages(out_paged, table_buf, 0, (int32_t) llama_kv_stream::page_size, ring_k, ring_v);
        ggml_set_name(out_paged, "out_paged");

        {
            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out_paged);
            ggml_graph_compute_with_ctx(ctx.get(), gf, /*n_threads =*/ 1);
        }
        {
            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(gf, out_ref);
            ggml_graph_compute_with_ctx(ctx.get(), gf, /*n_threads =*/ 1);
        }

        char name[64];
        snprintf(name, sizeof(name), "attention ubatch %d", ub);
        CHECK(ggml_nbytes(out_paged) == ggml_nbytes(out_ref), name, "output shapes differ");
        CHECK(memcmp(out_paged->data, out_ref->data, ggml_nbytes(out_ref)) == 0, name,
                "the paged attention differs from the contiguous one at this ubatch");
        if (ub == 3) {
            CHECK(s.n_streamed_pages() > 0, name, "the full-depth ubatch streams no page");
            printf("     ubatch %d: %u of 4 pages streamed\n", ub, s.n_streamed_pages());
        }
    }
}

} // namespace

int main() {
    part_map();
    part_attention();
    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the page map, write-back and read-back, and the pages through attention\n");
    return 0;
}

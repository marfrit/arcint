// Patch 0119's device-free cell: a paged FLASH_ATTN_EXT reads the same bytes the
// unpaged op reads, so its f32 output is bit-equal to the unpaged op's.
//
// The op carries a page table (I32 [n_pages, n_tables], one row used), a pool of
// page-size rows whose slots hold the logical pages in shuffled order with unused
// slots between them, and a ring that holds every ring_every-th logical page in
// reverse order behind an unused page. ggml_flash_attn_ext_set_pages attaches
// them; the CPU backend gathers the logical rows into contiguous K/V and runs
// the stock path on them.
//
// Every case builds two graphs from the same bytes: one unpaged over a
// contiguous K/V tensor, one paged over the pool and the ring. Then the outputs
// are compared byte for byte -- not an NMSE: the gather is a copy, so equality
// is expected, and the mutation of one table entry or of the ring order is what
// the comparison can fail on.
//
// Red case (branch kv-stream-red, compiled in): the CPU gather takes the pool
// for every entry (`e >= 0 ? pool : ring` -> `pool`), so a ring page reads
// pool rows that hold other pages or unused data. The cases with ring_every > 0
// fail; the commands are in the patch README section 0119.
//
// Not covered here: the OpenCL backend declines the op until its kernels learn
// the page table (patch 0119's ggml-opencl supports_op), and the cross-queue
// sync flags (ggml_flash_attn_ext_set_sync) are I2's, not this cell's.

#define GGML_USE_CPU 1

#include "ggml.h"
#include "ggml-cpu.h"

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

constexpr int64_t kPage = 256;

float pseudo(int64_t i) {
    return 0.5f*std::sin((double) i*0.017) + 0.25f*std::cos((double) i*0.031);
}

// one logical token's K (or V) row: hs * nh values, in the cache's token-major
// row layout (all heads of a token contiguous), stored in the tensor's type
void fill_row(ggml_type type, int64_t n, int64_t seed, std::vector<char> & out) {
    std::vector<float> src((size_t) n);
    for (int64_t i = 0; i < n; ++i) src[(size_t) i] = pseudo(i + seed);
    out.resize(ggml_row_size(type, n));
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) out.data(), n);
    } else {
        const size_t w = ggml_quantize_chunk(type, src.data(), out.data(), 0, 1, n, nullptr);
        GGML_ASSERT(w == out.size());
    }
}

struct kase {
    const char * name;
    int64_t      hs;   // head size
    int64_t      nh;   // KV heads
    int64_t      gqa;  // query heads a KV head
    int64_t      n_kv; // attended cells, a multiple of kPage
    int64_t      n_q;  // query rows (a verify round is 6)
    ggml_type    tk;
    ggml_type    tv;
    int64_t      ring_every; // every i-th logical page is in the ring; 0: none
};

struct buffers {
    std::vector<char> k_log; // n_kv token-major rows, the reference bytes
    std::vector<char> v_log;
};

void build_logical(const kase & c, buffers & b) {
    const int64_t row_k = ggml_row_size(c.tk, c.hs*c.nh);
    const int64_t row_v = ggml_row_size(c.tv, c.hs*c.nh);
    std::vector<char> rk, rv;
    for (int64_t t = 0; t < c.n_kv; ++t) {
        fill_row(c.tk, c.hs*c.nh, t*131, rk);
        fill_row(c.tv, c.hs*c.nh, t*131 + 7, rv);
        b.k_log.insert(b.k_log.end(), rk.begin(), rk.end());
        b.v_log.insert(b.v_log.end(), rv.begin(), rv.end());
    }
    GGML_ASSERT((int64_t) b.k_log.size() == c.n_kv*row_k);
    GGML_ASSERT((int64_t) b.v_log.size() == c.n_kv*row_v);
}

void copy_rows(char * dst, const char * src, int64_t row_bytes, int64_t n_rows) {
    memcpy(dst, src, (size_t) row_bytes*n_rows);
}

// a 4-D view of a 2-D token-major cache tensor: [hs, rows, nh, 1]
ggml_tensor * heads_view(ggml_context * ctx, ggml_type type, int64_t hs, int64_t nh, int64_t rows, const char * name) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, hs*nh, rows);
    ggml_set_name(t, name);
    return ggml_view_4d(ctx, t, hs, rows, nh, 1,
            t->nb[1], ggml_row_size(type, hs), ggml_nbytes(t), 0);
}

void run(const kase & c) {
    const int64_t n_pages = c.n_kv/kPage;
    const int64_t n_ring  = c.ring_every ? (n_pages + c.ring_every - 1)/c.ring_every : 0;
    const int64_t pool_rows = (n_pages + 3)*kPage;   // three unused slots between the pages
    const int64_t ring_rows = n_ring ? (n_ring + 1)*kPage : 0;

    buffers b;
    build_logical(c, b);

    // a page p goes to pool slot perm[p]; pages in the ring instead go to ring
    // index ring[p], in reverse logical order
    std::vector<int64_t> perm((size_t) n_pages);
    std::vector<int64_t> ring((size_t) n_pages, -1);
    {
        int64_t next = 0, r = 0;
        for (int64_t p = 0; p < n_pages; ++p) {
            if (c.ring_every && p % c.ring_every == 0) {
                ring[(size_t) p] = r++;
            } else {
                perm[(size_t) p] = next++;
            }
        }
        for (int64_t p = 0; p < n_pages; ++p) {
            if (ring[(size_t) p] >= 0) ring[(size_t) p] = n_ring - 1 - ring[(size_t) p];
        }
    }

    ggml_init_params ip = { /*.mem_size =*/ (size_t) 64*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hs, c.n_q, c.nh*c.gqa, 1);
    ggml_set_name(q, "q");

    // the reference: contiguous K/V, logical rows in order
    ggml_tensor * kc = heads_view(ctx, c.tk, c.hs, c.nh, c.n_kv, "k_ref");
    ggml_tensor * vc = heads_view(ctx, c.tv, c.hs, c.nh, c.n_kv, "v_ref");

    // the paged form: the pool, the ring, the table
    ggml_tensor * kp = heads_view(ctx, c.tk, c.hs, c.nh, pool_rows, "k_pool");
    ggml_tensor * vp = heads_view(ctx, c.tv, c.hs, c.nh, pool_rows, "v_pool");
    ggml_tensor * kr = n_ring ? heads_view(ctx, c.tk, c.hs, c.nh, ring_rows, "k_ring") : nullptr;
    ggml_tensor * vr = n_ring ? heads_view(ctx, c.tv, c.hs, c.nh, ring_rows, "v_ring") : nullptr;

    ggml_tensor * pages = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_pages, 2);
    ggml_set_name(pages, "pages");

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1);
    ggml_set_name(mask, "mask");

    // bytes: the pool and the ring carry the logical rows at their physical
    // rows, and everything else carries a constant so a wrong read is visible
    const int64_t row_k = ggml_row_size(c.tk, c.hs*c.nh);
    const int64_t row_v = ggml_row_size(c.tv, c.hs*c.nh);
    {
        const ggml_tensor * kc2 = kc->view_src; const ggml_tensor * vc2 = vc->view_src;
        copy_rows((char *) kc2->data, b.k_log.data(), row_k, c.n_kv);
        copy_rows((char *) vc2->data, b.v_log.data(), row_v, c.n_kv);
    }
    memset(kp->view_src->data, 0x5a, (size_t) ggml_nbytes(kp->view_src));
    memset(vp->view_src->data, 0x5a, (size_t) ggml_nbytes(vp->view_src));
    for (int64_t p = 0; p < n_pages; ++p) {
        const int64_t s = (int64_t) p*kPage;
        if (ring[(size_t) p] >= 0) {
            const int64_t d = ring[(size_t) p]*kPage;
            copy_rows((char *) kr->view_src->data + d*row_k, b.k_log.data() + s*row_k, row_k, kPage);
            copy_rows((char *) vr->view_src->data + d*row_v, b.v_log.data() + s*row_v, row_v, kPage);
        } else {
            const int64_t d = perm[(size_t) p]*kPage;
            copy_rows((char *) kp->view_src->data + d*row_k, b.k_log.data() + s*row_k, row_k, kPage);
            copy_rows((char *) vp->view_src->data + d*row_v, b.v_log.data() + s*row_v, row_v, kPage);
        }
    }
    if (n_ring) {
        memset((char *) kr->view_src->data + n_ring*kPage*row_k, 0x5a, (size_t) kPage*row_k);
        memset((char *) vr->view_src->data + n_ring*kPage*row_v, 0x5a, (size_t) kPage*row_v);
    }

    // row 0 of the table is garbage (would read the ring or unused pool slots),
    // row 1 is the map the graph uses
    int32_t * table = (int32_t *) pages->data;
    for (int64_t p = 0; p < n_pages; ++p) {
        table[(size_t) p] = (int32_t) (n_pages - 1 - p); // row 0: a wrong page of the pool
        table[(size_t) (n_pages + p)] = ring[(size_t) p] >= 0 ? (int32_t) (-1 - ring[(size_t) p]) : (int32_t) perm[(size_t) p];
    }

    // causal mask: key j is visible to query row i when j <= n_kv - n_q + i
    for (int64_t i = 0; i < c.n_q; ++i) {
        for (int64_t j = 0; j < c.n_kv; ++j) {
            ((ggml_fp16_t *) mask->data)[(size_t) (i*c.n_kv + j)] =
                j <= c.n_kv - c.n_q + i ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
        }
    }

    const float scale = 1.0f/std::sqrt((float) c.hs);

    ggml_tensor * out_ref = ggml_flash_attn_ext(ctx, q, kc, vc, mask, scale, 0.0f, 0.0f);
    ggml_set_name(out_ref, "out_ref");

    ggml_tensor * out_paged = ggml_flash_attn_ext(ctx, q, kp, vp, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_pages(out_paged, pages, /*table =*/ 1, (int32_t) kPage, kr, vr);
    ggml_set_name(out_paged, "out_paged");

    {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_ref);
        ggml_graph_compute_with_ctx(ctx, gf, /*n_threads =*/ 1);
    }
    {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_paged);
        ggml_graph_compute_with_ctx(ctx, gf, /*n_threads =*/ 1);
    }

    const size_t bytes = ggml_nbytes(out_ref);
    CHECK(ggml_nbytes(out_paged) == bytes, c.name, "output shapes differ");

    // a finite reference means the gather+stock path produced real numbers; the
    // comparison below is byte equality, so NaNs would compare equal and hide a
    // read of the 0x5a fill
    size_t n_bad = 0;
    for (size_t i = 0; i < bytes/sizeof(float); ++i) {
        if (!std::isfinite(((const float *) out_ref->data)[i])) ++n_bad;
    }
    CHECK(n_bad == 0, c.name, "the reference output is not finite");

    size_t first = bytes;
    for (size_t i = 0; i < bytes; ++i) {
        if (((const char *) out_ref->data)[i] != ((const char *) out_paged->data)[i]) { first = i; break; }
    }
    CHECK(first == bytes, c.name, "the paged output differs from the unpaged one");
    if (first != bytes) {
        printf("  first differing byte %zu of %zu\n", first, bytes);
    }

    ggml_free(ctx);
    if (first == bytes) printf("ok   %s\n", c.name);
}

} // namespace

int main() {
    // f16 K/V, the tiled path: 4 KV heads, 6 query heads a KV head, a 6-row verify
    run({ "f16 f16, 1024 keys, 6 rows, 4 KV heads x 6, ring every 2", 256, 4, 6, 1024, 6, GGML_TYPE_F16, GGML_TYPE_F16, 2 });
    // the agent's cache pair, the vec path: q8_0 K / q4_0 V
    run({ "q8_0 q4_0, 2048 keys, 6 rows, 4 KV heads x 6, ring every 3", 256, 4, 6, 2048, 6, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 3 });
    // no ring: every attended page is resident (the shallow-depth arm)
    run({ "q8_0 q4_0, 1024 keys, 2 rows, no ring", 256, 4, 6, 1024, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 0 });
    // one KV head, no GQA, the smallest geometry, ring every page
    run({ "f16 f16, 512 keys, 1 row, 1 KV head, ring every page", 64, 1, 1, 512, 1, GGML_TYPE_F16, GGML_TYPE_F16, 1 });

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the paged op's f32 bytes equal the unpaged op's\n");
    return 0;
}

// patch 0056: the expert cache's CPU miss rows (ggml-opencl-miss-cpu.cpp) against ggml-cpu's vec_dot, the
// weights in planes as the bank holds them, 1-6 tokens a pass; and the slow path's IQ4_NL rows out of a
// split bank. `red` reads the planes and the split as GGUF blocks: it must fail.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-opencl-miss-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <random>
#include <vector>
#include <chrono>
struct fd { int off, size; };
static int fields(ggml_type t, fd f[5]) {
    switch (t) {
        case GGML_TYPE_IQ2_XXS: f[0] = { 0, 2 }; f[1] = { 2, 64 }; return 2;
        case GGML_TYPE_IQ2_S:   f[0] = { 0, 2 }; f[1] = { 2, 32 }; f[2] = { 34, 32 }; f[3] = { 66, 8 }; f[4] = { 74, 8 }; return 5;
        case GGML_TYPE_Q2_0:    f[0] = { 0, 2 }; f[1] = { 2, 16 }; return 2;
        default: return 0;
    }
}
int main(int argc, char ** argv) {
    const bool red = argc > 1 && !strcmp(argv[1], "red");   // red case: the planes read as GGUF blocks
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    {
        // the slow path's IQ4_NL rows out of a split bank (two slices, as ggml_cl_usm_set_tensor lays a
        // USM tensor out: the scales, then the quants, each 128-aligned); `red` reads the split as blocks
        const int64_t ne0 = 640, ne1 = 2560, ns = 2, bpr = ne0 / QK4_NL, nblk = bpr * ne1 * ns;
        std::vector<float> wf(ne0 * ne1 * ns); for (auto & v : wf) v = nd(rng) * 0.02f;
        std::vector<uint8_t> gguf(nblk * sizeof(block_iq4_nl));
        ggml_quantize_chunk(GGML_TYPE_IQ4_NL, wf.data(), gguf.data(), 0, ne1 * ns, ne0, nullptr);
        const size_t qb = sizeof(block_iq4_nl) - sizeof(ggml_half);
        std::vector<uint8_t> usm(nblk * sizeof(block_iq4_nl) + 256);
        uint8_t * d = (uint8_t *) (((uintptr_t) usm.data() + 127) & ~(uintptr_t) 127);
        uint8_t * q = (uint8_t *) (((uintptr_t) (d + nblk * sizeof(ggml_half)) + 127) & ~(uintptr_t) 127);
        for (int64_t i = 0; i < nblk; ++i) { memcpy(d + i * 2, &gguf[i * sizeof(block_iq4_nl)], 2); memcpy(q + i * qb, &gguf[i * sizeof(block_iq4_nl) + 2], qb); }
        int bad = 0;
        std::vector<uint8_t> row(bpr * sizeof(block_iq4_nl));
        for (int64_t b = 0; b < ns; ++b) for (int64_t r : { (int64_t) 0, (int64_t) 1, ne1 / 2, ne1 - 1 }) {
            const int64_t i0 = (b * ne1 + r) * bpr;
            if (red) memcpy(row.data(), d + i0 * sizeof(block_iq4_nl), row.size());   // the defect: the split read as blocks
            else     ggml_cl_miss_cpu_join_iq4nl(d, q, i0, bpr, row.data());
            bad += memcmp(row.data(), &gguf[i0 * sizeof(block_iq4_nl)], row.size()) != 0;
        }
        printf("iq4_nl split rows: %d of %d differ %s\n", bad, (int) (4 * ns), bad ? "FAIL" : "ok");
        fails += bad != 0;
    }
    if (!ggml_cl_miss_cpu_ok()) { printf("rows skipped: no AVX2/FMA/F16C\n%s: %d failing\n", red ? "red" : "test", fails); return fails ? 1 : 0; }
    ggml_cpu_init();
    struct cs { ggml_type t; int64_t ne0, ne1; } cases[] = { { GGML_TYPE_IQ2_S, 2560, 640 }, { GGML_TYPE_IQ2_XXS, 2560, 640 }, { GGML_TYPE_Q2_0, 640, 2560 } };
    for (auto c : cases) {
        const size_t bs = ggml_type_size(c.t); const int64_t bpr = c.ne0 / ggml_blck_size(c.t), nblk = bpr * c.ne1;
        // real weights: quantize random floats (ggml's quantizer needs the i-quant tables)
        std::vector<float> wf(c.ne0 * c.ne1); for (auto & v : wf) v = nd(rng) * 0.02f;
        std::vector<uint8_t> gguf(nblk * bs);
        if (c.t == GGML_TYPE_IQ2_S || c.t == GGML_TYPE_IQ2_XXS) ggml_quantize_init(c.t);
        std::vector<float> im(c.ne0, 1.0f); ggml_quantize_chunk(c.t, wf.data(), gguf.data(), 0, c.ne1, c.ne0, ggml_quantize_requires_imatrix(c.t) ? im.data() : nullptr);
        std::vector<uint8_t> planes(nblk * bs);
        fd f[5]; const int nf = fields(c.t, f); size_t plane = 0;
        for (int q = 0; q < nf; ++q) { for (int64_t i = 0; i < nblk; ++i) memcpy(&planes[plane + i * f[q].size], &gguf[i * bs + f[q].off], f[q].size); plane += nblk * f[q].size; }
        const auto * tr = ggml_get_type_traits_cpu(c.t); const auto * tv = ggml_get_type_traits_cpu(tr->vec_dot_type);
        for (int nt = 1; nt <= 6; ++nt) {
            std::vector<std::vector<float>> x(nt, std::vector<float>(c.ne0));
            for (auto & v : x) for (auto & e : v) e = nd(rng);
            std::vector<std::vector<char>> aq(nt, std::vector<char>(ggml_cl_miss_cpu_act_size(c.t, c.ne0))), rq(nt, std::vector<char>(ggml_row_size(tr->vec_dot_type, c.ne0)));
            std::vector<std::vector<float>> out(nt, std::vector<float>(c.ne1)), ref(nt, std::vector<float>(c.ne1));
            const void * act[8]; float * op[8];
            for (int k = 0; k < nt; ++k) { ggml_cl_miss_cpu_quant(c.t, x[k].data(), c.ne0, aq[k].data()); act[k] = aq[k].data(); op[k] = out[k].data(); tv->from_float(x[k].data(), rq[k].data(), c.ne0); }
            const auto t0 = std::chrono::steady_clock::now();
            ggml_cl_miss_cpu_rows(c.t, planes.data(), !red, c.ne0, c.ne1, act, nt, op, 0, (int) c.ne1);
            // patch 0075: level 1's block-ahead hint cannot change a row's bits: the
            // same rows at level 0 and level 1 must be identical. (The red is the
            // ahead distance: ggml_cl_miss_cpu_prefetch_ahead() < 1 fails below.)
            if (!red && c.t == GGML_TYPE_IQ2_S && nt == 2) {
                std::vector<std::vector<float>> out_pf = out;
                ggml_cl_miss_cpu_set_prefetch(1);
                for (int k = 0; k < nt; ++k) op[k] = out_pf[k].data();
                ggml_cl_miss_cpu_rows(c.t, planes.data(), true, c.ne0, c.ne1, act, nt, op, 0, (int) c.ne1);
                ggml_cl_miss_cpu_set_prefetch(0);
                int differ = 0;
                for (int k = 0; k < nt; ++k) differ += memcmp(out[k].data(), out_pf[k].data(), (size_t) c.ne1 * sizeof(float)) != 0;
                printf("prefetch level 1 bits: %d of %d rows differ %s\n", differ, nt, differ ? "FAIL" : "ok");
                fails += differ != 0;
            }
            const double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6;
            double emax = 0, rmax = 0;
            for (int k = 0; k < nt; ++k) for (int64_t r = 0; r < c.ne1; ++r) {
                tr->vec_dot((int) c.ne0, &ref[k][r], 0, &gguf[r * bpr * bs], 0, rq[k].data(), 0, 1);
                emax = std::max(emax, (double) fabsf(out[k][r] - ref[k][r])); rmax = std::max(rmax, (double) fabsf(ref[k][r]));
            }
            const double rel = emax / rmax; const double tol = c.t == GGML_TYPE_Q2_0 ? 1e-3 : 1e-5;   // Q2_0: Strata's activation chunks, not ggml's q8_0
            const bool ok = rel < tol; fails += !ok;
            printf("%-8s nt=%d rel err %.2e (tol %.0e) %s  %.0f us\n", ggml_type_name(c.t), nt, rel, tol, ok ? "ok" : "FAIL", us);
        }
    }
    // patch 0075: the cold-page touch flows a slice's pages in. A 16-page
    // anonymous mapping, dropped, must be all absent before and all present
    // after one byte a page; a touch that strides 8192 leaves half absent (red).
    if (ggml_cl_miss_cpu_prefetch_ahead() < 1) {
        printf("prefetch ahead %d blocks FAIL\n", ggml_cl_miss_cpu_prefetch_ahead()); fails++;
    }
#if defined(__linux__)
    {
        const long pg = sysconf(_SC_PAGESIZE);
        const int np = 16;
        void * p = mmap(nullptr, (size_t) pg * np, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { printf("cold-page touch: mmap failed FAIL\n"); fails++; }
        else {
            unsigned char vec[16];
            madvise(p, (size_t) pg * np, MADV_DONTNEED);
            const int rb = mincore(p, (size_t) pg * np, vec);
            int before = 0; for (int i = 0; i < np; ++i) before += vec[i] & 1;
            ggml_cl_miss_cpu_touch(p, (size_t) pg * np);
            const int ra = mincore(p, (size_t) pg * np, vec);
            int after = 0; for (int i = 0; i < np; ++i) after += vec[i] & 1;
            const bool ok = rb == 0 && ra == 0 && before == 0 && after == np;
            printf("cold-page touch: %d of %d pages resident before, %d after %s\n", before, np, after, ok ? "ok" : "FAIL");
            fails += !ok;
            munmap(p, (size_t) pg * np);
        }
    }
#endif
    printf("%s: %d failing\n", red ? "red" : "test", fails);
    return fails ? 1 : 0;
}

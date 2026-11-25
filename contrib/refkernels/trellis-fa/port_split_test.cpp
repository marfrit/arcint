// port_split_test.cpp - the key split's rule (fa_port_gate.h: fa_port_splits, fa_port_split_wave,
// fa_port_split_blocks; patch 0223), on the model's graph export.
//
// Reads the same test-export-graph-ops files as port_gate_test.cpp and takes each FLASH_ATTN_EXT
// prompt line (220..223 with more than 8 rows): the global call (window 0) and the SWA ring (n_kv =
// rows + 1,024, window 1,024). Each line's own rows and kv heads give the call's work-groups
// (n_head_kv x ceil(rows / 8), qtw 1); its n_kv is widened to the depths 4,096 / 40,960 / 261,120 as
// the cache presents them (ceil((depth + rows) / 256) x 256 cells). The wave is
// fa_port_split_wave(n_cu, 256, G = n_head / n_head_kv): the B60 (160 compute units) and the B70 (256).
// Rows of 32 and 128 (a follow-up turn at depth) are the ub-512 line with its rows bent.
//
// Expected, from the design's table (the same rule computed independently, in Python):
//   global calls: the table below, row for row; every split walks at least 128 blocks;
//   SWA ring calls: p = 1 at every depth (a windowed call never splits);
//   the waves: B60 200, B70 320 at 256 GRF; 420 / 672 at 128 GRF (64 threads a core / 3).
// Red: built with -DFA_PORT_SPLIT_RED_NO_FLOOR the header drops the 128-block floor, the rule then
// splits the 4,096-deep global calls, and this test must FAIL there.
//
//   g++ -O2 -I <llama-tree>/ggml/include -I <llama-tree>/ggml/src/ggml-opencl port_split_test.cpp -o port_split_test
//   ./port_split_test fixtures/inkling-ub512.txt fixtures/inkling-ub1024.txt fixtures/inkling-ub2048.txt \
//                     fixtures/inkling-q8kv-ub512.txt fixtures/inkling-q8kv-ub1024.txt fixtures/inkling-q8kv-ub2048.txt
#include "fa_port_gate.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct src_t { int type; int64_t ne[4]; size_t nb[4]; };
struct line_t {
    int no = 0, op = 0, type = 0;
    int64_t ne[4] = {0, 0, 0, 0};
    std::vector<src_t> src;
};

static std::vector<line_t> read_lines(const char * path) {
    std::vector<line_t> out;
    std::ifstream f(path);
    std::string s;
    int no = 0;
    while (std::getline(f, s)) {
        ++no;
        std::istringstream is(s);
        line_t l; l.no = no;
        int64_t np = 0, ns = 0, dummy = 0;
        if (!(is >> l.op >> l.type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> np)) continue;
        for (int64_t i = 0; i < np; ++i) is >> dummy;
        if (!(is >> ns)) continue;
        for (int64_t i = 0; i < ns; ++i) {
            src_t x;
            is >> x.type >> x.ne[0] >> x.ne[1] >> x.ne[2] >> x.ne[3] >> x.nb[0] >> x.nb[1] >> x.nb[2] >> x.nb[3];
            l.src.push_back(x);
        }
        out.push_back(l);
    }
    return out;
}

enum { OP_FLASH_ATTN_EXT = 75 };

// the design's table: card, rows, p at the three depths
struct row_t { const char * card; int n_cu; int64_t rows; int p[3]; };
static const row_t TABLE[] = {
    { "B60", 160,   32, { 1, 6, 6 } }, { "B60", 160,  128, { 1, 3, 3 } }, { "B60", 160,  512, { 1, 3, 3 } },
    { "B60", 160, 1024, { 1, 3, 3 } }, { "B60", 160, 2048, { 1, 2, 2 } },
    { "B70", 256,   32, { 1, 10, 10 } }, { "B70", 256,  128, { 1, 5, 5 } }, { "B70", 256,  512, { 1, 3, 3 } },
    { "B70", 256, 1024, { 1, 3, 3 } }, { "B70", 256, 2048, { 1, 2, 2 } },
};
static const int64_t DEPTH[3] = { 4096, 40960, 261120 };

static int fails = 0;

static const row_t * expected(int n_cu, int64_t rows) {
    for (const row_t & r : TABLE) if (r.n_cu == n_cu && r.rows == rows) return &r;
    return nullptr;
}

// one call: the rule at the three depths against the table (global) or 1 (windowed)
static void check_call(const char * file, int no, int64_t rows, int64_t n_head, int64_t n_head_kv, int32_t window) {
    const int G = (int) (n_head / n_head_kv);
    for (int n_cu : { 160, 256 }) {
        const int wave = fa_port_split_wave(n_cu, 256, G);
        const int64_t nwg = n_head_kv * ((rows + 7) / 8);
        const row_t * e = expected(n_cu, rows);
        if (window == 0 && e == nullptr) {
            printf("  %s line %d rows %lld: FAIL, no table row\n", file, no, (long long) rows);
            ++fails; continue;
        }
        for (int d = 0; d < 3; ++d) {
            const int64_t n_kv = (DEPTH[d] + rows + 255) / 256 * 256;
            const int p = fa_port_splits(nwg, wave, n_kv, window, 0, 0);
            const int want = window != 0 ? 1 : e->p[d];
            const int bps = fa_port_split_blocks(n_kv, p);
            // a split walks at least 128 blocks (the floor), the splits cover every block, and none is past the call's
            const int64_t blocks = (n_kv + 31) / 32;
            const bool cover = (int64_t) bps * p >= blocks && (p == 1 || bps >= FA_PORT_SPLIT_MIN_BLOCKS);
            const bool good = p == want && cover;
            printf("  %s line %d %s wave %d rows %5lld nwg %5lld window %4d depth %6lld kv %6lld: p %2d (want %2d), %d blocks a split%s\n",
                   file, no, n_cu == 160 ? "B60" : "B70", wave, (long long) rows, (long long) nwg, window, (long long) DEPTH[d],
                   (long long) n_kv, p, want, bps, good ? "" : "   <-- FAIL");
            if (!good) ++fails;
        }
    }
}

int main(int argc, char ** argv) {
    // the waves, and the knobs' limits
    struct { int n_cu, grf, t, want; } W[] = { { 160, 256, 3, 200 }, { 256, 256, 3, 320 }, { 160, 128, 3, 420 }, { 256, 128, 3, 672 },
                                               { 160, 256, 6, 100 }, { 4, 256, 3, 0 } };
    for (auto & w : W) {
        const int got = fa_port_split_wave(w.n_cu, w.grf, w.t);
        printf("  wave n_cu %d grf %d threads %d: %d (want %d)%s\n", w.n_cu, w.grf, w.t, got, w.want, got == w.want ? "" : "   <-- FAIL");
        if (got != w.want) ++fails;
    }
    {
        // force sets p (windowed calls too), bounded by 16 and the call's blocks; cap bounds the rule; cap 1 is off
        struct { int64_t nwg, wave, n_kv; int32_t window; int cap, force, want; } K[] = {
            { 1024, 200, 41984, 1024, 0, 3, 3 }, { 1024, 200, 160, 0, 0, 3, 3 }, { 1024, 200, 64, 0, 0, 3, 2 },
            { 1024, 200, 41984, 0, 0, 40, 16 }, { 1024, 200, 41984, 0, 2, 0, 2 }, { 1024, 200, 41984, 0, 1, 0, 1 },
            { 1024, 0, 41984, 0, 0, 0, 1 }, { 1024, 200, 8160, 0, 0, 0, 1 }, { 1024, 200, 8192, 0, 0, 0, 2 } };
        for (auto & k : K) {
            const int got = fa_port_splits(k.nwg, k.wave, k.n_kv, k.window, k.cap, k.force);
            printf("  knobs nwg %lld wave %lld kv %lld window %d cap %d force %d: p %d (want %d)%s\n", (long long) k.nwg, (long long) k.wave,
                   (long long) k.n_kv, k.window, k.cap, k.force, got, k.want, got == k.want ? "" : "   <-- FAIL");
            if (got != k.want) ++fails;
        }
    }
    if (argc < 2) { printf("usage: port_split_test <export.txt> ...\n"); return 1; }
    for (int a = 1; a < argc; ++a) {
        const char * file = strrchr(argv[a], '/') ? strrchr(argv[a], '/') + 1 : argv[a];
        const std::vector<line_t> lines = read_lines(argv[a]);
        int n_global = 0, n_ring = 0;
        for (const line_t & l : lines) {
            if (l.op != OP_FLASH_ATTN_EXT || l.src.size() < 2) continue;
            const int64_t n_head = l.ne[1], rows = l.ne[2], n_kv = l.src[1].ne[1], n_head_kv = l.src[1].ne[2];
            if (rows <= 8) continue;                       // the decode step: not the port's
            const int32_t window = n_kv == rows + 1024 ? 1024 : 0;
            if (window) ++n_ring; else ++n_global;
            check_call(file, l.no, rows, n_head, n_head_kv, window);
            if (window == 0 && rows == 512) {
                // a follow-up turn at depth: the same call with 32 and 128 rows
                check_call(file, l.no, 32, n_head, n_head_kv, 0);
                check_call(file, l.no, 128, n_head, n_head_kv, 0);
            }
        }
        if (n_global != 1 || n_ring != 1) {
            printf("  %s: FAIL, %d global and %d ring prompt lines, expected 1 and 1\n", file, n_global, n_ring);
            ++fails;
        }
    }
    printf("%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

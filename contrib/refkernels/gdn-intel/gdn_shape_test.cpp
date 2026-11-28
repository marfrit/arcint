// gdn_shape_test.cpp - 0228: the Intel gated delta-net kernel's shape check (ggml-opencl-gdn-shape.h),
// on the CPU. The prompt kernel stages two token chunks in local float buf[2][CH_FLOATS] and reads
// both with 16-byte sub-group block reads; buf[1] starts 4 * CH_FLOATS bytes in, so a shape whose
// CH_FLOATS = tc * (2 * 128 + nsg * r_pp + 2) is not a multiple of 4 must be refused, else the reads
// on buf[1] are 8 bytes off a 16-byte boundary. The table runs the default shapes against the
// element, fit and local-memory rules, and the audit's odd TC.
//
// Build lines (the header is included by -I; the red build must exit 1):
//   g++ -std=c++17 -O2 -Wall -Wextra -I <llama-tree>/ggml/src/ggml-opencl gdn_shape_test.cpp -o t_green
//   ./t_green
//   g++ -std=c++17 -O2 -Wall -Wextra -DGGML_CL_GDN_SHAPE_RED_NO_ALIGN -I <llama-tree>/ggml/src/ggml-opencl gdn_shape_test.cpp -o t_red
//   ./t_red
#include "ggml-opencl-gdn-shape.h"

#include <cstdio>

struct shape_case {
    int r_tg, r_pp, nsg, tc, sg;
    size_t max_wg;
    uint64_t local_mem;
    bool want;
};

int main() {
    const shape_case cases[] = {
        {4, 4, 32, 8, 16, 1024, 131072, true},   // 1  the B60 default: CH_FLOATS 8 x 386 = 3,088
        {2, 2, 64, 16, 8, 1024, 131072, true},   // 2  the A770 default
        {4, 4, 32, 7, 16, 1024, 131072, false},  // 3  TC 7, ROWS 128: 2,702, % 4 = 2 (the audit's case)
        {4, 4, 32, 6, 16, 1024, 131072, true},   // 4  TC 6: 2,316 % 4 = 0
        {1, 1, 2, 7, 16, 1024, 131072, true},    // 5  ROWS 2: 7 x 260 = 1,820 % 4 = 0, odd TC aligned
        {1, 1, 1, 2, 16, 1024, 131072, false},   // 6  ROWS 1: 2 x 259 = 518 % 4 = 2
        {1, 1, 1, 4, 16, 1024, 131072, true},    // 7  ROWS 1, TC 4: 1,036 % 4 = 0
        {4, 4, 32, 0, 16, 1024, 131072, false},  // 8  tc 0
        {4, 4, 3, 8, 16, 1024, 131072, false},   // 9  128 % (4 x 3) != 0
        {4, 4, 32, 8, 16, 256, 131072, false},   // 10 nsg x sg = 512 > max_wg 256
        {4, 4, 32, 64, 16, 1024, 131072, false}, // 11 local memory: 2 x 64 x 386 x 4 = 197,632 > 131,072
        {32, 4, 32, 8, 16, 1024, 131072, false}, // 12 r_tg 32 > sg 16
        {3, 4, 32, 8, 16, 1024, 131072, false},  // 13 128 % 3 != 0
    };
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));
    int k = 0;
    for (int i = 0; i < n; ++i) {
        const shape_case & c = cases[i];
        const bool got = ggml_cl_gdn_intel_shape_ok(c.r_tg, c.r_pp, c.nsg, c.tc, c.sg, c.max_wg, c.local_mem);
        const bool ok = got == c.want;
        if (ok) ++k;
        std::printf("CASE %d r_tg=%d r_pp=%d nsg=%d tc=%d sg=%d -> got %d want %d %s\n",
                    i + 1, c.r_tg, c.r_pp, c.nsg, c.tc, c.sg, got ? 1 : 0, c.want ? 1 : 0, ok ? "OK" : "FAIL");
    }
    std::printf("GDN SHAPE: %d of %d cases right\n", k, n);
    return k == n ? 0 : 1;
}

// Patch 0212's device-free cell: the route an indexed-bias FLASH_ATTN_EXT decode takes
// (ggml/src/ggml-opencl/ggml-opencl-fa-bias-route.h, the function the dispatcher asks).
//
// 0143 compiles the bias variant of the GQA decode kernel at dispatch, on first use. Before
// 0212 a failed build there aborted the process ("an indexed-bias FLASH_ATTN_EXT reached the
// dispatcher without a kernel"); 1.8.0's review named it (S3). Now the call runs on the split
// kernel's bias variant, which supports_op has already built for the same shape. The card
// forces the failure with GGML_OPENCL_FA_BIAS_GQA_FAIL=1 (an #error in the kernel source);
// this cell drives the decision with the build's outcome as an input.
//
// Red: the same cell compiled with FA_BIAS_ROUTE_RED (the rule before 0212) is registered
// as fa-bias-route-red with WILL_FAIL: the failed-build cases read NONE, the abort.

#include "ggml-opencl-fa-bias-route.h"

#include <cstdio>

namespace {

int g_failures = 0;

#define CHECK(cond, what)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            printf("FAIL: %s\n", what);                                                \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

} // namespace

int main() {
    // the rows took the GQA kernel and its bias variant built: the variant is used
    CHECK(ggml_cl_fa_bias_route(1, 1, 1) == GGML_CL_FA_BIAS_ROUTE_GQA, "GQA rows, variant built: GQA");
    // the dispatcher does not build the split variant's lookup when the GQA one is taken:
    // split_ok is then 0 and must not matter
    CHECK(ggml_cl_fa_bias_route(1, 1, 0) == GGML_CL_FA_BIAS_ROUTE_GQA, "GQA rows, variant built, split unread: GQA");

    // S3: the GQA variant failed to build (or to create its kernel): the split variant
    CHECK(ggml_cl_fa_bias_route(1, 0, 1) == GGML_CL_FA_BIAS_ROUTE_SPLIT,
          "GQA rows, the GQA variant failed: the split variant, not the abort");

    // rows on the split kernel (1-3 rows, or a card without the GQA kernel): the split
    // variant whatever the GQA build would do (it is never asked for)
    CHECK(ggml_cl_fa_bias_route(0, 0, 1) == GGML_CL_FA_BIAS_ROUTE_SPLIT, "split rows: split");
    CHECK(ggml_cl_fa_bias_route(0, 1, 1) == GGML_CL_FA_BIAS_ROUTE_SPLIT, "split rows, a GQA kernel at hand: split");

    // no kernel at all (supports_op declines these shapes, reason no-bias-kernel): NONE,
    // the dispatcher's abort stays for a call that bypassed supports_op
    CHECK(ggml_cl_fa_bias_route(1, 0, 0) == GGML_CL_FA_BIAS_ROUTE_NONE, "GQA failed and no split: none");
    CHECK(ggml_cl_fa_bias_route(0, 0, 0) == GGML_CL_FA_BIAS_ROUTE_NONE, "no split: none");

    // the debug switch: on for a nonzero number only
    CHECK(!ggml_cl_fa_bias_gqa_fail(nullptr), "unset: off");
    CHECK(!ggml_cl_fa_bias_gqa_fail("0"), "0: off");
    CHECK(!ggml_cl_fa_bias_gqa_fail(""), "empty: off");
    CHECK(ggml_cl_fa_bias_gqa_fail("1"), "1: on");

    if (g_failures != 0) {
        printf("fa-bias-route: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("fa-bias-route: all cases pass\n");
    return 0;
}

// 0187's eligibility, host-side: the pure predicate against the real strides of the two copies
// the card's profile joined. Derived from the graph code, not guessed:
//  - src/models/inkling2.cpp:386 makes table = mul_mat(pad(attn_rel_bias), Rcur) = [1025, 24, T]
//    contiguous (nb 4, 4*1025, 4*1025*24); line 390's ggml_cont(ggml_permute(table, 1, 0, 2, 3))
//    leaves the source view [24, 1025, T] with nb {4*1025, 4, 4*1025*24, ...} and a contiguous
//    destination [24, 1025, T] (nb {4, 96, 98,400, ...}).
//  - mix() line 219 ggml_cont(ggml_transpose(states)) of the stack view [2304, k, T]
//    (nb1 2304*4, nb2 9,216*slots): the transpose view is [k, 2304, T] with nb
//    {9,216, 4, 9,216*slots, ...}, destination contiguous [k, 2304, T].
// The test also asserts the negatives: a contiguous source, a source whose slow step is not one
// fast row, and a destination that is not contiguous.
#include "ggml-opencl-cpy-t01.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static int failures = 0;

static void expect(const char * what, bool got, bool want) {
    if (got != want) {
        printf("FAIL %s: eligible=%d, want %d\n", what, (int) got, (int) want);
        ++failures;
    } else {
        printf("ok   %s: eligible=%d\n", what, (int) got);
    }
}

int main() {
    const int64_t T = 512, k = 4, nembd = 2304, heads = 24, bins = 1025, slots = 8;
    const size_t esz = sizeof(float);

    // (a) the bias table's copy: source a [1025, 24, T] product read as the (1,0)-permuted view
    const int64_t ne_s[4] = { heads, bins, T, 1 };
    const size_t nb_s[4] = { (size_t) bins * esz, esz, (size_t) bins * heads * esz,
                             (size_t) bins * heads * esz * (size_t) T };
    const int64_t ne_d[4] = { heads, bins, T, 1 };
    const size_t nb_d[4] = { esz, (size_t) heads * esz, (size_t) heads * bins * esz,
                             (size_t) heads * bins * esz * (size_t) T };
    expect("bias table copy (0,1)-swapped", ggml_cl_cpy_t01_eligible(0, 0, ne_s, nb_s, ne_d, nb_d), true);

    // the same bytes with a contiguous source: dim 0 fastest, so the pack kernel's case
    const size_t nb_c[4] = { esz, (size_t) heads * esz, (size_t) heads * bins * esz,
                             (size_t) heads * bins * esz * (size_t) T };
    expect("bias table copy, contiguous source", ggml_cl_cpy_t01_eligible(0, 0, ne_s, nb_c, ne_d, nb_d), false);

    // (b) the depth mix's copy: a [2304, k, T] stack view transposed to [k, 2304, T]
    const int64_t ne_s2[4] = { k, nembd, T, 1 };
    const size_t nb_s2[4] = { (size_t) nembd * esz, esz, (size_t) nembd * esz * (size_t) slots,
                              (size_t) nembd * esz * (size_t) slots * (size_t) T };
    const int64_t ne_d2[4] = { k, nembd, T, 1 };
    const size_t nb_d2[4] = { esz, (size_t) k * esz, (size_t) k * nembd * esz,
                              (size_t) k * nembd * esz * (size_t) T };
    expect("depth mix transpose", ggml_cl_cpy_t01_eligible(0, 0, ne_s2, nb_s2, ne_d2, nb_d2), true);
    expect("depth mix transpose, contiguous source", ggml_cl_cpy_t01_eligible(0, 0, ne_s2, nb_d2, ne_d2, nb_d2), false);

    // a source whose slow step is not one fast row: a strided-rows view, not a swap
    const size_t nb_s3[4] = { (size_t) heads * esz, esz, (size_t) heads * bins * esz,
                              (size_t) heads * bins * esz * (size_t) T };
    const int64_t ne3[4] = { heads, bins, T, 1 };
    expect("slow step not a fast row", ggml_cl_cpy_t01_eligible(0, 0, ne3, nb_s3, ne_d, nb_d), false);

    // a non-contiguous destination
    const size_t nb_d4[4] = { esz, (size_t) heads * esz * 2, (size_t) heads * bins * esz,
                              (size_t) heads * bins * esz * (size_t) T };
    expect("non-contiguous destination", ggml_cl_cpy_t01_eligible(0, 0, ne_s, nb_s, ne_d, nb_d4), false);

    // a non-f32 pair
    expect("f16 source", ggml_cl_cpy_t01_eligible(1, 0, ne_s, nb_s, ne_d, nb_d), false);

    if (failures) {
        printf("%d case(s) failed\n", failures);
        return 1;
    }
    printf("all cases pass\n");
    return 0;
}

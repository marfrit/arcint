// soa_route_test.cpp - 0230 / 0231: the write routes of a q8_0 / q4_0 tensor in the OpenCL set_tensor
// (ggml-opencl-soa-route.h), two tables.
//
// Table 1 (0230), ggml_cl_soa_write_route: a write that reaches the split conversion. The conversion uploads
// the WHOLE tensor from the caller's data, so only a whole write of a whole, contiguous base may take it: a
// view keeps 0209's refusal, a partial write is refused. Sizes: 8,704 bytes = 256 q8_0 blocks of 34 bytes;
// 4,608 = 256 q4_0 blocks of 18 bytes.
//
// Table 2 (0231), ggml_cl_soa_write_route2: the same write with the tensor's state. A tensor whose base is
// not split keeps table 1's route (its first whole write converts). A split base is never converted again: a
// whole write of it goes into its scales and quants (PLANES), in a weights buffer too; in a non-weights
// buffer a range of whole blocks inside the tensor and a view's do as well; in a weights buffer a view and a
// partial write keep table 1's refusals. A range that is not whole blocks or runs past the tensor is
// REFUSE_ALIGN, an Adreno layout REFUSE_LAYOUT. Shapes: q8_0 [256, 32] = 272 bytes a row, 8,704 bytes;
// q4_0 [256, 32] = 144 bytes a row, 4,608; the KV shape [4096, 64] = 4,352 (q8_0) / 2,304 (q4_0) bytes a row,
// 278,528 / 147,456 bytes. Not in the tables: a view or a partial write of a non-weights tensor that is NOT
// split is 0209's plain copy, which set_tensor takes before either function is asked.
//
// Expected: "SOA ROUTE: 8 of 8 cases right", "SOA ROUTE2: 27 of 27 cases right", rc 0.
// Red build -DGGML_CL_SOA_ROUTE2_RED (0230's routing: a range into a split tensor is refused, a whole write
// of a split base converts again): the fourteen PLANES rows of table 2 FAIL (1, 6, 7, 8, 9, 10, 17, 18, 19,
// 21, 22, 23, 24, 25), "SOA ROUTE2: 13 of 27 cases right", rc 1.
// Red build -DGGML_CL_SOA_ROUTE_RED (the routing before 0230, a partial write converts): table 1 cases 4, 5, 6
// and 8 FAIL, "SOA ROUTE: 4 of 8 cases right"; table 2 row 3 FAILS, "SOA ROUTE2: 26 of 27 cases right", rc 1.
//
//   g++ -std=c++17 -O2 -Wall -Wextra -I <llama-tree>/ggml/src/ggml-opencl soa_route_test.cpp -o soa_route_test
//   ./soa_route_test
#include "ggml-opencl-soa-route.h"

#include <cstdio>

struct row_t { int view; size_t offset, size, nbytes; ggml_cl_soa_route want; };

static const row_t rows[] = {
    { 0,    0, 8704, 8704, GGML_CL_SOA_CONVERT        },   // a whole write
    { 1,    0, 8704, 8704, GGML_CL_SOA_REFUSE_VIEW    },   // a view, written whole
    { 1,   34,  340, 8704, GGML_CL_SOA_REFUSE_VIEW    },   // a view wins over partial
    { 0,    0, 8670, 8704, GGML_CL_SOA_REFUSE_PARTIAL },   // short
    { 0,   34, 8670, 8704, GGML_CL_SOA_REFUSE_PARTIAL },   // offset, to the end
    { 0,   34,  340, 8704, GGML_CL_SOA_REFUSE_PARTIAL },   // a row range
    { 0,    0, 4608, 4608, GGML_CL_SOA_CONVERT        },   // q4_0: 256 blocks of 18 bytes, whole
    { 0,   18, 4590, 4608, GGML_CL_SOA_REFUSE_PARTIAL },   // q4_0, short
};

struct row2_t {
    int weights, view, contiguous, split, plain;
    size_t view_offs, offset, size, nbytes, base_nbytes, blk;
    ggml_cl_soa_route want;
};

static const row2_t rows2[] = {
    { 1, 0, 1, 1, 1,     0,     0,  8704,  8704,   8704, 34, GGML_CL_SOA_PLANES         },   //  1 weights, split, the whole base set again
    { 1, 1, 1, 1, 1,  1088,     0,   544,   544,   8704, 34, GGML_CL_SOA_REFUSE_VIEW    },   //  2 weights: a view of a split base, 0209's refusal
    { 1, 0, 1, 1, 1,     0,   272,   544,  8704,   8704, 34, GGML_CL_SOA_REFUSE_PARTIAL },   //  3 weights: a partial write of a split base, 0230's refusal
    { 0, 0, 1, 0, 1,     0,     0,  8704,  8704,   8704, 34, GGML_CL_SOA_CONVERT        },   //  4 not split: the first whole write
    { 0, 1, 1, 0, 1,  1088,     0,   544,   544,   8704, 34, GGML_CL_SOA_REFUSE_VIEW    },   //  5 not split: delegates (0209 took it before)
    { 0, 0, 1, 1, 1,     0,     0,  8704,  8704,   8704, 34, GGML_CL_SOA_PLANES         },   //  6 non-weights, split, the whole base set again
    { 0, 1, 1, 1, 1,  1088,     0,   544,   544,   8704, 34, GGML_CL_SOA_PLANES         },   //  7 rows 4..5: 1088 = 32 x 34, 544 = 16 x 34
    { 0, 1, 1, 1, 1,  1088,   272,   272,   544,   8704, 34, GGML_CL_SOA_PLANES         },   //  8 inside the view: 1360 = 40 x 34
    { 0, 0, 1, 1, 1,     0,   272,   544,  8704,   8704, 34, GGML_CL_SOA_PLANES         },   //  9 a partial write of the base: 272 = 8 x 34
    { 0, 1, 1, 1, 1,     0,     0,  8704,  8704,   8704, 34, GGML_CL_SOA_PLANES         },   // 10 a view of the whole base, written whole
    { 0, 1, 1, 1, 1,  1088,    17,   272,   544,   8704, 34, GGML_CL_SOA_REFUSE_ALIGN   },   // 11 1105 % 34 = 17
    { 0, 1, 1, 1, 1,  1088,     0,    35,   544,   8704, 34, GGML_CL_SOA_REFUSE_ALIGN   },   // 12 35 % 34 = 1
    { 0, 1, 1, 1, 1,  1088,   272,   544,   544,   8704, 34, GGML_CL_SOA_REFUSE_ALIGN   },   // 13 past the view: 816 > 544
    { 0, 1, 1, 1, 1,  8432,     0,   544,   544,   8704, 34, GGML_CL_SOA_REFUSE_ALIGN   },   // 14 past the base: 8976 > 8704
    { 0, 1, 1, 1, 0,  1088,     0,   544,   544,   8704, 34, GGML_CL_SOA_REFUSE_LAYOUT  },   // 15 row 7 with an Adreno layout
    { 0, 1, 0, 1, 1,  1088,     0,   544,   544,   8704, 34, GGML_CL_SOA_REFUSE_VIEW    },   // 16 row 7, not contiguous
    { 0, 1, 1, 1, 1,  1088,     0,     0,   544,   8704, 34, GGML_CL_SOA_PLANES         },   // 17 size 0
    { 0, 1, 1, 1, 1,   576,     0,   288,   288,   4608, 18, GGML_CL_SOA_PLANES         },   // 18 q4_0 rows 4..5: 576 = 32 x 18
    { 0, 0, 1, 1, 1,     0,   144,   288,  4608,   4608, 18, GGML_CL_SOA_PLANES         },   // 19 q4_0, a partial write of the base
    { 0, 1, 1, 1, 1,   576,     9,   144,   288,   4608, 18, GGML_CL_SOA_REFUSE_ALIGN   },   // 20 q4_0: 585 % 18 = 9
    { 0, 0, 1, 1, 1,     0,     0,  4608,  4608,   4608, 18, GGML_CL_SOA_PLANES         },   // 21 q4_0 non-weights, split, the whole base set again
    { 0, 1, 1, 1, 1, 34816,     0, 17408, 34816, 278528, 34, GGML_CL_SOA_PLANES         },   // 22 KV q8_0 rows 8..11: 34816 = 1024 x 34
    { 0, 1, 1, 1, 1, 34816, 17408, 17408, 34816, 278528, 34, GGML_CL_SOA_PLANES         },   // 23 KV q8_0 rows 12..15: 52224 = 1536 x 34
    { 0, 1, 1, 1, 1, 18432,  9216,  9216, 18432, 147456, 18, GGML_CL_SOA_PLANES         },   // 24 KV q4_0 rows 12..15: 27648 = 1536 x 18
    { 1, 0, 1, 1, 1,     0,     0,  4608,  4608,   4608, 18, GGML_CL_SOA_PLANES         },   // 25 q4_0 weights, split, the whole base set again
    { 0, 0, 1, 1, 0,     0,     0,  8704,  8704,   8704, 34, GGML_CL_SOA_REFUSE_LAYOUT  },   // 26 row 6 with an Adreno layout
    { 1, 0, 1, 0, 1,     0,     0,  8704,  8704,   8704, 34, GGML_CL_SOA_CONVERT        },   // 27 weights, not split: the first whole write
};

static const char * name(ggml_cl_soa_route r) {
    switch (r) {
        case GGML_CL_SOA_CONVERT:        return "CONVERT";
        case GGML_CL_SOA_REFUSE_VIEW:    return "REFUSE_VIEW";
        case GGML_CL_SOA_REFUSE_PARTIAL: return "REFUSE_PARTIAL";
        case GGML_CL_SOA_PLANES:         return "PLANES";
        case GGML_CL_SOA_REFUSE_ALIGN:   return "REFUSE_ALIGN";
        case GGML_CL_SOA_REFUSE_LAYOUT:  return "REFUSE_LAYOUT";
    }
    return "?";
}

int main() {
    const int n = (int) (sizeof rows / sizeof rows[0]);
    int right = 0;
    for (int i = 0; i < n; ++i) {
        const row_t & r = rows[i];
        const ggml_cl_soa_route got = ggml_cl_soa_write_route(r.view != 0, r.offset, r.size, r.nbytes);
        const bool ok = got == r.want;
        right += ok;
        printf("CASE %d view=%d offset=%zu size=%zu nbytes=%zu -> got %s want %s %s\n",
               i + 1, r.view, r.offset, r.size, r.nbytes, name(got), name(r.want), ok ? "OK" : "FAIL");
    }
    const int n2 = (int) (sizeof rows2 / sizeof rows2[0]);
    int right2 = 0;
    for (int i = 0; i < n2; ++i) {
        const row2_t & r = rows2[i];
        const ggml_cl_soa_route got = ggml_cl_soa_write_route2(r.weights != 0, r.view != 0, r.contiguous != 0, r.split != 0,
                                                               r.plain != 0, r.view_offs, r.offset, r.size, r.nbytes,
                                                               r.base_nbytes, r.blk);
        const bool ok = got == r.want;
        right2 += ok;
        printf("CASE2 %d weights=%d view=%d contiguous=%d split=%d plain=%d view_offs=%zu offset=%zu size=%zu nbytes=%zu "
               "base_nbytes=%zu blk=%zu -> got %s want %s %s\n",
               i + 1, r.weights, r.view, r.contiguous, r.split, r.plain, r.view_offs, r.offset, r.size, r.nbytes,
               r.base_nbytes, r.blk, name(got), name(r.want), ok ? "OK" : "FAIL");
    }
    printf("SOA ROUTE: %d of %d cases right\n", right, n);
    printf("SOA ROUTE2: %d of %d cases right\n", right2, n2);
    return right == n && right2 == n2 ? 0 : 1;
}

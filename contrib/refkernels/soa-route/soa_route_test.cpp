// soa_route_test.cpp - 0230: the split conversion's write route (ggml-opencl-soa-route.h), a table.
//
// ggml_backend_opencl_buffer_set_tensor sends a write into a q8_0 / q4_0 tensor that 0209's plain copy
// does not take (a weights tensor, or a non-weights tensor already split into scales and quants) to the
// conversion, which uploads the WHOLE tensor from the caller's data. Only a whole write of a whole,
// contiguous base may take it: a view keeps 0209's refusal, and a partial write (offset != 0 or fewer
// bytes than the tensor's) is refused too, since it read past the caller's data and re-cut the split
// layout. The route is a pure function, so this checks it alone, without a card or a backend.
//
// The table's sizes: 8,704 bytes = 256 q8_0 blocks of 34 bytes; 4,608 = 256 q4_0 blocks of 18 bytes.
// Expected: all 8 cases right, rc 0. Red build -DGGML_CL_SOA_ROUTE_RED (today's routing, a partial
// write converts): cases 4, 5, 6 and 8 FAIL, "4 of 8 cases right", rc 1.
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

static const char * name(ggml_cl_soa_route r) {
    switch (r) {
        case GGML_CL_SOA_CONVERT:        return "CONVERT";
        case GGML_CL_SOA_REFUSE_VIEW:    return "REFUSE_VIEW";
        case GGML_CL_SOA_REFUSE_PARTIAL: return "REFUSE_PARTIAL";
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
    printf("SOA ROUTE: %d of %d cases right\n", right, n);
    return right == n ? 0 : 1;
}

// Patch 0123zz's device-free cell: the paged FLASH_ATTN_EXT predicate's logic.
//
// Claude's window 226 found every paged case "not supported [OpenCL]" on the B60
// before the kernel had run: the predicate is in
// ggml/src/ggml-opencl/ggml-opencl-fa-paged.h, and supports_op calls it with the
// tensors' values. This cell drives the same function over the agent's shapes and
// flips each clause, so a predicate that is true where the kernel cannot read the
// case (or false where it can) is caught without a card.
//
// The kernel-side half -- a helper placed between
// `__attribute__((intel_reqd_sub_group_size(16)))` and `kernel void` takes the
// attribute for itself and makes the DPAS builtins fail, so the optional program
// build leaves the kernel null and every paged case is declined -- is asserted in
// tests/python/test_fa_gqa_i8.py's source-text cell (the kernel is not compiled
// on the recipe host).
//
// Red case (branch kv-stream-red6): the predicate's `kernels_ok` clause dropped,
// so a shape without its kernel passes; the "no kernel" checks fail.

#include "ggml-opencl-fa-paged.h"

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

// the agent: q f32, 24 query heads on 4 KV heads, 6 rows, head 256, q8_0 K /
// q4_0 V, a whole-page ring and a full-width mask
ggml_cl_fa_paged_shape agent_decode() {
    ggml_cl_fa_paged_shape s{};
    s.q_type = 0; s.k_type = 8; s.v_type = 4;          // F32, Q8_0, Q4_0
    s.q_ne1 = 6; s.q_ne2 = 24; s.q_ne3 = 1;
    s.k_ne0 = 256; s.k_ne2 = 4; s.k_ne3 = 1;
    s.v_ne0 = 256; s.v_ne2 = 4; s.v_ne3 = 1;
    s.table_ok = true; s.mask_ok = true; s.ring_ok = true;
    s.kernels_decode = true; s.kernels_prefill = true;
    return s;
}

} // namespace

int main() {
    // the arm that must proceed: the agent's decode shape
    CHECK(ggml_cl_fa_paged_ok(agent_decode()), "the agent's 6-row decode shape is accepted");
    CHECK(ggml_cl_fa_paged_kv_ok(agent_decode()) && ggml_cl_fa_paged_shape_ok(agent_decode()),
          "its KV pair and shapes pass their clauses");

    // each clause, flipped on its own, must refuse
    { auto s = agent_decode(); s.table_ok = false; CHECK(!ggml_cl_fa_paged_ok(s), "a broken page table is refused"); }
    { auto s = agent_decode(); s.k_type = 4; s.v_type = 8; CHECK(!ggml_cl_fa_paged_ok(s), "a q4_0 K / q8_0 V pair is refused"); }
    { auto s = agent_decode(); s.k_ne0 = 128; CHECK(!ggml_cl_fa_paged_ok(s), "head 128 is refused"); }
    { auto s = agent_decode(); s.q_type = 1; CHECK(!ggml_cl_fa_paged_ok(s), "q f16 is refused"); }
    { auto s = agent_decode(); s.mask_ok = false; CHECK(!ggml_cl_fa_paged_ok(s), "a compact mask is refused until I5"); }
    { auto s = agent_decode(); s.ring_ok = false; CHECK(!ggml_cl_fa_paged_ok(s), "a ring whose K/V pair is half present is refused"); }
    { auto s = agent_decode(); s.k_ne3 = 2; CHECK(!ggml_cl_fa_paged_ok(s), "a multi-stream K view is refused"); }
    { auto s = agent_decode(); s.q_ne1 = 512; s.kernels_prefill = false;
      CHECK(!ggml_cl_fa_paged_ok(s), "a prompt without the conversion+f16 kernel is refused"); }
    { auto s = agent_decode(); s.q_ne1 = 512; CHECK(ggml_cl_fa_paged_ok(s), "a prompt with those kernels is accepted"); }
    { auto s = agent_decode(); s.kernels_decode = false; s.kernels_prefill = false;
      CHECK(!ggml_cl_fa_paged_ok(s), "no kernel at all is refused"); }

    // 1-3 rows: the staged 1-3-row form is the decode kernel's, so they are accepted
    { auto s = agent_decode(); s.q_ne1 = 1; CHECK(ggml_cl_fa_paged_ok(s), "a 1-row decode is accepted"); }

    // the no-ring arm: every attended page is resident, still accepted
    { auto s = agent_decode(); CHECK(ggml_cl_fa_paged_ok(s), "a resident-only ubatch is accepted"); }

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the paged FA predicate accepts the agent's shapes and refuses each flipped clause\n");
    return 0;
}

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
// Window 228 exposed two wrong values: the KV pair compared literals (Q4_0 is 2 in
// ggml.h, not 4), so the agent's q8_0:q4_0 was declined while f16 passed; the test
// now names the enums. The companion crash was the device-side page table: the
// backend passed the tensor's base and ignored the used row, row 0 held INT32_MAX,
// and the kernel read a pool row of 2^31 rows out of bounds -- the table row's
// byte offset is passed now (or a host row is uploaded), see 0123zz.
//
// 0125 (I5 review D): the shape and the caps are one pair of plain structs,
// and `ggml_cl_fa_paged_form` returns the form the dispatcher takes -- 1 the GQA
// decode kernel, 2 the f16-block prefill conversion, 0 none. The dispatcher asks
// the same function before choosing, so supports_op cannot accept a shape the
// dispatcher would abort on: rows, mask heads, ALiBi/softcap/sinks and the
// work-group's 64-row cap are in the shape or the caps, not in a second copy.
//
// Red case (kv-stream-idxs-red, the 0119 review D): `ggml_cl_fa_paged_params_ok`
// dropped from the form's clauses, so a softcapped call passes and the param
// checks below fail. A second red: `c.tiles` ignored, so a 64-row decode on one
// tile passes and the row-cap check fails.

#include "ggml.h"
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
// q4_0 V, a whole-page ring, a full-width f16 mask over one stream, 8 tiles
ggml_cl_fa_paged_shape agent_decode() {
    ggml_cl_fa_paged_shape s{};
    s.q_type = GGML_TYPE_F32; s.k_type = GGML_TYPE_Q8_0; s.v_type = GGML_TYPE_Q4_0;
    s.q_ne1 = 6; s.q_ne2 = 24; s.q_ne3 = 1;
    s.k_ne0 = 256; s.k_ne2 = 4; s.k_ne3 = 1;
    s.v_ne0 = 256; s.v_ne2 = 4; s.v_ne3 = 1;
    s.mask_present = true; s.mask_type = GGML_TYPE_F16;
    s.mask_ne0 = 256*1024; s.mask_ne1 = 6; s.mask_ne2 = 1; s.mask_ne3 = 1;
    s.causal = false; s.n_batch = 1;
    s.table_ok = true; s.ring_ok = true;
    return s;
}

ggml_cl_fa_paged_caps agent_caps() {
    ggml_cl_fa_paged_caps c{};
    c.kernels_gqa_direct = true; c.kernels_gqa_staged = true; c.kernels_prefill = true;
    c.split_decode = true; c.tiles = 8; c.wide_on = true; c.intel_split_on = true;
    return c;
}

} // namespace

int main() {
    // the arm that must proceed: the agent's 6-row decode shape on the split kernel
    CHECK(ggml_cl_fa_paged_ok(agent_decode(), agent_caps()), "the agent's 6-row decode shape is accepted");
    CHECK(ggml_cl_fa_paged_form(agent_decode(), agent_caps()) == 1, "its form is the GQA decode");
    CHECK(ggml_cl_fa_paged_kv_ok(agent_decode()) && ggml_cl_fa_paged_shape_ok(agent_decode()),
          "its KV pair and shapes pass their clauses");

    // each clause, flipped on its own, must refuse
    { auto s = agent_decode(); s.table_ok = false; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "a broken page table is refused"); }
    { auto s = agent_decode(); s.k_type = GGML_TYPE_Q4_0; s.v_type = GGML_TYPE_Q8_0;
      CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "a q4_0 K / q8_0 V pair is refused"); }
    { auto s = agent_decode(); s.k_ne0 = 128; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "head 128 is refused"); }
    { auto s = agent_decode(); s.q_type = GGML_TYPE_F16; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "q f16 is refused"); }
    { auto s = agent_decode(); s.mask_type = GGML_TYPE_F32; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "an f32 mask is refused"); }
    { auto s = agent_decode(); s.ring_ok = false; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "a ring whose K/V pair is half present is refused"); }
    { auto s = agent_decode(); s.k_ne3 = 2; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "a multi-stream K view is refused"); }

    // 0119 review D: the params the paged kernels do not take
    { auto s = agent_decode(); s.max_bias = 8.0f; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "ALiBi (max_bias) is refused"); }
    { auto s = agent_decode(); s.logit_softcap = 30.0f; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "a logit softcap is refused"); }
    { auto s = agent_decode(); s.sinks_present = true; CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "attention sinks are refused"); }

    // the work-group's cap: (query heads / KV heads) * rows <= tiles * 8
    { auto s = agent_decode(); s.q_ne1 = 64; CHECK(ggml_cl_fa_paged_ok(s, agent_caps()), "a 64-row (head,token) verify is accepted"); }
    { auto s = agent_decode(); s.q_ne1 = 65; CHECK(ggml_cl_fa_paged_form(s, agent_caps()) == 2, "65 rows take the prefill conversion"); }
    { auto s = agent_decode(); auto c = agent_caps(); c.tiles = 4;
      CHECK(ggml_cl_fa_paged_form(s, c) == 2, "the same 6 rows on 4 tiles exceed the decode cap and take the conversion"); }
    { auto s = agent_decode(); s.q_ne1 = 96; s.q_ne2 = 96;  // 24x2 = 96 rows a tile cap of 8 tiles
      CHECK(!ggml_cl_fa_paged_ok(s, agent_caps()), "rows beyond the cap with no conversion are refused"); }

    // 9-64 rows: the wide GQA call needs one batch and one stream's mask
    { auto s = agent_decode(); s.q_ne1 = 20; CHECK(ggml_cl_fa_paged_form(s, agent_caps()) == 1, "20 rows take the wide GQA call"); }
    { auto s = agent_decode(); s.q_ne1 = 20; s.mask_ne2 = 2;
      CHECK(ggml_cl_fa_paged_form(s, agent_caps()) == 2, "20 rows over two mask heads take the conversion"); }
    { auto s = agent_decode(); s.q_ne1 = 20; s.n_batch = 2;
      CHECK(ggml_cl_fa_paged_form(s, agent_caps()) == 2, "20 rows on two K/V batches take the conversion"); }
    { auto s = agent_decode(); s.q_ne1 = 20; auto c = agent_caps(); c.wide_on = false;
      CHECK(ggml_cl_fa_paged_form(s, c) == 2, "with the wide verify off, 20 rows take the conversion"); }
    { auto s = agent_decode(); s.q_ne1 = 20; auto c = agent_caps(); c.split_decode = false;
      CHECK(ggml_cl_fa_paged_form(s, c) == 1, "without the split kernel the wide call still decodes"); }

    // prefill: the conversion + f16 prompt kernels carry every row count
    { auto s = agent_decode(); s.q_ne1 = 512; CHECK(ggml_cl_fa_paged_form(s, agent_caps()) == 2, "a 512-row prompt takes the conversion"); }
    { auto s = agent_decode(); s.q_ne1 = 512; auto c = agent_caps(); c.kernels_prefill = false;
      CHECK(!ggml_cl_fa_paged_ok(s, c), "a prompt without the conversion+f16 kernel is refused"); }
    { auto s = agent_decode(); auto c = agent_caps();
      c.kernels_gqa_direct = false; c.kernels_gqa_staged = false; c.kernels_prefill = false;
      CHECK(!ggml_cl_fa_paged_ok(s, c), "no kernel at all is refused"); }

    // 1 row: the staged 1-3-row form is the decode kernel's, so it is accepted
    { auto s = agent_decode(); s.q_ne1 = 1; CHECK(ggml_cl_fa_paged_ok(s, agent_caps()), "a 1-row decode is accepted"); }
    { auto s = agent_decode(); s.q_ne1 = 1; auto c = agent_caps(); c.split_decode = false;
      CHECK(ggml_cl_fa_paged_form(s, c) == 2, "a 1-row decode without the split takes the conversion"); }

    // causal attention has no split decode: it takes the conversion
    { auto s = agent_decode(); s.causal = true; CHECK(ggml_cl_fa_paged_form(s, agent_caps()) == 2, "a causal call takes the conversion"); }

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the paged FA predicate accepts the agent's shapes and refuses each flipped clause\n");
    return 0;
}

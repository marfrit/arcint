// The raw gating projections of a gated delta-net node, kept alive to it.
//
// Patch 0051's OpenCL fold computes the gating inside the GDN kernel from the
// raw projections alpha and b (vllm-xpu-kernels' delta stage takes a and b with
// A_log and dt_bias). The kernel reads them when the GDN runs, later than the
// allocator keeps them: it frees alpha after the ADD. Attaching them as extra
// sources of the GDN node keeps them alive; the op's semantics do not change and
// the CPU ignores them.
//
// The slots are the GDN family's contract (ggml.h, op_params[1] = mode):
// alpha rides src[8], b rides src[9]. src[6] is the rows form's I32 row index
// and src[6]/src[7] are the replay's records and control; an attached gating
// input on src[6] made a gathered node look like the rows form and every
// consumer that reads a row index there asserted on a F32 (patch 0081's red).
// A backend folds only when its sources are the chain's raw inputs.
//
// ggml.h only, so the same text serves the graph builders and the device-free
// cell (tests/test_gdn_gate_slots.cpp).

#ifndef GGML_GDN_GATE_INPUTS_H
#define GGML_GDN_GATE_INPUTS_H

#include "ggml.h"

static inline void gdn_keep_gating_inputs(ggml_tensor * gdn, ggml_tensor * g, ggml_tensor * b) {
    const auto unary = [](const ggml_tensor * t, ggml_unary_op u) { return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == u; };
    ggml_tensor * m = g->op == GGML_OP_RESHAPE ? g->src[0] : g;
    if (m->op == GGML_OP_MUL) {
        ggml_tensor * sp = unary(m->src[0], GGML_UNARY_OP_SOFTPLUS) ? m->src[0] : m->src[1];
        if (unary(sp, GGML_UNARY_OP_SOFTPLUS) && sp->src[0]->op == GGML_OP_ADD) {
            gdn->src[8] = sp->src[0]->src[0];
        }
    }
    if (unary(b, GGML_UNARY_OP_SIGMOID)) {
        gdn->src[9] = b->src[0];
    }
}

#endif // GGML_GDN_GATE_INPUTS_H

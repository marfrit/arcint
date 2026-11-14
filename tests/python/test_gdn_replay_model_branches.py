"""0086/0087: every model the replay gate admits builds its GDN state through the replay rows.

Text cell over the series' patched llama.cpp tree (LLAMA_SRC). The replay decision in
llama_memory_recurrent (LLAMA_GDN_REPLAY=1) admits qwen35, qwen35moe and qwen4exp; each one's
build_layer_attn_linear must read the base state gathered with `inp->s_rpl_rows` and must not
consult the rows form when replay is on. qwen35moe was missed by 0086/0087: it stayed on the
snapshot gather (`build_rs(inp, ...)`, which reads `s_copy`), so `rs_s_rpl_rows` was an input
tensor no node consumed. The graph allocator gives an unread input no buffer, and
`rs_set_input_rpl` then asserts `ggml_backend_buffer_is_host(s_rpl_rows->buffer)` on the first
decode: GGML_ASSERT(buffer) in ggml_backend_buffer_get_type (measured-here, the coder on the
A770, LLAMA_GDN_REPLAY=1; device-free the same abort on the CPU backend).

The sub-group contract (0127): `kernel_gated_delta_net_rpl` is the generation kernel's body
with a replay prologue and a record epilogue, and its `intel_sub_group_block_read/write` and
`sub_group_reduce_add` work over one sub-group of SG work-items, with every index `m*SG`. The
tg and pp kernels carry `intel_reqd_sub_group_size(SG)`; the replay kernel did not, so the
compiler need not choose SG (a smaller SIMD size leaves gaps and overlaps in the state rows).
The chain cases were red on the A770 and the B60 (max |diff| 1.3-1.6) while every no-replay
GATED_DELTA_NET case passed; the card arm is test-backend-ops with `-o 'GATED_DELTA_NET.*'`.

Red cases, run before the fixes: the qwen35moe branch dropped (the pre-fix tree) -> "qwen35moe
builds the replay state" fails and the `s_rpl_rows` count is 0 in that file (the mutation is the
fix's own hunk removed from 0000-07-gdn-and-fusions.patch); without 0127 the replay kernel's
declaration check fails, and on a card the chain cases were red on both the A770 and the B60
(max |diff| 1.3-1.6). 0128 adds the coder's geometry to those chain cases.
"""
import os
import sys

root = os.environ.get("LLAMA_SRC", "")
fail = 0


def check(name, ok):
    global fail
    print(("PASS " if ok else "FAIL ") + name)
    fail += 0 if ok else 1


def read(rel):
    return open(os.path.join(root, rel)).read()


mem = read("src/llama-memory-recurrent.cpp")
# the gate's architecture list is the set of models whose builder must carry the branch
for arch in ("LLM_ARCH_QWEN35", "LLM_ARCH_QWEN35MOE", "LLM_ARCH_QWEN4EXP"):
    check("the replay gate admits " + arch, arch in mem)
check("the gate requires the OpenCL backend", 'strcmp(ggml_backend_reg_name(reg), "OpenCL") == 0' in mem)
check("the gate is off unless LLAMA_GDN_REPLAY is non-zero",
      'getenv("LLAMA_GDN_REPLAY")' in mem and "atoi(env) != 0" in mem)

base = read("src/models/delta-net-base.cpp")
check("the replay op is built when the control is set",
      "if (inp->s_rpl_ctrl) {" in base and "ggml_gated_delta_net_replay(ctx0" in base)

REPLAY_STATE = """    const bool gdn_rpl = inp->s_rpl_ctrl != nullptr;
    const bool gdn_state_rows = !gdn_rpl && llama_gdn_state_rows_enabled(cparams, n_seqs, n_seq_tokens);"""
REPLAY_GATHER = """        state = build_rs(ssm_states_all, inp->s_rpl_rows, inp->s_rpl_main, hparams.n_embd_s(), n_seqs,"""

for model in ("qwen35", "qwen35moe", "qwen4exp"):
    src = read("src/models/%s.cpp" % model)
    check(model + " builds the replay state", REPLAY_STATE in src)
    check(model + " gathers the state with s_rpl_rows", REPLAY_GATHER in src)
    check(model + " reads s_rpl_rows (an unread input has no buffer)", "s_rpl_rows" in src)
    # the rows form stays off under replay
    check(model + " keeps the rows form off under replay", "!gdn_rpl && llama_gdn_state_rows_enabled" in src)

# 0127: the replay kernel's block I/O and its m*SG indices need the sub-group size pinned,
# as the tg and pp kernels do; without it the compiler's SIMD size need not be SG.
cl = read("ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl")
tbo = read("tests/test-backend-ops.cpp")
rpl = cl[cl.index("kernel void kernel_gated_delta_net_rpl(") - 400:]
rpl = rpl[:rpl.index("kernel_gated_delta_net_rpl(")]
check("0127: the replay kernel declares the sub-group size",
      "__attribute__((intel_reqd_sub_group_size(SG)))" in rpl)
check("0127: the generation kernel keeps its declaration",
      cl.count("__attribute__((intel_reqd_sub_group_size(SG)))") >= 3)
check("0127: the card chain runs the coder's geometry", "test_gdn_replay_chain(16, 2, 5, chain_a" in tbo)

sys.exit(1 if fail else 0)

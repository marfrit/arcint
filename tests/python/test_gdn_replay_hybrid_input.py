"""0091: the replay controls (0086) are filled by the hybrid memory inputs too. Text cell over the series' patched
llama-graph.cpp: a hybrid input that copies only s_copy leaves the controls unset and never calls rpl_take."""
import os
import re
import sys

src = open(os.path.join(os.environ.get("LLAMA_SRC", ""), "src/llama-graph.cpp")).read()
fail = 0


def check(name, ok):
    global fail
    print(("PASS " if ok else "FAIL ") + name)
    fail += 0 if ok else 1


def body(sig):
    i = src.index(sig)
    return src[i:src.index("\n}\n", i)]


check("one helper fills the controls", src.count("static void rs_set_input_rpl(") == 1)
check("rpl_take is called only from the helper", src.count("rpl_take(") == 1)
check("the recurrent input uses the helper", "rs_set_input_rpl(this, mctx, ubatch);" in body("void llm_graph_input_rs::set_input("))
for cls in ("llm_graph_input_mem_hybrid", "llm_graph_input_mem_hybrid_k", "llm_graph_input_mem_hybrid_iswa"):
    check(cls + "::set_input fills the replay controls",
          "rs_set_input_rpl(inp_rs.get(), mctx->get_recr(), ubatch);" in body(f"void {cls}::set_input("))
    check(cls + "::can_reuse checks the replay tensors",
          "rs_can_reuse_rpl(inp_rs.get(), mctx->get_recr(), params.ubatch)" in body(f"bool {cls}::can_reuse("))
sys.exit(1 if fail else 0)

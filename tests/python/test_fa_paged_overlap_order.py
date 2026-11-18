"""0131j: the KV window's copy-queue overlap keeps its orderings across a graph boundary.

Text cell over the series' patched llama.cpp tree (LLAMA_SRC), ggml-opencl.cpp. Three orderings
of GGML_OPENCL_FA_PAGED_OVERLAP=1 that the 1.7.0 release review found open (evidence class
`code`; none was seen failing on a card, where decode drains the host between graphs):

- consumed[] (each ring slot's last reader) was released at every new graph, so in the
  reused-prefetch branch node 1's copy had no wait at all: with an even attention chain slot 1's
  last reader is the previous graph's last FA, and a graph that follows without a host drain
  (consecutive no-logit ubatches) could overwrite the slot it still reads. Kept now.
- the consumed marker is waited on the copy queue, so the compute queue that holds it must be
  flushed (OpenCL's cross-queue rule; the cards' runtime happens to submit without it).
- the reuse key named the plan's (page, row) pairs, the chain length and the slot, not the layer:
  a graph in two scheduler splits of equal paged-FA count shares all three between the splits'
  node 0. The key now carries node 0's host store.

Red case, run before the fix: the tree with patches up to 0131i fails all three checks.
"""
import os
import re
import sys

root = os.environ.get("LLAMA_SRC", "")
fail = 0


def check(name, ok):
    global fail
    print(("PASS " if ok else "FAIL ") + name)
    fail += 0 if ok else 1


src = open(os.path.join(root, "ggml/src/ggml-opencl/ggml-opencl.cpp")).read()


def body(fn):
    i = src.index("static %s(" % fn)
    j = src.index("\n}\n", i)
    return src[i:j]


g = body("void ggml_cl_fa_paged_overlap_graph")
start = g[g.index("if (o.gen != ctx->graph_gen)"):g.index("o.gen  = ctx->graph_gen;")]
check("0131j: a new graph keeps consumed[] (the slots' last readers)",
      "clReleaseEvent(o.ready[i])" in start and "o.consumed[" not in start)
a = body("void ggml_cl_fa_paged_overlap_after")
m = a.index("clEnqueueMarkerWithWaitList(ctx->queue, 0, nullptr, &o.consumed[parity])")
check("0131j: the consumed marker is flushed on the compute queue before a copy waits it",
      re.search(r"CL_CHECK\(clFlush\(ctx->queue\)\);", a[m:]) is not None
      and a[m:].index("clFlush(ctx->queue)") < a[m:].index("ggml_cl_fa_paged_overlap_issue("))
same = body("bool ggml_cl_fa_paged_prefetch_same")
head = body("void ggml_cl_fa_paged_prefetch_head")
check("0131j: the prefetch reuse key names node 0's host store",
      "hk->data != o.pre_host_k) return false;" in same
      and "o.pre_host_k = dst->src[GGML_FA_SRC_HOST_K] ? dst->src[GGML_FA_SRC_HOST_K]->data" in head)

# 0131m: a paged call never takes 0028's row blocks (their 8-row re-entries ran outside the
# overlap's chain). Red: the 0131k tree, whose row-block condition has no kv_pages clause
fa = src[src.index("static void ggml_cl_flash_attn("):]
rb = fa[fa.index('getenv("GGML_OPENCL_FA_ROW_BLOCKS")'):]
rb = rb[:rb.index("for (int r0 = 0; r0 < n_q; r0 += 8)")]
check("0131m: the row blocks take no paged call", "kv_pages == nullptr" in rb)

# 0131n: the after-step runs on every exit of the dispatch (the conversion returned without it,
# window 171g). Red: the 0131m tree, which calls it at two of the exits and has no guard
fa_body = fa[:fa.index("\n}\n")]
k = fa_body.index("ggml_cl_fa_kv_stream_pages(backend_ctx, dst,")
after_copy = fa_body[k:]
check("0131n: the after-step is a guard set right after the paged copy, and no exit calls it by hand",
      "~paged_after_guard()" in after_copy and "ggml_cl_fa_paged_overlap_after(ctx, parity)" in after_copy
      and after_copy.count("ggml_cl_fa_paged_overlap_after(") == 1)

sys.exit(1 if fail else 0)

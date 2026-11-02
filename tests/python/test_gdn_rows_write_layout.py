"""0090: in the rows form the GDN kernels read the state from its cache row but write snapshots and dst in the
per-sequence layout (as the CPU reference does). Text cell over the series' patched kernel source."""
import os
import re
import sys

p = os.path.join(os.environ.get("LLAMA_SRC", ""), "ggml/src/ggml-opencl/kernels/gated_delta_net_intel.cl")
src = open(p).read()
fail = 0


def check(name, ok):
    global fail
    print(("PASS " if ok else "FAIL ") + name)
    fail += 0 if ok else 1


check("two kernels declare a read base for the rows form", src.count("const ulong state_base_rd = use_rows") == 2)
check("the per-sequence write base in all three kernels",
      src.count("const ulong state_base = ((ulong)seq * H_v + head) * state_size;") == 3)
check("the rows kernels read through state_base_rd", src.count("BLOCK_READ_F(data_state + state_base_rd") == 2)
check("no write uses the read base", not re.search(r"(snap_stride|s_off) \+ state_base_rd", src))
g = open(os.path.join(os.environ.get("LLAMA_SRC", ""), "ggml/src/ggml-opencl/kernels/gated_delta_net.cl")).read()
check("generic kernel: the final state goes to the per-sequence base",
      "data_dst[s_off + state_out_base + col * S_V" in g and "data_dst[s_off + state_base +" not in g)
check("generic kernel: snapshots use the per-sequence base", "target_slot * state_size_per_snap + state_out_base" in g)
sys.exit(1 if fail else 0)

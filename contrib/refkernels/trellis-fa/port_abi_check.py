#!/usr/bin/env python3
"""port_abi_check.py — the port kernel's call ABI, checked without a device.

The kernel signature in flash_attn_port.cl and the clSetKernelArg order in
ggml-opencl.cpp are written by hand on both sides; a swapped or missing argument
would not crash, it would answer garbage. So: parse the kernel's parameter list,
parse the launch's argument expressions in order, and compare each position
against the expression that parameter is supposed to receive. A mismatch is a
failed check; nothing here guesses on the kernel's side.

    python3 port_abi_check.py <llama-tree>
"""
import re
import sys

tree = sys.argv[1] if len(sys.argv) > 1 else "."
cl = open(tree + "/ggml/src/ggml-opencl/kernels/flash_attn_port.cl").read()
host = open(tree + "/ggml/src/ggml-opencl/ggml-opencl.cpp").read()

# --- the kernel's parameter list ---
m = re.search(r"kernel\s+void\s+kernel_flash_attn_port\s*\((.*?)\)\s*\{", cl, re.S)
if not m:
    print("FAIL: kernel_flash_attn_port's signature not found")
    raise SystemExit(1)
params = []
for chunk in m.group(1).split(","):
    toks = chunk.split()
    if toks:
        params.append((toks[-1], " ".join(toks[:-1])))      # name, type
print(f"kernel parameters: {len(params)}")

# --- the host's argument sequence for this kernel ---
s = host.index("cl_kernel kp = ")      # the f16 build or, for a q8_0 cache, the PORT_KV8 build: one argument list
seg = host[s:host.index("backend_ctx->enqueue_ndrange_kernel(kp, 3, gws, lws, dst);", s)]
args = re.findall(r"clSetKernelArg\(kp, a\+\+, sizeof\((?:cl_[a-z]+|int|float)\)\s*,\s*&([^)]+)\)", seg)
print(f"host clSetKernelArg calls: {len(args)}")

# --- what each position must carry ---
expect = {
    "q": "qsrc",            "q_offset": "qsrc_off",    "q_nb1": "qsrc_nb1",    "q_nb2": "qsrc_nb2",
    "k": "bk",              "k_offset": "ok",          "k_nb1": "k_nb1p",      "k_nb2": "k_nb2p",
    "v": "bv",              "v_offset": "ov",          "v_nb1": "v_nb1p",      "v_nb2": "v_nb2p",
    "dst": "bo",            "o_offset": "oo",          "o_nb1": "dst->nb[1]",  "o_nb2": "o_nb2p",
    "mask_void": "bm",      "mask_offset": "om",       "mask_nb1": "mask_nb1",
    "sink": "bs",           "sink_offset": "os",
    "table": "bt",          "table_offset": "ot",      "n_bins": "n_bins",
    "kpos": "bkp",          "kpos_offset": "okp",      "qpos": "bqp",          "qpos_offset": "oqp",
    "cflag": "bcf",         "cflag_offset": "ocf",     "rflag": "brf",         "rflag_offset": "orf",
    "n_q": "n_q",           "n_kv": "n_kv",            "n_head": "n_head",     "hkv": "n_head_kv",
    "qtw": "qtw",           "ngroups": "ngroups",      "kv_lo": "kv_lo",       "kv_hi": "kv_hi",
    "window": "window",     "scale_log2": "scale_log2",
}

fails = 0
if len(args) != len(params):
    print(f"FAIL: {len(params)} parameters, {len(args)} set")
    fails += 1
for i, (name, typ) in enumerate(params):
    want = expect.get(name)
    got = args[i] if i < len(args) else "<missing>"
    if want is None:
        print(f"FAIL: parameter {i} '{name}' has no expected host expression")
        fails += 1
    elif got != want:
        print(f"FAIL: argument {i} '{name}' gets &{got}, expected &{want}")
        fails += 1

# the kernel's size types: these must be ulong in the kernel, cl_ulong on the host
for i, (name, typ) in enumerate(params):
    if name.endswith(("_nb1", "_nb2", "_offset")) or name in ("q_offset", "k_offset", "v_offset", "o_offset"):
        if typ not in ("ulong",):
            print(f"FAIL: '{name}' is a {typ} in the kernel, the host passes sizeof(cl_ulong)")
            fails += 1
print(f"{len(params)} positions checked, {fails} failures")
raise SystemExit(1 if fails else 0)

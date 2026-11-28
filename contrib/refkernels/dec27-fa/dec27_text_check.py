#!/usr/bin/env python3
"""0227: what dec27_kernel_model.py takes on trust, read out of flash_attn_dec27.cl's own text.

The model walks the kernel's arithmetic with its own copy of the index formulas; a typo in the
kernel's hand-written tables would pass the model and fail on a card. This check reads the text:

1. the K table: DEC27_S_GROUP's nine K8W reads and the 17 dword arguments of each DEC27_PAIR - every
   argument `rX.sN` must be dword 17 p + i of the head (i = 0 .. 16, in DEC27_PAIR's parameter
   order E0 .. E7, X8, O9 .. O16), and every dword a read is used for must lie inside its 8;
2. the lane maps: DEC27_ORD, dec27_qdim, V4X, V4J and V4_ODIM are cut out of the text, compiled with
   the host compiler and tabulated; the model's own ORD / qdim / V4J must give the same tables, a
   block's two chunks must hold each of its 32 dims once (K's order and V's), and V's reads must sit
   on the 4-byte grid inside the head;
3. the kernel's parameter lists against the launch's clSetKernelArg sequences (count and order of
   the buffer / scalar kinds), when the backend source is given;
4. the block reads and writes by pointer: the kernel has three. The mask run's 16-half read stands
   under the alignment guard the model walks (`kg + 16 <= n_kv && mal[j]`, mal[j] the run's pointer
   on 4 bytes, except in the red build 7). The split state's dword write and read take a pointer
   into the state's float array (`st + c * QR * 16`, st a float pointer: 4-byte cells from the
   buffer's start). Another one fails this check until its pointer's alignment is shown: on the
   card a block read 2 bytes past a 4-byte boundary returns other cells (dec27_kernel_model.py,
   block_read_us).

    python3 dec27_text_check.py <tree>/ggml/src/ggml-opencl/kernels/flash_attn_dec27.cl [<tree>/ggml/src/ggml-opencl/ggml-opencl.cpp]
"""
import re, subprocess, sys, tempfile, os, json

import dec27_kernel_model as model


def fail(msg):
    print("FAIL:", msg)
    return 1


def main():
    src = open(sys.argv[1]).read()
    bad = 0

    # ---- 1. the K table ----
    m = re.search(r"#define DEC27_S_GROUP\(.*?\n((?:.*\\\n)+.*\n)", src)
    body = m.group(0).replace("\\\n", " ")
    reads = dict((name, int(w)) for name, w in re.findall(r"\b(r[a-i]) = K8W\((\d+), y\)", body))
    if sorted(reads.values()) != [0, 8, 16, 24, 32, 40, 48, 56, 60]:
        bad += fail(f"the K reads are at dwords {sorted(reads.values())}")
    pairs = re.findall(r"DEC27_PAIR\(acc, (\d), Q\d, Q\d, ([^)]*)\)", body)
    if len(pairs) != 4:
        bad += fail(f"{len(pairs)} DEC27_PAIR uses")
    params = re.search(r"#define DEC27_PAIR\(acc, p, QE, QO, ([^)]*)\)", src).group(1).replace(" ", "").split(",")
    want_params = ["E%d" % i for i in range(8)] + ["X8"] + ["O%d" % i for i in range(9, 17)]
    if params != want_params:
        bad += fail(f"DEC27_PAIR's dword parameters are {params}")
    used = set()
    for p, args in pairs:
        args = [a.strip() for a in args.split(",")]
        if len(args) != 17:
            bad += fail(f"pair {p}: {len(args)} dword arguments"); continue
        for i, a in enumerate(args):
            name, comp = a.split(".s")
            dword = reads[name] + int(comp)
            if int(comp) > 7 or dword != 17 * int(p) + i:
                bad += fail(f"pair {p} argument {i} is {a} = dword {dword}, want {17 * int(p) + i}")
            used.add(dword)
    if used != set(range(68)):
        bad += fail(f"the pairs use {len(used)} of the head's 68 dwords")
    # the body of DEC27_PAIR: which dwords are scales, splices and operands
    pb = re.search(r"#define DEC27_PAIR\(.*?\n((?:.*\\\n)+.*\n)", src).group(0).replace("\\\n", " ")
    for need in ("as_half((ushort) (E0))", "as_half((ushort) ((X8) >> 16))", "((E0) >> 16) | ((X8) << 16)",
                 "B16((uint4)(E1, E2, E3, E4))", "B16((uint4)(E5, E6, E7, kx0))",
                 "B16((uint4)(O9, O10, O11, O12))", "B16((uint4)(O13, O14, O15, O16))",
                 "nq[2 * (p)]", "nq[2 * (p) + 1]"):
        if need not in pb:
            bad += fail(f"DEC27_PAIR has no `{need}`")
    print(f"the K table: 9 reads, 4 pairs x 17 dwords, all 68 dwords once: {'ok' if bad == 0 else 'FAIL'}")

    # ---- 2. the lane maps, from the text ----
    defs = []
    for name in ("DEC27_ORD", "V4X", "V4J", "V4_ODIM"):
        defs.append(re.search(r"#define %s\(.*" % name, src).group(0))
    qdim = re.search(r"static inline int dec27_qdim\(.*?\n}\n", src, re.S).group(0)
    prog = "#include <stdio.h>\n" + "\n".join(defs) + "\n" + qdim + r'''
int main(void) {
    printf("{\"ord\": [");
    for (int h = 0; h < 16; ++h) printf("%d%s", DEC27_ORD(h), h < 15 ? ", " : "], \"qdim\": [");
    for (int c = 0; c < 16; ++c) for (int h = 0; h < 16; ++h) printf("%d%s", dec27_qdim(c, h), c * 16 + h < 255 ? ", " : "], \"v4x\": [");
    for (int b = 0; b < 8; ++b) printf("%d%s", V4X(b), b < 7 ? ", " : "], \"v4j\": [");
    for (int b = 0; b < 8; ++b) for (int l = 0; l < 16; ++l) printf("%d%s", V4J(b, l), b * 16 + l < 127 ? ", " : "], \"odim\": [");
    for (int c = 0; c < 16; ++c) for (int l = 0; l < 16; ++l) printf("%d%s", V4_ODIM(c, l), c * 16 + l < 255 ? ", " : "]}\n");
    return 0;
}
'''
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "t.c"), "w").write(prog)
        subprocess.run(["gcc", "-O0", "-o", os.path.join(d, "t"), os.path.join(d, "t.c")], check=True)
        tab = json.loads(subprocess.run([os.path.join(d, "t")], check=True, capture_output=True, text=True).stdout)
    b2 = 0
    if tab["ord"] != [model.ORD(h) for h in range(16)] or sorted(tab["ord"]) != list(range(16)):
        b2 += fail("DEC27_ORD is not the model's ORD, or not a permutation")
    if tab["qdim"] != [model.qdim(c, h) for c in range(16) for h in range(16)]:
        b2 += fail("dec27_qdim is not the model's qdim")
    for b in range(8):
        dims = sorted(tab["qdim"][32 * b:32 * b + 32])
        if dims != list(range(32 * b, 32 * b + 32)):
            b2 += fail(f"block {b}'s two Q chunks do not hold its 32 dims once")
        od = sorted(32 * b + x for x in tab["odim"][32 * b:32 * b + 32])
        if od != list(range(32 * b, 32 * b + 32)):
            b2 += fail(f"block {b}'s two output chunks do not hold its 32 dims once")
        x = tab["v4x"][b]
        if x % 4 != 0 or x + 16 > 144 or x != 18 * b + (2 if b & 1 else 4):
            b2 += fail(f"block {b}'s V read at byte {x}")
        if tab["v4j"][16 * b:16 * b + 16] != [model.V4J(b, l) for l in range(16)]:
            b2 += fail(f"V4J of block {b} is not the model's")
        # the quant byte a lane holds after the splice is the byte the read (or the splice) took
        for l in range(16):
            byte = (x + l) if (b & 1 or l < 14) else (18 * b + l - 12)
            if byte - (18 * b + 2) != tab["v4j"][16 * b + l]:
                b2 += fail(f"block {b} lane {l}: the read holds quant byte {byte - 18 * b - 2}, V4J says {tab['v4j'][16 * b + l]}")
    print(f"the lane maps (DEC27_ORD, dec27_qdim, V4X, V4J, V4_ODIM) from the text: {'ok' if b2 == 0 else 'FAIL'}")
    bad += b2

    # ---- 3. the parameter lists against the launch ----
    if len(sys.argv) > 2:
        cpp = open(sys.argv[2]).read()
        fn = cpp[cpp.index("static bool ggml_cl_flash_attn_dec27("):]
        fn = fn[:fn.index("\nstatic void ggml_cl_flash_attn(")]
        b3 = 0
        for kname, var in (("kernel_fa_dec27_q", "kq"), ("kernel_fa_dec27", "km"), ("kernel_fa_dec27_merge", "kg")):
            sig = re.search(r"kernel void %s\((.*?)\)\s*\{" % kname, src, re.S).group(1)
            kinds = []
            for prm in [p.strip() for p in sig.replace("\n", " ").split(",")]:
                kinds.append("cl_mem" if "*" in prm else "cl_ulong" if prm.startswith("ulong") else prm.split()[0])
            sets = re.findall(r"clSetKernelArg\(%s, a\+\+, sizeof\((\w+)\)" % var, fn)
            if sets != kinds:
                b3 += fail(f"{kname}: the kernel takes {kinds}, the launch sets {sets}")
        print(f"the three parameter lists against the launch's clSetKernelArg sequences: {'ok' if b3 == 0 else 'FAIL'}")
        bad += b3
    # ---- 4. the block reads by pointer ----
    b4 = 0
    code = "\n".join(l.split("//")[0] for l in src.split("\n"))
    uses = re.findall(r"intel_sub_group_block_(?:read|write)\w*\(", code)
    if uses != ["intel_sub_group_block_read_us(", "intel_sub_group_block_write8(", "intel_sub_group_block_read8("]:
        b4 += fail(f"the kernel's block reads and writes by pointer are {uses}: the mask run's read, the state's write and read are the ones shown")
    for need in ("intel_sub_group_block_write8((global uint *) (st + c * QR * 16), as_uint8(o[c]))",
                 "intel_sub_group_block_read8((global const uint *) (st + c * QR * 16))"):
        if need not in code:
            b4 += fail(f"the state's block access is not `{need}`")
    if len(re.findall(r"global (?:const )?float \* st = ", code)) != 2:
        b4 += fail("st is not a float pointer in the two kernels that use it")
    if not re.search(r"if \(kg \+ 16 <= n_kv && mal\[j\]\) \{\s*mu\[j\] = intel_sub_group_block_read_us\(\(const global ushort \*\) \(mrow\[j\] \+ kg\)\);", code):
        b4 += fail("the mask run's block read is not under `kg + 16 <= n_kv && mal[j]`")
    if not re.search(r"#ifdef DEC27_RED_MASK_ALIGN\s*mal\[j\] = true;\s*#else\s*mal\[j\] = \(\(size_t\) mrow\[j\] & 3\) == 0;\s*#endif", code):
        b4 += fail("mal[j] is not `((size_t) mrow[j] & 3) == 0` (true in the red build 7 alone)")
    print(f"the block reads and writes by pointer (the mask run's under its 4-byte guard, the state's on float cells): {'ok' if b4 == 0 else 'FAIL'}")
    bad += b4
    print("DEC27-TEXT:", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

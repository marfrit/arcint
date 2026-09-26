## Sibling report: run-to-run nondeterminism in the chunked GatedDeltaNet path on Xe2 (`bmg-g21`, Arc Pro B60) — execution level, one f16 ulp

Offered as a **sibling** of this issue, not a duplicate. #38099 reports **deterministic wrong values** for chunk ≥ 2 in the GPU chunked GatedDeltaNet unroll. What we measured on Xe2 is **run-to-run nondeterminism** in the same kernel family — a different failure mode, and I am posting it here because the localisation overlaps.

### Environment

- OpenVINO **2026.4.0-22849-71640275d29** (custom build; GPU plugin `5a6968ec28b0f104`, core `d5a67095ead5fbdf`)
- IGC **2.38.2** (`libigc.so.2.38.2+1782393643`), `ocloc` from `/usr/bin`
- Kernel **7.0.14-12-pve** (Debian 13), `xe` driver (`srcversion 898B416572903DCE3B55D5E`)
- Device: **Arc Pro B60 = `bmg-g21` = OpenVINO `GPU.0`**, f16, `ocl::paged_gated_delta_net::opt`, prefill chunk 512
- Model: a 48-layer hybrid with **36 GDN (linear-attention) layers and 12 full-attention layers** (full attention every 4th layer); the cut arms below are **single 1024-token forwards** (unchunked), not the served chunk-512 prefill
- **Control:** the same bytes, the same request and the same harness on an **Arc A770 (`ACM-G10`, `GPU.1`)** are **bit-identical** across repeats. The defect is Xe2-specific.

### Observation

Two identical forwards of the same prefill differ on the B60, **only in the GDN path**:

- `gated_delta_state_table.0` post-forward differs across repeats (5 distinct hashes in one process; 1–4 among repeats in cold processes) — while `conv_state_table.0` post-forward is **one value on every forward of every process**.
- The differing elements are confined to **`dim0 = row 0`** and always the **same 14 of 48 heads**: `[3,5,6,7,10,13,17,22,31,39,41,42,43,47]`.
- Within a head, the diffs are **scattered** over the whole 128×128 plane — no contiguous tile/lane block.
- `max |diff| = 9.7656e-04` = **one f16 ulp**; only the **count** of flipped elements varies (2423..3924); the head set never changes.

### Negatives (all measured)

- **Not a JIT/compiler difference.** `ocloc` compiles the captured bucket `src_002.cl` for `bmg-g21` twice to **byte-identical** binaries: `sha256 be20f259e064e763c47090615ee013da61d00daa66e2f65bbd894e1fb944dc78`, 64,712 B.
- **Not inter-kernel ordering.** An `LD_PRELOAD` interceptor calling `clFinish(queue)` after **every** one of 233 kernel enqueues leaves the repeats differing.
- **Not the upstream dense GEMM.** A minimal same-shape f16 MatMul (1024×10240×2560) is bit-identical ×8.
- **Not launch geometry.** `PagedGatedDeltaNetBaseGenerator::get_dispatch_data_func` derives `wgs.global = {sequences, head_nums, v_blocks * subgroup_size}` and `wgs.local = {1, 1, subgroup_size}` from static shapes plus the arch subgroup size; no `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`, no occupancy query; `!params.is_dynamic()` is asserted.
- **Not the reduction lowering.** The disassembled `sub_group_reduce_add` is a fixed register-halving tree at both widths — xe2/16: `add(8) + add(4) + add(1) + add(1)`; xe_hp/8: `add(4) + add(1) + add(1)` — with no SLM, no `barrier`, no send-to-SLM.
- **Not fixable via subgroup width.** `xe2` *requires* 16: a kernel carrying `intel_reqd_sub_group_size(8)` fails to compile on `bmg-g21`, `bmg-g31`, `lnl-m`, `ptl-h` with *"Kernel compiled with required subgroup size 8, which is unsupported on this platform"*.
- **Inputs are bit-identical.** The request's nine input ports — including both state tables, zeroed before each forward — hash identically across 8 repeats while the output differs.

### Conclusion offered

The variance is at **execution** level on Xe2, below the kernel-choice level, in the GDN arithmetic. The instruments (input/output port digests, per-row state diff, the `clFinish` shim, the `ocloc` two-build diff) are in hand, and I can supply a minimal standalone reproducer on request.

### Dated correction 2026-09-25 — the A770's die label (NOT POSTED)

[code] The text above names the control card as `acm-g12`. That is wrong: the
A770 is **ACM-G10** (DG2-512, PCI `0x56A0`). `acm-g12` is a DIFFERENT DG2 die —
DG2-256, 16 Xe-cores — shipped in Arc Pro A60 / A570M / A530M. Both dies are
**Xe-HPG**, which is the distinction the control actually rests on, so **no
observation or number changes**: the A770 control still reproduces
bit-identically across repeats and the defect remains Xe2-specific.

Draft follow-up comment, to be appended to the issue:

    Correction: the control card above is an Arc A770 = **ACM-G10** (DG2-512,
    PCI 0x56A0). The text says `acm-g12`, which is a different DG2 die
    (DG2-256, shipped in Arc Pro A60 / A570M / A530M). Both are Xe-HPG, so the
    observation is unaffected: the A770 control still reproduces
    bit-identically across repeats, and the defect remains Xe2-specific. No
    numbers change.

STATUS: **POSTED 2026-09-26** —
<https://github.com/openvinotoolkit/openvino/issues/38099#issuecomment-5843016598>
(operator decision 2026-09-26). The draft above is what was posted, verbatim;
no numbers change and the observation stands as recorded.

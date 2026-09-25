# Quantisation support by architecture — Alchemist vs Xe2, a 2026-09-25 survey

Scope: which weight formats each of three Intel-GPU runtimes computes
**natively on the device**, and which it falls back on (host/CPU dequant,
emulation, or refusal), split by architecture: **Alchemist / Xe-HPG**
(Arc A770, 16 GiB) against **Xe2 / Battlemage** (Arc B580, Arc Pro B60,
Arc Pro B70). Runtimes: (a) the OpenVINO GPU plugin, (b) IPEX-LLM
(formerly BigDL-LLM), (c) llama.cpp's SYCL backend. Formats: grouped int4
(u4/i4), int8, NF4, MXFP4, NVFP4, IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S,
Q4_K/Q5_K/Q6_K/Q8_0.

Device-free: no card window, no host touched. Access date **2026-09-25**
for every source. Every row names its source key (§Sources, which gives
the URL pinned to a commit sha, and that commit's date) and its evidence
class: `code` = the source was read at the named commit, `paper` = docs,
release notes, an issue or someone else's measurement, `measured-here` =
this repository's own card windows. **UNSOURCED** marks a cell nobody
could source; it is never filled by inference. Vendor claims and
independent measurements are labelled apart.

Silicon identity (`code`, [K1]): the A770 (PCI `8086:56A0`) is in
`INTEL_DG2_G10_D_IDS`, i.e. **ACM-G10**; the B60 (PCI `8086:E211`) is in
`INTEL_BMG_G21_IDS`, i.e. **BMG-G21**. The B70 is BMG-G31 per its
reporter's hardware line in [L9] (`paper`). The runtimes identify the
architectures as follows. OpenVINO: `gpu_arch::xe_hpg` / `gpu_arch::xe2`
([O4]). llama.cpp: `intel_gpu_acm_g10` / `intel_gpu_bmg_g21`, `_g31`, the
DPC++ `sycl_ext_oneapi_device_architecture` names ([L3], [L6]);
`intel_gpu_dg2_g10/g11/g12` are that extension's aliases for
`acm_g10/g11/g12` ([L11]). IPEX-LLM:
the device strings `"arc"` / `"bmg"` ([I3]).

---

## The matrix

### (a) OpenVINO GPU plugin — upstream `master` at `fe27528` (2026-09-24)

| format | Alchemist (xe_hpg) | Xe2 (xe2) | path | class | source |
|---|---|---|---|---|---|
| grouped int4 (u4/i4) | **native** | **native** | `MarkDequantization` for `{i8,u8,i4,u4,u2}` is registered with no arch gate. Its fold flag is `!supports_immad`, and `supports_immad` is set from the device's `CL_DEVICE_FEATURE_FLAG_DPAS_INTEL`, not from the arch enum. The oneDNN micro-GEMM for gather-matmul and the MoE 3-GEMM carry an explicit `case gpu_arch::xe_hpg` beside `xe2`; the difference is the subgroup width (8 vs 16), not the format set `{u4,i4,u8,i8}` | `code` | [O1] L677, L699; [O4]; [O5] |
| int8 (u8/i8) | **native** | **native** | same passes and kernels as u4/i4 | `code` | [O1] L677; [O5] |
| NF4 | **not a device data type** | **not a device data type** | `data_types_are_supported`: `case nf4: return false`, with no arch condition (checked by the program builder). Whether an nf4 decompression chain is constant-folded to f16 before that check, and so executes as f16, was not traced (**UNSOURCED**). The docs place NF4 on Core Ultra Series 2 NPUs only ("The NF4 data type is only supported on Intel® Core Ultra Processors Series 2 NPUs … and beyond") | `code` + `paper` | [O2] L209; [O6c] L135 |
| MXFP4 (f4e2m1 + f8e8m0) | **UNSOURCED** — see note (1) | **UNSOURCED** — see note (1) | the code points both ways, and no arch-specific execution was traced. Against: the `MarkDequantization` for `{f8e4m3,f8e5m2,f4e2m1,f8e8m0}` is registered only when `use_onednn && arch >= gpu_arch::xe3p`, which is later than both `xe_hpg` and `xe2`; the same gate stands at arcint's pin `71640275d29` (2026-08-21). For: the type is accepted (`case f4e2m1: return true`); `FC_COMPRESSED_WEIGHT_PATTERN` accepts f4e2m1; `ConvertFullyConnectedToFullyConnectedCompressed` is registered with no arch gate; `DynamicQuantizeFullyConnected` handles f4e2m1 weights with f8e8m0 (MX) scales under `supports_immad && use_onednn`, with no arch-enum gate | `code` | [O1] L679–684, L1741, L1818; [O2] L210; [O4] L34–38; [O7] L627–631; [O10] |
| NVFP4 | **UNSOURCED** — no NVFP4-specific GPU path found | **UNSOURCED** | the only NVFP4 line in the release notes is under **NNCF** in 2026.1 ("Added experimental support for NVFP4 data type"), not under the GPU plugin | `paper` | [O6a] L655 |
| IQ4_NL / IQ4_XS / IQ3_XXS / IQ2_S (GGUF) | **refused at read** | **refused at read** | the GGUF frontend's `gguf_fill_sym`/`gguf_fill_asym` take Q4_0/Q8_0/Q5_0/Q6_K/Q3_K/Q8_K and Q4_1/Q4_K/Q5_K/Q5_1/Q2_K. Every other type hits `OPENVINO_THROW("Unsupported tensor type …")`. A case-insensitive search for `iq2`/`iq3`/`iq4` over `src/frontends` and `src/plugins/intel_gpu` returns nothing | `code` | [O3] L778–811 |
| Q4_K / Q5_K / Q6_K / Q8_0 (GGUF) | **native after conversion** | **native after conversion** | converted **at model read** into the u4/i4/i8 + f16-scale (+zp) compressed-weight form, then executed on the device by the int4/int8 rows above. The GGUF block itself is not what reaches the device | `code` | [O3] (`fill_q4_k`, `fill_q6_k`, `fill_q8_0`) |

Note (1). The f4e2m1/f8 decompression is *not marked* by the xe3p-gated
pass on xe_hpg or xe2. The compressed-FC pattern and the dynamic-quantize
pass still accept f4e2m1 there. **UNSOURCED:** which of the two decides
what an MXFP4 FullyConnected executes on either arch (a compressed FC, or
constant-folded f16 weights), and whether oneDNN's own f4_e2m1 kernels add
a separate floor. Neither was traced or measured, so both cells stay
UNSOURCED rather than "native" or "not native".

Plugin documentation (`paper`): the plugin-library guide on quantised
models (`advanced-guides/quantized-models.rst`, [O11]) describes the
FakeQuantize/low-precision flow and names no GPU architecture. No doc page
was found that lists weight precisions per GPU architecture. The 2026.3
release block (4 August 2026) was not surveyed line by line
(**UNSOURCED** for this table).

Vendor statements (`paper`, 2026.2 release notes of 28 May 2026 [O6b]):
"The GPT-OSS-20B model now supports INT8 weight precision and runs on …
Intel® Arc™ GPUs" (generation not stated). Also, in 2026.1 (7 April
2026, [O6a] L589): "Preview: Experimental L0 backend support for Xe2+
GPUs". That is a runtime backend, not a weight format.

### (b) IPEX-LLM — `intel/ipex-llm` at `de6bce2` (2026-01-28), ARCHIVED

**Status (`code`, GitHub API [I5]):** `intel/ipex-llm` is `archived: true`
(last push 2026-01-28), and so is `intel/intel-extension-for-pytorch`
(last push 2026-03-30). Every row below is the frozen state. The named
successor does not cover the A770. In intel/llm-scaler#283 ([I7],
`paper`), project contributors answer that llm-scaler "is primarily
designed for Intel Arc B60 (and future products)" (2026-02-10). To a
question about the A770 and the B580 (2026-03-09) they answer "So far
llm-scaler … supports B60 (single and multi-GPU)" (2026-03-18). A
further follow-up the same day, asking about plans for consumer A- and
B-series cards, got no answer from the project.

**Kernel visibility:** the GPU compute kernels are **not in the
repository**. `setup.py` pulls them in as prebuilt `bigdl-core-xe-*`
wheels ([I4]). "Native on the device" is therefore the project's own claim
(`paper`), and **no kernel body is `code`-verifiable** for any row.

| format | Alchemist ("arc") | Xe2 ("bmg") | path | class | source |
|---|---|---|---|---|---|
| grouped int4 (`sym_int4`=q4_0, `asym_int4`=q4_1, `woq_int4`) | offered; kernel in binary wheel | offered; kernel in binary wheel | `ggml_tensor_qtype`. In `use_batch_forward`, the int4 clause of `hard_condition` has no device list. At batch 1 the extra clause sends `SYM_INT4`/`WOQ_INT4` to the batch kernel on `"bmg"`. This is kernel selection, not a support gate | `code` (dispatch) / `paper` (kernel) | [I1] L27–58; [I2] L292–326 |
| int8 (`sym_int8`=q8_0) | offered; batch kernel eligible (`hard_condition` lists `arc`, `pvc`, `mtl`, `arl`) | offered; **never** takes the batch kernel, at any batch size (`bmg` is not in that list) | kernel selection, not a support gate | `code` / `paper` | [I1]; [I2] L310–316 |
| NF4 (`nf4`) | offered; arch not stated | offered; arch not stated | qtype 10 | `code` (list) / `paper` | [I1] |
| MXFP4 / NVFP4 | **absent** | **absent** | the strings do not occur anywhere in the tree (the nearest is `fp4`, qtype 16, not MXFP4) | `code` | [I1] |
| IQ4_NL / IQ4_XS / IQ3_XXS / IQ2_S | **absent from the `ggml_tensor_qtype` path** | **absent from the `ggml_tensor_qtype` path** | that path has only `gguf_iq2_xxs`, `gguf_iq2_xs`, `gguf_iq1_s`, `gguf_iq1_m`. The one IQ4_XS mention in the tree is a GGUF filename in a vLLM test patch (`docker/llm/serving/xpu/docker/vllm_for_multi_arc.patch`). The separate llama.cpp/Ollama portable-zip binaries ([I6]) were not examined for IQ support (**UNSOURCED**) | `code` | [I1] |
| Q4_K / Q5_K / Q6_K / Q8_0 | offered (`q4_k`, `q5_k`, `q6_k`, `sym_int8`); `Q4_K`/`Q6_K` batch-kernel eligible | offered; `Q4_K`/`Q6_K`/`SYM_INT8` never take the batch kernel on `bmg` | qtypes 27, 28, 26, 8; the same `hard_condition` device list as the int8 row | `code` (list, dispatch) / `paper` (kernel) | [I1]; [I2] L310–316 |

The llama.cpp portable-zip quickstart says it was "verified on Intel Arc
A-Series GPU / Intel Arc B-Series GPU", with no per-format split
([I6], `paper`).

### (c) llama.cpp SYCL — `ggml-org/llama.cpp` at `84e76d8a` (2026-09-24)

The scheduler's CPU fallback happens where `supports_op` answers false.
For `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` (the MoE expert matmul),
the only **type** it refuses is `TQ2_0`/`TQ1_0`. The other refusals are
shape/layout cases ([L2] L6457–6485). Every format below is therefore
**scheduled on the device**. It then goes through one of three paths:
- DMMV at batch 1;
- MMVQ at batch ≤ 8, with per-type `vec_dot_*_q8_1` kernels;
- for larger batches, on-device f16 dequant followed by the oneDNN/oneMKL
  GEMM.

MMQ is disabled for **every** type (`ggml_sycl_supports_mmq` returns false,
"TODO: accuracy issues in MMQ", [L3] L4025–4029).

| format | Alchemist (acm_g10) | Xe2 (bmg_g21/g31) | path | class | source |
|---|---|---|---|---|---|
| grouped int4 (Q4_0/Q4_1) | **native** | **native** | DMMV + MMVQ. The reordered MMVQ (Q4_0) is enabled on every Intel GPU (`reorder = ext_oneapi_architecture_is(intel_gpu)`). One arch carve-out: on `intel_gpu_acm_g10` with Q4_0, the line `use = use || !use_mul_mat_vec_q` is skipped, which changes the DMMV/MMVQ choice. Its comment reads "Arc770 get benefit with Q4_0 by skipping MMVQ path". A speed heuristic, not a support gate | `code` | [L3] L174, L4064–4078, L4825–4830 |
| Q8_0 | **native** | **native** | DMMV + MMVQ, reorder-optimised on both. Independent report: Q8_0 runs ~4× slower than Q4_K_M on a B70 ([L9]) | `code` / `paper` | [L3] L4064–4078; [L9] |
| NF4 | **no such type** | **no such type** | ggml has no NF4 type | `code` | [L4] (no `GGML_TYPE_NF4`) |
| MXFP4 | **native** | **native** | MMVQ `vec_dot_mxfp4_q8_1`, also in the MoE (`mul_mat_vec_q_id`) switch; f16 dequant for large batches. No arch gate | `code` | [L4] mmvq.cpp L2638, L2843; [L5] L730 |
| NVFP4 | **native** | **native** | MMVQ, including the MoE switch; f16 dequant. No arch gate | `code` | [L4] mmvq.cpp L2651, L2848; [L5] L732 |
| IQ4_NL | **native** | **native** | MMVQ `vec_dot_iq4_nl_q8_1`, in the dense and MoE switches; f16 dequant. **Not** in the reorder set, **no** DMMV | `code` | [L4] vecdotq.hpp L1694, mmvq.cpp L2622, L2888; [L5] L728 |
| IQ4_XS | **native** | **native** | MMVQ `vec_dot_iq4_xs_q8_1`; same shape as IQ4_NL | `code` | [L4] vecdotq.hpp L1721, mmvq.cpp L2625, L2893 |
| IQ3_XXS | **native** | **native** | MMVQ `vec_dot_iq3_xxs_q8_1`; f16 dequant | `code` | [L4] vecdotq.hpp L1546, mmvq.cpp L2616, L2868; [L5] L722 |
| IQ2_S | **native** | **native** | MMVQ `vec_dot_iq2_s_q8_1`; f16 dequant | `code` | [L4] vecdotq.hpp L1492, mmvq.cpp L2613, L2863; [L5] L720 |
| Q4_K / Q5_K / Q6_K | **native** | **native** | DMMV + MMVQ, reorder-optimised on both archs. The doc lists this under "2026.04-05 — Optimize mul_mat by reorder feature for data type: Q4_K, Q5_K, Q6_K, Q8_0" | `code` + `paper` | [L3] L4064–4078; [L1] L55–57 |

The arch branches found in the backend concern **attention and
allocation**, not weight formats (`code`):
- **Alchemist gate:** `fattn-onednn.cpp` refuses the oneDNN fused SDPA on
  `intel_gpu_dg2_g10/g11/g12` for head size 64 only: "This is the improved
  SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we
  instead target only the failing shapes" ([L11] L68–77).
- **Xe2-only:** the flash-attention selector picks the TILE kernel for
  quantised-KV decode only on `bmg_g21`/`bmg_g31` ("TILE is faster for
  quantized KV decode on Xe2 (BMG); keep VEC on untested archs", [L6]
  L264–266).
- **Xe2/LNL:** the FA vec kernel uses a 256-thread work group on
  `bmg_g21`/`bmg_g31`/`lnl_m`, and 128 elsewhere ([L12] L19–22).
- **BMG-G31 only:** the maximum buffer allocation is cut to 60 % of the
  reported size, a workaround for compute-runtime#998 ([L3] L1014–1030).

The doc's verified-hardware table names "Arc A770, Arc A730M,
Arc A750" and "Arc B580" as `Support`. It has no B60/B70 row and no
per-arch format notes ([L1] L135–136, `paper`).

### Reference row — arcint's own patched plugin (`measured-here`)

This is not one of the three charter runtimes. It is recorded because it
is the only OpenVINO path that computes GGUF IQ blocks at all. Patch 0043
admits IQ4_NL / IQ3_XXS / IQ4_XS / Q8_0 as native expert formats. Patch
0045 decodes IQ4_NL / IQ3_XXS / Q8_0 inside the per-expert OpenCL kernel.
The native per-expert route has served on **both** cards: the A770 at
0.44 t/s request wall, and the B60 at 0.6 t/s decode, Paris
(`docs/campaigns/sub4bit-vram-kernel.md`, 2026-09-21/22 entries; DESIGN
§7.0.2ce). As of the base commit `e8b9241` (2026-09-24) the IQ2_S
native expert block decodes bit-exact against the CPU reference (a design
note, device-free). No served IQ2_S reading is recorded here. The fused MoE route has served on the A770: the depth-12 fused artifact
(`d12r fused`) ran at 26.6 t/s warm at ratio 99 + tier
(`docs/campaigns/sub4bit-vram-kernel.md` L333). That route sits behind the
`supports_immad && use_onednn` MoE gate (L647 at the pin [O7]; L699 on
`master` [O1]; the arcint patch series was not checked for touching it), so this is indirect
evidence that the A770 takes the DPAS branch. The flag's runtime value was
not read directly.

---

## MXFP4 / NVFP4 — Alchemist at all, or Xe2-only?

- **OpenVINO GPU plugin: undecided on both archs (UNSOURCED).** Evidence
  against: the compressed-weight marking for f4e2m1/f8e8m0/f8e4m3/f8e5m2 is
  gated on `arch >= xe3p`, which is later than Xe2, both on `master`
  (`fe27528`) and at arcint's pin (`71640275d29`) ([O1] L679–684, [O7]).
  Evidence for: `FC_COMPRESSED_WEIGHT_PATTERN` accepts f4e2m1,
  `ConvertFullyConnectedToFullyConnectedCompressed` has no arch gate, and
  `DynamicQuantizeFullyConnected` handles f4e2m1 with f8e8m0 scales under
  `supports_immad && use_onednn` ([O1] L1741, L1818, [O10]). All `code`;
  what executes was not traced. NVFP4 appears only as an experimental NNCF
  data type (`paper`, [O6a]).
- **llama.cpp SYCL: both archs, no gate.** MMVQ and f16-dequant kernels
  exist for both types, with no arch condition (`code`, [L4], [L5]).
  gpt-oss-20b's MXFP4 GGUF runs at 60.95 t/s decode on an A770, but under
  **Vulkan**, not SYCL ([L8], `paper`).
- **IPEX-LLM: neither format exists** (`code`, [I1]).
- The "Battlemage (Xe2) only, for now" quote from the llama.cpp SYCL
  discussion concerns oneDNN fused SDPA. The string is gone at
  `84e76d8a`. Its successor is the Alchemist shape gate in
  `fattn-onednn.cpp`, whose comment calls itself "the improved SPDA gate"
  that no longer gates "Alchemist GPUs from all SPDA features" ([L11]).
  The original quote itself was not located in any tree (**UNSOURCED**),
  so "superseded" rests on that comment. This is attention, not a weight
  format.

## Upstream tracker state

- **openvinotoolkit/openvino#38099** ([O8], `paper`, read through the API
  2026-09-25). **OPEN**, label `PSE`, opened 2026-09-12, updated
  2026-09-24. Replies from the Intel side:
  - `YuChern-Intel`, 2026-09-16: "please provide us a minimal code sample
    and IR model". The reporter's self-contained reproducer followed the
    same day.
  - `diego-villalobos` (GitHub profile company: Intel), 2026-09-24:
    `issuecomment-5817418707`, body "Ref. 195655". This is an internal
    tracking reference whose meaning is **UNSOURCED**.
  - Assignments on the timeline: `YuChern-Intel` assigned the issue to
    themselves and to `Munesh-Intel` on 2026-09-16; `diego-villalobos`
    assigned it to himself on 2026-09-24.

  The sibling comment `issuecomment-5751935449` (2026-09-20) exists. No
  linked PR and no duplicate marking were found on the timeline. The two
  cross-reference events there are arcint's own commits.
- **llama.cpp SYCL tracker** ([L7]–[L10], `paper`):
  - #19918 "[SYCL][Intel] Low performance on MoE models — SYCL is slower
    than VULKAN (A770)": 2026-02-26, closed as stale 2026-04-26. A
    contributor's reply: "SYCL backend is slower than Vulkan is the
    truth … SYCL was decreased the performance in last year".
  - #21893: B70 weight corruption without `GGML_SYCL_DISABLE_OPT=1` with
    Q6_K, pointing at the reorder path. Opened 2026-04-14, closed
    2026-05-04. That variable does not appear in the tree at `84e76d8a`.
  - #21517: Q8_0 ~4× slower than Q4_K_M on a B70. Opened 2026-04-06,
    closed 2026-04-07.
  - No open issue was found that names an IQ format as the trigger of a
    wrong result on Xe2. The earlier survey's "garbled output on B580
    with these types" issue (`docs/campaigns/research-sub4bit-weights.md`)
    was not re-located in this pass and stays **UNSOURCED** here.

## Measured numbers for a ~35B-A3B MoE on the A770

Someone else's measurements (`paper`), with the card named in the source.
None of them is `measured-here`.

| model / quant | runtime | card | decode t/s | prefill t/s | who | date | source |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b MXFP4 (GGUF) | llama.cpp **Vulkan** b7189 / b7209, `-ngl 100`, `-fa 0`, pp512/tg128 | 1× A770 | 60.95 / 54.20 | 884 / 885 | independent | 2025-11-30 | [L8] |
| "gpt-oss-20b" (unsloth GGUF) and "Qwen3.5-35B-A3B" (lmstudio-community GGUF), quant not stated | llama.cpp **SYCL** b8157 vs **Vulkan**, Windows. b8157 still carried the SYCL IQ refusal (see the correction) | A770 (issue title); device count not stated. The same reporter runs a 4× A770 host ([L13]) | **10 (SYCL) vs 68 (Vulkan)** | 600 vs 1100 | independent | 2026-02-26 | [L7] |
| gpt-oss-20b int4 (OV IR) | OpenVINO GenAI / OpenArc, Windows | A770 | 13 | — | independent | 2026-04-07 | [O9] |
| gpt-oss-20b int4 | OpenVINO GenAI 2026.2 nightly `dev20260427` | A770 | ~36 (Arc Pro B50: ~58) | — | **Intel-side** (repo contributor) | 2026-04-27 | [O9] `issuecomment-4329922747` |
| gpt-oss-20b | OpenVINO GenAI | "A770 (8GB)" | 15 | — | Intel-side (reporter questions it: the int4 model exceeds 8 GB) | 2026-04-14 | [O9] `issuecomment-4241274382` |
| Qwen3.5-35B-A3B (lmstudio-community) Q4_K_M / Q6_K / Q8_0 | llama.cpp **Vulkan** b8149, `-ngl 100`, pp512/tg128, Windows | **multi-GPU A770**: Q8_0 on 3× ("Found 3 Vulkan devices"), Q6_K and Q4_K_M on 2× (`GGML_VK_VISIBLE_DEVICES=2,3`), so the three quants are not like-for-like. The issue title names Qwen3-Coder-30B-A3B; the commands load Qwen3.5-35B-A3B | 49.18 / 44.18 / 38.88 (fa 0) | 242.54 / 175.02 / 612.14 | independent | 2026-02-25 | [L13] |
| gpt-oss-20b GGUF | llama.cpp SYCL vs Vulkan | **2× A770**, PCIe 3.0 | 25 (SYCL, stable) vs 26 → 10 (Vulkan, after 200 tokens) | — | independent | 2026-03-12 | [L7] `issuecomment-4045590622` |

Caveats:
- **[L7] is not a clean 35B-A3B row.** The issue lists **two** models,
  and its single text line "PP: 600 vs 1100 t/s / TG: 10 vs 68 t/s" does
  not say which model it belongs to. The per-model figures exist only in
  screenshots. It is A770 evidence for *a* 20–35B MoE on SYCL vs Vulkan;
  it is **not** A770 evidence for Qwen3.5-35B-A3B specifically.
- **No single-A770 row names Qwen3-30B-A3B or a 35B-A3B unambiguously**,
  in any format. That stays **UNSOURCED**. [L13] names Qwen3.5-35B-A3B,
  but on two or three A770s.
- **No IQ-format tokens/s on an A770 exists in any source found.**
  (Arc Pro B70 has one, IQ4_XS: Qwen3.5-27B dense at 17.52 t/s tg128,
  single GPU, llama.cpp SYCL, [L9]; Xe2, dense, not A770.)
- Xe2 comparison rows exist (for example Qwen3.6-35B-A3B Q4_K_M on a B70,
  llama.cpp Vulkan, 76 t/s after a Mesa upgrade; Qwen3-30B-A3B int4 on a
  B60 under OVMS, 67.95 t/s). Both come from independent blogs
  ([X1], [X2], `paper`, cross-runtime, self-described "directional").
- Estimator pages that print tokens/s without a measurement were
  excluded.

---

## Dated correction — the SYCL/IQ claim (2026-09-25)

**Claim as inherited:** "IQ formats have no SYCL kernels → host/CPU
dequant → 8–12 t/s".

**Correction:** at llama.cpp `84e76d8a` (2026-09-24) the claim is
**false**. Before 2026-03-22 it was **partly true**. Both halves are `code`.

- **Today.** Every IQ type has an MMVQ kernel, including in the MoE
  `MUL_MAT_ID` switch, plus a device f16-dequant, and `supports_op` never
  refuses it for `MUL_MAT`/`MUL_MAT_ID` ([L2], [L4], [L5]). This holds on
  Alchemist and Xe2 alike. The kernels themselves have been listed since
  the doc's "2024.4 — Support data types: GGML_TYPE_IQ4_NL,
  GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS,
  GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M"
  ([L1] L89–90, `paper`).
- **Until 2026-03-22.** `supports_op` returned **false** for every IQ type
  (IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S, IQ2_*, IQ1_*) whenever
  `b->ne[1] == 1 && ggml_nrows(b) > 1`. Such ops were scheduled off the
  SYCL device. The dates differ by op:
  - **`MUL_MAT`:** refused from at least b4500 (2025-01-17), the earliest
    tag read.
  - **`MUL_MAT_ID` (the MoE expert matmul):** refused only from
    2025-08-12 on. Before that, its branch tested `a = op->src[2]`, the I32
    ids tensor, so the IQ test could not fire. That was replaced by
    `a = op->src[0]` between #15092 (`3306ceab`, 2025-08-05) and #15151
    (`f4586ee5`, 2025-08-12).
  - **Removal, both ops:** the refusal is present at b8157 (the build of
    [L7]) and b8460. It was removed by ggml-org/llama.cpp#20803 "[SYCL]
    Support bf16 and quantized type of MUL_MAT", merged 2026-03-22
    (commit `f40a80b`), and is absent from b8480 on ([L14]).
- **Not traced.** Which model shapes met that condition (for example, the
  MoE `MUL_MAT_ID` at prefill against decode) was not traced, so how much
  of a given MoE forward fell back is **UNSOURCED**.

What the claim therefore gets wrong:
- It is wrong for any build after #20803 ("no SYCL kernels" was never
  literally true).
- Its **8–12 t/s** has no source tying it to the fallback. The nearest
  number, 10 t/s on an A770 under SYCL b8157 ([L7]), was measured on a
  pre-#20803 build, with the refusal in place. No source separates the
  refusal's share of that figure from SYCL's general speed on the A770.

What remains true today:
- IQ types get neither the reorder optimisation nor the DMMV path ([L3]).
- MMVQ quantises activations to q8_1.

Where the claim stood: the handoff for this survey names operator-local
packets. A literal search of this tree's `*.local.md` files on 2026-09-25
did not find the sentence, so this tracked statement is the correction of
record. A dated pointer to it was appended to the survey's operator-local
handoff packet.

---

## What this changes for the A770 performance option

1. **No off-the-shelf runtime computes IQ-class expert blocks natively on
   the A770 inside OpenVINO.** Upstream OpenVINO refuses them at GGUF
   read, and IPEX-LLM is archived with no IQ2_S/IQ3_XXS/IQ4_* in its
   `ggml_tensor_qtype` path at the frozen commit.
   arcint's patch series is the only OpenVINO path, and it serves on the
   A770 (`measured-here`, reference row).
2. **llama.cpp SYCL *does* compute IQ4_NL/IQ3_XXS/IQ2_S on the A770**
   in builds after #20803 (`code`). That makes it a *candidate*
   correctness comparand for the native route. It is not measured here
   on the A770. It cannot be bit-exact against an f16-activation kernel
   (MMVQ quantises activations to q8_1), and the non-fused MUL_MAT_ID path
   copies the ids to the host and waits on the stream ([L3] L5209–5216). As a *speed* comparand
   it is weak on the A770 (one report: 10 vs 68 t/s against Vulkan,
   [L7]), and no IQ-format A770 number exists to anchor it. An A770 speed
   bar for a native-format MoE has to be measured here; it cannot be
   borrowed.
3. **MXFP4/NVFP4 in OpenVINO: undecided on both archs.** One marker pass
   is gated to `xe3p` (later than Xe2), while the compressed-FC and
   dynamic-quantize paths accept f4e2m1 with no arch gate (`code`). What
   executes on either card is **UNSOURCED**. Deciding it takes a device
   leg, not more reading. In llama.cpp SYCL both formats have kernels on
   both archs (`code`).
4. **In the code read, the Alchemist/Xe2 splits are kernel-selection,
   attention and workaround branches, not format support.** Examples, not
   a census:
   - OpenVINO: micro-GEMM subgroup width 8 vs 16; LoRA horizontal fusion
     off on xe2 ("Temporary disabling for BMG due to regression", [O1]
     L1759–1761); int8 per-token dyn-quant forced to gs128 for `>= xe2`
     ([O1] L1828); the CM bidirectional-LSTM path for xe2 only ([O1] L1087).
   - llama.cpp SYCL: the Q4_0-on-`acm_g10` heuristic; the Alchemist
     head-64 oneDNN SDPA gate; the Xe2 FA TILE selector and work-group
     size; the BMG-G31 allocation cap.
   - IPEX-LLM: `bmg` excluded from the int8/K-quant batch kernel.

   No weight format was found that is Xe2-only in any of the three
   runtimes at the commits read (`code`). This negative covers the files
   named here only. oneDNN's vendored kernels and IPEX-LLM's binary
   wheels were not read.

## UNSOURCED

- What the OpenVINO GPU plugin executes for MXFP4 (f4e2m1 + f8e8m0) and
  f8 weights on xe_hpg and on xe2, and whether oneDNN's f4_e2m1 kernels
  carry their own arch floor (both MXFP4 cells).
- Any NVFP4-specific path in the OpenVINO GPU plugin.
- IPEX-LLM kernel bodies for every format (binary wheels), and hence
  native-vs-dequant for each of its rows on either arch. IQ support in
  IPEX-LLM's llama.cpp/Ollama portable binaries.
- Whether NF4 weights reach the OpenVINO GPU as f16 after constant
  folding.
- Which MoE shapes hit the pre-#20803 SYCL IQ refusal, and its share of
  any published A770 figure.
- An Intel announcement URL for IPEX-LLM's archival (only the GitHub
  `archived` flag was read).
- Any Intel-named successor to IPEX-LLM for A770 users (see the llm-scaler
  answer under (b): B60 only).
- The PR numbers that landed SYCL IQ kernels (2024.4) and SYCL fused MoE
  (2026.04-05).
- The current existence of `GGML_SYCL_DISABLE_OPT` ([L10] references it;
  absent at `84e76d8a`).
- The original "Battlemage (Xe2) only, for now" quote: not found at
  `84e76d8a`; its successor gate is cited ([L11]).
- An IQ-format wrong-output issue on Xe2 under SYCL (the 2026-09-05
  survey's B580 report was not re-located).
- A770 tokens/s for Qwen3-30B-A3B / Qwen3.5-35B-A3B / Qwen3.6-35B-A3B
  attributable to one model; any IQ-format tokens/s on an A770.
- The meaning of Intel's "Ref. 195655" on #38099.

---

## Sources

All accessed 2026-09-25. Commit dates are the committer dates of the
pinned sha.

| key | what | URL | date |
|---|---|---|---|
| O1 | OV GPU `transformations_pipeline.cpp` (L677–699, L1087, L1741, L1759–1761, L1818–1828) | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp#L677-L699 | 2026-09-24 |
| O2 | OV GPU `common_utils.cpp` (`data_types_are_supported`) | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/plugin/common_utils.cpp#L209-L210 | 2026-09-24 |
| O3 | OV GGUF frontend `gguf_quants.cpp` | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/frontends/gguf/src/quant/gguf_quants.cpp#L778-L811 | 2026-09-24 |
| O4 | OV `device_info.hpp` (arch enum, `supports_immad`); `ocl_device.cpp` L316 (DPAS flag) | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/include/intel_gpu/runtime/device_info.hpp#L34-L38 · https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/runtime/ocl/ocl_device.cpp#L316 | 2026-09-24 |
| O5 | OV micro-GEMM: `gather_matmul_gen_micro.cpp` L40–53, `moe_3gemm_gen_micro.cpp` L25–59 | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul/gather_matmul_gen_micro.cpp#L40-L53 · https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_gen_micro.cpp#L25-L59 | 2026-09-24 |
| O6a | OV release notes, 2026.1 block (NNCF NVFP4 L655; L0 Xe2+ L589) | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/docs/articles_en/about-openvino/release-notes-openvino.rst#L538-L660 | release 2026-04-07 |
| O6b | OV release notes, 2026.2 block (GPT-OSS-20B INT8 on Arc, L363) | same file, #L309-L363 | release 2026-05-28 |
| O6c | OV GenAI-on-NPU doc (NF4 NPU-only) | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/docs/articles_en/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.rst | 2026-09-24 |
| O10 | OV `compressed_weights_pattern.hpp` (f4e2m1 in `FC_COMPRESSED_WEIGHT_PATTERN`); `dynamic_quantize_fully_connected.cpp` L99–111 (f4e2m1 + f8e8m0 scales) | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/plugin/transformations/compressed_weights_pattern.hpp#L10-L17 · https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp#L99-L111 | 2026-09-24 |
| O11 | OV plugin-library guide, quantised models | https://github.com/openvinotoolkit/openvino/blob/fe2752839c9e5f2d17b6a027415d1581bd3cbd40/docs/articles_en/documentation/openvino-extensibility/openvino-plugin-library/advanced-guides/quantized-models.rst | 2026-09-24 |
| O7 | OV `transformations_pipeline.cpp` at arcint's pin | https://github.com/openvinotoolkit/openvino/blob/71640275d29692354223572e77ef717e92b891d9/src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp#L627-L631 | 2026-08-21 |
| O8 | openvino#38099 | https://github.com/openvinotoolkit/openvino/issues/38099 | 2026-09-12 … 2026-09-24 |
| O9 | openvino#35199 "GPT-OSS 20B A770 Low performance MoE" (open) | https://github.com/openvinotoolkit/openvino/issues/35199 | 2026-04-07 … 2026-04-28 |
| L11 | `fattn-onednn.cpp` Alchemist SDPA shape gate L68–77 | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/fattn-onednn.cpp#L68-L77 | 2026-09-24 |
| L12 | `fattn-vec.hpp` Xe2/LNL work group L19–22 | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/fattn-vec.hpp#L19-L22 | 2026-09-24 |
| L13 | llama.cpp#19887 (multi-A770 Vulkan, Qwen3.5-35B-A3B Q4_K_M/Q6_K/Q8_0) | https://github.com/ggml-org/llama.cpp/issues/19887 | 2026-02-25, closed 2026-04-12 |
| L14 | the pre-2026-03-22 SYCL IQ refusal and its removal: `supports_op` at b4500 and b8157; the MUL_MAT_ID `src[2]`→`src[0]` change between #15092 and #15151; PR #20803 | https://github.com/ggml-org/llama.cpp/blob/b4500/ggml/src/ggml-sycl/ggml-sycl.cpp · https://github.com/ggml-org/llama.cpp/blob/2943210c1eaa3fc9cc4f0ac6f0ae5f2ce2350f98/ggml/src/ggml-sycl/ggml-sycl.cpp#L4629-L4638 · https://github.com/ggml-org/llama.cpp/pull/20803 | 2025-01-17; b8157; merged 2026-03-22 |
| L1 | llama.cpp `docs/backend/SYCL.md` (news L55–57, L89–90; hardware L135–136) | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/docs/backend/SYCL.md | 2026-09-24 |
| L2 | `ggml-sycl.cpp` `supports_op` MUL_MAT/MUL_MAT_ID | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/ggml-sycl.cpp#L6457-L6485 | 2026-09-24 |
| L3 | `ggml-sycl.cpp` reorder flag L174, BMG-G31 alloc cap L1014–1030, MMQ off L4025–4029, reorder set L4064–4078, `acm_g10` Q4_0 L4825–4830 | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/ggml-sycl.cpp | 2026-09-24 |
| L4 | `mmvq.cpp` switches L2601–2651, L2843–2893; `vecdotq.hpp` L1407–1721; `ggml.h` type list | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/mmvq.cpp · https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/vecdotq.hpp | 2026-09-24 |
| L5 | `convert.cpp` `ggml_get_to_fp16_sycl` (from L656) cases L720–732 | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/convert.cpp#L656-L732 | 2026-09-24 |
| L6 | `fattn.cpp` Xe2 TILE selector L264–266 | https://github.com/ggml-org/llama.cpp/blob/84e76d8a23162eca70490da131945ebec1f09bf4/ggml/src/ggml-sycl/fattn.cpp#L264-L266 | 2026-09-24 |
| L7 | llama.cpp#19918 (A770, SYCL vs Vulkan, MoE) | https://github.com/ggml-org/llama.cpp/issues/19918 | 2026-02-26, closed 2026-04-26 |
| L8 | llama.cpp#17628 (A770 Vulkan, gpt-oss-20b MXFP4) | https://github.com/ggml-org/llama.cpp/issues/17628 | 2025-11-30 |
| L9 | llama.cpp#21517 (B70, SYCL, Q8_0 vs Q4_K_M vs IQ4_XS) | https://github.com/ggml-org/llama.cpp/issues/21517 | 2026-04-06 |
| L10 | llama.cpp#21893 (B70 SYCL reorder corruption) | https://github.com/ggml-org/llama.cpp/issues/21893 | 2026-04-14 |
| I1 | ipex-llm `ggml/quantize.py` (`ggml_tensor_qtype`) | https://github.com/intel/ipex-llm/blob/de6bce27133ab250f13fd5d549c197519ce16d30/python/llm/src/ipex_llm/ggml/quantize.py#L27-L58 | 2026-01-28 |
| I2 | ipex-llm `low_bit_linear.py` `use_batch_forward` | https://github.com/intel/ipex-llm/blob/de6bce27133ab250f13fd5d549c197519ce16d30/python/llm/src/ipex_llm/transformers/low_bit_linear.py#L292-L326 | 2026-01-28 |
| I3 | ipex-llm `transformers/utils.py` device names | https://github.com/intel/ipex-llm/blob/de6bce27133ab250f13fd5d549c197519ce16d30/python/llm/src/ipex_llm/transformers/utils.py#L174-L182 | 2026-01-28 |
| I4 | ipex-llm `setup.py` (binary `bigdl-core-xe-*` wheels) | https://github.com/intel/ipex-llm/blob/de6bce27133ab250f13fd5d549c197519ce16d30/python/llm/setup.py#L293-L322 | 2026-01-28 |
| I5 | GitHub API: `intel/ipex-llm`, `intel/intel-extension-for-pytorch` (`archived: true`) | https://api.github.com/repos/intel/ipex-llm · https://api.github.com/repos/intel/intel-extension-for-pytorch | pushed 2026-01-28 / 2026-03-30 |
| I6 | ipex-llm llama.cpp portable-zip quickstart | https://github.com/intel/ipex-llm/blob/de6bce27133ab250f13fd5d549c197519ce16d30/docs/mddocs/Quickstart/llamacpp_portable_zip_gpu_quickstart.md | 2026-01-28 |
| I7 | intel/llm-scaler#283 "Is llm-scaler a replacement for IPEX-LLM?" (open) | https://github.com/intel/llm-scaler/issues/283 | 2026-02-09; contributor replies 2026-02-10, 2026-03-18 |
| K1 | Linux `include/drm/intel/pciids.h` (DG2 G10 / BMG G21 ids) | https://github.com/torvalds/linux/blob/ca24e8d9fa48c7c121614c1a80971aecda640674/include/drm/intel/pciids.h | 2026-06-09 |
| X1 | B70 llama.cpp benchmarks (independent blog; Qwen3.6-35B-A3B Q4_K_M, Vulkan, Mesa ≥ 26.1, tg128 76.0) | https://jonathanmann.tech/blog/intel-arc-b70-llama-cpp-benchmarks/ | 2026-06-22, updated 2026-07-31 |
| X2 | "OpenVINO beats Vulkan on Arc B60" (independent blog; OVMS 2026.2.1 Qwen3-30B-A3B-Instruct-2507 INT4 67.95; Vulkan Qwen3-Coder-30B-A3B Q4_K_M 38.6; "directional, not precise") | https://localaifrontier.com/blog/openvino-beats-vulkan-on-arc-b60/ | 2026-07-18 |

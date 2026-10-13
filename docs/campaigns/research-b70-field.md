# Research: the B70 projects (Arc Pro B70, Xe2), read at source

Prompted by "The B70 Field Watch" No. 1 (2026-10-09). Four checkouts under
`~/src/ref-b70`, read 2026-10-10:
- `qwen38-flash-next-b70-offload` (sybil-solutions, `db27d3c`);
- `exl3xpu` (`15ded2f`);
- `qwen38-b70` (`0f9950b`);
- valarauca's llama.cpp SYCL rework (PR #1, branch `pr1`, `f7440ea`).

Mechanisms below are `code` (file:line in those trees). Every throughput or
KL figure is the authors' (`paper` for arcint).

**The card:** the B70 has 32 GB and a 608 GB/s spec, never measured in these
repos. Spec ratios to the B60: about 0.75x bandwidth and about 0.6x Xe2
cores. Scaled numbers below are estimates.

## What they measured

- **Flash-Next on one B70, experts on NVMe** (EXL3 3.05 bpw, SGLang +
  exl3xpu, 4x NVMe RAID0, PCIe 4.0 x16, no MTP; `docs/results.md:16-32`,
  2026-10-08).
  - Decode 25.6-30.6 t/s C1, prefill 1,135-1,193 t/s at 8k / 32k.
  - Decode is approximate: a pick in neither VRAM nor RAM becomes a
    zero-weight safe expert, 5-135 masked picks a step. Decode KL against
    the server's own prefill is 0.021-0.055 (top-20 KL).
  - The data race: the device sets RAM residency while the host punches the
    page (`experimental/n111-victim-ring/STATUS_N111.md`).
- **exl3xpu, dense Qwen3.8-27B** (EXL3 4.00 bpw, vLLM XPU, MTP k=3, fp8 KV;
  `recipe.json`, `README.md:39-42`).
  - Decode 63-91 t/s C1, depending on the workload.
  - Cold prefill 2,259 t/s at 32K and 911 at 254K with int8 x int8 oneDNN
    GEMMs and oneDNN SDPA (fp16 path: 1,434 / 650).
  - The int8 prefill's top-1 agreement against fp16 is 97.17 %.
  - No KL against BF16 anywhere.
- **valarauca:** llama-bench at depth 0 only. 27B Q4_0 pp512 354 -> 1,370;
  UD-IQ4_XS tg128 18.6 -> 25.4. KL against the CPU in commit messages.
  - The "4-5.5x" headline is in no file.
  - The f32-GEMM cause is real at the merge base (`ggml-sycl.cpp:2999-3003`):
    upstream SYCL ran prefill without XMX unless built with
    `GGML_SYCL_F16`.

- **Stock vLLM, dense Qwen3.8-27B, all-INT4** (a forum post of 2026-10-11,
  `Lets-Fafo/Qwen3.8-27B-FAFO-W4A16-G32-B70` on Hugging Face; Intel's
  `llm-scaler-vllm:0.26.0-b2` container, no custom kernels, eager mode, MTP
  off; all `paper`, no code read).
  - Weights: W4A16, group 32, BF16 scales, all 400 text projections, from
    the BF16 checkpoint without calibration; 18.66 GiB on the card;
    perplexity +3.3 % against BF16 on a small set (an INT4/FP8 mix: +0.25 %).
  - One stream: decode 26.4 t/s at 512 in, 20.2 t/s at 110k; prefill from
    TTFT about 1,670 t/s at 8,192, 1,517 at 32,768, 1,354 at 65,536, 1,173 at
    110,000 tokens.
  - The headline "1,798 t/s" is total throughput (prompt plus output) at 32
    clients with prefix caching; 296.5 output t/s sustained at 64 clients.
  - "ActiveKV" (KV kept on the host, the GPU holding a working set; peak
    31.89 -> 23.14 GiB, "exact outputs") is announced, without code or
    method.
  - A commenter's two B70s (FP8, TP2, MTP 2): prefill 2,910 t/s at 16.5k,
    1,533 at 131k, 1,213 at 197k; decode 91 -> 56 t/s over that range.

## Against arcint (B60)

- **Flash-Next prefill is our gap.**
  - The B70 recipe's 1,056-1,332 t/s scales to roughly 650-800 t/s on a
    B60. Strata measured 620 on our own card.
  - arcint is at 422 at 20k.
  - The references put their rate in non-MoE work and in large chunks:
    - sparse attention per query: 255 / 459 -> 56 / 66 ms a layer at
      8k / 32k, +32 % prefill;
    - fused hyper-connections: 945 -> 366 ms an 8k forward;
    - 8k-token chunks with every non-resident expert staged once a chunk
      into two VRAM buffers. MoE was 1.24 s of a 5.4 s 8k forward.
- **Dense 27B prefill:** arcint's 664 at 20k is about 70-80 % of the B70's
  fp16 path scaled, and above llama.cpp on the B70 scaled.
- **Dense 27B prefill at depth is our gap** (2026-10-11/12, arcint
  `measured-here` on the B60, MTP 5, q8_0:q4_0 KV; the stock-vLLM B70 numbers
  above, `paper`): 787-837 t/s at 4k against ~1,670 at 8k (about 1.3x after
  the B70's ~1.6x compute), 256 t/s at 128k against ~1,170 at 110k (about
  3x), 157 t/s at 237k. The gap grows with depth, so it is attention, not the
  GEMMs (`agent-context.md`).
- **Decode: no contradiction.**
  - Flash-Next: arcint 33-35 t/s exact against the B70's 25.6-30.6,
    approximate and without MTP.
  - Dense: exl3xpu with MTP scales to about 45-65 t/s on a B60, against
    arcint's 32 with MTP 5. The difference points at verify cost and
    acceptance, not at the weight kernel.

## Candidates for arcint (OpenCL), ranked

**Prefill:**
1. **Measure first:** bytes crossing PCIe a prefill token at arcint's ubatch
   (at 512 tokens x top-10 nearly every expert is touched every ubatch);
   whether the OpenCL copy queue overlaps compute (the B70 recipe measured
   no overlap within one process, `research/18-b70-ingest-ceiling.md:18-20`);
   and how the 12 sparse-attention layers run in arcint's llama.cpp path.
2. **Large chunks with per-layer expert staging:** read-ahead by layer, two
   VRAM staging buffers, a device pointer table, a grouped DPAS that decodes
   each expert once per 32-row block (`nvtier.py:529-640`,
   `exl3_moe.a8.sycl:398-635`).
3. **Strata's per-query split-K sparse attention.** fp32 SIMT, so it ports
   to OpenCL (`kernels/n107-qsa/csrc/qsa_row_sycl.sycl`; source of record:
   Strata `qsa_decode_attn.dp.cpp`).
4. **Hyper-connection fusion.**

**Decode:**
1. **Drop the `ksigns64` table from IQ2_XS / IQ3_XXS** for register sign
   masks and aligned SoA fields (valarauca `vecdotq.hpp:634-642`). IQ2_XS
   GEMV 214 -> 59 us on the B70. arcint's 0030 still reads the table. Cheap.
2. **IQ4 codebook in registers** (`iq4nl.hpp:14-44`): -22 % kernel time.
3. **MTP verify cost:** M <= 8 at about M=1 through DPAS or a multi-column
   GEMV; a pruned draft vocabulary.

**Not to copy:** the masked-pick tier (approximate); device-set residency
(the race); the memfd + SVM RAM tier (synchronous in-kernel reads); the EXL3
format (ALU-bound on the B60 at its trellis decode, a new loader, no BF16
KL).

**Runtime note:** compute-runtime 26.35 aborts every OpenCL and Level Zero
program on a B580 (Fedora 44, reported 2026-10-03). Hold 26.27.

Related: `research-infernix.md`, `strata-sycl-b60.md`,
`flash-next-llama-engine.md`, `prefill-expert-streaming.md`.

# Campaigns — one lever, one document

Every open defect or lever is a **campaign**: one document in this
directory, sufficient for a fresh session together with the DESIGN sections
and reference sources it cites. A closed campaign is a stub that says what
exists, where, and its number; its full history is in git. The campaigns of
the retired OpenVINO engine are at the tag `openvino-history`
(`docs/history.md`).

## Rules

- **One campaign = one defect or one lever, with its own gate.** Two levers
  with different owners or gates are two campaigns.
- **References first.** An open campaign names the reference implementation
  to follow, with source paths in `~/src/Strata-ref` or
  `~/src/FreeToken-ref` (or llama.cpp), the reference's measured effect, and
  its evidence class. Build the reference's mechanism, then measure it. See
  `research-reference-audit.md` and `CLAUDE.md` ("References first").
- **No "A but B".** A document never lists a known-working reference
  technique beside a reason arcint did not or cannot do it. An arcint
  deviation the operator decided is one sentence with its date.
- **The gate is on the record before the work starts**, and it can fail.
  Equivalence and quality gates use `CLAUDE.md`'s answer-level bar; speed
  gates are per phase (the target phase improves, the other stays within the
  run-to-run spread).
- **Evidence class on every disposition:** `paper`, `code` or
  `measured-here`.
- **The document is sufficient, and nothing in it is operator-local:**
  hosts, paths and unit names stay in the git-ignored `*.local.md` files.
- **Pipeline:** recon (the cited sources and the code) → design note when the
  change is more than a fix → red-first implementation → one card window at
  the end → review before commit → the patch README's section, a DESIGN §7
  line where the state of play moves, and a CHANGELOG line when it closes. Reviews are not skippable.
- **Invariants are not negotiable by an agent:** DESIGN §3.4 and §3.8, the
  §5 ladder, the measurement discipline in `CLAUDE.md`. A campaign that
  would trade one for a number records the trade, with the measured price of
  each side, and puts it to the operator. [Amended 2026-10-01. §3.4 was
  relaxed, and correctness is now judged at the answer, not at the bit
  (DESIGN §3.4 (amended 2026-10-01); `CLAUDE.md`). A reference mechanism that
  conflicts with an invariant is the operator's call
  (`research-reference-audit.md`).]
- **Status is the current state**, a few lines, rewritten as the campaign
  moves; a closed campaign becomes a stub. [Changed 2026-10-01 from an
  appended dated log; the logs up to b0447b8 are in git.]
- **Releases collect closed campaigns.** No release waits for a campaign,
  and no campaign is started to fill a release.

## Template

    # <slug> — <one-line charter>
    **Open** | **Closed <date>.** (one line)
    ## Charter
    ## Reference to follow      (source paths, measured effect, evidence class)
    ## Gate                     (answer-level bar; per-phase speed)
    ## Current state            (measured-here numbers with card and configuration)
    ## Where it lives

## Open

The build order of 2026-10-01 (the adaptive expert cache with its miss
split and a pinned bank, the doorbell, multi-draft MTP, prefill on the GPU)
is built on the libllama engine: 0021/0042 (0055's LFRU measured and dropped), 0056, `--llama-mtp`,
0032/0044. What is open follows the charter (`docs/milestone-1.0.0.md`) and
the audit's ranking (`research-reference-audit.md` §4).

| campaign | state | reference to follow |
|---|---|---|
| [flash-next-llama-engine](flash-next-llama-engine.md) | expert cache (0021): UD-Q3_K_XL 235 / 12.4-13.0 t/s; IQ2_XS (0022) 337 / 16.0 t/s, answers right; MTP via `--llama-mtp-gguf` 13.8 t/s (a verify ~3 steps; grouped verify next); 0042-0047 on decode (the swaps behind the graph, short prompts, the tables in one write, the inputs without a wait each: 33.1-37.3 t/s against Strata's 37.2-37.8 on this card, the rest is tokens a round) | Strata `src/core/expert_cache.cpp`, `generate.cpp:4413-4482`, `expert_source.cpp:1614-1696`; llama.cpp `--n-cpu-moe`, CPU `MUL_MAT_ID` |
| [strata-sycl-b60](strata-sycl-b60.md) | Strata's own engine (its SYCL port, upstream `sycl/`) on the B60 (IQ2_XS, MTP): 620.5 t/s prefill at 20k, 37.2-37.8 t/s decode, answers right, against arcint's 235 / 12.5-13.0 (`measured-here`); owed: IQ2_XS quality against UD-Q3_K_XL | Strata `sycl/`, `docs/INTEL.md` (`7ba023e`) |
| [llama-engine-kernel-gap](llama-engine-kernel-gap.md) | the libllama engine (patches 0001-0007) at 19.75 t/s decode / 521 t/s prefill (dense, B60) and 47.8 / 1,286 (coder, A770; 47.4 decode at 4k), against the retired OpenVINO engine's 24.0 / 1,141 and 43.9 / 1,379; open: prefill attention, dense decode bytes, launches; lever 7 (the int8 Q4_K GEMM, 84.5 TOPS): CUDA MMQ's shared decoded tile built and measured 46-59 % slower (2026-10-14), at its ceiling for this design: the per-(token, column) scale binds | OpenVINO `paged_gated_delta_net_opt.cl`, `paged_attention_opt.cl`, `moe_router_fused.cl`; arcint patch 0064; Strata `src/prefill/kernels.cu:380-423` |
| [dense-prefill-attention](dense-prefill-attention.md) | the dense agent's prompt attention at depth (B60): 56 % of device time at 58k; 0037 (quantized K/V to f16 in key blocks) and 0038 (vLLM's head-256 kernel with 2D block reads, the mask by blocks): q8_0:q4_0 prefill 425.9 -> 641.2 t/s at 58k, 249.0 -> 451.6 at 133k, 157 -> 324 at 232k, answers and KL unchanged; the A770 (0039/0040: Q out of local memory, 16 sub-groups): the coder's prefill 476.6 -> 685.5 t/s at 44k; the A770's attention is now staging-bound (load), f16 copies and mask classes lost there; 0045 (the keys split across work-groups when a wave runs part empty, llama.cpp CUDA's parallel_blocks): 228k 342.9 -> 370.1 t/s; next: the Q4_K GEMM (now the B60's largest kernel), V's staging on the A770; the B60's named leftovers sized at ~0 and the kernel at vllm's head-256 geometry (spent); the A770: 0048 (f16 V pairs, -20 to -28 %), quantized-V staging spent (every form lost or gained ~1 %, f16 conversion +1 % at 149 MiB left: not taken) | llama.cpp CUDA `fattn-common.cuh` `launch_fattn`; vllm-xpu-kernels `fmha_utils.hpp` `chunk_policy_head256` (`5210e3d`); llama.cpp CUDA `fattn-common.cuh` `parallel_blocks`; sycl-tla `xe_fmha_fwd_mainloop.hpp` (`7b56b93`); Vulkan `flash_attn_mask_opt.comp` |
| [gqa-small-t-decode](gqa-small-t-decode.md) | decode / MTP-verify attention in one pass over K/V per KV head (B60), patch 0016: a 6-row call at 131k 10.3 -> 4.83 ms (f16), served decode at 62,597 tokens 11.0 -> 15.6 t/s; open: 2.5x a 1-row call (bar 2x) | NInfer `src/ops/attention/causal_softmax/small_t_bf16.cuh`, `small_t.cu:99` |
| [kernel-autotune-ga](kernel-autotune-ga.md) | a GA over the llama engine's Intel kernel parameters, scored at the endpoint; against exhaustive ground truth (448 genomes) unseeded runs reached ranks 1, 2 and 1 in ~65 evaluations; 0014: coder prefill +6.4 % at 4,096 tokens (A770), dense +0.5 % (B60); open: decode kernels, IQ kernels, code-level genes | Kernel Tuner `kernel_tuner/strategies/genetic_algorithm.py` (`code`) |
| [mtp-cycle-wall](mtp-cycle-wall.md) | libllama engine: `--llama-mtp` served, dense 18-19.5 -> 31-35 t/s, coder 47.5 -> 69 t/s at the answer-level bar; Flash-Next 3 drafts with Strata's stop at 0.5 (37.9 against Strata's 37.8 t/s on its request) | Strata `src/core/mtp.cpp:771-820` (draft chain with `min_p`), `tools/mtp_fetch.py` |
| [dflash2-agent](dflash2-agent.md) | closed 2026-10-16: built (`--llama-dflash-gguf`), opt-in; against MTP 5 on the agent +27 % on code, -5 to -8 % on prose and 4k, -33 to -36 % at 130-200k, the reference likewise past ~60k (`measured-here`); the acceptance task 0/10 at 212,992 (MTP 10/10), not attributed; the unit keeps MTP 5 (operator) | llama.cpp `src/models/dflash.cpp`, `common/speculative.cpp:981-1429`, `tools/server` |
| [kv-checkpoint-restore](kv-checkpoint-restore.md) | in process (context checkpoints, 2026-10-05); across a restart open | Strata conversation cache (`src/core/conversation_snapshot.cpp`) |
| [lanes-agent-subagent](lanes-agent-subagent.md) | opened 2026-10-07: an agent lane and a smaller subagent lane in one process, picked by name, a shared KV pool; recon done | llama.cpp server `--parallel --kv-unified --kv-unified-per-slot` (`tools/server/server.cpp:170-177`, `server-context.cpp:4221-4229`) |
| [layer-split-two-cards](layer-split-two-cards.md) | opened 2026-10-07: Flash-Next across the B60 and the A770, a layer range, KV and expert cache per card; recon done, two OpenCL platforms (`measured-here`), design: a context per card (0023), an expert cache per card (0024) | Strata `src/program/generate.cpp:690-708, 2286-2391, 2860-2906`, `src/core/verify.cpp:815-820, 1128-1171`, `src/prefill/prefill.cpp:1866-1882` |
| [agent-context](agent-context.md) | closed 2026-10-12: the dense agent at 245,760 tokens with q8_0:q4_0 KV on the B60 (needle at 237k right, task 10/10, KL inside the bar, peak 24,195 of 24,480 MiB) | llama.cpp `-ctk`/`-ctv`; `contrib/llama.cpp` 0015 |
| [direct-submission-fault](direct-submission-fault.md) | mechanism on record; the N ≥ 5 confirmation on the current kernel owed | upstream `linux-7.1.y` ring-ordering fix |
| [imggen-q8_0-xmx-gemm](imggen-q8_0-xmx-gemm.md) | opened 2026-10-17: upstream's generic q8_0 GEMM was 68.17 % of the first profile, **80.9 %** of the 512² q8_0 run and **68.84 %** of the 1024² run with the fixed 0057; the q8_0 form fits 1024² (peak 17,137 MiB, step 13.14 s), the f16 form does not; recon/design note done, the XMX kernel next | arcint's K-quant XMX GEMM (0001/0009/0011-0014) and `kernel_mul_mm_f16w_f16` (0022); Vulkan's q8_0 path (B3: 1.78 vs 3.69 s a step) |
| [imggen-rope-apply-opencl](imggen-rope-apply-opencl.md) | lever fired, timing CONFIRMED (2026-10-17): ported the fork's `GGML_OP_ROPE_APPLY` to arcint's tree with an original OpenCL kernel (patch 0057) + the sd.cpp upstream-mode guard patch; the image gate caught a port defect (the OpenCL kernel wrote `dst` at element offsets on a char pointer), fixed; q8_0/f16 gates PASS (cos worst 0.999986/0.999987), the deliberate red case FAILs (0.890391); step 1.32 → 0.649 s with the fix, q8_0 3.69 → 3.14, 1024² 15.78 → 13.14 s (`measured-here`) | the fork's `GGML_OP_ROPE_APPLY` (`e05d18a1`, `ggml.c:4381`); sd.cpp `rope.hpp:1075`; alternatively the repeat/mul at device bandwidth |
| [imggen-vae-conv](imggen-vae-conv.md) | opened 2026-10-17: the VAE decode conv (`mul_mat_f16_f16`+`im2col_f16`) is 35.4 % of the f16 run; `--conv-direct vae=on` (the `GGML_OP_CONV_2D` direct kernel OpenCL supports) is 2.7x on the device and 1.83x on the wall with the same final image, but sd.cpp's auto probe never picks it for OpenCL (`measured-here`) | sd.cpp's `ggml_conv_2d_direct`/`sd_probe_conv_direct` (the Vulkan path times both forms); ggml Vulkan `conv2d_mm.comp`; arcint's XMX tiles |
| [imggen-attention-path](imggen-attention-path.md) | open 2026-10-17 (defect, rechecked): the same flags give two images across processes (512²: PASS cos 0.999987 vs FAIL cos 0.991745 at the same 0.649 s wall step), not reproduced in eight fresh-cache/global-cache/VAE arms (`mul_mm_f16w_f16` 1613 launches); the 1024² "sd-server outside the CLI cluster" was our own missing `-p` -- with the server's prompt the `sd-cli` image is bit-identical to the server (PSNR inf) and three server processes are identical; flash on/off and the fused-vs-generic RoPE both cluster (`measured-here`) | the probe `src/core/probe/flash_attn.cpp`, `flash_attention_for` (sd.cpp `f0aac52`); the 512 fresh-process variant |

The OpenVINO engine's closed campaigns (the tier, the static partition, the
host bank, the doorbell on the plugin, QSA, the u8:i4 KV) are listed in
`docs/history.md` and kept at the tag `openvino-history`.

FreeToken paths are under `~/src/FreeToken-ref/python/freetoken/`; Strata
paths under `~/src/Strata-ref/`.

## Closed

| campaign | what exists |
|---|---|
| [prefill-expert-streaming](prefill-expert-streaming.md) | `contrib/llama.cpp` 0032/0044: the bank stream; Flash-Next's 20k needle 408 -> 645 t/s (0.7.3), 690 in 0.7.12 (B60) |
| [llama-engine-kquant-kernels](llama-engine-kquant-kernels.md) | `contrib/llama.cpp` patch 0001: Intel K-quant matvec and XMX GEMM in ggml-opencl; dense 27B 10.3 -> 18.0 t/s decode (B60), coder 7.7 -> 37.7 (A770) |
| [pruefstand-cell-remote](pruefstand-cell-remote.md) | `ARCINT_ACCEPTANCE_PRUEFSTAND` manifest key, 10/10 |

## Research

| document | covers |
|---|---|
| [research-reference-audit](research-reference-audit.md) | Strata and FreeToken against arcint's record; the ranked levers |
| [research-hybrid-expert-execution](research-hybrid-expert-execution.md) | CPU/GPU expert execution, Flash-Next comparables, Strata read at source |
| [research-infernix](research-infernix.md) | Infernix (CUDA, Flash-Next on a 5090) at source: global LFRU cache, layer-walk prefill, shared prefix cache, CPU miss service; ranked for the B60 |
| [research-b70-field](research-b70-field.md) | The B70 projects at source (Flash-Next NVMe offload, exl3xpu, valarauca SYCL): Flash-Next prefill is our gap; per-layer staging, per-query sparse attention, register sign masks |
| [research-hyperqwen](research-hyperqwen.md) | HyperQwen (vLLM, Qwen3.8-27B on one 3090) read at source: draft vocabulary, multi-row verify, lookup drafting |
| [research-qsa](research-qsa.md) | serving Qwen Sparse Attention |
| [research-speculative-cycle](research-speculative-cycle.md) | MTP and speculative cycles |
| [research-sub4bit-weights](research-sub4bit-weights.md) | sub-4-bit weight formats and kernels |
| [research-kv-quantisation](research-kv-quantisation.md) | KV quantisation and prefill cost |
| [research-agent-lanes](research-agent-lanes.md) | serving an agent and its subagents from one model: per-slot caps, routing, batching lanes with speculative decoding, priority, prefix sharing on hybrid models; llama.cpp, vLLM, SGLang, ExLlamaV3, ollama, MLC-LLM read at source |
| [research-cold-start](research-cold-start.md) | kernel caches and warm-up |

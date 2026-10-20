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
| [clean-turns](clean-turns.md) | opened 2026-10-20: unwanted stops and unparseable tool calls in agent sessions; baseline from existing logs (4 empty-stop turns in 2653, 2 unterminated call blocks in 2395 served completions); step 1 masks the end-of-generation tokens while a think block is open, step 2 asks whether the pin's grammar machinery can constrain the call | llama.cpp `grammar-parser`, `llama-grammar`, server's grammar trigger; arcint `src/core/sampler.cpp`, `src/core/reasoning.cpp` |
| [layer-split-two-cards](layer-split-two-cards.md) | opened 2026-10-07: Flash-Next across the B60 and the A770, a layer range, KV and expert cache per card; recon done, two OpenCL platforms (`measured-here`), design: a context per card (0023), an expert cache per card (0024) | Strata `src/program/generate.cpp:690-708, 2286-2391, 2860-2906`, `src/core/verify.cpp:815-820, 1128-1171`, `src/prefill/prefill.cpp:1866-1882` |
| [agent-context](agent-context.md) | closed 2026-10-12: the dense agent at 245,760 tokens with q8_0:q4_0 KV on the B60 (needle at 237k right, task 10/10, KL inside the bar, peak 24,195 of 24,480 MiB) | llama.cpp `-ctk`/`-ctv`; `contrib/llama.cpp` 0015 |
| [direct-submission-fault](direct-submission-fault.md) | mechanism on record; the N ≥ 5 confirmation on the current kernel owed | upstream `linux-7.1.y` ring-ordering fix |
| [qwen3tts-card-real-time](qwen3tts-card-real-time.md) | **closed 2026-10-19, B3 PASS**: **B1 + placement/overlap done**: the card predictor is **272.4 ms/frame** vs **59.9 ms/frame** single-backend CPU (4.5×), device execution only 12 % of wall, host enqueue negligible → the ~6,600 tiny kernels/frame are the cost, not the splits; `0061` ARANGE cut 1007→299 splits for only −4.8 % (RTF 0.267→0.280); code2wav ~1.8 s/flush on card *and* CPU; best arm the **hybrid at 0.828** (mmproj graphs on CPU); card-only real time out of reach without a node-count change; the codec's flush cost is **fixed per call** (256 codes 1.79 s vs 992 codes 1.95 s) so chunking multiplies it, window <72 is not seamless (25,749/32768), and the async worker is refuted (0.536, frame drift) — both on the padded decoder before 0062, open again for the real-time milestone; codec-on-card 0.751; **the hybrid 0.828 is the accepted default (operator, 2026-10-19)**, and the hybrid is now the documented `--tts-gen-device cpu|gpu` flag, unit-carried, boot-checked (red `ARCINT_TTS_GEN_FORCE_GPU=1`, `tools/tts_placement_check.sh`; the card gate ran red-FAIL/green-PASS); **lever (a) explained** (CPU): the fixed cost is the always-72-frame `wav_tfm_swa` padded window, 99.6 % compute, not threads/build/alloc/im2col alone; the valid-frames lever is **landed as the engine default** (patch `0062-mtmd-qwen3tts-gen-valid-frames`; CPU flush proportional, 44 frames 1.11 s; the full-route card gate ran red/green: valid frames (default) 0.802 vs padded 0.729, both C4 PASS 0.0455, sample diff 191/32768, the patch arm bit-identical to the switch arm, the red `MTMD_TTS_WAV_PAD=1` identical to the padded arm); `0061` op suite green/red-green but **dropped** (its only beneficiary path is not adopted); **lever (b) landed**: the mmproj body requantized to Q8_0 with `gen.code.head`/`gen.code.embd` at f16 (predictor call bf16 105.4 -> 31.0 ms) measures **C4 PASS** (WER 0.0264, worst 0.333) and **B3 PASS** (**RTF 1.1291** before / **1.1597** after the card host's RAM upgrade, enlarged D8 set); the production `arcint-tts.service` points at the requantized mmproj; all-f16 fails B3 (0.8051) and f32-all is unmeasured, not re-run | the upstream `tools/tts/tts.cpp` / `mtmd-helper-gen.cpp` loop; `ggml/src/ggml-opencl/ggml-opencl.cpp` `supports_op` at the pin |
| [imggen-rope-apply-opencl](imggen-rope-apply-opencl.md) | `0057`, shipped in 1.1.0: 1024² step 4.77 -> 2.14 s against sd.cpp's generic RoPE (same window), gate PASS, red FAIL, `test_rope_apply` in the op suites (`measured-here`) | the fork's `GGML_OP_ROPE_APPLY` (`e05d18a1`) |
| [imggen-vae-conv](imggen-vae-conv.md) | adopted in 1.1.0: the unit pins `--conv-direct auto,vae=on` (1024² decode 15.41 -> 6.33 s, same image, 3.8 GiB less peak); `auto` on OpenCL picks im2col even when timed (the probe's small shapes); open: a shape-true probe, the direct kernel (`measured-here`) | sd.cpp `sd_probe_conv_direct`; ggml Vulkan `conv2d_mm.comp` |
| [imggen-attention-path](imggen-attention-path.md) | **closed 2026-10-18, no defect**: both "variants" were our own arm mismatches -- the 512² FAIL arms ran sd.cpp's default 20 steps (20 previews, 2.3x the DiT launches at the same step time) against an 8-step reference, and the 1024² CLI arms and CPU reference ran without `-p`; the gate tool now refuses arms that differ in prompt, size, steps, seed, sampler or models (`measured-here`) | — |
| [imggen-wan-vae-conv3d](imggen-wan-vae-conv3d.md) | **closed 2026-10-18**: 0059 puts the Wan VAE's 3D convolutions on the OpenCL implicit GEMM; op suite 276/276 pass, red FAILs 270; Qwen-Image runs (all 16.7 GB on VRAM, 512² 8 steps 55.7 s, Wan VAE 2.66 s; without 0059 the OpenCL graph crashes) and its German "WILLKOMMEN" lettering is legible at 512²/8 and 1024²/20 (`measured-here`) | CUDA `ggml-cuda/im2col.cu` `im2col_3d`; Vulkan `conv3d_mm.comp`; arcint's `conv2d_f16_f32.cl` tiling |
| [imggen-served-edit-divergence](imggen-served-edit-divergence.md) | open 2026-10-18 (1.1.1, released with this gate open by the operator): FLUX.2 klein's served edit (img2img + reference) on the B60 is 17.2 dB from the same request on a CPU sd-server, a local divergence; not the VAE (`measured-here`) | sd.cpp `routes_openai.cpp` (the edit route), `--clip-on-cpu` |
| [imggen-leaky-relu-esrgan](imggen-leaky-relu-esrgan.md) | **closed 2026-10-18**: 0060 adds `GGML_OP_LEAKY_RELU` (f32/f16, contiguous or strided) so sd.cpp's ESRGAN upscaler no longer falls to the CPU, where the split graph segfaulted; op suite 7/7 pass, red 7/7 FAIL, 512² -> 2048² in 27 s on the B60 (`measured-here`) | `ggml.c` `ggml_leaky_relu`; `unary_ext.cl` |
| [qwen3tts-german-genpath](qwen3tts-german-genpath.md) | **closed 2026-10-19**: the Q8_0 `mmproj`'s code predictor `a.gen.code` owned the unintelligible-never-stop German; the fix is the bf16 mmproj with a Q8_0 backbone (2.52 GB); the card arm C4 PASS (WER 0.0455, floor 0.20, `codec_eos` at 46-69 frames); the real-time lever is the separate open campaign | upstream `tools/tts/tts.cpp`, `tools/mtmd/mtmd-helper-gen.cpp` at the pin; Qwen `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

| [qwen3tts-voice-similarity](qwen3tts-voice-similarity.md) | opened 2026-10-19; **method measured**: the Seed-TTS-eval WavLM-large+ECAPA-TDNN SV (`wavlm_large_finetune.pth`, sha256 `51f07e3b…`, mirror `bezzam/wavlm_large_finetune_seed_tts_eval`, CC-BY-SA-3.0) is converted to safetensors in a torch container and embedded via the `huggingface/open-tts-leaderboard` HF-WavLMModel+ECAPA recipe, landed as `tools/tts_voice_sim.py`; the gate-voice clone vs its reference is **mean 0.598** (min 0.44, max 0.75, 46 sentences), above the measured CC cross pair **0.262** (two FLEURS files of one sentence, different speakers) and at the reference's published same-speaker VoxCeleb1 anchors 0.60–0.69; a same-speaker FLEURS pair is **not obtainable** (no speaker field in the local `de_de` dev TSV or canonical `google/fleurs`), and low-download CC speaker-labelled alternatives were not taken (Common Voice gated/empty; VoxPopuli `de` ungated but its smallest parquet 936 MB); open: the real recordings (operator) | `microsoft/UniSpeech` `downstreams/speaker_verification`; `github.com/BytedanceSpeech/seed-tts-eval`; `huggingface/open-tts-leaderboard` |

The OpenVINO engine's closed campaigns (the tier, the static partition, the
host bank, the doorbell on the plugin, QSA, the u8:i4 KV) are listed in
`docs/history.md` and kept at the tag `openvino-history`.

FreeToken paths are under `~/src/FreeToken-ref/python/freetoken/`; Strata
paths under `~/src/Strata-ref/`.

## Closed

| campaign | what exists |
|---|---|
| [imggen-q8_0-xmx-gemm](imggen-q8_0-xmx-gemm.md) | `0058`, shipped in 1.1.0: the 1024² step 10.95 -> **2.14 s** (5.1x), 512² 3.03 -> 0.476 s, gate PASS against the baseline arm, red FAIL (the last 16 of K zeroed); open: the tile search, the kernel at 69 % of a step and 34 % of the XMX peak (`measured-here`) | arcint's K-quant XMX GEMM; Vulkan `mul_mm.comp` |
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

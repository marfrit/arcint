# Campaigns — one lever, one document

Every open defect or lever is a **campaign**: one document in this
directory, sufficient for a fresh session together with the DESIGN sections
and reference sources it cites. A closed campaign is a stub that says what
exists, where, and its number; its full history is in git
(`git show b0447b8:docs/campaigns/<file>`) and in DESIGN §7.0.2x.

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
  the end → review before commit → a DESIGN §7.0.2x record and a CHANGELOG
  line when it closes. Reviews are not skippable.
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

The first four in the build order (`decision`, operator's architect,
2026-10-01): the cache (with its miss split and the bank pinning), the
doorbell, multi-draft MTP, prefill on the GPU. The rest follow the audit's
ranking (`research-reference-audit.md` §4).

| campaign | state | reference to follow |
|---|---|---|
| [flash-next-llama-engine](flash-next-llama-engine.md) | expert cache (0021): UD-Q3_K_XL 235 / 12.4-13.0 t/s; IQ2_XS (0022) 337 / 16.0 t/s, answers right; MTP via `--llama-mtp-gguf` 13.8 t/s (a verify ~3 steps; grouped verify next) | Strata `src/core/expert_cache.cpp`, `generate.cpp:4413-4482`, `expert_source.cpp:1614-1696`; llama.cpp `--n-cpu-moe`, CPU `MUL_MAT_ID` |
| [strata-sycl-b60](strata-sycl-b60.md) | Strata's own engine (its SYCL port, upstream `sycl/`) on the B60 (IQ2_XS, MTP): 620.5 t/s prefill at 20k, 37.2-37.8 t/s decode, answers right, against arcint's 235 / 12.5-13.0 (`measured-here`); owed: IQ2_XS quality against UD-Q3_K_XL | Strata `sycl/`, `docs/INTEL.md` (`7ba023e`) |
| [llama-engine-kernel-gap](llama-engine-kernel-gap.md) | the libllama engine (patches 0001-0007) at 19.75 t/s decode / 521 t/s prefill (dense, B60) and 47.8 / 1,286 (coder, A770; 47.4 decode at 4k), against the OpenVINO path's 24.0 / 1,141 and 43.9 / 1,379; open: prefill attention, dense decode bytes, launches | OpenVINO `paged_gated_delta_net_opt.cl`, `paged_attention_opt.cl`, `moe_router_fused.cl`; arcint patch 0064; Strata `src/prefill/kernels.cu:380-423` |
| [gqa-small-t-decode](gqa-small-t-decode.md) | decode / MTP-verify attention in one pass over K/V per KV head (B60), patch 0016: a 6-row call at 131k 10.3 -> 4.83 ms (f16), served decode at 62,597 tokens 11.0 -> 15.6 t/s; open: 2.5x a 1-row call (bar 2x) | NInfer `src/ops/attention/causal_softmax/small_t_bf16.cuh`, `small_t.cu:99` |
| [kernel-autotune-ga](kernel-autotune-ga.md) | a GA over the llama engine's Intel kernel parameters, scored at the endpoint; against exhaustive ground truth (448 genomes) unseeded runs reached ranks 1, 2 and 1 in ~65 evaluations; 0014: coder prefill +6.4 % at 4,096 tokens (A770), dense +0.5 % (B60); open: decode kernels, IQ kernels, code-level genes | Kernel Tuner `kernel_tuner/strategies/genetic_algorithm.py` (`code`) |
| [mtp-cycle-wall](mtp-cycle-wall.md) | libllama engine: `--llama-mtp` served, dense 18-19.5 -> 31-35 t/s, coder 47.5 -> 69 t/s at the answer-level bar; OpenVINO path: dense drafts one token a cycle, Flash-Next served with 3 drafts at ~3.6 plain steps a window | Strata `src/core/mtp.cpp:771-820` (draft chain with `min_p`), `tools/mtp_fetch.py` |
| [prefill-expert-streaming](prefill-expert-streaming.md) | prefill 63–68 t/s at 20k on the B60, CPU-tier-bound | Strata `src/prefill/prefill.cpp:71-104` (slot ring, chunks to 8,192, MMQ); FreeToken `moe/offload_cache.py` prefill overlap, `layers/moe.py:388-390` |
| [kquant-host-storage](kquant-host-storage.md) | native blocks on the host tier, quantised dot for decode-shaped calls (0043, 0074) | Strata `src/kernels/cpu/iq_avx2.cpp` (multi-token AVX2) |
| [kv-checkpoint-restore](kv-checkpoint-restore.md) | in process on both engines (libllama: context checkpoints, 2026-10-05); across a restart open | Strata conversation cache (`src/core/conversation_snapshot.cpp`) |
| [served-prefill-determinism](served-prefill-determinism.md) | B60 run-to-run variance, located in the GDN state; not gating | upstream openvinotoolkit/openvino#38099 |
| [lanes-agent-subagent](lanes-agent-subagent.md) | opened 2026-10-07: an agent lane and a smaller subagent lane in one process, picked by name, a shared KV pool; recon done | llama.cpp server `--parallel --kv-unified --kv-unified-per-slot` (`tools/server/server.cpp:170-177`, `server-context.cpp:4221-4229`) |
| [layer-split-two-cards](layer-split-two-cards.md) | opened 2026-10-07: Flash-Next across the B60 and the A770, a layer range, KV and expert cache per card; recon done, two OpenCL platforms (`measured-here`), design: a context per card (0023), an expert cache per card (0024) | Strata `src/program/generate.cpp:690-708, 2286-2391, 2860-2906`, `src/core/verify.cpp:815-820, 1128-1171`, `src/prefill/prefill.cpp:1866-1882` |
| [direct-submission-fault](direct-submission-fault.md) | mechanism on record; the N ≥ 5 confirmation on the current kernel owed | upstream `linux-7.1.y` ring-ordering fix |

FreeToken paths are under `~/src/FreeToken-ref/python/freetoken/`; Strata
paths under `~/src/Strata-ref/`.

## Closed

| campaign | what exists |
|---|---|
| [llama-engine-kquant-kernels](llama-engine-kquant-kernels.md) | `contrib/llama.cpp` patch 0001: Intel K-quant matvec and XMX GEMM in ggml-opencl; dense 27B 10.3 -> 18.0 t/s decode (B60), coder 7.7 -> 37.7 (A770) |
| [expert-hot-set-lru](expert-hot-set-lru.md) | patch 0076: adaptive expert cache, non-blocking admission, Strata's RAM exchange and RAM budget; 20k needle decode 7.5 -> 10.3 t/s (B60) |
| [tier-handoff-doorbell](tier-handoff-doorbell.md) | patch 0077: the decode step submitted ahead, the tier fed through mapped memory; 10.7 -> 12.7 t/s needle, 14.0 -> 16.0 long answer (B60) |
| [test-ladder-close](test-ladder-close.md) | acceptance references filled; DESIGN §7.0.2aj–al |
| [prefill-fallback-tristate](prefill-fallback-tristate.md) | patch 0019, `ExpertWeightsSide` |
| [turnstile-wall-time](turnstile-wall-time.md) | synchronised turnstile test, `free_port` in `roundtrip.sh` |
| [pruefstand-cell-remote](pruefstand-cell-remote.md) | `ARCINT_ACCEPTANCE_PRUEFSTAND` manifest key, 10/10 |
| [u8i4-prefill-price](u8i4-prefill-price.md) | patch 0020: u8:i4 on micro-SDPA at u8's prefill rate |
| [u8i4-deep-prefill-fault](u8i4-deep-prefill-fault.md) | no fault on the micro path through 118k; generic-path belt kept |
| [static-partition-cold-start](static-partition-cold-start.md) | `--fit-ledger-dir` and the pre-warm forward |
| [static-partition-prefill](static-partition-prefill.md) | patches 0037/0042: grouped prefill split by residency |
| [partition-seeding](partition-seeding.md) | patch 0046 census seed, `MOE_CPU_TIER_SEED` |
| [sub4bit-vram-kernel](sub4bit-vram-kernel.md) | per-expert dispatch with in-kernel native decode (0038–0058, 0067); 35B fully resident on the A770 |
| [serving-shape-logits](serving-shape-logits.md) | fill fixes; served d48 against the f32 reference |
| [ple-disk-backend](ple-disk-backend.md) | `--ngram-staging-rows`: 26.82 GiB pin → 2.9 MiB staging |
| [nvme-direct-expert-tier](nvme-direct-expert-tier.md) | patches 0048/0049: arcwell load-time fill, LISBON-001 |
| [host-expert-bank](host-expert-bank.md) | patch 0072 RAM bank (`+p23`), patch 0074 quantised decode dot (`+p25`) |
| [dense-q8-flash-next](dense-q8-flash-next.md) | `--dense-q8`, the `d48q8` artifact |
| [qsa](qsa.md) | `--qsa` served with a compressed block-key cache, non-default |
| [hybrid-expert-fetch](hybrid-expert-fetch.md) | merged into `expert-hot-set-lru` 2026-10-01 (the miss split) |

## Research

| document | covers |
|---|---|
| [research-reference-audit](research-reference-audit.md) | Strata and FreeToken against arcint's record; the ranked levers |
| [research-hybrid-expert-execution](research-hybrid-expert-execution.md) | CPU/GPU expert execution, Flash-Next comparables, Strata read at source |
| [research-hyperqwen](research-hyperqwen.md) | HyperQwen (vLLM, Qwen3.8-27B on one 3090) read at source: draft vocabulary, multi-row verify, lookup drafting |
| [research-qsa](research-qsa.md) | serving Qwen Sparse Attention |
| [research-speculative-cycle](research-speculative-cycle.md) | MTP and speculative cycles |
| [research-sub4bit-weights](research-sub4bit-weights.md) | sub-4-bit weight formats and kernels |
| [research-kv-quantisation](research-kv-quantisation.md) | KV quantisation and prefill cost |
| [research-agent-lanes](research-agent-lanes.md) | serving an agent and its subagents from one model: per-slot caps, routing, batching lanes with speculative decoding, priority, prefix sharing on hybrid models; llama.cpp, vLLM, SGLang, ExLlamaV3, ollama, MLC-LLM read at source |
| [research-cold-start](research-cold-start.md) | kernel caches and warm-up |

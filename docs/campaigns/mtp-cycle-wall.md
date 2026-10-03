# mtp-cycle-wall — draft several tokens a cycle with the model's own MTP head, as Strata does

**Open.** Lever 3 of `research-reference-audit.md` §4. Covers the dense agent
and Flash-Next.

## Charter

Each speculative cycle drafts a chain of tokens with the MTP head, continuing
while the last draft is confident enough, and verifies the whole chain in one
main-model forward, so a cycle yields more tokens than its cost.

## Reference to follow

**Strata** (`~/src/Strata-ref`):
- **The chain** (`code`: `src/core/mtp.cpp:771-820`, `MtpDrafter::draft`).
  Draft 0, then further drafts while the last draft's probability is at least
  `min_p`, up to `max_drafts`; per-request `spec_min_p`
  (`src/program/generate.cpp:361`, `:5297-5299`).
- **The verify window** (`code`: `src/core/verify.cpp`). The last accepted
  token plus the drafts run through all 48 layers as one window; draft and
  verify passes are captured graphs.
- **The head for Flash-Next** (`code`: `tools/mtp_fetch.py`). The 31 `mtp.*`
  tensors (~5 GB) come from the BF16 checkpoint by HTTP range reads, checked
  by sha256; `tools/draft_vocab.py` builds a reduced draft vocabulary.
- **Effect** (`paper` §3.3): up to 3 drafts, a 50 % confidence gate, output
  identical to plain greedy; 1.6–1.8× decode with CPU-resident experts.
  Community Flash-Next reports with MTP read 1.4–1.7× decode
  (`research-hybrid-expert-execution.md`, comparables table).

Other engines chain the MTP head the same way (vLLM `num_speculative_tokens`,
TensorRT-LLM chained modules; `research-speculative-cycle.md`).

### The libllama engine against Strata and NInfer (read at source 2026-10-03)

Strata at `c499bd1`, NInfer (`~/src/ninfer`, dense 27B on one GPU) at
`d9dbe1ce`, all `code` unless marked.

- **The verify matmul: followed.** Both read each weight once for the
  window's rows: Strata `native_mmvq_multi_kernel` (2-8 columns, DP4A on
  Q8_1 activations, `src/kernels/cuda/native_mmvq.cu:775-1062`), NInfer
  `q4_rowsplit_gemm_simt` (up to 16 tokens, weights decoded once,
  `src/ops/linear/q4/q4_rowsplit_gemm_simt.cuh`). Ours is
  `contrib/llama.cpp` 0009 (fp16 DPAS, 2-16 columns): an 8-row dense step
  1.21x a 1-row step on the B60 (`measured-here`); Strata's planning prior
  is 1.45x at 4 rows (`include/strata/spec/controller.hpp:27`, a prior,
  not a measurement). NInfer on an RTX 5090: 77.6 t/s plain at a 7,680-token
  prompt and 167.0 t/s with 3 drafts at 3.17 tokens a round
  (`docs/performance.md:461, 480`, `doc-measured`), which makes a round about
  1.47 plain steps. That ratio is arithmetic on two rows taken at different
  prompt lengths, so it is an upper estimate.
- **Draft 0 from the catch-up batch: followed (2026-10-03).** Strata runs
  the MTP over the window pairing row t with the target's pick at row t, so
  the accepted row's output is draft 0 (`mtp.hpp:67-77`, `mtp.cpp:771-800`);
  NInfer does the same inside one round graph (`mtp_impl.h:85-200`).
  `src/exec/llama_spec.cpp` now does it after the verify walk. It needed
  `contrib/llama.cpp` 0010: the pin's masked MTP context returned token 0's
  nextn row for a batch of several tokens with one output (`code`:
  `src/models/qwen35.cpp`, the capture before the output selection). Without
  0010, drafts after the first chained from the wrong row: acceptance 78 ->
  57 % (dense, 3 drafts), 91 -> 75 % (coder, 2). With it acceptance matched
  the old drafter's exactly on the same prompts: 804/1,026, 302/332
  (`measured-here`). The saving is small on the dense model, 1.1 ms a cycle:
  the forward that went computed no logits. What a draft step costs is the
  248k-row head, next item.
- **Confidence-gated chain: deviation, open.** Strata stops at a draft
  probability below `min_p` 0.5 and sizes the next window to it
  (`mtp.cpp:804`, `generate.cpp:5297-5300`). Ours drafts a fixed N.
  Reason: not built yet; no invariant.
- **Draft vocabulary: followed (2026-10-03).** The references draft from a
  subset of the vocabulary:
  - Strata: 106,299 rows (`mtp.hpp:156`, `tools/draft_vocab.py`);
  - NInfer: a 131,072-row Q4 shortlist with an id remap
    (`tools/convert/qwen3_6_27b/draft_head.py`, `speculative_round.cuh:566`);
  - HyperQwen: 40,960 rows (`research-hyperqwen.md`).

  `--llama-mtp-vocab FILE` takes such a list (int32s or JSON), and needs no
  llama.cpp patch:
  - The MTP context returns its head input rows (unmasked nextn) and no
    logits, so libllama skips the 248k-row head.
  - `DraftHead` (`src/exec/llama_spec.cpp`) reads those rows of
    `output.weight` from the GGUF, from whichever shard of a split file holds
    it, onto the model's card. It multiplies a draft step's row by them and
    takes the argmax on the host.

  Served, on the same prompt at temperature 0 (`measured-here`):

  | model | drafts | full head | HyperQwen's 40,960 | Strata's 106,299 |
  |---|---|---|---|---|
  | dense 27B (B60) | 4 | 47.1 t/s, 72.1 % | 50.8 t/s, 68.0 % | 50.8 t/s, 71.5 % |
  | dense 27B (B60) | 5 | 48.7 t/s, 65.5 % | 52.6 t/s, 61.0 % | 52.6 t/s, 64.6 % |
  | coder (A770) | 4 | 73.6 t/s, 77.6 % | 79.3 t/s, 73.3 % | 76.7 t/s, 74.2 % |

  (Each cell: decode rate, draft acceptance.)

  Drafting time a cycle halves (dense, 4 drafts: 4.66 -> 2.16 s over the
  run). The acceptance task scored 10/10 in every one of these runs, at
  temperature 0 and sampled. Coverage risk is on record at Strata: CJK and
  Cyrillic were added after Chinese answers drafted nothing. Neither subset
  has been tried here on a non-English or non-code workload.
- **Recurrent-state rollback: deviation, open.** Strata and NInfer record
  the window's per-token gated delta-net inputs and replay the accepted
  prefix (`verify.hpp:13-17`; NInfer `core/gdn_replay_records.h:36-40`,
  `ops/gdn_replay.h:20-45`); llama.cpp at the pin writes a state snapshot
  per row (`src/models/delta-net-base.cpp:564-590`). Reason: the pin's
  mechanism; its cost at 4-8 rows is not measured here.
- **Acceptance: equivalent.** NInfer's rejection sampling with greedy drafts
  accepts at our sample-and-compare rate; Strata's coupled drafting
  (`coupled_draft.hpp`, one uniform shared by draft and verify row) raises
  sampled acceptance and needs a position-keyed RNG. Not built.

## Gate

At the served depth, on the same card and window: decode with MTP faster than
plain; prefill within the run-to-run spread; the answer-level bar
(`CLAUDE.md`) between MTP on and off — for the dense agent the `agent-dense`
cell and Prüfstand 10/10, for Flash-Next the 20,085-token needle and window-0
KL no more than 0.03 nats above plain's, argmax down at most 1 point.

## Current state

- **libllama engine (2026-10-03), dense 27B and coder: served.**
  `--llama-mtp N` (`docs/llama-engine.md`, MTP): llama.cpp's single-head
  `draft-mtp` rebuilt on libllama (`src/exec/llama_spec.cpp`), the verify's
  rows sampled by arcint's sampler, the rollback on the device through the
  target's recurrent snapshots. `measured-here`, served, the acceptance
  task: dense (B60, 3 drafts) 18.0-19.5 -> 30.8-35.0 t/s, 69-86 %
  accepted, 10/10 at temperature 0, 13 of 20 sampled at 10/10 (plain arms
  10, 11, 15 of 20); coder (A770, 2 drafts, up to a 16k context) 47.5 ->
  68.9 t/s, 86-90 % accepted, 10/10 and 6 of 6 sampled. On the dense model
  a cycle (3 drafts, a 4-row verify) took 102 ms -- verify 90, drafting 11.5
  -- against a 51 ms plain step, ~2.0 steps for 3.6 tokens (the K-quant
  matvec takes 4 columns, the decode attention up to 8 rows), against the
  OpenVINO path's ~3.6 steps a 4-token window below. Flash-Next on this
  engine: not yet.
- **Dense 27B agent** (`measured-here`, 24 GB card, DESIGN §7.0.2ag): MTP
  drafts one token a cycle; at 77,134 tokens it accepts 90.8 % and decodes
  4.9 t/s against plain 15.3 t/s, a cycle of ~390 ms; DFlash reads 18.8 t/s.
  The MTP state is charged (`kMtpStateBytesPerToken`, `src/exec/fit.h`) and
  the drafters' rotary runs in f32 (`ARCINT_DRAFT_ROPE_F16=1` reverts).
- **Flash-Next, served (2026-10-03).** `tools/export_mtp_flash_next.py`
  builds Strata's draft layer (`code`: `include/strata/core/mtp.hpp:1-20`,
  `src/core/mtp.cpp:413-585`) from the 31 `mtp.*` tensors with the serving
  shape's own emitters (hyper-connections, attention, `emit_moe_tiled` with
  u4 group-128 experts: export_mtp's MoE lowering compiled but failed every
  one-token infer, on the stock plugin too); the layer's K/V state takes a
  `kv_len` input, so speculative cells drop on the next committed feed
  (`measured-here`, CPU: equal to a fresh prefix, max diff 0.0). The runtime
  exposes `layer47/out`, primes the layer over the prompt, chains up to 3
  drafts from the layer's own residual while the draft probability is
  >= 0.5 (`ARCINT_MTP_DRAFTS`, `ARCINT_MTP_MIN_P`), and verifies through the
  checkpoint-row window. B60, d48q8, 12.8e9-byte expert pool (the layer and
  its head take ~2.2 GiB of VRAM), `measured-here`: acceptance 76.2 / 69.5 /
  69.2 % at depths 1 / 2 / 3; the repeat request 19.1 t/s against 13.2
  plain on the same pool, a 500-token free answer 10.9 against 13.3, the
  needle right in both. A 4-token window costs ~3.6 plain steps here (the
  CPU tier's work grows with the window's tokens); Strata's costs ~2x a step
  (`paper`), and its mechanism for that is the miss split (P1).
- **Flash-Next, before**: the head's 31 `mtp.*` tensors were fetched on 2026-09-11
  (`tools/fetch_safetensors_tensors.py`, 4.856 GiB, sha-checked) but never
  exported or served for Flash-Next; Strata's `tools/mtp_fetch.py` reads the
  same tensors. The MTP layer reuses
  the QSA indices across draft steps (`paper`, `research-qsa.md`).
- **Preconditions met:** DESIGN §3.4 (amended 2026-10-01) and `CLAUDE.md` judge MTP
  on against off at the answer-level bar; byte cells stay tripwires.

## Where it lives

`src/exec/backend_ov.cpp` (the MTP path, `ARCINT_PROFILE_CYCLE`,
`ARCINT_DRAFT_*`), `src/exec/fit.h`, `tests/equivalence/run.sh` (MTP
section), `tests/acceptance/cells.json` (`agent-dense`); the Flash-Next
exporter in `tools/q4e/` for the head.

Full history: `git show b0447b8:docs/campaigns/mtp-cycle-wall.md`.

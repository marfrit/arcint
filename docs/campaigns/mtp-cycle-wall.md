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

## Gate

At the served depth, on the same card and window: decode with MTP faster than
plain; prefill within the run-to-run spread; the answer-level bar
(`CLAUDE.md`) between MTP on and off — for the dense agent the `agent-dense`
cell and Prüfstand 10/10, for Flash-Next the 20,085-token needle and window-0
KL no more than 0.03 nats above plain's, argmax down at most 1 point.

## Current state

- **Dense 27B agent** (`measured-here`, 24 GB card, DESIGN §7.0.2ag): MTP
  drafts one token a cycle; at 77,134 tokens it accepts 90.8 % and decodes
  4.9 t/s against plain 15.3 t/s, a cycle of ~390 ms; DFlash reads 18.8 t/s.
  The MTP state is charged (`kMtpStateBytesPerToken`, `src/exec/fit.h`) and
  the drafters' rotary runs in f32 (`ARCINT_DRAFT_ROPE_F16=1` reverts).
- **Flash-Next**: the head's 31 `mtp.*` tensors were fetched on 2026-09-11
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

# mtp-head-fit — refit the MTP head against what the served model writes

**Open.** Step one measured (2026-11-06); recon done, E0 under way; the E1 tooling is built (capture hook, replay, eval, linear fit; section "E1 tooling"), its card window not yet run.

## Charter

The coder's draft head (`blk.40` in its GGUF, provenance not recorded) sets how many tokens a verify yields. One lever: raise acceptance by fitting the head on the coder's own output. The served path does not change. The data is generated, never session content, and the corpus and any fitted head stay private: only this method is published (operator, 2026-11-06).

## Reference to follow

HASS (arXiv 2408.15766, `paper`): ranking-distillation loss and context-aligned training on self-generated prefixes, no added inference cost, 2.81-4.05x from acceptance alone, on other models and hardware. EAGLE-3 (arXiv 2503.01840, `paper`). HyperQwen lists MTP head fine-tuning among what it tried and lost (`docs/optimizations.md`, `doc-measured`, no detail read here). Neither has been tried in arcint: no verdict in either direction.

## Gate

Held out on generated text not used in the fit: first-draft acceptance up by at least 5 points on code and thinking, measured with `ARCINT_MTP_TRACE`; then served decode on the deployed flags per the answer-level bar in `CLAUDE.md`, acceptance task included. A verify still checks every draft, so answers cannot get worse through the head; the task runs anyway.

## Current state (`measured-here`, A770, 2000 MHz, the coder's file, arcint 1.5.4 + 0101, 24 generated prompts, one run an arm, q8_0:q4_0)

Kept given reached, by draft position, greedy with 4 drafts a verify:

| Text | rounds | 1 | 2 | 3 | 4 | tokens a verify |
|---|---:|---:|---:|---:|---:|---:|
| code | 293 | 90 % | 81 % | 78 % | 68 % | 3.59 |
| tool calls | 55 | 89 % | 82 % | 82 % | 79 % | 3.69 |
| thinking | 1,144 | 80 % | 74 % | 75 % | 72 % | 3.15 |
| prose | 1,134 | 69 % | 56 % | 53 % | 43 % | 2.36 |

Deployed sampling with the cost policy: first draft 92, 88, 78, 71 % in the same order, so sampling costs the first draft nothing here; tokens a verify 3.52, 3.45, 2.65, 2.03, at 3.0, 3.0, 2.2, 1.5 drafts a round, 29 ms a round against 37.

What a fit could buy (`inferred`, arithmetic on the rows above, a verify's cost held): 10 points at every position takes thinking from 3.15 to 3.83 tokens a verify (+21 %) and prose from 2.36 to 2.81 (+19 %); code has less room. Whether a fit reaches 10 points is unknown.

## Prior art and what it leaves open

HyperQwen fit its head by KL over the draft vocabulary, depth-2 chains, 7M tokens: KL halved, top-1 against the true token unchanged at 0.685, acceptance within noise (`paper`; `drafter/README.md`, `train_mtp.py`). That negative does not cover three things (`code`): its head matched its trunk; its data is sampled at model defaults and scored against the sampled token, which caps top-1 by the sampling noise; its loss has no top-K term. HASS continued DeepSeek-R1's official MTP for 2 epochs with CE plus top-10 and a feature term: +8.47 % at T=0, +8.18 % at T=1 (`paper`, repo README; `code`, `main_hass.py`). The draft vocabulary, HyperQwen's largest gain, is not our limit: coverage on the coder's own text is 97.7 %, and 525 more rows from 92k tokens reach 98.6 % for about +1 % tokens a verify (`measured-here`).

Provenance (`measured-here` on samples of the coder's F16 against the base 35B's BF16; `paper` from the source's card): non-expert tensors equal the base within F16 rounding and kept experts match, so the trunk's weights did not change, only 72 experts a layer dropped and routing went to top-10. The head was presumably trained on the 256-expert, top-8 trunk (`inferred`) and now reads rows from the pruned one, with its own FFN pruned too.

## What a fit could buy (`inferred`, arithmetic on the position table, which reproduces its tokens a verify)

| Text | first draft for +10 %, others held | first draft 100 % | first draft +5 | all four +5 | all four +5, width re-chosen |
|---|---:|---:|---:|---:|---:|
| code | impossible | +8.0 % | +4.0 % | +9.9 % | +9.1 % |
| tool | impossible | +9.0 % | +4.1 % | +10.2 % | +10.2 % |
| thinking | 91.7 % | +17.1 % | +4.3 % | +10.0 % | +5.8 % |
| prose | 80.9 % | +26.0 % | +4.2 % | +8.8 % | +5.5 % |

Planning range 0 to +8 % decode, more likely under +5 %: a judgement, which only a run settles.

## Next, cheapest falsifier first

- **E0, the head's bytes.** The base 35B's MTP shard against the coder's F16 `blk.40`: is the head the base's, and which experts did it keep?
- **E1, capture, evaluation and a linear refit.** A hook `ARCINT_MTP_CAPTURE` next to `ARCINT_MTP_TRACE` writes the target's row and the head's row per position; no llama.cpp patch (`llama_get_embeddings_nextn`, unmasked under `--llama-mtp-vocab`). Replay the corpus through it, check each re-tokenized count. Report top-1 against the sampled token and the target's argmax, at simulated round starts too (must reproduce the served table), and E[max p_T] at deployed sampling, per text class on data held out by task string, because the corpus crosses tasks with languages. Fit `A = (Z'Z + lambda I)^-1 Z'H`, `W' = W_d A` in closed form. +2 points or more held out on code or thinking means the output side is miscalibrated: continue. KL down with top-1 flat is HyperQwen's result on the output side only.
- **E2, no-training arms on the served trace.** The pruned head at F16 (quantization's cost), the pruned head at top-8, the base head at top-8. A gain would need no training.
- **The full fit, only after these.** CE plus top-10 and an unrolled 3-step chain, `eh_proj`, `attn_output` and norms before any experts: 92k tokens are far too few for 3.1 M weights an expert. 15-20 agent hours, 2-5 card hours.

Risks: overfitting; quantization erasing the fit (the served Q4_K head is 0.075 relative from F16; measure the delta against that); wider verifies eating the gain, which is why thinking's +10 % becomes +5.8 %; prefill rows against verify rows. A rows refit needs a sidecar flag on `DraftHead`, since those rows are shared with the target, and that conflicts with "the served path does not change": the operator's call if it comes to it.

## E1 tooling (`code`; built 2026-11-06, device-free tests green, no card run yet)

- **Capture.** `ARCINT_MTP_CAPTURE=<base>` (`src/exec/mtp_capture.h`, wired in `llama_spec.cpp` `decode_multi`): per prompt position the target's row `h` and the head's row `z`, raw f32, plus a token index; format and alignment rule in the header file. Both rows are post-norm vectors that `output.weight` multiplies and both predict the next token: `z(p)` is the head's output for the entry `(tok p, h(p-1))`. Off: one getenv at construction and a pointer test. On: needs `--llama-mtp-vocab` (refused at start without it), holds the 0067 skip off, and the backend starts every request on an empty lane so no position is skipped by prefix reuse. Only the prompt pass is captured, not verify rows.
- **Replay.** `tools/mtp_capture_replay.py` sends each corpus text as a token-array prompt with `max_tokens` 1, thinking text as its own class, tool class dropped, and re-tokenised counts checked against `tokens`. Rows are found afterwards by the hash of each text's ids, so stray requests and retries do not shift anything.
- **Eval and fit.** `tools/mtp_fit.py eval|linear`; numpy only; `W_d` read from `output.weight` of the served GGUF (Q6_K on the coder; the Q6_K and Q4_K decoders agree with the F16 file at 0.019 and 0.075 relative, the latter the figure above). Splits are by task, stratified by class, ordered by a hash.
- **What the capture cannot give.** Only depth 1 is captured: a deeper draft of the served chain feeds the head its own row, the capture holds the target's. The round-start simulation therefore reproduces the first-draft column of the served table; depths 2 to 4 are the head with a teacher-forced input, an upper bound.
- **What the corpus allows.** After the tool class is dropped, 61 k tokens train, 14 k validate, 17 k test (by task); code 39 task keys, prose 30, thinking 16. The held-out thinking class is 3 tasks (3.6 k rows): a gate of 2 points on thinking is inside its noise until more thinking text exists.

## Where it lives

The measurement's scripts move to `tools/` with the corpus step. `ARCINT_MTP_TRACE`: `src/exec/mtp_trace.h`.

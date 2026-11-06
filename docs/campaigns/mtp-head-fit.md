# mtp-head-fit — refit the MTP head against what the served model writes

**Open.** Step one measured (2026-11-06); recon done, E0 under way; the E1 tooling is built (capture hook, replay, eval, linear fit; section "E1 tooling"), its card window not yet run.

## Charter

The coder's draft head (`blk.40` in its GGUF, provenance not recorded) sets how many tokens a verify yields. One lever: raise acceptance by fitting the head on the coder's own output. The served path does not change. The data is generated, never session content, and the corpus and any fitted head stay private: only this method is published (operator, 2026-11-06).

## Reference to follow

HASS (arXiv 2408.15766, `paper`): ranking-distillation loss and context-aligned training on self-generated prefixes, no added inference cost, 2.81-4.05x from acceptance alone, on other models and hardware. EAGLE-3 (arXiv 2503.01840, `paper`). HyperQwen lists MTP head fine-tuning among what it tried and lost (`docs/optimizations.md`, `doc-measured`, no detail read here). Neither has been tried in arcint: no verdict in either direction.

## Gate

Held out on generated text not used in the fit: first-draft acceptance up by at least 5 points on code and thinking, measured with `ARCINT_MTP_TRACE`; then served decode on the deployed flags per the answer-level bar in `CLAUDE.md`, acceptance task included. A verify still checks every draft, so answers cannot get worse through the head; the task runs anyway.

Operator's decisions, 2026-11-06: (1) a measured gain with nothing degrading is adopted at any size; +5 points is the line at which the expensive fit is rethought, not a condition for adoption. (2) The full fit runs only if E1 shows room; a null E1 and E2 end the campaign without it, and the record must say the reference's mechanism (a top-K loss on a chain) was not tried, so that is no verdict on it. (3) A refit of the draft head's rows may add a small sidecar flag on `DraftHead`, if E1 points there, although the served path then changes.

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

HyperQwen fit its head by KL over the draft vocabulary, depth-2 chains, 7M tokens: KL halved, top-1 against the true token unchanged at 0.685, acceptance within noise (`paper`; `drafter/README.md`, `train_mtp.py`). That negative does not cover three things (`code`): its head matched its trunk; its data is sampled at model defaults and scored against the sampled token, which caps top-1 by the sampling noise; its loss has no top-K term. HASS continued DeepSeek-R1's official MTP for 2 epochs with CE plus top-10 and a feature term: +8.47 % at T=0, +8.18 % at T=1 (`paper`, repo README; `code`, `main_hass.py`). The draft vocabulary, HyperQwen's largest gain, is not our limit: coverage on the coder's own text is 97.7 %, and 525 more rows from 92k tokens of its own answers reach 98.6 % on 24 held-out prompts (`measured-here`). Greedy with 4 drafts, two pairs, identical: tokens a verify 2.87 -> 2.90 (+1.0 %), code 3.58 -> 3.66 (+2.2 %); deployed sampling 2.45, 2.48 -> 2.42, 2.50, inside its spread; a round's cost unchanged, 1 MiB more. Adopted on the coder under the any-gain rule. The list stays private.

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

- **E0, the head's bytes: done (`measured-here`, the coder's F16 against the base 35B's MTP shard, `tools`-free comparison, sampled 1,048,576 values a matrix).** Every matrix of `blk.40` equals the base's within F16 rounding (worst relative 7.6e-8), the 184 router rows and `down` experts are base rows, and the norms are the base's plus exactly 1.0 (correlation 1.0000), the converter's RMS-norm form. So the head is the base's, pruned the same way, and never refit: it was trained on the 256-expert, top-8 trunk (`inferred`) and reads a 184-expert, top-10 one.
- **E1, done (`measured-here`, the served Q6_K head rows, 91k positions of generated text, held out by task; capture on the A770, analysis on the CPU).** The tool reproduces the served first draft within 1.2-1.6 points on all rows, and a one-position shift of the rows collapses it to 0.6 %. Room at the first draft under deployed sampling, the head's chance of being drawn against the target's best: code 89.3 of 95.1 %, thinking 85.0 of 92.1, prose 73.4 of 87.6. Top-1 against the target's argmax, all positions: 90.4, 87.3, 75.8 %. The closed-form least squares made the head worse even on its training rows (code 90.5 -> 84.4 % train, 80.3 held out). Its smallest lambda won every time, so it is plain OLS; its objective is the wrong one: L2 on the rows, equal to L2 on the logits while the draft rows are full rank, is dominated by 40,000 tail logits the head was never trained to get right, and it trades the top for the tail. It says nothing about the head. A correction fit to the target's distribution from the identity, Adam on 2048 x 2048 and stopped on validation, is the fair form: best arm code 90.4 -> 91.0, prose 75.8 -> 76.4, thinking 87.3 -> 86.9, sampled pick +0.6, +0.4, -0.2 points, while training KL halves in the higher-rate arm. A sharpened target at the high rate never beat the identity. So the +2 line is not met on these arms: HyperQwen's result, on the output side. The second-model review's limits on that reading: these arms stop on 3,400 validation rows (standard error 0.56 points), the best arm had not converged, and the sharpened target ran only at a rate the other arm showed too high; closing arms at lower rates on all validation rows are running. Held out: 8 code, 6 prose, 3 thinking tasks; thinking's move is inside its noise. It tests a linear map of the head's row only, not the head's attention or FFN and not a top-K loss on a chain. `tools/mtp_fit.py`, `tools/mtp_capture_replay.py`, `ARCINT_MTP_CAPTURE`; depths 2-4 are upper bounds, since the capture feeds the target's row.
- **E2, no-training arms: done (`measured-here`, the B60, greedy, 4 drafts, policy off, 24 held-out prompts, q8_0:q4_0 KV, `--llama-mtp-gguf`; `tools/coder_mtp_gguf.py`).** Tokens a verify, all / code / thinking / prose, first draft kept, ms a round: the served head 2.87 / 3.67 / 3.17 / 2.35, 75 %, 27.1; its own bytes through a head file, identical, 27.4 (the path's control); the pruned head rebuilt, twice, 2.86, 74 %; pruned at top-8, 2.88, 75 %, 27.0; pruned at F16, 2.90, 75 %, 43.2; **the base's own head, all 256 experts at top-8, in the served quant: 3.14 / 4.08 / 3.42 / 2.59, 80 %, 27.0, +9.4 %**, with every position up: 80, 75, 73, 71 against 75, 70, 69, 67. So the draft loss is the pruning of the head's experts, not its quantization and not the routing. The trunk's pruning stays as served. No training, no verdict on a fit.
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

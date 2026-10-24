# loop-guard — detect a reasoning loop, escalate, end the turn cleanly

**Shipping in 1.2.3 (2026-10-24).** A coder model can circle inside its think block until
the token cap: no answer, a wasted round. The operator approved an escalation
ladder **detect → nudge → slap → abort**, behind a switch that is off by
default.

## Charter

- **Detection inside the think block only.** A rolling window of reasoning
  lines (a line ends at a newline; leading and trailing ASCII whitespace is
  stripped, then only lines longer than 20 characters count) whose repeat
  share reaches a threshold. Incremental, no whole-text
  re-scan per token. The default match rule is **near** (`--loop-guard-match
  exact|near`): a line also counts as a repeat when its character-3-gram set
  has Jaccard >= 0.8 with one of the last 400 distinct counted lines; exact
  counts identical text only. Near catches a counting loop whose lines differ
  only in a number word. Detection counts once per crossing, with hysteresis
  at threshold - 0.1, so a nudge's own dip does not count a second loop.
- **nudge:** inject one fixed, short reasoning sentence as if the model had
  generated it; it lands in the KV, the history, `reasoning_content` and the
  stream. Keep watching.
- **slap:** after the grace, force the think-block's closer **by token id** as
  the next token, ahead of any draft token in flight. The model then answers.
- **abort:** if the answer phase itself loops after a slap, end the turn on a
  distinct finish reason, `loop`.
- **Switches:** `--loop-guard off|detect|nudge|slap|ladder` (default off),
  `--loop-guard-match exact|near` (near), `--loop-guard-window N` (200),
  `--loop-guard-threshold T` (0.8), `--loop-guard-grace K` (40); request field
  `loop_guard` overrides the flag
  for that request. Off is today's behaviour exactly: no detector, no forced
  token, no new finish reason.
- **Observability:** detections, nudges, slaps, aborts in the per-request stats
  (the decode line), a `loop_guard` object in the response, and the mode and
  thresholds in `/props`.

## Reference to follow

- vLLM issue #52673 / PR #52677 "ThinkingLoopBreaker" (`paper`): exact
  token-cycle detection in the reasoning section (cycle 4–128 tokens, ≥ 3
  repeats in a 512-token window, checked every 16 tokens after a minimum of
  512 think tokens), action = force the reasoning-end sequence, applied at both
  the main and the rejection-sampler hook so it works with speculative
  decoding, counter `thinking_loop_breaks_total`, per-request opt-out. The
  token-cycle detector is not built here; the line detector is, and either may
  trigger (a follow-up can add the cycle test beside it).
- llama.cpp issue #21264 (`paper`): line-level detection with a temporary
  temperature boost instead of truncation. That "shake" is not a rung here;
  the line detector is reusable for it.
- arcint evidence (`measured-here`): forcing the closer at the loop onset gave
  fully correct answers in 8 of 8 looping coder runs, so the slap rung is the
  one that matters most. The card ladder fixed 5 of 5 live looping tasks (three
  by slap, two after the nudge); the one miss was a counting loop whose exact
  share stayed at 0.50. Offline over 42 reasoning texts (window 200, lines
  > 20 chars): exact clean <= 0.65, loopers 1.0, the counting loop 0.71; near
  clean <= 0.67, loopers 1.0, the counting loop 0.97. Masking digits and number
  words was rejected: one clean numeric enumeration reached 0.89.

## Gate

Unit suite (device-free): the detector fires at the expected line and not
before; a varied stream of the same length never fires; the nudge text appears
once in the reasoning; after the grace the next drawn token is the closer id
and content flows; a looping answer after a slap ends on `loop`; off is
byte-identical; a request field overrides the flag; counters reach the stats
line and the response object. A card window later measures the coder's 30-task
battery: loops that previously hit the cap answer instead, and answers that
already came right are unchanged (the answer-level bar in `CLAUDE.md`).

## Current state

Built, default off; the line match defaults to near. Detection, nudge (forced
token ids), slap (forced closer id), and the content-phase abort are
unit-tested on synthetic streams, including a counting loop that exact misses
and near catches, the near false-positive guard, and the detection hysteresis. The
speculative lane is handled at the token boundary the verify walk samples:
a forced token that differs from the draft breaks the walk on the mismatch, so
the remaining drafts in flight are dropped and only the emitted prefix is
accepted.

Card battery (`measured-here`, 2026-10-23/24, the coder on the A770, its
card's thinking row, seed 1, 64k cap, `ladder`; 30 post-cutoff
competitive-programming tasks, judged in a network-less sandbox against a
reference and generated tests). Screening without the guard: 19 fully
correct, 2 partly wrong, 9 at the cap (8 loops, 1 long thinking). Round 1,
exact match: 6 of 7 live loops fixed, all fully correct; the counting loop
stayed at an exact share of 0.50. Round 2, near match: 9 of 9 cap runs fully
correct (4 by the nudge, 4 by the slap, 1 did not loop), one detection per
loop. False-positive run on the other 21 tasks: 17 run before the window
closed, 16 fully correct, 1 wrong with no guard event (run-to-run variance);
on the near build without stripping, two triggers on code drafts written out
several times (indented short lines passed the 20-character bar), silent with
stripped lines; one real loop there caught and fixed by the nudge. The last
four tasks on the 1.2.3 pre-tag build (2026-10-24): three fully correct with
no guard event; one looped this time and climbed the whole ladder, its answer
looping after the slap (code comments re-checking one contradiction, each
line six times), and the abort ended it without a program (the screening,
guard off, had answered it partly wrong). In all, 21 of 21: 19 fully correct,
1 wrong with no guard event, 1 aborted on a real loop. The operator keeps the
full ladder on the coder.

## Where it lives

`src/core/sampler.{h,cpp}` — the detector, the ladder state and the forced
draw; `src/core/sampling.{h,cpp}` — the mode, the match rule, the thresholds and
the request override;
`src/core/chat.cpp` — the `loop_guard` request field; `src/config.{h,cpp}` —
the flags and `/props` defaults; `src/exec/backend_llama.cpp` — nudge
tokenisation, per-request counters, `FinishReason::Loop` checks and the
spec-lane comment; `src/exec/verify_walk.h` — the forced-token boundary and the
abort latch; `src/api/handlers.cpp` — the decode suffix, the response object
and `/props`; `tests/test_loop_guard.cpp` (red-first cases),
`tests/test_props.cpp`.

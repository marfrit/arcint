# loop-guard — detect a reasoning loop, escalate, end the turn cleanly

**Shipping in 1.2.3 (2026-10-24).** A coder model can circle inside its think block until
the token cap: no answer, a wasted round. The operator approved an escalation
ladder **detect → door → nudge → slap → abort**, behind a switch that is off
by default. The door rung (2026-10-24) lets the model close by its own draw
before anything is injected.

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
- **door:** while the rung is active and the think block is open, mask every
  token whose decoded piece consists only of newline characters and has two or
  more of them (the double, triple ... newline tokens). The single-newline
  token stays legal. The masked id set is computed once per vocabulary from the
  pieces the backend owns and handed to the sampler as the nudge ids are. This
  is the newline trap (`measured-here`, operator, offline): BPE fuses line
  breaks, so after a sentence a done-but-cannot-stop looper takes the double
  newline with ~0.95 while the closer after a single newline has ~0.998; the
  joint chance of closing per round is 0.03-0.05. In clean reasoning the
  closer after a single newline is ~0 (mean 0.0002) until the real end. The
  offline estimate with single line breaks: the looper would close by itself
  within ~25 lines at 50 %. Nothing else changes -- no injected text, no
  forced token; the model draws normally and closes, or not, by itself.
- **nudge:** inject one fixed, short reasoning sentence as if the model had
  generated it; it lands in the KV, the history, `reasoning_content` and the
  stream. Keep watching.
- **slap:** after the grace, force the think-block's closer **by token id** as
  the next token, ahead of any draft token in flight. The model then answers.
- **order:** the door opens at detection and stays open through nudge and slap
  (it helps both). Under `ladder` the nudge comes after
  `--loop-guard-door-grace` more counted lines (default 20) if the loop
  persists; the slap keeps its own grace after the nudge (40). When the block
  closes the door is gone; content is never touched.
- **abort:** if the answer phase itself loops after a slap, end the turn on a
  distinct finish reason, `loop`.
- **Switches:** `--loop-guard off|detect|nudge|slap|door|ladder|ladder-nodoor`
  (default off), `--loop-guard-match exact|near` (near),
  `--loop-guard-window N` (200), `--loop-guard-threshold T` (0.8),
  `--loop-guard-grace K` (40), `--loop-guard-door-grace K` (20); request field
  `loop_guard` accepts the same mode names and overrides the flag for that
  request. `door` is detect + the mask only (never nudge/slap); `ladder`
  includes the door; `ladder-nodoor` is today's ladder exactly (nudge at
  detection, no mask), so a card test can compare both on one binary. `nudge`
  and `slap` stay as they were (no door). Off is today's behaviour exactly: no
  detector, no mask, no forced token, no new finish reason.
- **Observability:** detections, nudges, slaps, aborts, `self_closes` (the
  block closed by the model's own draw while the door was open, not by a slap)
  and `door_masked` (draws at which the mask was active and a masked token sat
  in the row's top-k candidates; whether the model would otherwise have taken
  it is not knowable) in the per-request stats (the decode line), a
  `loop_guard` object in the response, and the mode, thresholds and
  `door_grace` in `/props`.

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

Built, default off; the line match defaults to near. Detection, door, nudge
(forced token ids), slap (forced closer id), and the content-phase abort are
unit-tested on synthetic streams, including a counting loop that exact misses
and near catches, the near false-positive guard, the detection hysteresis, the
door's mask and self-close, a slap's forced closer not counted as a
self-close, the door grace in the ladder order beside
`ladder-nodoor`, and `ladder-nodoor` never opening the door. The door ids
are scanned at model load; the `door_masked` counter reads the draw's own
candidate set. Evidence class:
the mechanism is `code` (this tree); the line-end probabilities and the closing
estimate are `measured-here` (operator, offline, recorded above). The
speculative lane: the door mask applies inside `sample()`, so the verify walk
sees it like any other mask -- a draft token the mask forbids is a mismatch and
drops the rest; the forced-token boundary of the nudge/slap is unchanged.

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

Door card run (`measured-here`, 2026-10-24, the same coder and window,
seed 1, 64k cap, `ladder` with the door; the 9 tasks that had hit the cap):
all 7 looping runs ended fully correct, all 7 closed by the model's own draw
-- 5 by the door alone within seconds of opening, 2 after the nudge -- and no
forced token in the whole round. The same set on the no-door ladder needed
the slap 4 times. The door battery's remaining gap: lines the vocabulary
spells as one newline stay unmasked, and a loop that does not produce a
line-with-newline pattern can still reach the nudge and the slap (that path
is measured, not owed).

## Open points

The two review notes on the door handover are closed (2026-10-24); neither
changes the draw.

- **Door-id scan, thread safety: closed.** The ids are now scanned once at model
  load, where the vocabulary becomes known, by a pure
  `scan_loop_guard_door_ids(Tokenizer&)`; the per-request path only reads the
  result, so no request-time lazy cache exists and lanes cannot race on it. The
  load-time log line stays. A device-free test scans a stub-shaped tokenizer
  from one call and from two concurrent calls and compares the ids; evidence
  `code`, unit suite.
- **`door_masked` without a full vocabulary pass: closed.** The mask keeps its
  `-inf` writes where they were and stores only the masked ids' logits at mask
  time. The draw then reads that small list against the floor of the candidate
  set it collects anyway (the effective row's top-k, or the argmax on a greedy
  row), so the counter no longer walks the vocabulary per draw. The stored
  logits are pre-penalty and the floor is post-penalty; a masked id whose
  penalty would have lowered it can over-report, which is the counter's
  documented slack. A fixed-row test proves the drawn ids are unchanged from
  the pre-change sampler; evidence `code`, unit suite.

## Where it lives

`src/core/sampler.{h,cpp}` — the detector, the door mask, the ladder state and
the forced draw; `src/core/sampling.{h,cpp}` — the mode, the match rule, the
thresholds and the request override;
`src/core/chat.cpp` — the `loop_guard` request field; `src/config.{h,cpp}` —
the flags and `/props` defaults; `src/exec/backend_llama.cpp` — the
vocabulary scan for the door ids, nudge tokenisation, per-request counters,
`FinishReason::Loop` checks and the spec-lane comment; `src/exec/verify_walk.h` — the forced-token boundary and the
abort latch; `src/api/handlers.cpp` — the decode suffix, the response object
and `/props`; `tests/test_loop_guard.cpp` (red-first cases),
`tests/test_props.cpp`.

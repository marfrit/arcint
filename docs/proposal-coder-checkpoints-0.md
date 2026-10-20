# Unit change proposal — coder: `--llama-checkpoints 0`

Status: **applied by the operator, 2026-10-20** (the flag is in
`arcint-coder-llama.service`'s `ExecStart`; the unit has run with it since
12:48 UTC and every sampled batch below was taken with it in place). This
document is the evidence for that change, and it now also records the pass bar
the operator set on the same day — see "The bar is a MEAN, not a count".
Every number is `measured-here`, 2026-10-20, on the coder host's A770
(`card0`, PCI `8086:56a0`, GPU.1), `arcint 1.2.1 (83d7609)`, sha256
`86b33f88fd4f609a`, model `qwen36-coder-iwo3-c5f495ac-shq8.gguf`
(`27931997cfeec31f`).

## The change

Add one flag to `ExecStart`:

    --llama-checkpoints 0

Nothing else changes: same GGUF, same `--n-ctx 98304 --llama-mtp 4
--llama-kv q8_0:q4_0`, same port and served name.

## What it buys

The acceptance task, thinking off, which is the cell's setting
(`~/pruefstand-coder/run.sh` sets `DECKEL=4000 OHNE_DENKEN=1`):

| arm | T=0 (5 runs) | 30 runs at 0.7 | mean |
|---|---|---|---|
| unit as it runs now (checkpoints on) | 0 of 5 at 10/10, all 9/10 | 26 at 10/10, 3 at 9, 1 at 8 | 9.833 |
| `--llama-checkpoints 0` | **5 of 5 at 10/10** | 27 at 10/10, 3 at 9 | 9.900 |

Greedy goes from a reproducible 9/10 to a reproducible 10/10. The failure it
removes is one bug: the model writes the CRLF branch of its line walker with
`i = i + 1` and the bare-LF branch without it, so the parser spins on LF input
and the harness burns its instruction budget. With checkpoints on the model
emits that bug five times identically; with checkpoints off it does not emit
it at all. Same prompt, same GGUF, same sampler — only where the prefill
batches break differs, because checkpoints force batch boundaries near the
prompt's end and at user messages.

This reproduces what `docs/llama-engine.md` recorded on 0.5.12 (9 against 10,
"one case flipping with the batch layout") two engine versions later on 1.2.1.
It is not a new effect; it is the same one, still live.

VRAM: 15,879 MiB peak against 15,877 for the unit — 2 MiB. Nothing paged, GTT
32 MiB, host RSS 711 MiB, `MemAvailable` never below 90,967 MiB.

## What it costs — this is the part to weigh

Checkpoints exist for production traffic. Counting that traffic honestly means
excluding my own acceptance batteries, which dominate the journal: of 153
prefill lines since 2026-10-13, **145 are prompts under 1,000 tokens** (the
battery's 282-token task, mine). The production traffic is the **8 large
prefills**: 24,069 / 26,871 / 29,465 / 30,442 / 32,480 / 32,496 tokens, plus
two around 1.5k.

Of those 8, **5 resumed from a checkpoint** ("`slot: lane 0 resumes from its
checkpoint at N tokens (62.8 MiB)`") and reused 88-94 % of their prompt; they
prefilled 11,983 tokens between them instead of ~175,000. `docs/llama-engine.md`
measured what that resume is worth on this model: a 24.3k-token follow-up
**34.06 s cold against 0.54 s warm**, an edited message 34.06 s against 0.48 s.
The unit's own cold prefill at that depth is 835-865 t/s, which lands in the
same place.

So the flag costs roughly **34 s of re-prefill per long follow-up or edited
message, and this unit saw 5 of those in seven days**. That is a small bill on
this unit's actual traffic — but it is traffic-dependent, and an agent harness
pointed at the coder in earnest would raise it by an order of magnitude. The
flag also frees the host memory the checkpoints held: 62.8 MiB each, up to 32
a lane.

## The bar is a MEAN, not a count (`decision`, operator, 2026-10-20)

The pass bar for the coder service, as the operator set it after seeing this
batch's spread:

- **greedy: 5 of 5 runs at 10/10**, and
- **mean ≥ 9.8 over 30 sampled runs at 0.7**.

The earlier "at least 27 of 30 at 10/10" wording is **dropped**, on this
session's own measurement: the no-knob anchor scored 26/30 and 25/30 in two
batches of the same configuration, so a count bar would fail the production
unit on most measurements of itself. A mean is the right statistic because it
weights a 0/10 run correctly — see below.

### What the mean bar actually demands (`measured-here`, arithmetic)

Scores are 0–10 over 30 runs, so `mean = (Σ scores)/30`. One zero-score run
caps the batch mean at `10 × 29 / 30 = 9.667`. **A single 0/10 run makes
mean ≥ 9.8 arithmetically unreachable**, whatever else happens:

| zeros in the batch | highest mean attainable | mean ≥ 9.8 |
|---|---|---|
| 0 | 10.000 | needs ≥ 24 of 30 at 10/10 |
| 1 | 9.667 | **impossible** |
| 2 | 9.333 | **impossible** |

So the mean bar is not softer than the count bar — it is a **zero-rate bar**
wearing a friendlier face. Passing means: no zero-score run in 30, and at
least 24 of the 30 at 10/10.

### The zero is a different defect from the loop bug

The 9/10 losses are the parser's LF branch spinning (`ZEITLIMIT: Instruktions-
budget erschoepft`) — a bug in the *generated program's logic*, costing one
point. The 0/10 runs are **programs that do not parse**: `LADEFEHLER: :50:
syntax error near 'end'`, `:19: syntax error near 'else'`. Different defect,
different cost, and under the mean bar it is the one that decides the gate.

Zero-runs at 0.7 are not new to this repository: `docs/llama-engine.md:99-101`
records the dense 27B sampling `9, 0, 10, 10, 0, 10`, and `:109` records
builds whose KL agreed to four places scoring `10/10` and `0/10`. The defect
is the task's, at temperature, not this unit's.

### Production measured against the mean bar

Seven 30-run batches of the shipping configuration (`--llama-checkpoints 0`,
no sampler knob, thinking off, cap 4,000), same day, same unit:

| batch | mean | ±SE | hist |
|---|---|---|---|
| window leg | 9.900 | 0.056 | 27×10, 3×9 |
| anchor 1 | 9.867 | 0.063 | 26×10, 4×9 |
| anchor 2 | 9.800 | 0.088 | 25×10, 4×9, 1×8 |
| mean-bar 1 | 9.900 | 0.056 | 27×10, 3×9 |
| mean-bar 2 | **9.233** | 0.462 | 25×10, 3×9, **2×0** |
| zero-rate anchor 1 | 9.833 | 0.061 | 25×10, 5×9 |
| zero-rate anchor 2 | 9.800 | 0.088 | 25×10, 4×9, 1×8 |
| **pooled, n = 210** | **9.762** | **0.071** | 180×10, 26×9, 2×8, **2×0** |

**The bar is not established for production as it runs today.** Pooled
9.762 ± 0.071, 95 % CI [9.62, 9.90] — the bar at 9.8 sits inside that
interval, so the honest verdict is *not established*, not *failed*. Batch
means scatter 9.233–9.900 (sd 0.29 between batches, four times the
within-batch SE), and all of that scatter is the zero-rate.

At the measured rates — zero-rate **0.0095 a run** (2 of 210), perfect-rate
0.865 among non-zero runs — a batch clears the bar with probability ≈ **0.68**:
0.75 that no zero appears in 30, times 0.90 that ≥ 24 of them are perfect.
That second factor is the ceiling: **even with a zero-rate of exactly nil this
configuration clears the bar only about 90 % of the time.** To clear it 80 %
of the time the zero-rate must be ≤ 0.0039 a run (1 in 254); it is currently
1 in 105.

Settling the bar to ±0.03 would take about **1,190 sampled runs** (≈ 40
batches of 30, ≈ 4.7 h of the unit). Nobody should spend that to argue about
a tenth of a point; the honest report is the interval above.

### Consequence for the flag under debate

`--llama-checkpoints 0` is **not supported by the mean bar**. Its seven
batches pool to 9.762; the one pre-change batch measured on this unit
(checkpoints on) scored 9.833. The difference, 0.071 ± 0.128, is not
significant — but the flag's justification was greedy 10/10, and the mean bar
does not contradict that either. The flag is carried for the greedy cell; the
sampled mean is orthogonal to it and is governed by the zero-rate, which the
flag does not touch.

Two arms were measured against that demand and both lose:

- **Thinking on** (`enable_thinking: true`, cap 16,000): greedy 5/5 at 10/10,
  sampled **15 of 30** at 10/10, mean 7.133, 6 zeros. Two zeros were the model
  spending the entire budget reasoning and never answering; four were the
  answer wrapped in an answer-tag with no code fence, which the harness cannot
  extract. Those four were re-scored after stripping the wrapper: 10/10, 8/10,
  8/10, 5/10 — so one perfect answer was lost to the extractor and three were
  genuinely poor. Corrected: 16 of 30. Cost: 8,073 tokens and 126 s an answer
  against 531 tokens and 7 s.
- **KV precision**: not run, deliberately. `docs/llama-engine.md` already ran
  this task at 30 runs per arm across KV types (q8_0 15 of 30 mean 7.4; f16 19
  of 30 mean 8.13, at 65,536 tokens) and recorded the gap as under one standard
  error, reversing in a second batch. The arm's buffers say the change would
  fit (KV is 780 MiB at `q8_0:q4_0`, 1,201 MiB of headroom, so V at q8_0 costs
  about +220 MiB), but it is recorded as **not tried**, not as verified
  against.

## Sampler knobs at the card temperature — measured, no lever found

The next lever the operator named: `min_p`, `top_p`, `top_k`, one variable per
arm, 30 sampled runs at 0.7 each. No card window: arcint reads all three
**per request** (`src/core/chat.cpp:142-147`), so the knob rides in the request
body and the unit is never touched.

**A check that can fail first: are the knobs honoured at all?** Two probes,
`wp/coder-quality/honour.py` (`measured-here`):

- **Parse probe.** A mistyped value must be refused. `top_k: "not-a-number"`
  → `HTTP 400 top_k must be an integer`; likewise `top_p` and `min_p`. A key
  that was being ignored would have answered 200, and every arm built on it
  would have been the baseline wearing a hat.
- **Degenerate-limit probe.** `top_k=1`, `min_p=0.99` and `top_p=1e-6` each
  collapse the distribution to the argmax, so a sampled request must reproduce
  the greedy answer byte for byte. All three did, **3/3 each**.

One structural fact from the code: `sampler.cpp:100` returns the argmax
**before** any knob runs when temperature is 0. These knobs cannot move the
greedy arm at all — they are levers for the sampled bar only.

| arm, 30 runs at 0.7 | at 10/10 | mean | what broke |
|---|---|---|---|
| anchor, no knob | 26 | 9.867 | 3× `nur LF`, 1× `einfach CRLF` |
| `min_p 0.05` | 27 | 9.600 | one 0/10 (Lua syntax error) |
| `min_p 0.1` | **29** | 9.967 | looked best |
| `top_p 0.9` | 21 | 8.467 | 4× 0/10 |
| `top_p 0.95` | 22 | 7.933 | 6× 0/10 |
| `top_k 20` | 26 | 9.867 | same as anchor |
| `top_k 40` | 25 | 9.800 | same as anchor |
| **`min_p 0.1`, second batch** | **22** | 9.733 | **did not replicate** |
| **anchor, second batch** | **25** | 9.800 | **beat it** |

Scored against the mean bar instead of a count: `min_p 0.1` pooled over four
batches (n = 120) gives **9.767 ± 0.089** against production's **9.762 ±
0.071** — a difference of −0.005 ± 0.114. The knob is neutral on the mean too.

A second hypothesis was formed from this sweep and then killed by its own
test. `min_p 0.1` had produced **no** zero-score run in its first 60 runs while
the no-knob arm produced 2 in 150, which under a mean bar is the only
difference that matters. A balanced design (two batches of each, interleaved,
`zerorate.sh`) settled it: `min_p 0.1` produced a zero in its fourth batch
(mean 9.500) and the overall zero-rates came out **0.0095 a run** for no-knob
against **0.0083 a run** for `min_p 0.1` — 3 zeros in 330 runs, indistinguishable.
The knob does not suppress the defect. Recorded because the hypothesis was
reasonable and the check that killed it was cheap.

`min_p 0.1` winning by three runs and then losing by three is the same shape
the KV-precision A/B has in `docs/llama-engine.md:214-219`: pooling the six
neutral arms (both anchors, both `min_p`, both `top_k`) gives 180 perfect runs
of 210, p̂ = 0.857, and the spread of a 30-run arm at that rate is ±1.9 runs —
so a three-run gap is noise. At p̂ = 0.857 the odds of any arm reaching 27/30
are 36 %, of 30/30 1.0 %.

**Why top_p collapses is worth recording — it is not the loop bug.** The 0/10
runs under `top_p 0.9/0.95` are syntactically invalid Lua:
`LADEFEHLER: :19: syntax error near 'else'` in six of seven cases, at the same
line of different programs. Tightening top_p pushes the model into malformed
control flow, which scores zero instead of nine. So top_p is not neutral here,
it is harmful; `min_p` and `top_k` are neutral.

**Disposition:** no sampler knob moves the sampled mean. The lever that does
exist is the one already applied, `--llama-checkpoints 0`, and it moves the
greedy cell, not the mean.

## Recommendation

Under the mean bar the decision splits cleanly, because the two criteria are
governed by two different defects:

1. **Greedy (5 of 5 at 10/10): apply `--llama-checkpoints 0`.** Already in the
   unit, free in VRAM (+2 MiB), removes a reproducible defect rather than
   masking it, and costs about 34 s of re-prefill per long follow-up — a small
   bill on this unit's measured traffic (5 such follow-ups in seven days). The
   mean bar neither supports nor contradicts it.
2. **Sampled mean >= 9.8: not established, and no knob found here establishes
   it.** Production pools to 9.762 +/- 0.071 over 210 runs. Sampler knobs are
   measured and neutral (`min_p`, `top_k`) or harmful (`top_p`). The bar's
   binding constraint is the **zero-rate** — 1 run in 105 emits Lua that does
   not parse — and its ceiling: P(>= 24 perfect) = 0.90 even at a zero-rate of
   nil. State the bar with its measured probability (about 0.68 of a batch) or
   move the model; do not treat 9.8 as a property of this configuration.
3. **The lever the mean bar points at, and it is not a card lever.** Removing
   the two zeros from those same batches lifts the mean from 9.762 to **9.856
   +/- 0.026**, clearing the bar with the interval off it. That is reachable
   only by rejecting an unparseable answer and regenerating it: a syntax guard
   (`luac5.4 -p` is on the host, and `pruefstand.lua` proves the same thing by
   loading the file) in front of the coder, not inside the engine. Price: one
   extra generation in about 1 in 105 requests, roughly 7 s. **This is a
   conflict between a measured gain and a DESIGN invariant, so it is the
   operator's call, not mine**: a retry loop in the served path changes what
   the service promises about determinism, and the answer-level bar judges
   correctness at the answer, which such a guard would be intervening in. Not
   built, not claimed done — recorded as the only measured route to 9.8.
4. **`--llama-mtp 0` is now the weakest candidate, not the strongest.** It was
   listed as the untouched lever while the count bar was in force. Under the
   mean bar the zeros dominate, and `docs/llama-engine.md:440-444` records a
   drafter producing 0/10 twice with the same bytes, so MTP 0 might reduce
   them — but the zeros seen here are syntax errors at :19/:50/:60 in the
   middle of a program, not drafter artefacts, and the arm costs a card window
   plus 70 -> 47 t/s. Spend it only for the batch-layout measurement itself.

## Reproduce

On the coder host, `~/wp/coder-quality/`: `leg-ckpt0b.sh` is the window leg
(sampler started before the leg, anchored to `^/usr/bin/arcint ` plus the
arm's own flag substring — the flash-next unit on the other card holds fds on
both render nodes, so a bare anchor misattributes its memory). Logs in
`out-ckpt0b/`; the thinking arms are `think-t0.log`, `think-t07.log` and
`accept-think.py` in the same directory. `base-t0.log` and `base-t07.log` are
the baseline.

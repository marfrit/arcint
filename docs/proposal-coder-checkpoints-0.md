# Unit change proposal — coder: `--llama-checkpoints 0`

Status: **applied by the operator, 2026-10-20** (the flag is in
`arcint-coder-llama.service`'s `ExecStart`; the unit has run with it since
12:48 UTC and every sampled batch below was taken with it in place). This
document is the evidence for that change, and it now also records the pass bar
the operator set on the same day and revised the same evening to a
non-inferiority rule — see "The bar, as the operator finally set it".
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

## The bar, as the operator finally set it: NON-INFERIORITY (`decision`, 2026-10-20)

The probabilistic form the operator chose is not "the batch mean must reach
9.8". It is:

- **greedy: 5 of 5 runs at 10/10**, and
- **over at least 90 pooled runs at 0.7, the bar counts as FAILED only if the
  upper 95 % bound of the mean lies below 9.8.**

That is a non-inferiority test: the unit is presumed to meet 9.8 unless the
measurement demonstrates it cannot. It is the right shape given the scatter
measured below — a single unlucky batch can no longer fail the unit, and a
configuration can only fail it when enough runs place its whole interval
under the bar.

**Production's verdict under that rule: not failed.** `measured-here`, the
eight interleaved batches of the shipping configuration, n = 240, mean 9.771:

| n | SD used | SE | upper 95 % bound | verdict |
|---|---|---|---|---|
| 240 | 0.976 (pooled per-run) | 0.063 | **9.874** | **not failed** |
| 240 | 1.210 (batch-implied, overdispersed) | 0.078 | **9.899** | **not failed** |
| 90 | 0.976 | 0.103 | 9.940 | not failed |
| 30, the worst batch measured (9.233) | 0.976 | 0.178 | 9.526 | would fail on its own — which is why the rule sets a 90-run floor |

Both SDs clear it, the conservative one by 0.10. The greedy half is met
outright (5/5 at 10/10, three times running). **The coder unit passes the bar
as the operator has now set it.**

What it would take to fail this unit — the run count at which the upper bound
drops under 9.8, for a given true mean:

| true mean | pooled SD | overdispersed SD |
|---|---|---|
| 9.70 | 258 runs | 396 |
| 9.75 | 1031 | 1585 |
| 9.77 (the measured point estimate) | 2864 | 4402 |
| 9.80 | never | never |

And the protection it buys: the worst batch measured in this session, 9.233,
would fail the unit under a naive per-batch reading (its own upper 95 % bound
is 9.53 pooled, 9.60 overdispersed — well under the bar). Under the rule as
set it cannot, because the floor is 90 pooled runs.

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
builds whose KL agreed to four places scoring `10/10` and `0/10`. But the
historical record does **not** carry this failure mode: scoring all 653
`antwort-*` answers under `~/pruefstand` (`measured-here`), 43 score 0, of
which **14 are syntax errors and 29 are other** (mostly the 2026-10-03
population, which is a different task mix: 33 runs score 2/10), and **not one
of the 653 contains a Python-ism**. The `pass`-in-Lua mode is specific to this
model and this configuration, not to the task.

### Every zero in the sweep, classified by cause (`measured-here`)

The operator's check of `mb-2-08.lua` and `mb-2-13.lua` found the mechanism:
with thinking off the model keeps reasoning aloud inside the CR/LF branch —
27 and 19 comment lines against 7 in a passing answer — and then writes the
Python statement `pass`, once preceded by `-- placeholder, will rewrite
below`. Confirmed by reading the files: `mb-2-08.lua:59` is
`pass -- placeholder, will rewrite below`, `mb-2-13.lua:49` is a bare `pass`,
comment lines 25 and 19 against 10 in a passing answer. So the zero mode is
**thinking leaking into the code, then a Python-ism**.

Classifying all 510 sweep runs (`classify.py`), so an arm is never credited
with fixing a defect it never touched:

| arm | n | mean | zeros | syntax | cap | extraction |
|---|---|---|---|---|---|---|
| anchor (no knob, pooled) | 60 | 9.833 | 0 | 0 | 0 | 0 |
| anchor 1 / anchor 2 | 30 / 30 | 9.833 / 9.800 | 0 | 0 | 0 | 0 |
| mean-bar 1 / mean-bar 2 | 30 / 30 | 9.900 / **9.233** | 0 / **2** | 0 / **2** | 0 | 0 |
| `min_p 0.1` (4 batches) | 120 | 9.767 | 1 | **0** | 1 | 0 |
| `min_p 0.05` | 30 | 9.600 | 1 | 1 | 0 | 0 |
| `top_k 20` / `top_k 40` | 30 / 30 | 9.867 / 9.800 | 0 | 0 | 0 | 0 |
| `top_p 0.9` | 30 | 8.467 | 4 | 2 | 2 | 0 |
| `top_p 0.95` | 30 | 7.933 | 6 | 6 | 0 | 0 |
| thinking on | 60 | 8.483 | 6 | 4 | 0 | 2 |
| **total** | **510** | | **20** | **15** | **3** | **2** |

**The discriminator is the Python-ism, not the digression alone.** Across the
sweep: 15 syntax zeros, of which **26.7 % contain a Python statement** —
against **0.0 % of all 477 non-zero runs** (`pass`, `elif`, `None`/`True`,
`def`, `.append(`, `#`). Comment density by itself barely separates (median 8
lines in the syntax zeros against 7 in the perfects); the digression that
matters is a *run* of consecutive comment lines, and it tracks the score:

| longest comment run in the emitted code | n | mean | syntax zeros | perfect |
|---|---|---|---|---|
| < 5 lines | 419 | 9.671 | 7 | 85.0 % |
| 5–7 lines | 22 | 8.455 | 3 | 72.7 % |
| ≥ 8 lines | 9 | **6.333** | 1 | 44.4 % |

### Ranking the knobs by the zero cause, which is the only thing the mean bar sees

The zero-rate cannot rank arms: at ~1 % a run, showing a halving needs ~4,100
runs an arm (~11 h each). The syntax-zero counts, pooled over every batch of
this sweep, do separate one arm:

| arm | syntax zeros | rate | Fisher p vs no-knob |
|---|---|---|---|
| no-knob (production) | 2 / 180 | 0.0111 | — |
| `min_p 0.1` | 0 / 120 | 0.0000 | 0.52 |
| `top_k 20/40` | 0 / 60 | 0.0000 | 1.00 |
| `top_p 0.9/0.95` | **8 / 60** | **0.133** | **0.0003** |

So the ranking is: **`top_p` is the only knob that measurably changes the
zero-rate, and it raises it twelve-fold** (8 syntax zeros in 60 runs against 2
in 180, Fisher p = 0.0003). `min_p` and `top_k` are indistinguishable from the
no-knob arm and must not be credited with anything. The earlier
`min_p 0.1` digression signal (4 digressions in 120 runs against 15 in 180,
Fisher p = 0.094) is suggestive and **not** significant; it is recorded as a
hypothesis for a future arm, not as a finding.

### Production measured against the mean bar

Seven 30-run batches of the shipping configuration (`--llama-checkpoints 0`,
no sampler knob, thinking off, cap 4,000), same day, same unit. (An eighth
batch of the same configuration ran later, interleaved with the
system-message arms; the eight pooled are tabulated in "The shipped
configuration, pooled over eight interleaved batches" below, and supersede
the pooled figure here. The zero-rate quoted here, 1 in 105, becomes 1 in 120
over the eight.)

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

**Production is not shown to meet the bar** *(under the pre-non-inferiority
reading; see "The bar, as the operator finally set it" above for the verdict
under the rule that now governs — not failed)*. Pooled 9.762 ± 0.071, 95 % CI
[9.62, 9.90]: the bar at 9.8 sits inside that interval, and the pooled point
estimate is within one SE of it. That is *not shown to meet*, not *fails* —
the measurement cannot resolve a tenth of a point at n = 210. What the
measurement does resolve is the scatter: batch means run 9.233–9.900 (sd 0.29
between batches, four times the within-batch SE), and all of that scatter is
the zero-rate.

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

`--llama-checkpoints 0` is **not supported by the mean bar**. Its eight
batches pool to 9.771 (CI [9.64, 9.87]); the one pre-change batch measured on
this unit (checkpoints on) scored 9.833. The difference, 0.062 ± 0.128, is not
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

## The bar is inside the noise of one measurement of it (`measured-here`)

Five independent 30-run batches at 0.7 of **one and the same configuration**
(the shipped unit, `--llama-checkpoints 0`, no sampler knob) — the first five
of the eight tabulated later; the eight give SD 0.221, the five 0.286, and the
conclusion is the same either way:

| batch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| mean | 9.900 | 9.233 | 9.900 | 9.800 | 9.867 |
| at 10/10 | 27 | 25 | 27 | 25 | 26 |
| zeros | 0 | 2 | 0 | 0 | 0 |

Batch-mean SD **0.286** over these five, **0.221** over all eight. So a single
30-run measurement of the production unit carries a 90 % interval of
**± 0.36–0.47** around its own mean, and the bar 9.8 sits well inside it: four
of these five batches reach 9.8, one (9.233) misses it by 0.57, and nothing
about the unit changed between them. The spread is the zero mode arriving or
not arriving in a given batch, which is what a 30-run batch is a draw from.

Runs per arm needed for 80 % power at alpha = 0.1, computed from the measured
variance (`measured-here`, arithmetic). Two SDs are given: the pooled per-run
SD of the 240-run histogram (**0.976**), and the larger SD implied by the
between-batch scatter (**1.21**, i.e. treating the batch-to-batch excess as
real overdispersion). The honest column is the second:

| to detect | per arm (pooled SD) | per arm (batch-implied SD) |
|---|---|---|
| 0.30 points of mean | 131 | 201 |
| 0.20 points | 295 | 453 |
| 0.15 points | 524 | 806 |
| 0.10 points | 1178 | 1813 |
| zero-rate 0.83 % → 0.30 % | 2473 | — |
| zero-rate 0.83 % → 0.00 % | 741 | — |

At 30 runs per arm the 90 % interval is ± 0.36–0.47; at 300, ± 0.11–0.15.
**A 30-run arm cannot rank two configurations of this unit at the resolution
the bar is set at**, and a 60-run arm cannot either (± 0.26–0.33). Any A/B
proposed against this bar must say how many runs it needs, or it measures
nothing.

This also settles the shape of the bar itself. As written ("mean >= 9.8 over
30 sampled runs"), the production unit fails it about 1 time in 3 measured
batches of the configuration that currently meets the greedy half. Either
state the bar with its probability ("mean >= 9.8 with the 90 % interval off
it", which needs ~300 runs), or set it where one batch can decide it.

## What this does to the quant A/B that was proposed

The proposed design — arm Q = `qwen36-coder-Q8_0.gguf` on the A770 with
CPU-MoE offload, arm S = production shq8, 60 runs each at 0.7, decide the
lever on a Fisher test at p < 0.1 — cannot answer the question it asks. Three
independent reasons, each checked:

1. **Power.** With production's measured zero-rate of 0.83 %, the proposed
   rule has power 0.000 at 60 runs per arm, 0.005 at 150, 0.05 at 300 and
   0.26 at 750 (simulation, 8000 draws, α = 0.1). Even a quant that removed
   the zero mode **entirely** needs ~740 runs per arm to show it at p < 0.1.
   The mean is no better a detector of it: a zero costs 9.77 points of a
   30-run batch's total, so 0.83 % → 0.30 % lifts the mean by 0.05 points,
   which needs ~4,400 runs per arm. The test would come back "no difference"
   whatever the truth, and that "no difference" would be read as a verdict.
2. **The arm is not the same deployment.** `qwen36-coder-Q8_0.gguf` is
   25.99 GiB; expert bodies are 91 % of its weight bytes (23.68 GiB), and the
   A770 has 17.08 GB. Serving it means `--llama-cpu-moe` moving ~10 GiB of
   experts to host memory — a 5700X, 8 cores, against production's all-on-card
   path. The reference for that route is `measured-here`: 14.6 t/s on the same
   CPU family at 26k (`docs/campaigns/research-hybrid-expert-execution.md:270`)
   against the 70 t/s production decodes at. So arm Q would change the
   quantization AND the execution route AND the decode speed by ~5x, in one
   arm. A difference could not be attributed to the quant.
3. **The prior art already covers the axis, and its own confound runs the
   other way.** `docs/llama-engine.md:596-620` (`measured-here`, 2026-10-05)
   searched `c5f495ac` *against the Q8_0 logits*: Q8_0 is the KL and top-1
   reference there (c5f495ac KL 0.02305 / top-1 95.33 %, shq8 0.02367 /
   95.43 %, Q4_K_M 0.02374 / 95.53 %). Its answer-level column reads c5f495ac
   9/10 at T=0, shq8 10/10, Q4_K_M 10/10 — quant flipping a deterministic
   answer without moving KL. But that table predates 83814b3 (2026-10-16, the
   drafter's state through checkpoints) and was recorded in b642716, the same
   commit that introduced `--llama-checkpoints` with default 32 — so that
   table ran with checkpoints ON, the configuration this proposal shows scores
   9/10 at T=0. Its per-GGUF differences are therefore partly the checkpoint
   configuration, not the quant.

What would settle the quant question, if it is to be run: the same execution
route in both arms. The A770 cannot hold all-Q8_0 at 98,304 tokens, so the
fair comparison is on the B60 (24 GiB, and `arcint-flashnext` currently holds
it), or at a context short enough that both quants sit on one card.
**The operator closed this axis on 2026-10-20: not run.** It is closed as
*answered by these numbers*, not as forbidden — and reopening it has a stated
cost, so it cannot be reopened casually.

### How the quant axis would be reopened

The defect of the proposed A/B was that it changed quant, execution route and
decode speed in one arm. Reopening it means removing that confound first.

1. **Same engine, both quants, no offload in either arm.** Not arcint on the
   A770 — it cannot hold the 25.99 GiB Q8_0 at any useful context. The place
   is **`bosch`, the DGX Spark (ASUS Ascent GX10, GB10 Grace Blackwell,
   compute capability 12.1, 20-core aarch64, 121 GiB unified memory),
   llama.cpp built with CUDA** (`~/src/llama.cpp-latest` is there; a
   `-DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DCMAKE_CUDA_ARCHITECTURES=121` recipe
   already exists on that host, fleet record 2026-09-09, `code`). Both files
   fit there outright — Q8_0 25.99 GiB, shq8 13.59 GiB — so both arms run
   all-on-card, same route, same link, differing only in the bytes. Neither
   coder GGUF is on `bosch` yet; they would have to be copied (26 + 14 GiB).
2. **Run count taken from the power simulation, targeting power ≥ 0.8 at
   α = 0.1.** The originally proposed 60/arm had power 0.000; **750/arm
   reaches only 0.26** — nowhere near enough. Simulated (Fisher exact on the
   zero-rate, 1200–1500 draws, production rate 0.83 %):

   | alternative | runs per arm for power ≥ 0.8 |
   |---|---|
   | quant removes the zero mode **entirely** (0.83 % → 0 %) | **~850** |
   | quant cuts it to a third (0.83 % → 0.30 %) | **~2800** |

   The honest number for a realistic gain is therefore **2800 runs per arm**:
   5600 runs, about **11 hours of serving** at the ~7 s a run costs here, on a
   card that is not the production card. The cheaper variant, 850 per arm
   (~3.3 h total), detects only a quant that makes the defect disappear.
3. **Judge the zero-rate, not the mean.** The same alternative is far cheaper
   to see in the zero-rate than in the mean: 0.83 % → 0.30 % shifts the mean
   by only 0.052 points (a zero costs 9.77 points of a 30-run batch's total),
   which needs ~6800 runs per arm on a mean test (overdispersed SD) against
   ~2800 on the Fisher. If the axis is reopened, the zero-rate is the
   statistic to run it on.
4. **What a result would carry.** A CUDA/llama.cpp result is a statement about
   the **weights**, not about arcint: it transfers to production only as a
   model-level fact, and the answer-level bar would still have to be
   re-measured on arcint on the A770 before anything moved. The KL and top-1
   columns of `docs/llama-engine.md:611-615` stay the cheap first filter —
   unaffected by the checkpoint confound noted there — with this caveat:
   Q8_0 is already their reference, so an A/B that starts from Q8_0 starts
   from the reference itself and cannot improve on it in those columns.

## The prompt lever, measured — no gain (`measured-here`)

Four arms, 30 runs each, **round-robin interleaved** so any drift across the
hour hits every arm equally, temp 0.7, thinking off, cap 4,000, live
production unit, no card window (a `system` message is honoured by the engine:
`src/core/chat.cpp:14` admits `system`/`developer`, `chat.cpp:483` renders it;
a live probe returned the instructed token). Driver
`~/wp/coder-quality/accept-sys.py`, logs `sys-screen.log`, files `out-sys/`.

| arm | system message | mean | at 10/10 | zeros | syntax-bad | Python-ism | comment-run ≥ 5 |
|---|---|---|---|---|---|---|---|
| anchor | none | 9.833 | 26 | 0 | 0 | 0 | 1 |
| neutral | "Du bist ein hilfreicher Assistent…" | 9.933 | 28 | 0 | 0 | 0 | 2 |
| short | finished runnable Lua, no planning comments, no placeholders, no `pass`/`elif`/`None` | 9.867 | 27 | 0 | 0 | 0 | 1 |
| strict | answer **only** with a complete Lua program in one code block, no planning text, no comment blocks, no placeholders, no Python syntax, write out every branch | 9.500 | 25 | **1** | **1** | 0 | **0** |

Read it honestly: **nothing here is a lever.** The spread across four arms
(9.500–9.933) is smaller than the spread of repeated batches of one
configuration (9.233–9.900), the ordering is not monotone in instruction
strength, and `neutral` — an instruction with no content relevant to the
defect — beats the two arms that name it. The one zero sits in the strictest
arm, and Fisher on 1/30 against 0/30 is p = 1.0: it licenses no claim either
way. The paired proxy that the cause analysis predicted would move — the
comment-run signature — does move in the intended direction (`strict` 0 runs
with ≥ 5 consecutive comment lines, the only arm with none), but the zero it
produced is **not** the Python-ism mode:

```
luac5.4: out-sys/strict-15.lua:73: '::' expected near 'end'
```

The answer used `goto continue_loop` and wrote the label as `::continue_loop`
without the closing `::`. That is a Lua slip in a construct the passing
answers also use, with no comment block and no Python-ism in sight. So
suppressing the digression signature did not remove a zero; it coincided with
a different one. The zero mode is not one defect with one cause reachable
from the prompt.

**Decision (operator, 2026-10-20): client-side only.** arcint stays as it is;
whatever a caller wants in a `system` message is the caller's business. The
engine's behaviour is unchanged, nothing is injected into the served path, and
no client of the coder service gets a new sentence it did not ask for.

## Decisions on the record (operator, 2026-10-20)

| question | decision | what it means |
|---|---|---|
| the mean bar | **non-inferiority** | greedy 5/5 at 10/10, and over ≥ 90 pooled runs at 0.7 the bar counts as failed only if the **upper 95 % bound of the mean** lies below 9.8. Production's verdict under it: **not failed** (upper bound 9.874 / 9.899 conservative, n = 240) |
| the system message | **client-side only** | arcint unchanged, nothing injected in the served path |
| syntax guard + regenerate | **rejected** | no retry loop in the served path — and see below: at greedy a retry returns the same bytes, so it would not terminate |
| the proposed quant A/B | **dropped, axis closed** | not run; see the power and confound arguments above |
| the 2026-10-05 quant table | **annotate the confound** | done in `docs/llama-engine.md`, no re-measurement |

The rejection of the syntax guard matters for how the remaining gap is read:
the one intervention that cleared the bar with its interval off it
(9.856 ± 0.026) is off the table **by choice**.

### Why the guard-and-regenerate route is closed

The reason first recorded for refusing it (a determinism promise) does not
survive checking; the corrected reason is stronger, so the decision stands and
the argument changes. Evidence: `code` plus `measured-here`, 2026-10-20.

- DESIGN §3.4 point 3: *"Deterministic replay is a cheap default, not a law"
  … may be dropped by a mechanism that buys measured speed.* It is not an
  invariant.
- DESIGN §3.6 says an unseeded request gets a logged seed. **It does not get
  a logged seed**: `src/exec/backend_llama.cpp:903-907` draws one from
  `std::random_device` when `seeded` is false, and no code path in `src/`
  logs it (searched for every `seed` site with a logging call; the only hit
  is a comment in `src/core/sampler.h`). The seed is accepted per request
  (`src/core/chat.cpp:154-156`, `src/core/sampling.cpp:73-75`) and never
  surfaces in the API or `/props`.
- Measured on the live unit at temp 0.7, three requests with
  `"seed": 424242`: **byte-identical output, 3/3** (sha256
  `690d5110…` on all three). Two unseeded requests at the same temperature:
  two different outputs. So a seeded request IS reproducible today and an
  unseeded one already is not — the sampled path has no reproducibility for
  a retry to break.

**Therefore the guard-and-regenerate proposal fails for a mechanical reason,
not a promise-breaking one.** At T=0 the decode is argmax: regenerating the
same request returns the same bytes, so a retry loop that triggers on
`luac5.4 -p` would spin forever on the deterministic defect — the exact
defect `--llama-checkpoints 0` already removed. To make the retry terminate
at greedy it must perturb something (bump the seed, nudge the temperature),
and *that* is the change that alters what a greedy request returns. At T=0.7
unseeded it changes nothing at all, because nothing is reproducible there to
begin with.

Two consequences worth keeping:

1. **If reproducibility at sampled temperature is wanted, it is a one-line
   client change, not a service change**: send a `seed`. It already works,
   byte for byte. The missing piece on the engine side is that §3.6 promises
   the seed of an unseeded request is logged and it is not — a documentation
   claim without a code path behind it.
2. **The project's own release gate is not this bar.** DESIGN §5 gates
   `pruefstand` at **at least 9/10** on the coder through the deployed
   package, 10/10 the target, and the equivalence cells run greedy. The
   shipped configuration is 5/5 at 10/10 greedy, so that gate is green
   deterministically. The mean-over-30-at-0.7 bar is stricter than anything
   the release procedure measures; the number that would actually turn a
   release red is the zero-rate, ~1 in 120 sampled runs.

## The shipped configuration, pooled over eight interleaved batches (`measured-here`)

Eight 30-run batches at 0.7 of the shipping configuration (`--llama-checkpoints
0`, no sampler knob, thinking off, cap 4,000), all measured interleaved with
other arms so no batch sits at a special time of day:

| batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| mean | 9.900 | 9.900 | 9.233 | 9.867 | 9.800 | 9.833 | 9.800 | 9.833 |
| at 10/10 | 27 | 27 | 25 | 26 | 25 | 25 | 25 | 26 |
| zeros | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

- pooled **n = 240**, mean **9.771**, histogram 206×10, 29×9, 3×8, 2×0
- perfect-rate 85.8 %, **zero-rate 0.83 %** (1 in 120)
- pooled binomial SE 0.063 → 90 % CI **[9.667, 9.874]**
- between-batch SE 0.078 → 90 % CI **[9.642, 9.899]**
- 7 of 8 batches reach 9.8; the eighth is the one where two zeros landed
- simulation of the pooled per-run distribution: **P(a 30-run batch clears
  9.8) = 0.65**

Under a per-batch reading the bar would be unresolved: the interval contains
9.8, the point estimate is below it by half an SE, and a single batch of it
passes about two thirds of the time. Anyone reporting this unit as "meets 9.8"
or "misses 9.8" from one 30-run measurement is reporting noise. Under the
non-inferiority rule the operator set the same evening, the same data gives a
verdict: **not failed** — upper 95 % bound 9.874 (9.899 on the conservative
overdispersed SD), both above 9.8, on n = 240.

## Recommendation

**As the bar now stands (non-inferiority, operator 2026-10-20), the shipped
configuration passes it**: greedy 5/5 at 10/10, and 240 pooled sampled runs
whose upper 95 % bound (9.874, or 9.899 on the conservative overdispersed SD)
sits above 9.8. Nothing further is owed to the bar. What follows is what the
session learned about the margin, not a list of outstanding work.

Under the mean bar the decision splits cleanly, because the two criteria are
governed by two different defects:

1. **Greedy (5 of 5 at 10/10): apply `--llama-checkpoints 0`.** Already in the
   unit, free in VRAM (+2 MiB), removes a reproducible defect rather than
   masking it, and costs about 34 s of re-prefill per long follow-up — a small
   bill on this unit's measured traffic (5 such follow-ups in seven days). The
   mean bar neither supports nor contradicts it.
2. **Sampled mean >= 9.8: under the non-inferiority rule, NOT FAILED.** Eight
   interleaved batches pool to **9.771** over 240 runs; the upper 95 % bound
   is 9.874 (9.899 conservative), above the bar. Under the earlier reading the
   same data said "unresolved" — 90 % CI [9.64, 9.87] contains 9.8 — and that
   is exactly the ambiguity the rule removes. No lever found here moves the
   underlying number: `top_p` raises the syntax-zero rate twelve-fold
   (p = 0.0003), `min_p`/`top_k` are indistinguishable from no knob, and a
   `system` message aimed straight at the defect shows no gain (above). The
   margin's binding constraint is the **zero-rate** — 1 run in 120 emits Lua
   that does not parse — and its ceiling: P(>= 24 perfect) = 0.90 even at a
   zero-rate of nil. As a per-batch event, this configuration clears 9.8 in
   about **0.65 of 30-run batches**; the rule deliberately stops treating that
   number as a pass or a fail.
3. **The levers that remain are the model and the task, not the serving
   path.** The zero mode is known mechanically in one of its forms: with
   thinking off the model renders its reasoning as comment blocks inside the
   CR/LF branch and then emits the Python statement `pass`
   (`mb-2-08.lua:59`: `pass -- placeholder, will rewrite below`). Nothing in
   the sampler touches that, and the measurement agrees: no knob lowers the
   syntax-zero rate, `top_p` raises it, and a `system` message that names
   `pass`, `elif` and `None` by name does not lower it either — the one zero
   it produced was a Lua label written without its closing `::`, a second
   defect with a different signature. Of the two routes that looked live:
   **reject-and-regenerate is refused** (operator: no retry loop in the
   served path; and `code` says it would not terminate at greedy anyway — see
   "Why the guard-and-regenerate route is closed" above) and **the system
   message is client-side only** (operator: arcint injects nothing). What is
   left with evidence behind it is a different model or a different quant at
   the same deployment — and the quant axis is closed above, on grounds of
   power and confound, with the cost of reopening it stated there (same engine
   on `bosch`, ~2800 runs per arm for power ≥ 0.8). The bar is now a
   non-inferiority rule and production is **not failed** under it, so none of
   this is outstanding work. Not built, not claimed done.
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
the baseline. The sampler-knob arms are `sweep-sampler.log`, `repl-minp.log`
and `zerorate.log` (driver `accept-extra.py`, knobs passed per request through
`frage-extra.py`'s `EXTRA`). The system-message arms are `sys-screen.log` with
`out-sys/` (driver `accept-sys.py`, request builder `frage-sys.py`, arms in
the `ARMS` variable of `sys-screen.sh`); it scores every run in-process and
carries the `syntax` / `pyism` / `comment_run` proxies per run. `aufgabe.txt`
is a symlink to the untouched yardstick in `~/pruefstand`.

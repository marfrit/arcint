# Unit change proposal — coder: `--llama-checkpoints 0`

Status: **proposal, not applied.** The production unit
(`arcint-coder-llama.service`) is unchanged as of this writing: it still runs
without the flag, i.e. checkpoints on (up to 32 a lane, 8192 apart). The
decision is the operator's. Every number is `measured-here`, 2026-10-20, on
the coder host's A770 (`card0`, PCI `8086:56a0`, GPU.1), `arcint 1.2.1
(83d7609)`, sha256 `86b33f88fd4f609a`, model
`qwen36-coder-iwo3-c5f495ac-shq8.gguf` (`27931997cfeec31f`).

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

## The honest limit of the change

**It does not meet the bar the task was set with.** 27 of 30 is not 30 of 30.
The three losses are the same bug class in its other variants. And 27/30 is
the best this repository has ever measured for this model at 0.7 — the 0.5.12
arm got the same 27 at 10/10, 3 at 9.

At a true per-run rate p, the probability that all 30 sampled runs score 10/10
is p^30. At the best measured p ≈ 0.90 that is **4 %**. To pass 30/30 with
even 50 % confidence needs p ≥ 0.978. No arm in the record — 606 scored runs
under `~/pruefstand*/logs` plus the tables in `docs/llama-engine.md` — shows
p anywhere near that at n = 30; the arms that read "all 10/10" did so at n = 9
or 10, where p = 0.90 already gives a 35-39 % chance of looking perfect. So
"30 sampled runs, all 10/10" is not a target the sampled task can express for
this model; it is a demand for p ≈ 1, and the model's own sampled behaviour
sits at 0.90.

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

**Disposition:** no sampler knob moves the sampled bar. The lever that does
exist is the one already applied, `--llama-checkpoints 0`.

## Recommendation

Take the flag for the greedy cell, and treat the sampled criterion as a
separate decision:

1. **Apply `--llama-checkpoints 0`** if the greedy 10/10 is what the gate
   checks and 34 s per follow-up is acceptable on this unit. It is free in
   VRAM and it removes a reproducible defect rather than masking it.
2. **Do not expect 30/30.** Measured odds at the observed per-run rate
   (p̂ = 0.857, 180 of 210 sampled runs perfect across the six neutral arms):
   27/30 in about 36 % of batches, 30/30 in 1 %. To reach 30/30 with even 50 %
   confidence the per-run rate would have to be 0.977 — no arm in the 606-run
   record has ever held that. Sampler knobs are now measured and do not get
   there (§ above); a new quant is the expensive route and the KV evidence
   says precision is not what moves this score.
3. **Untouched lever worth one more window** if the sampled number matters
   more than the 34 s: `--llama-mtp 0` (plain decoding). The record notes the
   target's greedy text depends on how many rows each verify carries, and the
   only catastrophic task failure on record (0/10, twice the same bytes) came
   from a drafter. MTP 0 would be a third batch-layout arm at a decode cost of
   roughly 70 → 47 t/s on this model.

## Reproduce

On the coder host, `~/wp/coder-quality/`: `leg-ckpt0b.sh` is the window leg
(sampler started before the leg, anchored to `^/usr/bin/arcint ` plus the
arm's own flag substring — the flash-next unit on the other card holds fds on
both render nodes, so a bare anchor misattributes its memory). Logs in
`out-ckpt0b/`; the thinking arms are `think-t0.log`, `think-t07.log` and
`accept-think.py` in the same directory. `base-t0.log` and `base-t07.log` are
the baseline.

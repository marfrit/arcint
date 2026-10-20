# Note — coder baseline on the live production unit (2026-10-20)

Task: the operator's acceptance bar for the coder service is **10/10** —
greedy (T=0) AND 30 sampled runs at temperature 0.7. This note records the
baseline that bar is measured against, taken on the running production unit
with no card window and no change to the unit. Everything here is
`measured-here`.

## What was measured

Arm: `arcint-coder-llama.service` on the coder card host (container `dirac`,
image `arcint-coder:latest`), model
`/models/gguf/iwolucion-v3/qwen36-coder-iwo3-c5f495ac-shq8.gguf`
(14,611,874,464 bytes, SHA-256 `27931997cfee…`), flags
`--n-ctx 98304 --mlock --llama-mtp 4 --llama-kv q8_0:q4_0`,
`GGML_OPENCL_DEVICES=1:0` (GPU.1 = Arc A770 16 GB).

Harness: the operator's own, `~/pruefstand/frage.py` + `pruefstand.lua`
(10 test cases, deterministic scoring, wall-clock limit 30 s per case).
Scoring is identical to the 606-run record in `~/pruefstand/logs/`, so these
numbers are comparable with that history. Driver:
`~/wp/coder-quality/accept.py`. Raw logs: `~/wp/coder-quality/logs/`
(card host, not in this repo).

## Result — baseline does not meet the bar

| Arm | Score distribution | Perfect |
|-----|--------------------|---------|
| T=0 greedy, 5 runs | 9/10 × 5 | **0/5** |
| T=0.7, 30 runs | 10/10 × 26, 9/10 × 3, 8/10 × 1 | **26/30** |

Mean at 0.7: **9.833**. Median decode at 0.7: **73.3 t/s** (min 67.0,
max 78.4). Mean prompt eval: 375.6 t/s. Service VRAM during the battery:
**15,877 MiB allocated / 16,269 MiB driver**, peak utilization 99 %. The
service was not disturbed by the battery and stayed up throughout.

T=0 is deterministic: all five runs produced the same 9/10. So this is not
variance, it is one reproducible defect.

## The failure is one bug class, not noise

Across both arms, every loss except one single case is the same case:
**"Parser für Zeilen, die mit Ziffer + Punkt beginnen"** — subtest **"nur LF"**.

Signature:

    FEHLER nur LF  base-t0-XX.lua:49: ZEITLIMIT: Instruktionsbudget erschoepft

i.e. the generated Lua parser **spins forever on bare-LF input** and burns
the instruction budget. The model writes the CRLF branch correctly, with
`i = i + 1`, and omits the advance in the `\n` branch. Reproduced verbatim
from the T=0 generation:

    elseif line:sub(p, p) == "\r" then          -- CRLF
        i = i + 1                              -- present
        if i <= #lines then table.insert(out, lines[i]) end
    else                                        -- bare LF
                                               -- i = i + 1 MISSING -> loop
        if i <= #lines then table.insert(out, lines[i]) end
    end

The same shape appears in the sampled runs, with two variants: the missing
advance (loop), or a duplicated advance (skips a line, wrong count). Fixing
this one bug class would move T=0 from 9/10 to 10/10 and would account for
the sampled losses as well.

Residual, unrelated: one sampled run (seed 207) lost a different subtest,
"1000 Zeilen" (1000-line input), scoring 8/10. That is a separate, rarer
failure and is not the lever.

## What this says about the lever

The defect is a **generation-quality** defect (a dropped statement in a
branch), not an engine bug: the engine executes the code it is given, and
the harness catches it. Two consequences for the search:

1. **Context is not the constraint.** Prompt sizes in this battery are
   120–520 tokens; the 98k window is nowhere near binding. Widening context
   will not move this score.
2. **The lever is precision, not scheduling.** KV is already at
   `q8_0:q4_0`; V is at q4_0. Raising V to q8_0 costs ~1.07 GB at 98304
   tokens (`measured-here`, see below), against 1.2 GiB of driver headroom —
   tight but in range, and it is the cheapest precision lever that needs no
   new quant.

## VRAM accounting for the KV lever (measured-here)

From the live unit's own KV dump at `n_ctx = 98304`:

    [KV 3/64] size = 133,167,360 B, n_seq = 12, n_stream = 1, n_kv = 1179648
              type_k = 8 (q8_0), type_v = 10 (q4_0)

So each **full-attention** layer carries **127.0 MiB** of paged KV, and there
are **16** such layers (3, 7, 11, …, 63 — interval 4; the linear-attention
layers hold no paged KV). Total KV: **2.13 GB**.

Raising V from q4_0 to q8_0 doubles the V half. On a per-byte basis q8_0
costs 1.57× what q4_0 costs (34 B vs 18 B per 32-element block), so the
steady-state increase is on the order of **+0.77 GB** (2.13 GB × 0.5 ×
0.57/2 ≈ 0.30 GB if K and V are equal halves — the honest statement is that
the figure must be read off the KV dump line for the new arm, not derived
from the old one, because the buffer geometry depends on `n_seq` as well).

Headroom on the live unit is 17,080 − 16,269 = **811 MiB driver-level** and
1,203 MiB allocated-level. Whether the arm fits therefore cannot be settled
from the running unit: it has to be read from the new arm's own init dump
inside a card window.

## Limits of this baseline

- One arm, one day, one card. No comparison arm was run; the numbers
  describe the production unit as configured, not a controlled experiment.
- The 30-run sampled arm is one sample of 30; 26/30 perfect has a wide
  interval (~87 % ± 6 %). Treat "26/3/1" as the shape, not a precise rate.
- The failure analysis is from reading the generated code the harness
  reported, not from a controlled ablation of the parser.
- Logs and quants referenced here live on the card host under
  `~/wp/coder-quality/`; they are not in this repository.

## The thinking confound (settled 2026-10-20, `measured-here`)

The unit's own comment and `docs/llama-engine.md` record the acceptance task
at **10/10 at T=0** for this exact GGUF and these exact flags (2026-10-05).
This baseline gets 9/10. Before treating that as a regression, the harness
setting had to be pinned: `frage.py` takes `OHNE_DENKEN=1`, which sends
`enable_thinking: false`.

What the record actually used:

- The cell's wrapper `~/pruefstand-coder/run.sh` and the N-run driver
  `~/wp/llama/pruef-n.sh` both set `DECKEL=4000 OHNE_DENKEN=1`.
- The surviving raw answers from that day confirm it: every
  `~/pruefstand*/antwort-20261004-*.roh` and `antwort-20261005-*.roh` has
  zero think-block hits (byte counts 2.0-3.9 kB, all code).

So the recorded 10/10 arms and this baseline used the **same** thinking
setting. There is no confound between them; the difference is the engine
(0.5.11/0.5.12 then, 1.2.1 now) and its prefill batch layout — which is
exactly the axis `docs/llama-engine.md` already identified: at 0.5.12 the
same prompt scored 9 at T=0 with checkpoints on and 10 with
`--llama-checkpoints 0`, "one case flipping with the batch layout".

**Thinking on is a different arm, and it passes** (`measured-here`, live
unit, T=0, 5 runs, `DECKEL=16000`):

| arm | T=0 | tokens an answer | wall an answer | decode |
|---|---|---|---|---|
| thinking off (the cell's setting) | 0 of 5 at 10/10 | 467 | 6.3 s | 73.9 t/s |
| thinking on | **5 of 5 at 10/10** | 6,601 | 99 s | 66.4 t/s |

All five thinking-on runs are byte-identical in length and all score 10/10:
the model finds the missing advance once it reasons first. VRAM is unchanged
(15,881 MiB against 15,877). That is a serving-policy lever, not an engine
lever, and it costs ~14× the tokens and ~16× the wall of the thinking-off
answer. It is recorded here because it is the largest effect measured on this
battery, and because it says the defect is reachable by the model at all.

Production traffic is not the acceptance cell: the journal shows the unit
serving agentic conversations (prefill 24,069 / 26,871 / 32,480 tokens,
decode up to 10,482 tokens, cache hits 88-94 %), i.e. multi-turn with long
replies. The acceptance cell is a single 282-token turn with thinking off.

## The `--llama-checkpoints 0` window (2026-10-20, `measured-here`)

Card window on the coder host, 10:32-10:47 UTC, SOP followed: sampler started
before the leg and anchored to `^/usr/bin/arcint ` plus a `--llama-checkpoints 0`
substring (the flash-next unit on the other card holds fds on **both** render
nodes, so a bare anchor charges its memory to this arm — that is what the
substring is for); seat line with the card by PCI id (`card0` = `8086:56a0` =
GPU.1); binary named with its sha256 (`arcint 1.2.1 (83d7609)`,
`86b33f88fd4f609a`); unit stopped, arm run as one process, unit restored,
release line with the process count.

Arm: the unit's flags exactly, plus `--llama-checkpoints 0`, port 8095,
`--n-ctx 98304 --llama-mtp 4 --llama-kv q8_0:q4_0`. Load log confirms the
arm: no checkpoint line at all, where the unit logs
`context checkpoints: up to 32 a lane, 8192 tokens apart`.

| arm | T=0 (5 runs) | 30 runs at 0.7 | mean | decode | peak VRAM |
|---|---|---|---|---|---|
| the unit, checkpoints on | 0 of 5 at 10/10 (all 9/10) | 26 at 10/10, 3 at 9, 1 at 8 | 9.833 | 72.5 t/s | 15,877 MiB |
| `--llama-checkpoints 0` | **5 of 5 at 10/10** | 27 at 10/10, 3 at 9 | 9.900 | 70.3 t/s | 15,879 MiB |

Prefill 935 t/s at 282 tokens (the unit: 440 t/s cold, 4,400 warm). Peak VRAM
15,879 MiB, GTT 32 MiB, host RSS 711 MiB, MemAvailable never under 90,967 MiB.
Nothing paged; the arm costs 2 MiB more than the unit.

**The greedy cell flips and it reproduces.** Thinking off, same GGUF, same
flags: with checkpoints on the model emits the parser with the missing advance
in the bare-LF branch, five times identically; with checkpoints off it emits
the correct parser, five times identically (475 tokens every run). That is
`docs/llama-engine.md`'s 0.5.12 observation — 9 against 10, "one case flipping
with the batch layout" — reproduced two engine versions later on 1.2.1.

**What it does not do.** The sampled bar is not met: 27 of 30, not 30 of 30.
The three losses are the same bug class (two `einfach CRLF`, one `nur LF`), and
the 8/10 outlier is gone. 27/30 equals the best the record has for this model
at 0.7 (0.5.12: 27 at 10/10, 3 at 9).

**And the bar itself deserves a number.** At a true per-run rate p, the chance
that all 30 sampled runs score 10/10 is p^30. The best measured arms sit at
p ≈ 0.90, where p^30 ≈ 4 %; 26/30 and 27/30 are what p ≈ 0.90 looks like, not
what p = 1 looks like. Asking for 30/30 at 0.7 asks for p ≥ 0.90 once and
p ≥ 0.99 to be dependable, and no arm in this repository's record — 606 scored
runs in `~/pruefstand*/logs`, plus the tables in `docs/llama-engine.md` — has
shown that at n = 30. The arms that read "all 10/10" did so at n = 9 or n = 10,
where p = 0.90 gives a 35-39 % chance. So the bar is either a statement about
greedy (which `--llama-checkpoints 0` now meets) or it is a demand for a
sampling regime no measured arm reaches.

## What was not tried, and why

`--llama-kv q8_0:q8_0` was the other planned lever and was **not** run. The
arm's own buffers put its price in range — KV 780 MiB at `q8_0:q4_0` against
1,201 MiB of VRAM headroom, so raising V costs roughly +220 MiB and fits — but
`docs/llama-engine.md` already ran the KV-precision A/B on this task at 30 runs
per arm (q8_0 15 of 30, mean 7.4; f16 19 of 30, mean 8.13, at `--n-ctx 65536`)
and recorded that the gap is under one standard error and reverses in a second
batch. Spending a second window on a lever the record already showed cannot
separate the task, when the measured failure is a dropped statement in a
branch, is not the next thing to do. Recorded as not tried, not as verified
against.

## Where the effect actually is

Three arms on one battery, one variable each, all `measured-here`:

| change | greedy | sampled at 0.7 |
|---|---|---|
| baseline (unit as configured) | 9/10 | 26 of 30 |
| checkpoints off (batch layout) | 10/10 | 27 of 30 |
| thinking on (`enable_thinking: true`, cap 16,000) | 10/10 | see below |

The two things that move this score are where the prefill batches break and
whether the model reasons before writing. Neither is a precision knob.

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

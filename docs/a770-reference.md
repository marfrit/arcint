# The Arc A770's last reference (2026-11-19)

arcint was developed and gated on two cards: an Arc A770 16 GB (Alchemist,
Xe-HPG, ACM-G10) and an Arc Pro B60 24 GB (Battlemage, Xe2). The A770 left
the development host on 2026-11-19; an Arc Pro B70 32 GB takes its slot.
**1.7.0 is the last release gated on an A770.** From the next release on,
Alchemist is **untested**: its code paths stay in the kernel series (the int8
DPAS paths without local memory, 0099 and 0102-0104; the sub-group 16 tiles;
the A770 rows of the op suites), nothing is removed, but no release gate
runs them on a card. A report from an Alchemist card is welcome; treat the
rows below as the last measured state, not as a promise for later releases.

Everything below is `measured-here` on the 1.7.0 package (tag v1.7.0,
bef9627e), the A770 behind a PCIe 3.0 x4 chipset link, its GT clock pinned at
2000 MHz, the xe kernel driver, Intel's compute runtime from the distribution
(Debian 13). The model is the coder the A770 served in production: the
Qwen3.6-27B-A3B-Coder (`qwen35moe`, 184 experts, MTP layer) as the searched
per-layer quant mix `iwo4-36d0d52d-shq8` (13.45 GB), with the deployed
service's own flags: 196,608 tokens of context, q8_0:q4_0 KV, MTP 4 with the
cost policy and the base model's head, the GDN replay
(`LLAMA_GDN_REPLAY=1`), `--prefill-ubatch 512`.

## Served, at the service's full context

One sequence of public source text (the acceptance corpus), each depth
prefilled on top of the last, then 256 greedy tokens at that depth; a needle
sentence sits at token 104,000 and is asked for at the end:

| depth (tokens) | prefill of the segment | decode | draft tokens accepted |
|---|---|---|---|
| 4,096 | 1,344.4 t/s | 76.42 t/s (33.7 ms a step) | 62 % |
| 65,536 | 604.5 t/s | 38.34 t/s (83.1 ms) | 69 % |
| 131,072 | 284.2 t/s | 23.64 t/s (108.9 ms) | 62 % |
| 192,000 | 185.1 t/s | 17.24 t/s (121.2 ms) | 53 % |

- The needle at token 104,000 of 192,000: found.
- Card memory at its peak: 15,983 MiB resident in VRAM, 32 MiB in GTT (the
  card's OpenCL-usable memory is 15,473.6 MiB of the driver's 16,288; DESIGN
  §3). No eviction.
- `/props` downgrades: none; no `DOWNGRADE` line, no assertion in the server
  log; the capital answered right.
- The acceptance task (a Lua CSV parser to RFC 4180, scored by executing it):
  **10/10**, 520 tokens in 5.6 s.
- The same chain on a development build of the same patches four days
  earlier: prefill 1,344.5 / 604.6 / 283.7 / 184.6 t/s, decode 78.22 / 38.74 /
  23.79 / 17.67 t/s. Prefill agrees to 0.2 %; decode varies with the answer
  text and the drafts it accepts.

From the 1.7.0 release gate the day before (the same card, flags and tree;
CHANGELOG 1.7.0): prefill at 41k tokens 892.35 t/s; decode on three
300-token answers 50.80 / 54.45 / 48.70 t/s; 12 answers 76.25-76.32 t/s; the
acceptance task 10/10 three times.

## llama-bench, the service's file

The release tree's `llama-bench`, all layers on the A770, flash attention on,
MTP off, 3 repetitions:

| KV | pp512 | tg128 |
|---|---|---|
| f16 | 1,827.00 ± 8.26 | 55.44 ± 0.03 |
| q8_0:q4_0 | 1,815.46 ± 6.90 | 54.08 ± 0.02 |

Llama 2 7B Q4_0 on the same card in the 1.7.0 gate: 2,608.53 / 70.21 served
(f16 KV), 2,375.13 / 66.00 at q8_0:q4_0.

## KL against the F16 model

Reference logits from the coder's F16 GGUF (52.5 GB) on the CPU; the
quantized file on the A770 against them, `llama-perplexity
--kl-divergence`, 2,048-token chunks at batch 2,048. Two text panels: the
holdout and fitness slices the quant mix was searched on (the search itself
scored against a Q8_0 teacher, not the F16).

| panel | chunks | mean KL (nats) | median | 99th pct | same top token | PPL / PPL(F16) |
|---|---|---|---|---|---|---|
| holdout | 8 | 0.0406 ± 0.0017 | 0.0118 | 0.406 | 92.74 % | 4.737 / 4.683 (+1.2 %) |
| fitness | 32 | 0.0571 ± 0.0025 | 0.0130 | 0.410 | 92.17 % | 7.125 / 7.039 (+1.2 %) |

The same file on the CPU, holdout panel: mean KL 0.0413 ± 0.0017, 92.80 %
same top token. The card's arm is 0.0007 nats lower, well inside the
answer-level bar (0.03 nats, CLAUDE.md); the gap from F16 is the
quantization's, not the card's.

The xe kernel log counted no wedge, CAT error or job timeout over the run.

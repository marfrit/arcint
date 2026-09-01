# DFlash2 pairing probe: the draft head transfers to our int4 target (2026-09-01)

## The question

`incoai/Qwen3.8-27B-DFlash2` is a public block-diffusion draft head for
Qwen3.8-27B (5 non-causal layers, block 8, a top-16 candidate-path selector,
two-tap dynamic convolutions; ICML 2026, z-lab/dflash). Its card reports
acceptance lengths of 3.7–4.8 per verification cycle at seven drafts — double
what our MTP head yields per pass — measured on an H200 against the bf16
target. The head conditions on the target's hidden states (layers 5, 19, 33,
47, 61) — features our artifact produces through int4 weights. **Do those
features still pair?** The 3.6 MTP lesson applies: acceptance is the oracle,
and it is measured, not assumed.

## Method (pre-registered: pairing works at >= 2.5 on code; <= 1.5 means the
features do not transfer)

Teacher-forced offline replay, no serving integration, no GPU touched:

- `tools/dflash_probe_dump.py` (OpenVINO, CPU): drives the b7c1 stateful IR
  greedily for 192 tokens and records, per position, the residual stream
  entering layers {6, 20, 34, 48, 62}'s input_layernorm (HF's
  `hidden_states[layer_id+1]` for the head's `target_layer_ids`) plus the
  input embeddings.
- `tools/dflash_probe.py` (torch, CPU): a minimal reimplementation of the
  DFlash2 forward (the venv's transformers 5.0.0 predates the reference's
  cache API) — KV-injection attention over [context features | noise block],
  grouped dynamic convolutions, the greedy selector path, logits through the
  artifact's own extracted lm_head. Per cycle it proposes 7 tokens and counts
  the prefix that matches the target's recorded greedy continuation; advance
  is accepted+1, exactly the online loop under greedy.
- **Null control**: the same replay with the feature rows shuffled. A probe
  that cannot fail measures nothing.

## Result (B60-class artifact `qwen38-b7c1-ov`, int4, CPU-extracted features)

| arm | code prompt | prose prompt |
|---|---|---|
| paired | **3.39** (56 cycles, 134/392 drafted accepted) | **3.76** (51 cycles, 141/357) |
| shuffled (null) | 1.23 | 1.10 |

Acceptance length = accepted+1 per verification cycle, the card's metric. The
paired arms sit in the published band (card: 3.74 MT-Bench … 4.79 MBPP on
bf16/H200 with xhigh reasoning); the shuffled arms collapse to ~1.1, so the
number is carried by the features, not by the selector guessing from the
anchor. **The int4 target's features pair.**

## What this is worth, and what it costs

Per cycle the online loop pays one M=8 verify forward (dense model — the MoE
M>1 subbuffer issue does not apply) plus one 5-layer draft forward, and gets
~3.4-3.8 tokens against the MTP head's 1.93 per M=2 pass. The remaining build
is real: an OV export of the draft (KV-injection attention, dynamic convs,
selector), five intermediate outputs on the paged graph, the verify loop at
depth 8 (the checkpoint-row machinery already supports it via `--draft N`),
and a memory plan — the draft is 3.85 GB bf16 (~1 GB int4) and the B60 at
155k context has no headroom, so context shrinks or the draft parks on the
other card (`--mtp-device` precedent; ~250 KB of features cross per cycle).
Greedy-only under our accept-only-if-equal rule, like the MTP head — the
upstream head also supports lossless *sampled* speculation, but that is an
acceptance-rule change and explicitly not on the table.

Weights: `dirac:/models/gptq/qwen38-dflash2/` (sha-checked download of the
HF repo). Probe artifacts: `dirac:/tmp/dflash-{code,prose}.npz`.

## From probe to serving (2026-09-01, same day)

The build: `tools/export_dflash.py` exports the head to OpenVINO twice — a
stateless graph (the parity instrument) and a stateful one for serving, whose
per-layer context K/V live in graph state and only ever receive **accepted**
positions, so the drafter needs no rollback at all. The selector runs on the
host from raw sidecars; logits go through the artifact's own extracted
lm_head. arcint wires it as `--dflash DIR` (`--dflash-device` to park it on
the other card): five residual taps on the paged graph (`dflash_feats`,
layers 5/19/33/47/61 concatenated), a feature window of 2048 positions, and
the existing checkpoint-row verify at depth 7.

**The f16 lesson, measured stage by stage:** the head's residual stream
peaks at ~128k — past f16's 65504 — while its input starts at 0.05, and on
GPU-f16 the drafter emitted one constant junk token ("$", 0/2800 accepted)
while CPU (f32) was fine. Input pre-scaling does nothing (the norms
renormalise it away), and a plain 1/64 fold broke the *first* norm instead
(the noise embeddings' squares sank below eps: acceptance 3.39 → 2.02 on
both devices). What works is exact arithmetic, not clamping: fold 1/4 into
the residual writers (stream peak → 32k, fits f16) and give each norm an
input pre-scale with eps·pre² — `rms(c·x, c²·eps) ≡ rms(x)` identically —
×64 for the tiny layer-0 input, ×1/256 for the big stream norms. After
that, **GPU-f16 is cycle-exact with CPU** (3.28 code / 3.62 prose; the ~3%
against the torch probe's 3.39/3.76 is f16 re-rounding of folded weights
flipping near-tie cycles).

## Serving measurements (B60, `qwen38-b7c1-ov` int4, u8 KV, 400 tokens,
greedy, `--repetition-penalty 1.0`, 32768 ctx unless stated)

| arm | t/s | accept | max ctx (from the reservation) |
|---|---|---|---|
| plain (`--mtp off`) | 24.0 | — | 199,712 |
| MTP head (`--mtp on`) | 33.0 | 76.7% (1 draft/pass) | 155,680 |
| **DFlash2 int4, same card** | **44.8** | 3.13/cycle (273/896) | **136,640** |
| DFlash2 int4, draft on A770 | 39.8 | 3.13/cycle | 171,904 |

**+87% over plain and +36% over the MTP head on one card**; the draft costs
~63k tokens of context headroom there (int4 draft 1.2 GB + feats buffer +
draft state). Parking the draft on the A770 buys 35k of that back at −5 t/s
of PCIe round-trips per cycle — and its output is **byte-identical** to the
same-card arm, so placement is a pure capacity/speed trade. Speculative
output differs from plain greedy at the documented M>1 near-tie level (the
same property the MTP head has always had; README's caveat applies
unchanged — serve without a drafter for bit-exact reproducibility).

Open items, honestly: acceptance under the production thinking template and
prefix-cache hits (a warm hit starts the drafter with no features for the
cached prefix) are unmeasured; the drafter is greedy-only by the acceptance
rule; and the Prüfstand has not scored the dflash arm yet.

## Coexistence on the A770: loads, drafts, and wedges under load (2026-09-01)

Question: coder at 48k context on the A770 with the agent's draft beside it.
The arithmetic said no (coder at 48k leaves ~0.9 GiB of the card's 15.11;
the draft package wants ~2.3 — 0.90 GiB int4 draft + 1.27 GiB target
lm_head + state). The measurement said something more useful, in two parts:

1. **Light traffic works.** Both processes loaded — the reservation
   arithmetic was pessimistic about what the allocator actually found — and
   the agent drafted at 39.6 t/s (4.06 accepted/cycle on an easy prompt)
   while the coder answered beside it.
2. **Concurrent load wedges both.** A 35k-token coder prefill fired
   alongside drafted agent generation: neither process ever logged a
   request line, both curls ran to their 800 s timeouts, and the kernel
   logged no GPU hang or reset. Both test processes then **ignored
   SIGTERM** (stuck in driver waits); one kept holding the A770 and took
   the production coder down with it — its replacement wedged the same way
   until the zombie was SIGKILLed, after which everything recovered clean.

Verdict: two arcint processes sharing the A770 is load-ordering roulette
with no failure signal from the kernel, and a hang that outlives SIGTERM.
The one-process-per-card rule stays; the supported cross-card layout is the
one measured earlier — draft on a card that is otherwise idle. The int4
draft's lm_head (1.27 GiB) is the largest movable piece if a shared-card
variant is ever wanted; it is not wanted now.

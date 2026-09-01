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

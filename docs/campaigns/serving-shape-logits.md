# serving-shape-logits — the served Flash-Next artifact's logits carry no information about the model; find the layer, or the fill, that loses it

## The defect, as measured

`measured-here`, 2026-09-18, Arc Pro B60, the 48-layer artifact in the
fused shape (`qwen3.8-flash-next-d48f`, tree da52858) served with
`marfrit-openvino +p18` at `--offload-ratio 99 --moe-cpu-tier`, KV u8,
f16 inference, one lane, from the NVMe: the llama.cpp logits capture of
the same GGUF (`qwen4exp-c2735-chunks2`, two windows of 2,735 tokens)
replayed through the server — mean KL(reference ‖ served) **12.42 / 12.16
nats** (window 0, below / above the midpoint) and **12.40 / 12.34**
(window 1), max 26–35, argmax agreement **0.0007 and 0.0000**. For scale, ln(248320) = 12.42 nats is
the KL a reference of zero entropy would score against a uniform served
distribution; with an argmax agreement of zero the served logits carry no
information about the model's own. The depth-12 rung read 11.93 / 11.63 on
2026-09-13 (`docs/window-051.md`) and the depth-4 rung about the same;
those were read as "44 (36) layers missing". A truncated prefix of a deep
model scores like that too, so those rungs could not decide between
"missing layers" and "a broken artifact"; the full-depth rung decides:
the artifact is broken, and nothing says at which depth.

The served output is deterministic: France raw, greedy, 8 tokens is
byte-identical across two cold starts and across two storage paths, and
warm repeats are identical. Whatever is wrong is wrong the same way every
time.

## Known against hypothesised

Known (`measured-here`): the fused MoE route computes the HF-exported 35B
correctly on the same plugin and card (Paris, 23.6 t/s); the CPU tier
matches the device kernel at depth 12 (KL 0.020 on the France prefill,
decode steps within 0.4 logits); the unfused and fused routes disagree at
depth 12 (KL 0.147 on the same prefill) — both are approximations and
neither has been compared to a reference at that depth. Known (`code`):
every emitted piece (GDN, attention, hyper-connections, PLE, MoE, the
n-gram gather) has a numpy reference and a bit-exact contract test at the
suite's reduced geometry (`tests/python/`, `q4e/ref_*.py`), on synthetic
weights; the real-geometry fill (`q4e/gguf_feed.py`, `q4e/expert_fill.py`,
`tools/export_serving_artifact.py`) has its own cells (the expert fill's
codes are pinned bit for bit against the C++ unpack; the dense fill is
checked against the GGUF's own dequantisation in `test_expert_fill.py`'s
shard-gated cells) but no cell compares a served forward at real geometry
against any reference. Known (`measured-here`, 2026-09-13): the n-gram
table's binding and the PLE hash ordinal were measured as mechanism, never
as values.

Hypothesised (unranked; none measured): the real-geometry structural
parameters the reduced geometry cannot exercise (rope span and theta, the
GDN geometry at real head counts, the hyper-connection lowrank, the
attention interval, `layer_types`); the dense fill's key mapping at real
geometry (a transposed or misnamed tensor passes shape checks and fails
values); the PLE / n-gram table's row addressing at the real vocabulary
(a wrong hash ordinal reads the wrong rows and produces deterministic
noise); the expert fill's expert ordering across layers; the served
binary's port feeding (`inputs_embeds` from the embedding model, the
position ids, the conv mask) against what the emitter expects.

## Gate

A served forward at depth 48 whose logits match the reference: mean KL
within the 0.5.1 bar (re-derived from this model's own reference
round-trip, not the 35B's 0.0599), argmax agreement above 0.9 on both
capture windows, and the Paris line — the model's own token for the
capital of France — or a named refusal that says which layer refuses.
Byte-identical across two cold starts (§3.4) stays required.

## Entry criteria

Met: a full-depth artifact that compiles and serves on one card
(8.06 GiB device at ratio 99 with the tier, +p18), an NVMe copy that makes
a 2,735-token replay a 20-minute cell, the reference capture staged where
the container reads it, the replay and compare tools (`tools/kld_served.py`),
a two-dump diff (`tools/logits_dump_diff.py`), and the cut localiser in
the boot driver (`--cut layerN/out`). Unmet: a per-layer reference at real
geometry — the numpy `ref_backbone` fed with the real GGUF at depth 1 (the
reduced-geometry tests already wire the same modules), or a llama.cpp
hidden-state tap.

## Scope — in / out

In: localising the loss layer by layer (depth 1 first: the embedding, the
first GDN block, the first MoE, the PLE layer) against a real-geometry
reference; the fill's key mapping and the structural parameters at real
geometry; the port feeding of the served binary; the fix in the emitter or
the fill; the re-export; the gate measurement.

Out: the MoE route's own numerics (fused against unfused at depth 12 is a
separate, smaller question once the artifact computes the model); the
residency stream's speed; the B60/A770 kernel differences.

## Where it lives

`tools/q4e/` (the emitter and the fill), `tools/export_serving_artifact.py`,
`tools/boot_serving_shape.py` (`--cut`, `--dump-logits`, `--stage forward`),
`tools/kld_served.py`, `tools/logits_dump_diff.py`, `docs/window-050.md`
and `docs/window-051.md` (the depth-4 and depth-12 records that were read
as depth), `docs/design-qwen-flash-next.md`; this campaign's own status
below; `sub4bit-vram-kernel.md` (status 2026-09-17/18, how the full-depth
serve was reached).

## Pipeline for this campaign

Recon: read the depth-4 KLD record and the feed-the-ports measurement of
2026-09-13 again with this verdict in hand → the reference at depth 1: the
numpy backbone over the real GGUF for one short prompt, against the served
cut at `layer0/out` (values, not shapes) → walk forward layer by layer
until the first divergence, then inward (embedding, norm, attention or
GDN, hyper-connection, MoE, PLE) → the fix, red-first in the reduced
geometry where the same bug can be planted → re-export at depth 4, KLD
against the reference (a depth-4 rung of a correct emitter must beat the
uniform floor by a wide margin even with 44 layers missing — that is the
cheap signal) → full depth, the gate → review, DESIGN record, CHANGELOG.

## Invariants

DESIGN §3.4 (history-independent greedy output) and §3.8; the measurement
discipline in `CLAUDE.md`; a served number without the reference beside
it is mechanism, not an answer.

## Status

- 2026-09-18 — opened from the full-depth KLD of the night before: the
  served logits carry no information at depth 48 (KL 12.4 nats, argmax
  agreement 0); the depth-12 and depth-4 rungs read the same and could not
  distinguish "missing layers" from "broken" — full depth does. Nothing
  localised yet.
- 2026-09-18, 01:00–02:00 — **localised and fixed at layer 0, three ways**
  (`measured-here`, B60 for the cuts, CPU for the references, France ids =
  the served 5-token prefill). The cut ladder (`--cut layerN/out
  --cut-prune --probe`, tree fd54fb6) against two references: the artifact's
  layer 0 reproduces the pin's modules fed from the GGUF (corr 0.9988) — the
  emitter was right; the pin-based reference departs from llama.cpp's tap at
  the first hyper-connection mix — the FILL was wrong. Whole tensors
  (llama-eval-dump) then found: (1) folded norm gammas (1 + w) and ssm_a =
  −exp(A_log) fed as stored → `gguf_feed` kinds `gamma1`/`neglog`; (2) the
  output gate is a sigmoid, not the pin's silu default → `output_gate_type`;
  (3) value head h pairs with key head h % 16, not h // 3 → `gdn_key_head_map:
  tiled`. Each red-first against the GGUF's own values or the transcription;
  with all three the layer-0 output matches llama.cpp at corr 0.9999 (max
  |diff| 0.004 on values of mean 0.008). Every earlier parity leg read 0.0
  before and after: both sides shared the feed. DESIGN §7.0.2bz. Next: the
  depth-4 re-export through all three fixes, its cut ladder against
  llama.cpp's l_last-0..3 (the PLE and the first attention layer included),
  then full depth and the KLD gate.
- 2026-09-18, 02:07 — **the served-shape graph agrees with llama.cpp at
  every cut of a depth-4 re-export** (`measured-here`, B60, f16, the paged
  pass, France ids; whole tensors on both sides): layer0/out corr 0.99924,
  after the PLE 0.99942, layer1/out 0.99918, layer3/out 0.99873 (through
  the first full-attention layer). The GDN, the PLE, the hyper-connections,
  the fused MoE route and the attention layer are all right on the card.
  Full-depth re-export running; the gate (the KLD replay, the Paris line)
  follows on it.
- 2026-09-18, 03:53 — **the Paris line, served at depth 48** (`measured-here`,
  B60, the d48g artifact from the NVMe, +p18, ratio 99 + tier, KV u8, f16):
  ` Paris. Paris is a city in France`, cold and warm byte-identical, a
  coherent 64-token continuation, the chat form reasoning in the model's
  own voice. The KLD replay (2 × 2,735 ids of the llama.cpp capture) is
  running for the gate number.
- 2026-09-18, 04:27 — **the KLD gate on the corrected full-depth artifact**
  (`measured-here`, B60, d48g from the NVMe, +p18, ratio 99 + tier, KV u8,
  f16, one lane, chunk 512; the llama.cpp capture's 2 × 2,735 ids replayed):
  mean KL(reference‖served) 0.635 / 0.437 (window 0, below / above the 2051
  boundary), 0.829 / 0.922 (window 1); ALL 0.732 / 0.680; argmax agreement
  0.73 / 0.71; prefill 846 s and 1,066 s. Against last night's 12.4 nats and
  0.000: the model. Against the bar (0.0599, another model's number) and
  against what a faithful f16 serve of the same GGUF should read: an order
  of magnitude of residual. Position-resolved: median ≈ 0.2 per row in every
  128-token bucket, no rise with position, no chunk-boundary staircase,
  below ≈ above the QSA boundary, no turn-boundary token in either window —
  a uniform per-token discrepancy. Suspect: the f16 inference precision
  compounding over 48 layers (the card's layer-3 cut sits at corr 0.9987 ≈
  5% RMS against llama.cpp where the CPU f32 reference reads 0.9999 ≈ 1.3%).
  The precision leg (f32 / KV f16 cuts at depth 4) measures it next.
- 2026-09-18, 04:50 — **the residual, named; localisation complete**
  (`measured-here`, CPU): with `--precision f32` refused by the fused MoE
  route (`MoERouterFused` has no f32 layout) and `--paged-kv f16` inert at
  five tokens, the CPU plugin's f32 GatherMatmul route reads layer 0 at
  corr 0.99925 against llama.cpp — the card's f16 figure to the fourth
  digit — where the pin's modules fed with the EXACT dequant read 0.99991.
  What the serving graph holds that the reference does not is the expert
  repack: `expert_fill.quantise_group_affine`, u4 codes over groups of 128,
  of experts the checkpoint ships as IQ3_XXS (gate, up) and IQ4_NL (down).
  Its relative RMS error on blk.0's experts is 0.13 / 0.13 / 0.11 (gate /
  up / down), the matmul output on random activations inherits it, and the
  uniform per-token residual at depth 48 (median KL ≈ 0.2) follows. The
  precision, the KV cache, the chunking, the QSA seam and the PLE boundary
  are excluded by measurement. This campaign's question — where the served
  model loses the model — is answered three times over (the fill's folds,
  the gate, the pairing) plus once for the residual; the residual's remedy
  is `sub4bit-vram-kernel`'s own goal, the native sub-4-bit expert kernel.

- 2026-09-19 — **closed.** The residual's remedy landed in
  `sub4bit-vram-kernel` (patch 0043, the native expert formats: 0.54 → 0.42
  mean on window 0 against llama.cpp's capture), and the question this
  campaign could not settle against that capture — what is ours and what is
  llama.cpp's — is settled by a better yardstick: the pin's own full-depth
  f32 forward (`tools/ref_forward_stream.py`, run on a Spark). Against it
  the served artifact reads mean 0.37 / median 0.18 / argmax 0.83 on window
  0 where llama.cpp reads 0.34 / 0.065 / 0.80 (`measured-here`); the
  block-level arithmetic is exact to 0.2% through 24 layers. The remaining
  term is a long-context floor (the f16 recurrent state or the f16
  long-context attention; chunks and KV precision excluded) and lives on in
  `sub4bit-vram-kernel`'s status. Every instrument this campaign built
  (`llama-eval-dump`, `llama_tap_compare.py`, the cut ladder, the KLD replay
  and its position-resolved reading) stays in use.
  [DATED 2026-09-28: the f32 reference captures
  (`tools/ref_forward_stream.py`, every capture since af465dc, 2026-09-19) ran
  each sparse-attention indexer on byte garbage: `gguf_feed` cast the BF16
  indexer projections' raw bytes to f32 until 2026-09-28 (`measured-here`).
  Rows below position 2,051 are valid, because the selection keeps every
  complete block whatever the scores (`code`). Rows at or above it are void. A
  capture scores rows 1,368-2,734, 684 of them at or above 2,051, so every
  whole-window mean, median or argmax against the reference is void as quoted,
  and so is any above-boundary figure, with its attribution to the
  dense-for-sparse price. The below-2,051 figures stand; the re-capture is
  owed (`docs/campaigns/qsa.md`).]

## Gate, re-read

The Paris line is served at depth 48. The KLD number against llama.cpp's
capture (0.73 nats u4, 0.42 native) is bounded below by llama.cpp's own
distance from the model (mean 0.34 on the same rows), so it is no longer the
gate; the gate is the reading against the model's own f32 reference capture
(on the dev host), where the artifact stands at 0.37
/ 0.18 with its long-context term named as the next work.

[DATED 2026-09-28: the f32 reference captures (`tools/ref_forward_stream.py`,
every capture since af465dc, 2026-09-19) ran each sparse-attention indexer on
byte garbage: `gguf_feed` cast the BF16 indexer projections' raw bytes to f32
until 2026-09-28 (`measured-here`). Rows below position 2,051 are valid,
because the selection keeps every complete block whatever the scores (`code`).
Rows at or above it are void. A capture scores rows 1,368-2,734, 684 of them
at or above 2,051, so every whole-window mean, median or argmax against the
reference is void as quoted, and so is any above-boundary figure, with its
attribution to the dense-for-sparse price. The below-2,051 figures stand; the
re-capture is owed (`docs/campaigns/qsa.md`).]

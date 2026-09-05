# 0.5.0 — Qwen Flash Next support

Recorded 2026-09-05. The charter: **arcint serves the Qwen Flash Next
checkpoint the operator named**, on both cards, under the §5 gates —
allowlisted, exported, fitted, drafted where it ships a drafter, measured.
The name is the operator's; the first recon step pins it to a checkpoint
(its Hugging Face id, `config.json`'s `model_type` and `architectures[0]`)
and records the architecture against what this engine already serves,
because nothing below may assume what that file says.

## Where the repository stands

- The allowlist (`src/core/artifact.cpp`) admits one family: hybrid
  GatedDeltaNet + full attention at `full_attention_interval = 4`, MoE
  (Qwen3.6-35B-A3B with 256 experts; Qwen3.6-27B-A3B-Coder, pruned to 184)
  and dense (Qwen3.8-27B), all
  `*ForConditionalGeneration` VLM exports with the vision tower reported
  and never loaded (`docs/model_requirements.md` §1). A checkpoint outside
  that list is refused by name.
- Everything architecture-specific is read from `config.json` at load:
  layer types (or the interval that derives them), head geometry for the
  fit's own terms (`packed_values_scratch_geometry` wants
  `num_attention_heads` and `head_dim`), `num_experts`; MTP presence
  (`has_mtp_head`) comes from the MTP layer IR being in the directory, the
  DFlash pairing from `--dflash`.
- Exporting is optimum-intel plus this repository's own tools
  (`tools/export_mtp.py`, `tools/export_dflash.py`, the tiled MoE
  lowering); a new architecture is first an exporter question, then a
  graph-audit question (paged-attention ports, GDN ledger rows, logits
  slice), then a fit question, then a drafter question.

## What recon has to establish, in this order

1. **The checkpoint**: id, size class, expert count and active count,
   attention/linear-attention pattern, head geometry, context, drafter
   heads (MTP layer? DFlash pairing available?), and whether it ships as
   safetensors only, GGUF only, or both — the last decides whether 0.4.0
   is a prerequisite or merely a neighbour.
2. **Deltas against the served family**: every `config.json` key the
   loader reads that this checkpoint sets differently or not at all
   (`layer_types`, `full_attention_interval`, `linear_*`, `head_dim`,
   `num_experts`, `moe_intermediate_size`, sliding-window or gated-attention
   fields). Each delta is either "read already" or a change with a red
   case.
3. **Export**: does optimum-intel at the pinned OpenVINO nightly export
   it; does the tiled MoE lowering apply; which precisions
   (`--quant q4|q8`) are meaningful for its expert size; whether the
   export needs a `tools/export_*.py` of its own.
4. **Fit**: the reservation terms at its geometry on both cards (the
   16 GiB card's admitted depth at u8 and u8:i4, the 24 GB card's), the
   expert-offload and host-tier paths if it does not fit resident.
5. **Drafting**: an MTP head if present (measured against plain at depth,
   as `mtp-cycle-wall` demands), a DFlash drafter only if a checkpoint of
   that lineage exists for it.

## Gate

- Loads on both cards from an allowlisted directory with the banner,
  `/props` and refusals of the served family; a wrong file refused by
  name.
- Prüfstand through the served endpoint at the artifact's own sampling
  defaults and greedy, the score recorded with card, precision and depth;
  the §5 equivalence bar (cold/warm, one lane/two, paged/stateful) byte-
  exact.
- The depth ladder cell on both cards at `u8` and `u8:i4`; a fit line for
  each card at auto-fit and at an explicit depth.
- Any drafter it ships measured against plain at the same depth before it
  is served by default (the M11 rule: MTP is off at depth on the dense
  27B because it never beat plain there).

## Entry criteria

The checkpoint on the dev host; the pinned runtime able to export it (or
the export blocker named); `docs/model_requirements.md` current, so the
deltas are against a written record and not memory.

## Pipeline

Recon → design note `docs/design-qwen-flash-next.md` (the deltas, the
exporter plan, the fit at both cards' budgets) → red-first implementation
(allowlist and loader deltas with unit tests; exporter changes with a
device-free selftest) → one card window per card → review → DESIGN
`§7.0.2x`, CHANGELOG, `model_requirements.md` gains the row, README's model
table gains the entry.

## Size

Unknown until recon step 1; medium if the architecture is the served
family with different numbers, large if it introduces a layer type the
exporter or the plugin has never lowered.

## Status

- 2026-09-05 — recorded; nothing started. Independent of 0.4.0 unless the
  checkpoint ships only as GGUF.

# FIX B — config delta ledger: Qwen Flash Next against the served family

## Checkpoint identification

Searched Hugging Face for "Qwen Flash Next", "Qwen4", and combinations with
"config.json"; also tried `Qwen3-Flash-Next`. The checkpoint that came back
is:

- **repo id**: `Qwen/Qwen3.8-Flash-Next` (also mirrored as `-FP8` and
  `-NVFP4` quantised variants under `Qwen/`, `Inferact/`, `RadixArk/`,
  `unsloth/`, `nvidia/`, `AtomicChat/`; a `QwenLM/Qwen3.8-Flash-Next` GitHub
  repo exists too). This BF16 repo's `config.json` is what this ledger is
  built from, fetched directly
  (`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`).
- **top-level `model_type`**: `qwen4_exp`
- **`text_config.model_type`**: `qwen4_exp_text`
- **`architectures[0]`**: `Qwen4ExpForConditionalGeneration`

The model card describes it as an early, explicitly experimental preview of
the Qwen4 architecture: 512 experts (10 routed + 1 shared active per token),
125B total / 6B activated, a 51B-parameter n-gram embedding indexed at layer
2, a 1-layer MTP head, and "Qwen Sparse Attention" (QSA) replacing plain
full attention on the interval layers. All of that matches the recon and
the fetched `config.json` (`num_experts: 512`, `num_experts_per_tok: 10`,
`shared_expert_intermediate_size: 640`, `ngram_*` keys, `mtp_num_hidden_layers:
1`, `head_dim: 256` / `num_key_value_heads: 2` on the full-attention layers,
`linear_num_key_heads: 16` / `linear_num_value_heads: 48` /
`linear_key_head_dim: 128` / `linear_value_head_dim: 128` /
`linear_conv_kernel_dim: 4` on the linear-attention (GDN) layers).

For the "served value" column, the served family's own public config.json
files were fetched the same way for comparison:
`Qwen/Qwen3.8-27B` (dense, `qwen3_5`/`Qwen3_5ForConditionalGeneration`) and
`Qwen/Qwen3.6-35B-A3B` (MoE, `qwen3_5_moe`/`Qwen3_5MoeForConditionalGeneration`).
These are the public upstream configs for the architecture family arcint's
allowlist entries are built from — the actual served artifacts are custom
exports (pruned experts, quantised) of the same architecture, so geometry
fields not touched by pruning/quantisation (attention shape, layer types,
MoE/embedding config keys) should match; `models/allowlist-raw.json` and
`src/core/model_registry.cpp` were the source for the values that are
export-specific (`hidden`, `layers`, `experts`, hashes).

## What the loader actually reads

Two files were read to establish this: `src/core/artifact.cpp`
(`load_artifact`) and `docs/model_requirements.md` §1. A repo-wide grep for
JSON key lookups (`.value(`, `.contains(`, `int_or`, bracket access) turned
up two more read-sites the docs don't mention:

- `src/exec/fit.h` (`packed_values_scratch_geometry`): `num_attention_heads`,
  `head_dim` — sizes the u8:i4 packed-values scratch-buffer cap.
- `src/exec/backend_ov.cpp`: `moe_intermediate_size` (expert host-slot
  byte sizing); and, GGUF-open path only (`apply_gguf_weights`, dense
  `qwen35` template only per `docs/model_requirements.md` §6 — MoE files
  are stage 2/3, not yet served) — `num_hidden_layers`, `hidden_size`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`,
  `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`,
  `linear_value_head_dim`, `full_attention_interval`, `vocab_size`.

`model_type` itself is stored (`Artifact::model_type`,
`ModelEntry::model_type`) but never branches control flow anywhere in
`src/` — confirmed by grep; no `if (model_type == ...)` exists.

## Delta table

| config.json key | served value (qwen3.8-27b dense / qwen3.6-35b-a3b MoE) | Flash-Next value | classification | red case |
|---|---|---|---|---|
| `model_type` (top level) | `qwen3_5` / `qwen3_5_moe` | `qwen4_exp` | read already (stored, not branched on) | — |
| `text_config.model_type` | `qwen3_5_text` / `qwen3_5_moe_text` | `qwen4_exp_text` | read already (via `tc.value` fallback) | — |
| `architectures[0]` | `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` | `Qwen4ExpForConditionalGeneration` | read already (stored as `ov_arch`) | — |
| `text_config.num_hidden_layers` | 64 / 40 | 48 | read already | — |
| `text_config.hidden_size` | 5120 / 2048 | 2560 | read already | — |
| `text_config.max_position_embeddings` | 262144 / 262144 | 262144 | read already | — |
| `text_config.num_experts` | absent (dense) / 256 | 512 | read already (sets `moe = true`) | — |
| `text_config.full_attention_interval` | 4 / 4 | 4 | read already | — |
| `text_config.layer_types[]` | 16×`full_attention`/48×`linear_attention` pattern / 10×`full_attention`/30×`linear_attention` pattern | same two-string vocabulary (`full_attention`, `linear_attention`), interval 4, 48 entries | read already (string set unchanged; the semantic content of a `full_attention` layer differs — QSA replaces plain attention on those layers, which is FIX C's kernel-audit scope, not a config-reading delta) | — |
| `text_config.eos_token_id` | 248044 (int) | 248044 (int) | read already | — |
| `text_config.num_attention_heads` | 24 (dense) / 16 (MoE) | 24 | read already (`fit.h` scratch-geometry cap) | — |
| `text_config.head_dim` | 256 (dense) / **absent** (MoE — `packed_values_scratch_geometry` returns `nullopt` today for the served MoE family) | 256 (present) | read already — and now succeeds where it silently no-ops for the served MoE family; not a regression | — |
| `text_config.num_key_value_heads` | 4 (dense) / 2 (MoE) | 2 | read already (GGUF geometry check only; N/A — MoE GGUF not served, §6) | — |
| `text_config.vocab_size` | 248320 / 248320 | 248320 | read already (GGUF geometry check only, same N/A) | — |
| `text_config.moe_intermediate_size` | absent (dense) / 512 | 640 | read already (expert host-slot byte sizing; 25% larger per-expert slots × 2× more experts — the fit arithmetic in FIX C/D must account for this) | — |
| `text_config.shared_expert_intermediate_size` | absent (dense) / **512, already present in the served MoE family** | 640 | new key, not read — pre-existing gap, not introduced by Flash-Next: the host-slot sizing formula in `backend_ov.cpp` already ignores this key for the served 35B-A3B; out of this delta's scope | — |
| `text_config.num_experts_per_tok` | absent (dense) / 8 | 10 | new key, not read (no code path needs the active-expert count; the host/device slot formulas size all resident experts, not the per-token active subset) | — |
| `text_config.linear_num_key_heads` / `linear_num_value_heads` / `linear_key_head_dim` / `linear_value_head_dim` / `linear_conv_kernel_dim` | 16 / 48 / 128 / 128 / 4 (MoE) — dense has no linear-attention layers, keys absent | 16 / 48 / 128 / 128 / 4 | new key, not read outside the GGUF (N/A) path — values match the served MoE family exactly | — |
| `text_config.mtp_num_hidden_layers` | 1 / 1 (flat scalar; not read by loader — MTP presence is file-existence-detected) | 1 (flat, unchanged) | new key, not read | — |
| `text_config.mtp` (nested object: `hybrid`, `layer_types`, `mtp_use_hidden_state_from_layer`, `num_hidden_layers`, `rope_theta`) | absent | present | new key, not read | — |
| `text_config.ngram_size`, `ngram_vocab_size_base`, `heads_per_ngram`, `make_ngram_vocab_size_divisible_by`, `split_ngram_parts`, `ple_layer_ids`, `ple_conv_kernel_size`, `ple_embed_dim` (the 51B n-gram embedding) | absent | present | new key, not read | RED-B-01 |
| `text_config.indexer_budget`, `indexer_compress_ratio`, `indexer_head_dim`, `indexer_kv_heads`, `indexer_n_heads` (QSA indexer) | absent | present | new key, not read | — (see note below) |
| `text_config.hc_count`, `hc_lowrank` (gated residual) | absent | present | new key, not read | — |
| `text_config.rope_parameters` (nested: `mrope_interleaved`, `mrope_section`, `partial_rotary_factor`, `rope_theta`, `rope_type`) | present, identical shape | present, identical values (`rope_theta` 10000000, `mrope_section` [11,11,10]) | new key, not read (neither served nor Flash-Next config is read here) | — |
| `generation_config.json` (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty`) | present for served exports | not fetched — this repo's `Qwen/Qwen3.8-Flash-Next` listing was not checked file-by-file for it | read already if present, optional either way (`load_artifact` tolerates its absence) | — |

## Red-case appendix

**RED-B-01 — the 51B-parameter n-gram embedding has no representation in
`load_artifact`.**

`load_artifact` (`src/core/artifact.cpp`) has exactly one precedent for "a
component every served checkpoint ships but the loader does not compile":
the vision tower, handled by `scan_unloaded_vision_irs` /
`kUnloadedVisionIrBaseNames` (`artifact.cpp:38-67`) — it stat's the three
vision IR base names on both extensions and records what is actually on
disk (`UnloadedIr{name, size}`) precisely because a first version silently
under-reported an IR's presence ("2 files, 1.7 MiB" against a 457 MB merger
on disk — see the comment at `artifact.cpp:38-49`).

Flash-Next's config declares an n-gram embedding (`ngram_size: 3`,
`ngram_vocab_size_base: 20000000`, `heads_per_ngram: 8`,
`make_ngram_vocab_size_divisible_by: 128`, `split_ngram_parts: 128`,
`ple_layer_ids: [2]`, `ple_conv_kernel_size: 4`, `ple_embed_dim: 2560`) that
the model card sizes at 51B parameters — larger than the entire weight file
of any artifact currently in the allowlist (the biggest, `qwen36-35b-a3b-ov`,
is 18.6 GB int4; 51B parameters at int4 alone is over 25 GB). Nothing in
`load_artifact` looks for it: it is not in the required-file list, not
scanned the way the vision tower is, and `a.weights_bytes` is computed from
`file_size(a.language_model_bin)` alone. If a Flash-Next export ships the
n-gram table as its own IR pair (as the vision tower is shipped, and as
its 51B size relative to the 6B activated width suggests it would be, to
keep it off the compute graph), an artifact missing that IR would load
without any diagnostic, and `weights_bytes` would report a small fraction
of what is actually meant to be on disk — the failure mode the vision-tower
fix above was already written to prevent, recurring for a component that,
unlike the vision tower, is *not* reserved/refused — it is load-bearing for
generation quality on this checkpoint, not an optional modality.

*What the test must assert*: given a synthetic Flash-Next-shaped artifact
directory whose `config.json` declares n-gram embedding keys, and (a) an
n-gram embedding IR file present, or (b) that file absent —
`load_artifact` must, in case (a), account for its bytes in a field the
caller can see (either folded into a corrected `weights_bytes` or reported
the way `unloaded_vision_irs` reports the vision tower), and in case (b),
either refuse the artifact by name or report the component's absence
explicitly — never silently produce a `weights_bytes` figure that omits it
without saying so. The red case must fail against today's `artifact.cpp`
(which does neither) before any fix, per this repository's "a test must be
able to fail" rule.

**Note on the QSA indexer keys (`indexer_budget`, `indexer_compress_ratio`,
`indexer_head_dim`, `indexer_kv_heads`, `indexer_n_heads`)**: these are
small (MQA-shaped, 1-4 heads at head_dim 128) relative to the n-gram table,
and every existing hybrid-attention component this loader already serves
(GDN's own state tensors, the `full_attention` layers themselves) is fused
into the single `openvino_language_model.xml/.bin` graph rather than
shipped as a separate IR. Absent an actual Flash-Next OpenVINO export to
inspect, there is no evidence the indexer is exported any differently — so
this is logged as "new key, not read" without a red case, rather than
asserting a break that has not been observed. If a real export later shows
the indexer as a separate artifact file, this note is what should become
RED-B-02.

## What did not work / was not checked

- The `generation_config.json` for `Qwen/Qwen3.8-Flash-Next` was not
  fetched — only `config.json` was pulled from the repo. If its sampler
  defaults are absent or malformed that would show up under the existing
  "read already, optional" handling, not as a new finding here.
- No Flash-Next OpenVINO IR export exists to inspect (no `--gguf`/`--model`
  directory, no `arch_hash` to pin) — every classification above is derived
  from `config.json` and the model card text alone, not from a real
  artifact directory. `n_layer`/`n_gdn_layer`/`n_attn_layer` arithmetic,
  hashes, and `weights_bytes` therefore cannot be measured, only reasoned
  about from the declared `layer_types[]`.
- The dense `qwen3.8-27b` config's `head_dim` (256, present) versus the
  served MoE family's `head_dim` (absent) was confirmed by fetching
  `Qwen/Qwen3.6-35B-A3B/config.json` directly, not assumed from the recon.

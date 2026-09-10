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

## FIX C — GDN geometry and sparse-attention kernel audit

### Scope correction against the assignment

The assignment names "patches 0021-0030" as the GDN/paged-attention kernel
series. Reading `contrib/packaging/marfrit-openvino/patches/README.md` (the
authoritative per-patch record; `patches/` at the repository root is a
browsing mirror of the same files, per its own header) shows 0021-0030 are
the **GGUF K-quant fully-connected kernel** series (`FullyConnectedKQuant`,
Q4_K/Q5_K/Q6_K decode and tiled-prefill variants) -- weight-format kernels
for the GGUF-open dense (`qwen35` template) path, unrelated to GDN state or
paged-attention geometry. They are **N/A to this audit**: FIX B established
that the GGUF-open path in `apply_gguf_weights` (`backend_ov.cpp`) serves
only the dense `qwen35` template (`docs/model_requirements.md` §6; MoE GGUF
files are stage 2/3, not yet served), and Flash-Next is a MoE checkpoint --
it would load through the IR path, not GGUF-open, so 0021-0030 never touch
it under any configuration this repository drives today. Kept in the table
below, marked N/A, rather than silently dropped.

The patches actually keyed to attention/GDN/paged-KV geometry are
**0008-0020** (paged-KV precision, asymmetric kernel plan, micro-SDPA) and
**0032-0035** (micro-SDPA prefetch/alignment/tail correctness); **0003-0007**
and **0011-0019** are the MoE offload-tier series (expert count/size, not
attention geometry, but directly priced by Flash-Next's expert-count and
`moe_intermediate_size` growth -- covered in the Fit section). One more
GDN-specific kernel exists outside the numbered patch series entirely: the
`LgcPermute` custom-layer head-major swap in `backend_ov.cpp` (`route_head_
swap_permutes`), bound through arcint's own `CustomLayer` OpenCL kernel
config (`cfg.custom_kernels`), not through a patch to the pinned OpenVINO
build. It is in the table below because it is a GDN kernel, even though it
carries no patch number.

### Kernel verdict table

| patch | kernel | Flash-Next geometry | verdict | red case |
|---|---|---|---|---|
| (none -- `backend_ov.cpp` `route_head_swap_permutes` / `LgcPermute`) | GDN head-major transpose `(0,2,1,3)` -> custom OpenCL kernel | Hardcoded match on `in[2]==32 && in[3]==128` (the served GDN reshape's own dims, not `num_attention_heads`/`head_dim` from config) | fits, contingently | Flash-Next's GDN layers are declared byte-identical to the served MoE family (`linear_num_key_heads` 16 / `linear_num_value_heads` 48 / both dims 128 / conv kernel 4, FIX B) -- so whatever internal reshape the exporter produces here should also land on `[.., 32, 128]` for Flash-Next, since nothing about that geometry changed. The literal is a shape MATCH, not a computed one: it silently stops routing to the fast kernel (falls back to the generic `Transpose`, not a break, per its own selectivity comment) the moment any exporter revision changes this specific reshape's trailing dims -- including a Flash-Next export that shapes the GDN head-major tensor differently for reasons unrelated to `linear_*` config (e.g. the export batches K/V heads into one combined axis before the swap for QSA/GDN interop). RED-C-01: compile a synthetic Flash-Next-geometry graph, feed it through `route_head_swap_permutes`, and assert the routed count equals the number of GDN layers (36) -- today this can only be checked once a real Flash-Next IR exists, so the red case is currently un-runnable and should be logged as such, not skipped. |
| 0003 | MoE batched-GEMV expert-mask subbuffer cache | `expert_num` read from `config.num_expert` (compiled graph config, itself sourced from `text_config.num_experts` = 512) -- no hardcoded expert count | fits | none -- the patch's own diff uses `config.num_expert` throughout; 512 replaces 256 as a plain runtime value. The `README`'s "512 calls per layer" is the SERVED number, not a compiled-in constant. |
| 0004 | MoE OTD perf counters | instrumentation only | fits | none |
| 0005-0007 | MoE OTD device-resident slot pool / async upload / redundant-finish removal | `slots(m) = ceil(num_expert*(100-ratio)/100)` (fit.h `expert_slot_bytes`), `per_expert_bytes` from the IR walk or `3*hidden*moe_intermediate*bytes_per_weight` | fits mechanically, **unmeasured at Flash-Next's admission point** | 0005-0007 were tuned and measured (10.4-15.5 t/s) at the served 256-expert / 1.5 MiB-per-expert-layer geometry with offload ratios 50-75%. Flash-Next's per-expert-layer slot is 2,457,600 B (see Fit section) and needs an offload ratio in the high-80s to low-90s percent just to keep the resident slot pool under a few GiB -- a working-set fraction (8-15% of 512 experts resident) well past anything 0005-0007's plateau probe or async-upload batch sizing was tuned against. RED-C-02: run the plateau probe (`MOE_OTD_PERF_LOG`) at `--offload-ratio` in the high 80s against a synthetic 512-expert graph and assert the probe still converges (does not oscillate/thrash) and the async-batch upload path still completes within one inference step's budget -- this is a capacity/tuning question, not a correctness break, and there is no existing test at this ratio to point to. |
| 0008 | `VALUE_CACHE_PRECISION` config knob | orthogonal to head/expert geometry | fits | none |
| 0009 | Asymmetric KV kernel-plan decline (mixed key/value packing on decode fast path) | orthogonal to head/expert geometry -- gates on `kv_cache_dt`/`value_cache_dt` equality, not head count | fits | none -- this decline is unaffected by Flash-Next's `(24, 2, 256)` full-attention shape vs. the served `(24, 4, 256)` / `(16, 2, ~128)` shapes; the guard is precision-keyed, not geometry-keyed. |
| 0010 | Per-side u8:i4 decode-read/write kernel | `num_kv_heads`/`head_size`/`num_blocks` all read as runtime parameters through the kernel's own port shapes | **fits, verified** | RED-C-03 CLOSED GREEN (2026-09-09): 7/7 cells at `(24, 2, 256)` pass at 1e-2 tolerance under all three page orders. The previously untested GQA group size 12 is now measured. |
| 0011-0012 | MoE host CPU tier kernel (AVX2/scalar) + decode-split wiring | `num_expert`, `per_expert_bytes` runtime params; LRU probe geometry-agnostic | fits mechanically, **same admission-ratio caveat as 0005-0007** | See RED-C-02 -- 0011-0012's own measured cells (15.0-15.5 t/s at ratio 50/75) are the SAME two ratios 0005-0007 were measured at; Flash-Next's arithmetic (Fit section) needs a ratio in the high 80s-low 90s, unmeasured for the host tier's own readback/overlap timing (0017's `usm_host` hoist was tuned at ratio 50/75 too). |
| 0013 | MoE routing histogram | instrumentation only | fits | none |
| 0014 | `assign_impl` output-layout adoption (DFlash2 K/V state chain) | DFlash2 is a separate drafter lineage (`--dflash`), not part of Flash-Next's own MTP head | N/A | out of scope -- Flash-Next serves through its OWN 1-layer MTP head (`mtp_num_hidden_layers`, `text_config.mtp`), not the DFlash2 draft head this patch fixes. |
| 0015-0016 | Paged-attention bounded partials, intermediate sizing | `PAGED_ATTENTION_MAX_PARTITIONS`, `get_internal_buffer_descs` sized from `num_of_partitions`/output dtype -- no head/expert-count literal | fits | none in the diff; the buffer terms scale from `n_ctx`/chunk, which are runtime values regardless of model architecture. |
| 0017-0019 | MoE host-tier readback decomposition, static partition, prefill-fallback tristate | `splitmix64(seed, layer_key, expert)` ranking is `expert`-count-agnostic (any `expert` value ranks); `slots` from the same `expert_slot_bytes` formula as 0005-0007 | fits mechanically, same ratio caveat as 0005-0007/0011-0012 | See RED-C-02. Additionally: 0018's own header records "five load-time bugs" surfaced only once the offload ratio was pushed hard enough for a 100%-pinned pool to be reachable at all -- exactly the regime Flash-Next's arithmetic requires by default, not as an edge case. A fresh load-time bug hunt at Flash-Next's admission ratio should be budgeted as likely, not assumed clean by analogy to the 256-expert result. |
| 0020 | Micro-SDPA admits u8 keys / i4 values on prefill (was OCL-fallback only) | Selector gates on precision, not head count; measured at `(16, 2, ~128)`-class (coder) and `(24, 4, 256)`-class (agent/drafter) shapes only | **fits, verified** | RED-C-03 CLOSED GREEN (2026-09-09): 7/7 cells at `(24, 2, 256)` pass. See RED-C-03 measurement record below. |
| 0021-0030 | GGUF K-quant `FullyConnectedKQuant` decode/tiled-prefill kernels | N/A -- GGUF-open serves the dense `qwen35` template only (FIX B §"What the loader actually reads") | N/A | Flash-Next is MoE; it does not reach `apply_gguf_weights` under any configuration this repository drives today. Listed per the assignment's own patch range, not because these kernels touch GDN/attention geometry. |
| 0031 | `fc-deterministic-gemm` (oneDNN split-K determinism) | Applies to the GGUF-open f16-activation compressed FC form only | N/A | same reason as 0021-0030. |
| 0032 | Micro-SDPA K-tile prefetch bounds fix (the CAT-error fix) | Regression test at `(24, 4, 256)` only | **fits, verified** | RED-C-03 CLOSED GREEN (2026-09-09): 7/7 cells at `(24, 2, 256)` pass. The prefetch bounds fix generalises to `num_kv_heads=2` as the code-read predicted. |
| 0033 | Micro-SDPA V-operand alignment fix (`alignment_for_ld` keyed to `head*2`, capped 128) | `head` here is `head_dim` (256 for both the tested dense shape and Flash-Next); `num_kv_heads` does not enter `alignment_for_ld` | fits | Confirmed from the patch diff: `alignment_for_ld` is a pure function of `head` (== head_dim) and the value precision, with `num_kv_heads` used only for buffer indexing (`num_kv_heads * block_size * head_size` strides) elsewhere in the same file -- these two roles are independent. Flash-Next's head_dim (256) is the exact value the fix was written and tested against, so this specific bug class is closed for Flash-Next's shape. Still covered by RED-C-03 for the combination as a whole (indexing correctness at `num_kv_heads=2`, not the alignment constant). |
| 0034 | Micro-SDPA tail test (reads past `seq_len`) + by-token reproducer (disabled) | Tail test run at `(24, 4, 256)`; by-token reproducer at `(32, 2, 128)` default and `(24, 4, 256)` served | fits (tail test), **open, disabled finding (by-token)** | The by-token pairing itself is not reached by Flash-Next (arcint never selects BY_TOKEN keys per the patch's own note), so this finding does not block Flash-Next. Carried in the table for completeness since the assignment asked for every kernel in the series; not a Flash-Next-specific concern. |
| 0035 | BY_TOKEN test harness key-fill fix | Test-only, no served kernel change | N/A to Flash-Next serving path | none |

**RED-C-03 CLOSED GREEN (measured 2026-09-09).** The (24, 2, 256) geometry
passes all 7 cells (ascending, reversed, gapped page orders at two
subsequence sizes plus the 3×1000-token long-context pair) at 1e-2
tolerance against the float reference, on the dev host's GPU. Patch 0036
(the fixture) applied cleanly to ovsrc-m18, built incrementally in
build-m18 (`ENABLE_TESTS=ON`), and ran in 9.3 s total (39 cases including
the pre-existing 0033/0035 suite; 21 in the main suite, all 21 passed).
The "fits, unverified" verdict for 0010/0020/0032/0033 at Flash-Next's
full-attention shape is now **fits, verified by measurement** — the third
combination `(24, 2, 256)` exercises every stride, loop-bound and
alignment path those patches parameterise over `num_kv_heads` at a GQA
group size (12) no prior cell tested.

Three pre-existing failures in the 0035 by-token suite (cases /7-/9,
original (32, 2, 128) geometry, "reverse" page order) are the known
BY_TOKEN block-size gate from patch 0034: "Incorrect block size for Paged
Attention operation for key cache quant mode BY_TOKEN. Expected 16, but
got 32." These are not Flash-Next-related and not reached by arcint's
serving path (which never selects BY_TOKEN keys).

### Sparse attention (QSA)

**What it is** (from the config keys plus public documentation -- no arcint
code reads or reasons about these keys today, confirmed by grep for
`indexer_` across `src/`): Qwen Sparse Attention replaces plain full
attention on Flash-Next's 12 `full_attention`-labelled layers with a
block-sparse, indexer-gated mechanism. A lightweight indexer (`indexer_n_
heads`, `indexer_kv_heads`, `indexer_head_dim` -- small, MQA/GQA-shaped
relative to the main attention) scores the KV sequence at 4-token block
granularity (`indexer_compress_ratio` 4 -- one compressed key per four real
tokens) and selects the top `indexer_budget` blocks (reported at 512 blocks
/ 2,048 tokens) for the query to actually attend over. The main attention
for that layer then runs only over the selected, dynamically-varying
(per-query) subset of the KV cache, not the full causal context.

**Whether the OpenVINO op set can express it**: not as the single fixed
pattern arcint's export pipeline currently relies on. The whole paged-KV
serving path in `backend_ov.cpp` depends on `ov::pass::SDPAToPagedAttention`
(`backend_ov.cpp:2533`) rewriting a plain, statically-shaped scaled-dot-
product-attention subgraph (fixed causal mask, one KV cache, no per-query
selection) into the plugin's `PagedAttention` op. QSA is not that subgraph:
it needs (1) a second, smaller attention-like computation (the indexer) over
compressed keys, (2) a `TopK`/`Gather`-shaped block-selection step whose
output varies per query and per decode step, and (3) a main attention whose
KV extent is that dynamic selection rather than "everything up to `past_
len`". OpenVINO's generic op set (`MatMul`, `Softmax`, `TopK`, `Gather`) can
express steps (1) and (2) as ordinary ops; step (3) is exactly the paged/
block-sparse attention shape that `PagedAttentionOptImpl` and its kernels
(the entire 0008-0020/0032-0035 series this audit just read) were built to
serve for a FIXED, non-selective KV window -- none of that code has a
"gather these specific blocks by index per query" mode. Building QSA's
step 3 out of unfused generic ops (materializing the selected KV subset per
query via `Gather` before an ordinary dense attention over just that subset)
would work functionally but forfeits every paged-KV optimization this
repository's whole patch series exists to deliver -- no in-place cache
paging, no micro-SDPA, no block-table reuse across decode steps -- and its
cost model is unmeasured and likely to reintroduce exactly the kind of
per-inference host round trips patches 0003-0007 exist to remove.

**What the export would lower it to**: unknown without a real Flash-Next
OpenVINO IR export to inspect -- the honest position FIX B already recorded
for the rest of Flash-Next's new config surface, and it applies doubly here
since QSA has no analogue in either served checkpoint's graph. optimum-intel
(or whatever exporter produces the IR) has to have made a choice already;
until that export exists, this audit cannot say whether QSA is lowered to
(a) a set of generic ops arcint's transform passes have never seen and would
need new pattern-matching to route to any paged path at all, or (b) a
single fused custom op (the way `FullyConnectedKQuant` and `LgcPermute`
already are, in this same codebase) that the exporter and arcint would both
need to agree on -- nothing in the exporter or in optimum-intel's public
tree was inspected as part of this audit (out of scope: no network access
to a real Flash-Next IR from this pass, same limitation FIX B logged).

**Verdict: refuses.** Absent a QSA-aware transform pass and a QSA-aware
kernel (neither exists in this repository or in the patch series today),
the honest behavior is an explicit refusal by name at load, not a silent
fallback to the unfused generic-op path described above (which would serve
wrong-shaped or catastrophically slow attention without saying so -- the
same failure class RED-B-01 flagged for the n-gram embedding). **What the
red case must assert**: given a synthetic Flash-Next-shaped `config.json`
(`indexer_budget`/`indexer_compress_ratio`/`indexer_head_dim`/`indexer_kv_
heads`/`indexer_n_heads` present) paired with an IR graph containing an op
this repository's transform passes do not recognize as either plain SDPA or
an already-known custom op, `load_artifact` or the graph-rewrite pass must
refuse the load by name (citing QSA/the indexer keys), not compile a graph
that silently drops the indexer's selection and either runs full attention
over everything (wrong output, unmeasured cost) or crashes on an unmatched
op type deep in `SDPAToPagedAttention`. This is RED-C-04; like RED-C-03 it
cannot be made to run today because no Flash-Next IR export exists to
supply the graph half of the test -- logged as open rather than skipped.

### Fit: admitted depth at Flash-Next geometry

Per-expert-per-layer slot bytes (`3 * hidden * moe_intermediate_size *
bytes_per_weight`, the `config`-source fallback formula in `backend_ov.cpp`
around the MoE host-slot ledger, `int4` => `bytes_per_weight = 0.5`):

| | hidden | moe_intermediate_size | bytes/expert/layer (int4) | num_experts | moe layers | full residency (int4) |
|---|---|---|---|---|---|---|
| served MoE (35B-A3B) | 2048 | 512 | 1,572,864 B (1.50 MiB) | 256 | 40 | 16,106,127,360 B (15.00 GiB) |
| Flash-Next | 2560 | 640 | 2,457,600 B (2.34 MiB) | 512 | 48 | 60,397,977,600 B (56.25 GiB) |

The per-expert-layer slot grows **1.5625x** (both `hidden` and `moe_
intermediate_size` grow 25%, and the two factors compound -- FIX B's delta
table calls out the `moe_intermediate_size` growth alone as "25% larger
per-expert slots"; the `hidden_size` growth compounds on top of it, so the
per-slot growth is 56%, not 25%, and this is worth a correction note against
that table) on top of 2x the expert count and 1.2x the layer count (48 vs.
40) -- full residency is **56.25 GiB**, more than double what the served
model needs (15.00 GiB) and beyond either card's total VRAM outright (the
24 GB card, the larger of the two, has room for well under half of it before
any KV pool, activation reservation, or non-expert weight is even counted).

Working backward from `expert_slot_bytes(num_expert, ratio_pct, per_expert_
bytes, moe_layers) = ceil(num_expert*(100-ratio_pct)/100) * per_expert_bytes
* moe_layers` (`fit.h`), the offload ratio Flash-Next needs for a given
device-resident expert-slot budget:

| device-resident expert-slot budget | resident slots | admitted `ratio_pct` |
|---|---|---|
| 4 GiB | 36 of 512 | ~93% |
| 6 GiB | 54 of 512 | ~89% |
| 8 GiB | 72 of 512 | ~86% |
| 10 GiB | 91 of 512 | ~82% |
| 12 GiB | 109 of 512 | ~79% |

Every measured cell for the offload-tier patches (0005-0007, 0011-0012,
0017-0019) sits at `ratio_pct` 50 or 75 -- the served model's own working
point, where up to half the experts stay resident. Flash-Next's arithmetic
puts the admitted ratio in the high 80s to low 90s **by construction**, not
as an edge case reachable only under unusual configuration: on the smaller
(16 GiB-class) card, where the served model already runs its offload dial
near its tightest setting, an 8-10 GiB expert-slot budget is itself
optimistic once the KV pool, drafter/MTP state, and the paged-attention
scratch terms (`packed_values_prefill_scratch_bytes_ex`, now correctly
priced for Flash-Next since `head_dim` is present -- see FIX B) are
subtracted from the same VRAM pool. No card in scope admits Flash-Next with
anything like the resident fraction the offload-tier code has actually been
tuned and byte-verified at; this is the practical bottleneck the kernel
audit above keeps deferring to RED-C-02, not a kernel that refuses outright.

A second, narrower fit finding, independent of the MoE arithmetic: `fit.h`'s
`kMtpStateBytesPerToken = 8192` (comment: "4 heads * 256 head_dim * 2 (K+V)
* 4B (f32)") is a **literal constant matching the served dense drafter's
own measured MTP-analogue geometry**, not a value derived from `config.json`
at all -- confirmed by grep, nothing in `fit.h` or `backend_ov.cpp` reads an
MTP head/kv-count/head-dim key before using this constant (`mtp_state_bytes`
just multiplies it by `n_ctx`). Flash-Next's own MTP head is described by
the new, unread `text_config.mtp` object (`hybrid`, `layer_types`, `mtp_
use_hidden_state_from_layer`, `num_hidden_layers`, `rope_theta` -- FIX B) --
whether its state is `(4 heads, head_dim 256)`-shaped like the dense
drafter, GDN-shaped like the base model's own `linear_*` layers (`hybrid`
suggests it may mix both), or something else entirely is not established by
anything read in this pass. If it differs, this constant silently mis-
prices the MTP-state term folded into `fterms.kv_bytes_token` at every load
that serves Flash-Next with MTP on -- no warning, no refusal, the same
failure shape RED-B-01 and RED-C-04 both flag elsewhere in this delta.
**RED-C-05**: given a Flash-Next artifact with MTP enabled, assert that the
per-token MTP state charge folded into the fit is derived from the actual
compiled MTP-head state tensor shapes (the way `kv_bytes_token_` itself is
already derived from the compiled model's `conv_state_table.`/`gated_delta_
state_table.`-prefixed tensors at `backend_ov.cpp:3150-3294`, generically),
not from a constant calibrated against a different checkpoint's drafter.
Un-runnable today for the same reason as RED-C-01/03/04: no Flash-Next IR
exists to compile and inspect.

### What did not work / was not checked

- No Flash-Next OpenVINO IR export exists (same limitation FIX B recorded),
  so every "fits"/"needs patch" verdict above that depends on an actual
  compiled graph (the `LgcPermute` shape match, the QSA lowering question,
  the MTP-state geometry) is reasoned from config and patch-diff reading
  alone, not measured against a real artifact. RED-C-01, RED-C-03, RED-C-04
  and RED-C-05 are all logged as currently un-runnable for this reason,
  per this repository's "report what did not work" rule, rather than
  skipped silently.
- The exporter/optimum-intel side of the QSA lowering question (what op(s)
  a real export actually emits for the indexer and the block-selective
  attention) was not inspected -- no such exporter code was fetched or read
  as part of this pass.
- The offload-ratio arithmetic in the Fit section is analytic (the same
  `expert_slot_bytes` formula the codebase itself uses for its own
  ceiling estimate), not a measured card run -- there is no Flash-Next
  artifact to load and no plateau probe to run against one yet.

### FIX C addendum (2026-09-09): synthetic-fixture buildability audit

The "un-runnable until a Flash-Next IR exists" verdict above was too broad.
Every GDN kernel red case in the 0021-0030 patch series was written as a
synthetic fixture -- a toy graph or toy test-harness geometry built by hand,
never a real export -- and the same is true of most of FIX C's own red
cases once each is checked against the test infrastructure that would carry
it, rather than assumed blocked by analogy to RED-C-01/RED-C-04. Read for
this pass: `tests/test_gguf_graph.cpp` (arcint's own synthetic-`ov::Model`
IR-pass tests, CPU-only, no device), `contrib/packaging/marfrit-openvino/
patches/README.md` and the 0032-0035 patch diffs themselves (the OpenVINO
GPU plugin's own unit-test harness), and the state of `~/ovsrc` and
`~/ovsrc-pkg` on the dev host.

**RED-C-03 -- buildable today, and the concrete shape is now known.** The
paged-attention/micro-SDPA regression harness this case names does not live
in this repository at all: it is `src/plugins/intel_gpu/tests/unit/
test_cases/paged_attention_gpu_test.{h,cpp}` inside the pinned OpenVINO
tree, carried forward by patches 0032-0035. That harness builds its KV
cache, block tables and reference output entirely from test parameters
(`num_heads`, `num_kv_heads`, `k_head_size`, `v_head_size`, a subsequence
list, a page order) -- it never touches an exported IR or a real checkpoint.
Patch 0033 already added the served-geometry case as a small static
factory:

    static paged_attention_test_params u8i4_mixed_micro_served(...) {
        paged_attention_test_params p = u8i4_mixed_micro_params(subsequences);
        p.num_heads = 24; p.num_kv_heads = 4;
        p.k_head_size = 256; p.v_head_size = 256;
        p.page_order = page_order;
        return p;
    }

RED-C-03's cell is the same factory with `num_kv_heads = 4` changed to `2`,
instantiated under the same three page orders (`""`, `"reverse"`, `"gap"`)
0033 already exercises for the served shape, and compared to the harness's
own float reference at the suite's existing 1e-2 tolerance -- exactly the
pattern already in `INSTANTIATE_TEST_SUITE_P(..., paged_attention_
u8i4_mixed_micro_test, ...)`. No Flash-Next export, no `qwen4_exp`
transformers support, and no change to arcint's own `tests/` tree are
needed; the fixture is a few dozen lines added as a new numbered patch
(e.g. `0036-...`) touching only `paged_attention_gpu_test.{h,cpp}`, the
same file pair 0032-0035 already touch. This is *not* a job for
`test_gguf_graph.cpp` or a new file in arcint's own `tests/`: that file
exercises `src/exec/gguf_graph.cpp`'s IR-rewrite pass (CPU-only, no kernel
dispatch), which the paged-attention kernel path never goes through --
`gguf_graph.cpp` and `paged_attention_gpu_test.cpp` are different
codebases (this repository vs. the patched OpenVINO tree) testing different
layers (graph rewrite vs. compiled-kernel correctness).

What is genuinely missing is infrastructure, not geometry: `~/build-ov-
selftest` does not exist on the dev host, and `~/ovsrc/build-dbg` is a
stale, barely-configured tree (370 MiB, no `ov_gpu_unit_tests` binary,
default `-DENABLE_TESTS=OFF` per `build-openvino.sh`). Every prior red case
in 0015-0035 was validated by reconfiguring a *separate* build of `~/ovsrc`
(or a scratch copy) with `-DENABLE_TESTS=ON` and building the
`ov_gpu_unit_tests` target by hand each time -- there is no persisted
test-enabled tree to build against today. Standing that build up (and
running it under the test-window ritual, since it needs a free GPU) is the
actual next step, not an IR export.

**RED-C-02 -- also buildable today, no export needed.** The plateau probe
(`MOE_OTD_PERF_LOG`) and the offload-ratio machinery it drives
(`expert_slot_bytes` in `fit.h`) take `num_expert`, `per_expert_bytes` and
`moe_layers` as plain parameters; nothing in the probed path reads a
compiled Flash-Next graph. A synthetic 512-expert configuration at
Flash-Next's own `per_expert_bytes` (2,457,600 B, from the Fit section
above) and `--offload-ratio` in the high 80s/low 90s is expressible without
any checkpoint. Not audited further in this pass beyond confirming the
inputs are synthesizable; the actual probe run still needs both GPUs free
(the test-window ritual) and was not executed here.

**RED-C-01 and RED-C-04 -- genuinely blocked, but not for the reason
stated.** Both read as "un-runnable until IR exists" in the table above,
and that remains correct, but the mechanism is narrower than "no IR": both
tests hinge on a fact only a real exporter run can supply. RED-C-01 asks
whether the *actual* trailing dims Flash-Next's exporter produces for the
GDN head-swap reshape land on the hardcoded literal `[.., 32, 128]` in
`route_head_swap_permutes` -- a synthetic graph can be built with any
trailing dims we choose, including `[.., 32, 128]` itself, but that would
only prove the pass matches what we assumed, not what the real exporter
emits; the open question is the exporter's behavior, which cannot be
synthesized. RED-C-04 is the same shape of gap: it asks whether a real QSA/
indexer export contains an op this repository's passes do not recognize,
and a hand-built graph can trivially be made to contain (or not contain)
an unrecognized op either way, again proving nothing about what the real
exporter does. A synthetic fixture *can* still exercise the general refusal
mechanism itself (feed `load_artifact` a synthetic `config.json` carrying
the indexer keys alongside a graph with a deliberately-unrecognized op, and
assert refusal-by-name rather than silent fallback) -- that is worth
building as a mechanism test, but it is a weaker claim than RED-C-04 as
written, which is about what the real exporter's output actually contains.

**RED-C-05 -- likely buildable, not attempted in this pass.** The claim is
that `fit.h`'s MTP-state charge should be derived from compiled
`conv_state_table.`/`gated_delta_state_table.`-prefixed tensor shapes the
way `kv_bytes_token_` already is (`backend_ov.cpp:3150-3294`), not from the
`kMtpStateBytesPerToken = 8192` literal. Since that derivation already
walks the compiled model's own tensors by name/prefix generically, a
synthetic `ov::Model` carrying result tensors named with those prefixes at
a chosen (non-dense-drafter) shape should be enough to exercise it, in the
same style as `test_gguf_graph.cpp`'s toy templates. Not read closely enough
in this pass to commit to the exact fixture shape; flagged as the next one
to scope, not as blocked.

**Summary**: of FIX C's five red cases, RED-C-02 and RED-C-03 are buildable
as synthetic fixtures today with no export dependency (RED-C-03's shape is
now fully specified above), RED-C-05 is likely buildable pending a closer
read of `backend_ov.cpp:3150-3294`, and RED-C-01/RED-C-04 remain genuinely
blocked on a real export -- not because no test infrastructure exists, but
because the fact under test (what the real exporter actually emits) cannot
be conjured synthetically without begging the question. Test code itself
is not written as part of this pass, per instruction; this is the
buildability assessment to review before it is.

## FIX A — Export harness and upstream blocker

**The blocker (updated 2026-09-09).** Two upstream gaps, not one. The
original framing ("no transformers release carries qwen4_exp") was stale:
HF Transformers contributed `qwen4_exp` on 2026-08-26 (PR #48337), and
`transformers.models.qwen4_exp` imports cleanly in transformers 5.17.0.
The actual blocker chain, verified in a fresh venv on the dev host:

  1. **optimum-intel 2.1.0 pins `transformers<5.6,>=4.51`.**
     `pip install optimum-intel` resolves to transformers 5.5.4, which does
     NOT carry `qwen4_exp` (`ModuleNotFoundError` on import). The cap is
     an explicit `<5.6` in optimum-intel's own requirements, not a solver
     accident.

  2. **Force-upgrading transformers to 5.17.0 breaks optimum-intel 2.1.0
     at import time.** The crash site is
     `optimum.intel.openvino.modeling_visual_language` (line 40), which
     does `from transformers.models.qwen2_vl.modeling_qwen2_vl import
     Qwen2VLModel, VisionRotaryEmbedding` — `VisionRotaryEmbedding` was
     removed or renamed in transformers 5.17.0, so the import fails with
     `ImportError: cannot import name 'VisionRotaryEmbedding'`. This
     triggers before any qwen4_exp code path is reached: the
     `OVModelForCausalLM` import itself crashes.

Both gaps are upstream (optimum-intel's version cap and its stale
qwen2_vl import); nothing in arcint's own code is implicated.

**The harness.** `tools/export_flash_next.py` is landed and runs today. It
checks transformers-support before touching the network: import
`transformers`, record `__version__`, then import
`transformers.models.qwen4_exp`. On failure it raises `AssertionError`
naming the installed transformers version, the missing module
(`transformers.models.qwen4_exp`) and class (`Qwen4ExpForConditionalGeneration`),
and states plainly that this is the known upstream blocker recorded here —
not an arcint defect. The script exits 1 on that path (a red case that can
actually fail, per this repository's measurement-discipline rule). Once
the import succeeds, the same script falls through to the real
optimum-intel export and needs no rewrite — only whatever ordinary
follow-up a first real qwen4_exp export surfaces (a config key
optimum-intel's OpenVINO mapping does not yet know, say), which is exactly
the kind of thing FIX B's delta table above is already tracking.

**The watcher.** `tools/watch_flash_next_export.py` runs the same
transformers-support check (not a full export — no network or GPU needed
to answer "has upstream shipped it yet") and appends one dated line to a
git-ignored log (`tools/flash-next-watch.log`, listed in `.gitignore`) —
"still blocked: <reason>" or "OPEN->CLOSED <version> now carries
qwen4_exp". It always exits 0: unattended, cron-driven, "still blocked" is
the expected outcome and should not alarm anything. A weekly cadence is
suggested in the script's own docstring as a crontab line; the actual
crontab entry (which host, which account, which venv it checks) is
operator infrastructure and belongs in `CLAUDE.local.md`, not here.

**The two options, once upstream is unblocked or if it stalls too long:**

  (a) **Vendor the modeling module** from the HF `transformers` PR or the
      model repo's own `modeling_qwen4_exp.py` (several checkpoints ship a
      `trust_remote_code` module alongside the weights) directly into a
      local shim transformers can import, unblocking export without
      waiting on a transformers release. Risk: an unreviewed, possibly
      still-changing implementation becomes a dependency arcint has to
      track and re-vendor on every upstream revision until it lands
      properly.

  (b) **Wait for upstream.** No extra maintenance burden, but the 0.5.0
      milestone (FIX C's kernel work, FIX D's n-gram offload) stays
      export-blocked for as long as it takes.

**Due:** at design freeze for 0.5.0, if upstream has not landed
`qwen4_exp` by then, the operator picks (a) or (b) — this memo does not
decide it.

**Current status (2026-09-09):** harness exists. The transformers-support
check gate fires and exits 1 under the production venv (transformers 5.5.4,
capped by optimum-intel 2.1.0's `<5.6` pin). Under a force-upgraded venv
(transformers 5.17.0), the gate passes but `OVModelForCausalLM` import
crashes immediately (`ImportError: cannot import name
'VisionRotaryEmbedding'` from `transformers.models.qwen2_vl` -- stale
optimum-intel import, not a qwen4_exp issue). Both blockers are upstream;
the export path is unblocked when optimum-intel releases a version that
both lifts the `transformers<5.6` cap and fixes its own
`modeling_visual_language.py` imports.

Watcher (`tools/watch_flash_next_export.py`, log path
`tools/flash-next-watch.log`, suggested weekly cadence in its docstring)
is not yet added to any crontab — that step is operator-side, tracked in
`CLAUDE.local.md`. Neither of the two options above has been chosen; none
of this repository's other Flash-Next work (FIX B's delta table above,
FIX C's kernel audit, FIX D's n-gram offload) waited on this decision.

**Decision memo (2026-09-10): watcher stays, construction path is an
arcint-original `qwen4_exp` export shim.** The two-link upstream chain
(optimum-intel's `<5.6` transformers cap AND its stale `VisionRotaryEmbedding`
import) is not a gate on our work, only on acceptance against the real
checkpoint. The construction path is the same pattern this repo already
uses for `tools/export_mtp.py` and `tools/export_dflash.py`: read the
checkpoint's `config.json`, walk its `state_dict` directly, build the
`ov::Model` from config geometry and HF weights, write the multi-component
IR layout — bypassing optimum-intel's export pipeline entirely.

**Shim landed (`tools/export_qwen4_exp.py`, 2026-09-10):** the argument
parser (`--checkpoint`, `--out`, `--moe-lowering`, `--rope`, `--dry-run`);
a `translate_config()` that walks the checkpoint's `text_config` and
returns the arcint-internal geometry dict the graph builder keys off
(`n_layer`, `n_embd`, `n_head`, `n_head_kv`, `head_dim`, `n_ff`,
`rms_norm_eps`, `rope_theta`, `max_position_embeddings`,
`tie_word_embeddings`, `layer_types`, `full_attention_interval`, the
GDN linear-attention heads and head-dims and conv kernel, MoE keys
`num_experts` / `moe_topk` / `moe_intermediate_size` /
`num_shared_experts` / `moe_norm_topk`, `mtp_layers`); and a
`write_output_layout()` that (a) passes the checkpoint's own
`config.json`, `chat_template.jinja`, `tokenizer.json` and
`tokenizer_config.json` through verbatim — the arcint loader
(`src/core/artifact.cpp`) reads HF-native keys from `text_config` and
hashes `chat_template.jinja`, so a fabricated stand-in produces an
artifact whose hashes diverge from every existing pin, (b) writes an
`arcint.json` sidecar carrying the derived geometry and the shim's own
knobs (`moe_lowering`, `rope`), and (c) invokes the caller's
`component_writer(out, geo, options)` and asserts every entry in
`REQUIRED_OUTPUTS` is present after it returns — the layout is not
silently half-built. 23 unit tests cover the three surfaces
(`tools/test_export_qwen4_exp.py`), all pass on this repository's
Python 3 stdlib alone (no venv). The backbone graph reconstruction
(`build_backbone_ir()`) refuses at runtime with a named
`NotImplementedError` naming the watcher and the geometry it was
handed; the graph is the watcher-gated increment.

**Acceptance against the real checkpoint awaits the watcher.** Watcher
stays running; nothing about the shim replaces it. Once either upstream
gap clears, `build_backbone_ir()` walks the state_dict layer by layer
and produces `openvino_language_model.xml` tensor by tensor, the same
shape `export_mtp.py` uses for its 1-layer head. None of the argument,
config-passthrough or layout code is affected by that work.

## FIX D — N-gram table host-offload and dequantise-on-gather

### The tensor

`per_layer_token_embd` -- Flash-Next's per-layer n-gram embedding table. Not
attention, not an expert: a lookup table read by `ggml_get_rows` (the
reference implementation's own op for it), one row per token id, 160
elements wide per row. Two figures for its size were logged together in the
recon (HANDOFF-0.5.0.local.md): "51.2 G elements, 97.7 GiB BF16" -- these
are the same tensor rounded two different ways (51.2 G is the headline
count rounded to one decimal; 97.7 GiB divides back to a cleaner element
count, ~52.45 G, and is the more precise of the pair). `src/exec/fit.h`'s
`kFlashNextNgramElements` is derived from the 97.7 GiB figure for exactly
that reason -- see its own comment.

At 51-52 G elements this table alone is larger than either card in scope
(the served family's own weights, by comparison, fit inside a single card).
It is host-resident by design, not as a fallback: nothing in this
repository loads a table this size onto a card, and nothing should.

### The format: 32-element blocks, not K-quant

The shipped GGUF quantises this table Q4_0. Q4_0, Q4_1 and Q8_0 -- the three
precisions this section prices -- all block at **32 elements**, one
constant-width scale (and, for Q4_1, one min) per block, no per-superblock
scale-of-scales packing. This is deliberately different from the K-quant
family (Q4_K/Q5_K/Q6_K, `core/gguf_dequant.cpp`'s existing
`dequantize_row_q4_k`/etc, and the `FullyConnectedKQuant` op,
`exec/kquant_op.h`) that the rest of this codebase's GGUF-native path
already decodes: K-quant blocks at **256** elements, and 160 (this table's
row width) is not a multiple of 256 -- a K-quant block would cross a row
boundary, which GGUF's own layout rule forbids ("a quantization block runs
along dims[0] and never crosses a row boundary", `core/gguf.h`). 160 = 5 x
32 exactly, so every row is a whole number of 32-element blocks and the
existing K-quant decoders and op simply do not apply here; this is why FIX D
needed its own kernel rather than reusing `FullyConnectedKQuant`.

Byte layout (from `core/gguf.h`'s own type table, unchanged by this work):

| format | block elements | bytes/block | per-block layout |
|---|---|---|---|
| Q4_0 | 32 | 18 | f16 `d` (scale) + 16B of 4-bit nibbles, code centered at 8 (`(nibble-8)*d`) |
| Q4_1 | 32 | 20 | f16 `d` (scale) + f16 `m` (min) + 16B of 4-bit nibbles (`nibble*d + m`) |
| Q8_0 | 32 | 34 | f16 `d` (scale) + 32B signed int8 (`qs[l]*d`) |

### What landed: the gather-with-dequant kernel (harness, red-first)

`src/exec/ngram_gather.h`:

- `gather_dequant_scalar` -- the reference path, a thin wrapper over
  `lgc::gguf::dequantize_row` (per row, per gathered index). This function
  did not cover Q4_0/Q4_1 before this work -- only Q8_0 and the four
  K-quant types plus the float pass-throughs -- so FIX D's first change was
  adding `dequantize_row_q4_0`/`dequantize_row_q4_1` to
  `core/gguf_dequant.cpp` (transcribed from ggml-quants.c's own
  `dequantize_row_q4_0`/`q4_1`, the same way every decoder already in that
  file is transcribed and commented). This keeps one reference dequantizer
  per type, reused rather than re-derived, matching that file's own stated
  purpose ("these exist to check a fixture and, later, real tensors").
- `gather_dequant` -- the fast path: AVX2 block kernels
  (`dequant_block_q4_0_avx2`, `_q4_1_avx2`, `_q8_0_avx2`) dispatched per
  gathered row, falling back to `gather_dequant_scalar` whenever AVX2 is
  unavailable at runtime, the type is outside this kernel's three formats,
  or the row width is not a whole multiple of 32.

**AVX2 only, gated by cpuid at runtime, never assumed at compile time.** The
dev container's host is a Zen 3 part: AVX2 yes, AVX-512 no -- AVX-512 there
is a SIGILL, not a graceful degrade. The three AVX2 block-dequant functions
are marked `__attribute__((target("avx2")))` (GCC/Clang function
multiversioning) rather than relying on a blanket `-mavx2` build flag, so
this header compiles into a binary built with no `-march`/`-mavx2` at all;
`gather_dequant` calls into them only after `cpu_has_avx2()`
(`__builtin_cpu_supports("avx2")`, a runtime cpuid check) returns true. No
`fma` target string appears anywhere in this file: every block dequant does
an explicit multiply then an explicit add (two separate instructions, two
separate roundings), matching the scalar reference's own separate
operations term for term -- an FMA'd multiply-add rounds once where a
multiply-then-add rounds twice, and the two are not always bit-identical,
which would have broken the byte-exact check below for no reason connected
to an actual kernel bug.

**The red-first test**, `tests/test_ngram_gather.cpp`: a synthetic
1000-row x 160-column table, quantised into all three formats from known,
per-dimension-varying float input (not a constant fill -- see the file's own
comment on why: a constant-per-dimension fill hid a real defect elsewhere in
this codebase's history, `HANDOFF-0.4.7.local.md`'s BY_TOKEN NaN record) via
test-local quantizers transcribed from ggml-quants.c's
`quantize_row_q4_0`/`q4_1`/`q8_0`. For a sample of gathered row indices
(including both table edges), the test asserts, `memcmp`-exact:

1. `gather_dequant`'s AVX2-or-scalar dispatch output against
   `gather_dequant_scalar`'s forced-scalar output, for the identical input
   -- the kernel-correctness case this section exists to make pass;
2. `gather_dequant_scalar`'s own per-row output against calling
   `lgc::gguf::dequantize_row` directly on the same row bytes -- an
   independent check that does not route through either of the two
   functions the first check compares, so a bug shared by both cannot hide
   from it;
3. a loose (1.0f) tolerance check of the dequantized values against the
   known source floats, catching a wrong-scale or wrong-sign defect the
   byte-exact checks (which only prove internal *consistency*, not
   *correctness* against ground truth) would not.

On a host without AVX2, the AVX2-specific cases skip by name
(`SKIP_UNLESS(ngram::cpu_has_avx2(), ...)`); a fourth case
(`ngram_gather_scalar_path_matches_dequantize_row_on_any_host`) runs
unconditionally so the suite is not entirely skip-gated on such a host (this
repository's own ctest invocation runs `--max-skips 0`, so an all-skip file
would itself be a red flag caught by CI, not a silent pass).

All cases pass on the dev container (AVX2 present, confirmed via
`cpu_has_avx2()` returning true in the same test run). No microbench
(token-gathers/s) has been taken yet -- see "What's blocked", below.

**FIX D verification (2026-09-10, the dev host, build from source at HEAD `67ef4d9`):**
All 9 TEST() cases from `tests/test_ngram_gather.cpp` (commit `3cc08f9`)
executed for the first time on the dev host. Binary built from source
(`cmake -B build -DCMAKE_BUILD_TYPE=Release`, GCC 14.2.0), checksums of
all five source files (`test_ngram_gather.cpp`, `ngram_gather.h`,
`gguf_dequant.cpp`, `gguf_dequant.h`, `fit.h`) verified byte-identical
between the git HEAD and the dev host's non-git copy. Result:

    host: the dev container (hostname verified)
    date: 2026-09-09T23:25:14+0000
    binary: build/arcint-test (md5 415fabb648eefe10093977244d1dfb78)
    filter: ngram → 9 cases run, 0 failed, 0 skipped

Per-case (each run individually, same binary):

    ngram_gather_q4_0_avx2_matches_scalar_reference_byte_exact     PASS
    ngram_gather_q4_1_avx2_matches_scalar_reference_byte_exact     PASS
    ngram_gather_q8_0_avx2_matches_scalar_reference_byte_exact     PASS
    ngram_gather_scalar_path_matches_dequantize_row_on_any_host    PASS
    ngram_table_bytes_matches_the_recon_quoted_figures              PASS
    ngram_table_bytes_unknown_type_prices_as_nothing                PASS
    host_ram_fit_refuses_q8_0_ngram_table_on_the_48gib_dev_container   PASS
    host_ram_fit_refuses_q4_1_ngram_table_plus_an_oversized_expert_pool PASS
    host_ram_fit_admits_q4_1_ngram_table_on_a_128gib_unit_host         PASS

No skips (AVX2 present on the Zen 3 host). **FIX D's red-first kernel
tests are now verified by execution, not only by code review.**

### The memory budget: n-gram table vs. expert pool vs. host RAM

`src/exec/fit.h` adds the host-RAM side of the fit arithmetic (everything
above it in that file prices a *card's* memory; this is the first term that
prices the *host's*):

- `ngram_table_bytes(ggml_type, n_elements)` -- `ceil(n_elements /
  block_size) * bytes_per_block`, reading `block_size`/`type_size` from
  `lgc::gguf::type_info` (the same table `core/gguf.cpp` already carries;
  no new numbers). Returns 0 for a type the reader does not know, matching
  this file's convention for "nothing to price" rather than throwing.
- `kFlashNextNgramElements` -- Flash-Next's own element count, derived from
  the recon's 97.7 GiB BF16 figure (see above).
- `host_ram_fit_must_refuse(ngram_bytes, expert_pool_bytes,
  other_resident_bytes, host_ram_bytes, margin_bytes)` -- `true` when the
  sum would overrun the host's RAM budget. Pure arithmetic, no throw (this
  file stays testable without a host throughout); the load-time refusal
  *by name*, with an actual host's numbers, is backend_ov.cpp's job at the
  call site, not landed as part of this pass (no artifact exists yet to
  load).
- `HostRamFit` / `host_ram_fit(...)` -- the itemized version of the same
  check (required bytes, signed headroom/shortfall, the `refuse` bool),
  for a fit-doc line or a load-time log to name every term.

**The fit table**, `ngram_table_bytes` evaluated at `kFlashNextNgramElements`
(~52.45 G elements) for each candidate precision, against the two hosts in
scope:

| precision | table size | dev container (48 GiB RAM) | fits alongside FIX E's expert pool? |
|---|---|---|---|
| Q4_0 (shipped) | 27.48 GiB (~29.5 GB decimal) | fits alone | ~20.5 GiB left for everything else |
| Q4_1 | 30.53 GiB | fits alone | ~17.5 GiB left -- **the same RAM FIX E's host-resident expert pool wants** (HANDOFF-0.5.0.local.md's own framing) |
| Q8_0 | 51.90 GiB | **does not fit** | refused outright, `host_ram_fit_must_refuse` is unconditionally true regardless of `expert_pool_bytes` |

A unit host with materially more RAM (this section's own test fixture uses
128 GiB as an illustrative, not measured, second row) has enough headroom
for Q4_1 plus a generous expert-pool allocation with margin to spare --
`tests/test_ngram_gather.cpp`'s
`host_ram_fit_admits_q4_1_ngram_table_on_a_128gib_unit_host` case pins this
so the budget check is not read as "refuses everything unconditionally."
No real unit-host RAM figure was measured for this table; the 128 GiB row
is a fixture bound, not a claim about actual unit hardware.

vLLM's own `VLLM_PLE_CPU_OFFLOAD=1` (auto-enabled on their reference
80 GB-class cards, gated on host RAM >= 51 GB) is the closest published
comparison point: their threshold sits almost exactly at this table's Q8_0
size, which is consistent with Q8_0 being the precision they expect to need
that much host RAM for in the first place.

### What's blocked, and why

Everything gated on an actual Flash-Next artifact is not landed, for the
same reason FIX A/B/C all name: no export exists yet
(`transformers.qwen4_exp` is absent from every pinned transformers version
checked, FIX A's own blocker). Specifically:

1. **Microbench numbers (token-gathers/s against the host's DRAM
   ceiling)** -- needs the real table (or a full-size synthetic one built to
   the real element count) resident in host RAM on the dev host, and a
   timed loop of `gather_dequant` calls at realistic batch/index
   distributions. The 1000-row fixture above proves correctness, not
   throughput at scale; a throughput number taken on it would not be a
   measurement of anything the real table's access pattern implies (a
   1000-row table fits entirely in L2/L3 cache, defeating the DRAM-ceiling
   question this item exists to answer).
2. **Integration**: the gather path in the graph (`ggml_get_rows` analogue
   at the right op level, wired to the budget check above) is not built --
   there is no Flash-Next IR to wire it into, and no served checkpoint in
   this repository's existing families uses a table this shape, so there is
   nothing to integrate against today. FIX D's own item 3 stays open until
   FIX A's export blocker clears.
3. **The refusal case as a load-time behavior** (not just the pure
   arithmetic's boolean, which is covered) -- `backend_ov.cpp` does not yet
   call `host_ram_fit_must_refuse` anywhere, because there is no load path
   that reads a table this shape yet.
4. **FIX E crossover**: HANDOFF-0.5.0.local.md is explicit that "FIX D and
   FIX E contend for one DRAM budget, and the crossover calibration must be
   measured with both paths live, not FIX E alone" -- this section's fit
   table treats the expert pool's RAM draw as a given input
   (`expert_pool_bytes`), not something FIX D calibrates; the live,
   both-paths-active DRAM-bandwidth measurement is FIX E's own "Done when"
   item 1, not duplicated here.

## FIX E — Expert-pool CPU executor design

### Scope and status

Design only, per the handoff's own split: what follows is landable without
a Flash-Next artifact (calibration methodology, the worker-pool
architecture, the overlap strategy, the DRAM-contention accounting, the
FreeToken comparison protocol). The kernel work itself — AVX2 dequant
routines for Q4_K/Q5_K/Q6_K, the pinned pool's thread-affinity plumbing,
the gate-weighted partial-output combine — is explicitly out of scope for
this pass (see "What's landable now" below) and follows once FIX A's
export blocker clears.

The existing host CPU tier (patches/0011-0012, extended by 0017-0019) is
the starting point, not a green field: `MOE_CPU_TIER`, the AVX2/scalar
grouped-int4 kernel, the mmap weight accessor, the M14 perf counters, the
static-partition residency fix (0018/0019) all exist today and are
measured on the served 35B MoE (`docs/design-qwen-flash-next.md` FIX C
above cites 15.0/15.5 t/s at ratio 50 with the host tier on, against
10.4/10.6 without it). FIX E is the design for what changes on top of
that tier to make it Flash-Next-shaped: 512 experts at 6% activation
instead of the served model's smaller pool, K-quant GGUF dequant instead
of the grouped-int4 layout the existing kernel targets, and a bandwidth
calibration step the existing tier does not do at all — 0011-0012 pick
their CPU/device split by static residency partition (0018's F2), not by
a measured transfer-vs-compute crossover. FreeToken's contribution is
exactly that crossover heuristic; this section is where arcint adopts the
idea, re-derived from the paper's description rather than the reference
CUDA implementation (Apache-2.0 fork-tax avoidance, per the handoff).

### Why a new dequant path at all

The existing patches/0011 kernel operates on arcint's own grouped-int4
layout — the layout the served model's export produces. Flash-Next, if
and when it ships as GGUF (the export blocker in FIX A above is silent on
container format; a GGUF community quantisation is the likelier first
artifact to exist, independent of the optimum-intel path), carries
K-quant blocks (Q4_K/Q5_K/Q6_K: a two-level structure, per-superblock
scale/min in a higher precision than the per-sub-block quantised
weights, packed at 256-weight superblock granularity). That is a
different bit layout from arcint's grouped-int4 tables, not a
reparametrisation of the same one — the existing kernel's inner loop
cannot read it. This design proposes new AVX2 dequant kernels, one per
K-quant type, sharing the existing tier's thread pool, mmap accessor and
perf-counter scaffolding rather than replacing any of it.

### Bandwidth calibration methodology

FreeToken's `ft bench bw` heuristic (paper §4, "Bandwidth-Adaptive
Execution") measures, at load time, the achievable PCIe host-to-device
transfer rate (`B_P`) and the achievable CPU expert-execution rate
expressed as an equivalent bandwidth (`B_H`), then routes each cache-miss
token to whichever path — stream the weight over PCIe and execute on
device, or execute in place on the CPU — is cheaper for that token's
expert size at that moment. arcint needs the same two numbers, measured
on arcint's own hardware and kernels rather than assumed from FreeToken's
published figures (per this repository's measurement-discipline rule:
every number is named against the card, the host and the configuration
it was taken on).

What to measure, concretely:

- **`B_P` — PCIe stream rate.** Time a repeated non-blocking upload of a
  known expert-slot-sized buffer (per-expert bytes from `fit.h`'s
  `expert_slot_bytes` arithmetic) through the existing staging-ring path
  (patches/0006), amortised the same way 0006 already amortises real
  traffic — batched, not per-tensor. Measured, not read off a spec sheet:
  the design doc's own FIX C section above already treats PCIe 4.0 x16's
  "~28 GB/s" as a nameplate figure to be checked against, not trusted.
- **`B_H` — CPU-execution equivalent bandwidth.** Time the new AVX2
  K-quant dequant-and-GEMV kernel over a representative expert (same
  slot size, same activation count) on the pinned worker pool at steady
  state (warm caches, no first-request JIT cost — see
  `feedback-first-request-compiles-kernels` in memory: a fresh process
  pays a one-time kernel-compile/warmup cost that must not leak into a
  steady-state rate). Expressed as bytes-of-weight-processed per second
  so it is directly comparable to `B_P`.
- **DRAM achieved bandwidth**, independently of both of the above: a
  saturating multi-thread streaming-read microbenchmark (e.g. a
  many-thread sequential-read sweep sized well past the 32 MiB L3, so it
  measures DRAM and not cache) run twice — once with FIX D's n-gram
  streaming path idle, once with it active — because `B_H` above is
  itself DRAM-bound (the dequant kernel is a bandwidth-bound gather over
  mostly-cold weight pages) and shares the same DDR4 channel FIX D's
  table lookups stream through. This is the number the crossover
  calculation actually needs to stay honest under contention — see
  "DRAM contention with FIX D" below.

**DRAM read bandwidth measurement (2026-09-10, dev host = 5700X 8-core,
DDR4, 8 of 16 logical cores online in the container):**

| Workload     | Threads | Buffer  | MB/s aggregate | MB/s per thread |
|--------------|---------|---------|---------------:|----------------:|
| memcpy       |       1 | 256 MiB |         20,025 |          20,025 |
| memcpy       |       6 | 256 MiB |         35,267 |           5,878 |
| memcpy       |       1 | 1 GiB   |         19,363 |          19,363 |
| memcpy       |       6 | 1 GiB   |         37,435 |           6,239 |
| AVX2 dequant |       1 | 256 MiB |          9,405 |           9,405 |
| AVX2 dequant |       6 | 256 MiB |         41,699 |           6,950 |
| AVX2 dequant |       1 | 1 GiB   |          9,394 |           9,394 |
| AVX2 dequant |       6 | 1 GiB   |         46,423 |           7,737 |

Method: `gcc -O2 -mavx2 -mf16c -mfma`, pinned threads (cores
1,3,4,6,8,10), buffers allocated with `aligned_alloc(64)`, 2 warmup + 8
measured iterations (best of 8). Dequant kernel: Q4_0-style (18-byte
blocks, f16 scale + 16 packed-nibble bytes → AVX2 widen + FMA,
accumulate to registers, no write-back — pure source-read bandwidth).
memcpy is read+write (measures DDR4 channel saturation including the
write-back traffic the dequant path avoids).

Key numbers for FIX E's crossover: single-thread dequant reads the
source buffer at **9.4 GB/s** (compute-bound — below the ~20 GB/s
single-thread memcpy, so the ALU is the bottleneck, not DRAM). At 6
threads the aggregate dequant throughput is **42–46 GB/s** (exceeds
memcpy's 35–37 GB/s because dequant has no write-back traffic). A
256 MiB expert slab takes ~27 ms to dequant on 1 core, ~6 ms on 6.

**Consequence: the host feed at ~45 GB/s is ~11% of the A770's measured
414–418 GB/s demand; the q\* policy must therefore be card-resident-first.**
Every expert that fits on the card must stay there — CPU execution is not
a competitive alternative to card-resident compute at this bandwidth ratio,
only to PCIe streaming of cold experts.

The crossover point `q*` FreeToken derives (paper's own term, `q* ~=
m·B_P/B_H` where `m` is the expert's byte size) becomes, once both rates
are measured on arcint's own hardware and kernel: below `q*` tokens
routed to a given cold expert in a step, stream-and-execute-on-device is
cheaper; at or above it, execute-in-place on the CPU pool is cheaper.
This is a per-load, per-hardware constant (a function of the two
measured rates and the expert byte size), not a per-token decision
computed at serving time from scratch — it is computed once at load and
consulted, the same shape the existing static-partition design (0018)
already uses for its own load-time decision.

### Worker-pool architecture

A persistent pool of `N` OS threads, created once at model load (not
per-request, not per-layer) and pinned one-to-one to physical cores via
the existing `core/affinity.h` (`pin_current_thread`, already used by
`--pin-dispatch` — see `src/exec/backend_ov.cpp` around the dispatch-pin
code) — pinned to physical cores specifically, not SMT siblings: two
threads sharing one physical core's execution units contend for the same
AVX2 ALU and L1 rather than adding throughput, and the dev host's 8
physical / 16 SMT-thread topology gives exactly 8 independent execution
units to place work on, not 16. `moe_cpu_tier_threads_` (already a
config field, `MOE_CPU_TIER_THREADS`) is the existing knob this reuses;
its default ("auto") should resolve to the physical core count, not the
logical one, on hosts where that distinction matters — the dev host is
one such host and should not silently oversubscribe to 16.

Each worker owns:

- an mmap window into the GGUF weight file (extending the existing
  0011 mmap accessor to a K-quant-block-aware stride instead of the
  grouped-int4 stride it uses today);
- a per-K-quant-type AVX2 dequant-and-accumulate kernel (Q4_K, Q5_K,
  Q6_K each need their own inner loop — the superblock scale/min
  layout differs per type, not just the sub-block bit width) that
  dequantises a block on the fly and accumulates directly into the
  gate-weighted partial-output buffer, rather than materialising a
  dequantised f32/f16 weight tensor in DRAM first. Dequant-on-the-fly is
  the point: materialising first would double the DRAM traffic this
  design is trying to keep under the calibrated `B_H`, once for the
  dequant write and again for the GEMV read.
- a lock-free work queue (or the existing tier's queue primitive if one
  already exists in 0011-0012's thread-pool code — re-use over
  reinvention per this repository's own default) fed `(layer, expert,
  token-batch)` triples by the dispatch path.

### Gate-weighted partial outputs

Each active expert's contribution to a token's MoE output is scaled by
that token's routing gate weight before combination (standard MoE
combine, unchanged by where the expert executes). What FIX E adds is that
this combine has to be identical in shape whether the expert ran on
device or on the CPU pool: the CPU kernel's accumulation buffer layout
must match what `mlp_reduce` (patches/0012's join point) already expects
from a host excursion, so a token whose 10 routed experts split
CPU/device combines through the same reduce path already wired by 0012,
with the K-quant kernel and calibrated `q*` split as the only new inputs
to that decision — not a new combine path. This reuses 0012's join
rather than adding a second one, which matters directly for FIX D's
determinism concern below: two independent combine paths would be two
independent places for a floating-point-order divergence to hide.

### Overlap strategy

The served model's existing host-tier overlap (0012: "how the host
excursion overlaps the GPU work and joins before `mlp_reduce`") is the
template — GPU-resident attention and GPU-resident experts for a layer
proceed on the device queue while the CPU pool works its own assigned
experts for the same layer concurrently, joining only at the reduce.
Flash-Next's much higher offload ratio (high 80s-low 90s percent, per
FIX C above) means the CPU pool is doing much more of the per-layer work
than the served model's ratio-50/75 measurements exercised, so the
overlap window that mattered little at ratio 50 (a small CPU tail
finishing after a much larger GPU-resident majority) is the dominant cost
at ratio ~90 (a small GPU-resident minority finishing well before a much
larger CPU tail). The design implication: at Flash-Next's ratio, GPU
attention finishing early and idling while the CPU pool is still the
long pole is the expected steady state, not an edge case — the
per-layer, per-step split has to be sized so the CPU pool's finish time
tracks the GPU's attention-plus-resident-expert finish time as closely as
the calibrated `q*` split allows, rather than treating GPU idle time at
this ratio as a bug to chase.

### DRAM contention with FIX D

FIX D's n-gram table offload and FIX E's CPU expert pool are both,
fundamentally, DRAM-bandwidth-bound gather workloads over the same
dual-channel DDR4 bus on the same host, active in the same decode step
(the n-gram lookup happens at layer 2's PLE embedding per the config
keys FIX B's delta table already reads — `ple_layer_ids`,
`ple_embed_dim` — while MoE layers, including any of the 48 that route
through the CPU pool, are spread across the rest of the stack). Neither
path saturates the bus alone by construction — the calibration
methodology above measures `B_H` and DRAM bandwidth twice, once with
FIX D idle and once active, specifically because the two are not
independent: a `B_H` measured with FIX D's streaming path off overstates
what's actually available once both are live in the same served step.
This design does not attempt to prescribe a fixed split of DRAM budget
between the two features here — that's an implementation-time tuning
question — but it names the two measured rates (`B_H`-with-ngram-idle and
`B_H`-with-ngram-active) as the pair the calibration profile below must
carry, so the crossover `q*` used at serving time is the contended one,
not the idealized one.

### Calibration profile format

One profile per host, recorded (not merely spec-sheet-derived) at
calibration time, carrying at minimum:

- host identity as an opaque calibration-profile label (never a real
  hostname per this repository's public-repo rule — an operator-chosen
  tag, with the real host recorded only in `CLAUDE.local.md` if needed
  for reproduction);
- CPU ISA level actually used (`AVX2`, since the dev host has no
  AVX-512) and physical-core / SMT-thread counts separately;
- measured DRAM bandwidth, n-gram streaming path OFF;
- measured DRAM bandwidth, n-gram streaming path ON (FIX D's contention
  case);
- measured `B_P` (PCIe stream rate) on the card in scope, named with the
  card;
- measured `B_H` per K-quant type (Q4_K/Q5_K/Q6_K each dequant at a
  different rate — a single averaged number would hide which type
  dominates a given expert mix), both with and without FIX D active;
- the derived `q*` crossover per K-quant type, both contention cases;
- date and the `marfrit-openvino` patch level the measurement was taken
  against, per this repository's changelog convention.

**Done-when item (1)** requires this profile to exist for both unit
hosts and the dev container, with the dev-host row explicit on the
figures above (AVX2, 8 physical cores, DRAM measured both ways). Not
done in this pass — this section defines the format and methodology; the
harness that runs and records it is implementation work, listed under
"What's landable now" below.

### FreeToken comparison methodology

Per `feedback-no-baseline-claim-without-survey` and the
`reference-freetoken-edge-moe` memory note, no arcint claim about serving
Flash-Next stands alone — it is stated against FreeToken's own published
row for the closest comparable case. FreeToken's paper reports Qwen3.6-
35B-A3B (the served model's own family, not Flash-Next — FreeToken's
own model list does not cover Flash-Next per the memory note) at 77-83
tok/s on an RTX 5090 with 8 GB-96 GB card targets; the appropriate
comparison card class for the dev host's setup is the smaller end of
that range, not the 5090 row, since neither card in scope here is a
5090-class device.

The claim row required by done-when item (4) must state, side by side:

| | FreeToken (published) | arcint (measured) |
|---|---|---|
| served model | (their row's model) | Flash-Next, once exported |
| card | (their row's card, named) | the card in scope, named |
| precision | (their row's, if stated) | the KV/weight precision arcint served at |
| offload ratio / resident fraction | (if stated) | measured, per FIX C's arithmetic |
| decode rate | 77-83 tok/s (35B-A3B, RTX 5090) | measured |

Two honesty constraints apply directly: FreeToken's 35B-A3B row is the
served model's family, not Flash-Next, so an arcint Flash-Next number
compared against it is a cross-model comparison and must say so plainly
rather than implying an apples-to-apples read; and if FreeToken has no
row at all comparable in card class or model to what arcint actually
measures, the claim row must say that gap out loud rather than pick the
nearest FreeToken number silently. Per DESIGN §7.0.1, a mechanism
narrated without a matching measurement gets retracted rather than
edited around — this applies equally to a comparison row assembled from
a paper's headline figure without checking whether the comparison is
actually apples-to-apples.

### What's landable now vs. what needs the export

**Landable now, no Flash-Next artifact required:**

- This design section itself.
- The calibration harness: the DRAM-bandwidth microbenchmark (both n-gram
  states), the `B_P` PCIe-stream measurement (against any expert-sized
  buffer, not specifically Flash-Next's — the existing served-model
  expert size already exercises the same staging-ring path), and the
  calibration-profile recording format above. None of this depends on
  Flash-Next's config or weights.
- The K-quant AVX2 dequant kernels' correctness (Q4_K/Q5_K/Q6_K
  bit-layout decode against known-good reference vectors) — a GGUF
  K-quant block is a fixed, documented bit layout independent of which
  model's weights fill it, so a synthetic or third-party GGUF file
  exercises the kernel today.
- The microbench required by done-when item (2) — "CPU-executor token
  rate vs PCIe-stream rate, crossover measured" — against the **served**
  35B MoE, using its existing expert sizes and either its existing
  grouped-int4 kernel (0011) or a synthetic K-quant re-encode of the same
  weights for the new kernel path. This is explicitly what done-when
  item (2) asks for ("microbench on the served 35B MoE's experts"), so it
  needs no Flash-Next artifact at all and should be run before FIX A
  unblocks, not after.

**Needs the export (blocked on FIX A):**

- Done-when item (3), the served-family hybrid run against the
  equivalence suite's byte-exactness gates — "served-family" reads as
  the served 35B MoE here too per the same logic as item (2), so this
  may also be landable pre-export; if it in fact requires Flash-Next
  specifically, it is blocked the same way FIX C's RED-C-01/03/04/05
  are, on FIX A.
- Wiring the calibrated `q*` split into Flash-Next's actual 512-expert,
  10+1-active routing at serving time — needs a compiled Flash-Next
  graph to route against, same blocker as everywhere else in this
  document that depends on a real artifact.
- The final FreeToken claim row's arcint-side numbers for Flash-Next
  specifically (the table's right column) — the methodology and left
  column are landable now; the right column's actual Flash-Next figures
  are not.

## §6 — MoE GGUF is not served: exact refusal path

The FIX B delta table's parenthetical ("MoE GGUF not served, §6") and
`model_requirements.md` §6 both name this gap. This section records the
exact loader behaviour, verified from code, not documentation:

**The GGUF serving path refuses a MoE file at the architecture gate.**
`gguf_geometry()` (`src/core/gguf_map.cpp:41-42`) checks
`general.architecture` from the file's metadata against the only accepted
value, `"qwen35"` (stage 1, dense):

    if (arch != "qwen35")
        throw std::runtime_error(
            "gguf: architecture '" + arch + "' is not served "
            "(stage 1 serves qwen35)");

A MoE file from the served family declares `general.architecture =
"qwen35moe"`. The check is a hard `std::runtime_error` throw before any
tensor matching or weight reading begins — not a silent fallback.

A second refusal would fire if the architecture gate were bypassed:
`gguf_apply_to_template()` (`src/exec/gguf_graph.cpp:318`) throws for any
IR constant whose module name has no entry in `kLayerModules`
(`gguf_map.cpp:22-35`). That table maps only the dense MLP projections
(`ffn_gate`, `ffn_up`, `ffn_down`); MoE expert tensors (`ffn_gate_exps`,
`ffn_up_exps`, `ffn_down_exps`, `ffn_gate_inp`, shared-expert variants)
are absent, producing `gguf_module_map() == nullptr` and the throw:

    if (!found) throw std::runtime_error(
        "gguf: no tensor map for IR constant " + name);

**Consequence for FIX 8.2**: the artifact
`Qwen3.6-27B-A3B-Coder-Q4_K_S.gguf` (14.03 GiB, MoE, `qwen35moe`) is
refused at `gguf_geometry()`. The dense `Qwen3.8-27B-UD-Q3_K_XL.gguf`
on the dev host is also ruled out: Q3_K is in `model_requirements.md` §6's
own "not yet served" list (stages 2/3).

The candidate served case was the on-disk
`Qwen3.5-2B-Q4_K_M.gguf` (1.19 GiB, dense, `general.architecture =
"qwen35"`, `qwen35.full_attention_interval = 4`, Q4_K_M is a served type).
Verified on the dev host 2026-09-09T19:42:34+00:00: GGUF magic ok, 24 blocks,
head_count 8, head_count_kv 2, key_length 256, value_length 256.

**Window B measurement (2026-09-09, GPU.1 = A770 16 GiB):**
The template-IR geometry acceptance step (`gguf_apply_to_template`) **refuses
the 2B** — no matching 2B template IR exists on disk; the only available IR
is the 27B artifact:

    --gguf Qwen3.5-2B-Q4_K_M.gguf is not this artifact's architecture:
    layers: file 24, artifact 64; hidden size: file 2048, artifact 5120;
    attention heads: file 8, artifact 24; kv heads: file 2, artifact 4;
    GDN value heads: file 16, artifact 48

The 27B Q4_K_M (15.93 GiB) passes both the architecture gate and geometry
matching against the 27B IR, repacks 497 projections (288 repacked, 209
native, 38 s on 8 workers), but fails compilation with
`CL_OUT_OF_RESOURCES` — the file alone exceeds the card's usable VRAM
before KV pool or scratch allocation.

**FIX 8.2 consequence**: GGUF serving on the 16 GiB card requires either
(a) exporting a 2B template IR (a new artifact, not on disk today), or
(b) a Q4_K_S requant of the 27B (~14.5 GiB target) — which is marginal
at best given VRAM overhead beyond raw weights. Neither path is zero-prep;
the "remaining assumption" from the metadata check is answered: refusal.

**Path (a) attempted (2026-09-10):** A 2B template IR was exported via
optimum-intel 2.1.0 (`OVModelForVisualCausalLM`, transformers 5.2.0) to
the multi-component layout arcint expects (language_model, text_embeddings,
tokenizer, detokenizer, chat_template). The 2B was added to the model
registry (`qwen35-2b-ov`, `n_layer = 24`, `n_embd = 2048`,
`full_attention_interval = 4`). The GGUF file loaded — geometry matched,
the process compiled and served on GPU.1 (1.76 GiB device-resident, 9.1 s
compile). However: **0 projections repacked** (0 repacked, 0 native rows).
The GGUF contributed only the embedding rows (Q6_K, 248320 × 2048,
397 MiB) and 79 norm comparisons; all projection weights served from the
template's int8_asym constants.

Root cause: `gguf_apply_to_template` matches weight constants by the
`_openvino_orig_weight` suffix, which the AWQ export path produces. The 2B
IR was exported with optimum-intel's default int8_asym per-channel
quantisation, which does not create these marker constants. The 27B's 497
repacked projections work because that artifact was AWQ-quantised. The 2B
needs an AWQ export (or the export must otherwise preserve the original
weight constants with the marker suffix) for GGUF projection replacement
to function.

**Marker forensics (2026-09-10):** The 27B artifact's `openvino_config.json`
records `quant_method: "awq"`, `bits: 4`, `group_size: 64`, `sym: false`,
`ratio: 1.0`, `optimum_version: "2.3.0"`, `transformers_version: "5.2.0"`,
`dataset: "fleetcode"`, `num_samples: 32`. The AWQ weight-compression pass
(implemented by NNCF, invoked through optimum-intel's
`OVWeightQuantizationConfig`) inserts FakeQuantize nodes around each
projection constant; the original weight gets the `_openvino_orig_weight`
suffix, and the FQ subgraph carries `/scale`, `/zero_point`,
`/fq_weights_1`. The 27B has 2,485 such constants = 497 projections × 5
FQ components each. The 2B's bare `OVModelForVisualCausalLM.from_pretrained
(export=True)` (no `quantization_config`) produced int8_asym constants
folded directly, with no FakeQuantize subgraph and no marker suffix.

**2B AWQ re-export attempts (2026-09-10):** Three paths tried, all blocked:

1. *Python API, CausalLM path* (`OVModelForCausalLM` +
   `OVWeightQuantizationConfig(quant_method="awq", bits=4, dataset="wikitext2")`):
   export succeeds, produces int4-compressed weights, but 0
   `_openvino_orig_weight` constants. The CausalLM path folds compressed
   weights directly without FakeQuantize subgraphs.

2. *CLI, VL path with local corpus* (`optimum-cli export openvino --task
   image-text-to-text --awq --dataset ~/fleetcode`): rejected by
   `OVWeightQuantizationConfig.post_init()` (optimum-intel 2.0.0+) which
   validates dataset names against a hardcoded allowlist — visual LLMs
   accept only `{'contextual'}`, LLMs accept `{'c4', 'c4-new', 'auto',
   'gsm8k', 'wikitext2'}`. The 27B's `"dataset": "fleetcode"` predates
   this validation.

3. *CLI, VL path with predefined dataset* (`--dataset contextual`):
   `contextual` is deprecated and falls back to `textvqa`, which hits a
   `video_processor_class` NoneType error in transformers 5.2.0 (the
   Qwen3.5 processor has no video processor class attribute).

**Root cause:** the `_openvino_orig_weight` naming is produced only by the
VL export pipeline's FakeQuantize insertion, but that pipeline's calibration
path is broken for Qwen3.5 on the current toolchain. The CausalLM path
compresses correctly but uses different constant naming that
`gguf_apply_to_template` does not match. Fix options: (a) patch
optimum-intel to accept local datasets for VL models, (b) backport to the
optimum-intel version used for the 27B, or (c) extend `gguf_map.cpp` to
match the CausalLM path's constant names.

**Export-tooling fixes (2026-09-10, `tools/export_2b_awq.py`):** all three
fronts addressed:

- **Front 2 (dataset allowlist):** `bypass_dataset_validation()` sets
  `config.dataset` after `post_init()` has run. Tested: the bug
  (custom string rejected) and the fix (arbitrary string/None accepted)
  both verified on the dev host's venv (optimum-intel 2.0.0). 3/3 tests
  pass.

- **Front 3 (VL calibration bug):** `make_text_calibration_data()` builds
  calibration samples from `AutoTokenizer` alone, never instantiating
  `AutoProcessor` or touching `video_processor_class`. Deterministic,
  in-vocab-range, correct tensor shape. 3/3 tests pass.

- **Front 1 (CausalLM naming):** moot IF the VL path produces markers
  once fronts 2+3 are fixed. The export script uses `OVQuantizer.quantize()`
  with explicit `calibration_dataset` and AWQ config; whether this produces
  `_openvino_orig_weight` FakeQuantize subgraphs (as the 27B's VL export
  did) or folds weights directly (as the CausalLM path did) is **not yet
  verified by a real export run**. `count_orig_weight_markers()` checks
  the result and warns if markers are 0. 2/2 utility tests pass.

**Remaining gap:** the full export (`python3 tools/export_2b_awq.py`)
has not been run. It needs the Qwen3.5-2B checkpoint downloadable from HF
and ~10-30 min CPU time on the dev host; no GPU window. If the export
produces 0 markers, the gap narrows to front 1 alone: extend
`gguf_graph.cpp` to match CausalLM-style constant names (option (c) above).

All 8 tests: `python3 tools/test_export_2b_awq.py` on the dev host, 0.015 s,
2026-09-09T23:xx+0000, optimum-intel 2.0.0, transformers 5.0.0.

**Optional follow-on stress case** (if path (b) is pursued): a Q4_K_S
requant of the on-disk `Qwen3.8-27B-Q4_K_M` (17.1 GB → target ≤ 14.5 GiB)
via `llama-quantize` (`--allow-requantize`, built from llama.cpp HEAD
2026-09-09) exercises real served-weight scale (414–418
GB/s regime) on the 16 GiB card. Run off-peak with `--threads 6`, never
the closing artifact, never mid-day on the serving container.

**Full-export attempt (2026-09-10).** `tools/export_2b_awq.py` was run
against the checkpoint on the newer stack that produced the 27B AWQ
(optimum-intel 2.1.0 + transformers 5.2.0). Phase 1 (plain VL export)
completes. Phase 2 (OVQuantizer AWQ) dies with
`AutoProcessor.from_pretrained(config.processor)` reaching a 404 on
`None`: the VL calibration path still instantiates the processor even
when the caller supplies a pre-tokenized `list[dict]` as
`calibration_dataset`, because `OVQuantizer._prepare_visual_causal_lm_calibration_data`
routes by model class (`OVModelForVisualCausalLM`), not dataset shape.
Front 3 as written is insufficient: **AWQ via optimum-intel 2.1.0's
Python API is not reachable for `qwen3_5` on this toolchain without
patching optimum-intel.** Path (a) from the fix options above stays
open; path (c) is landed below.

**Plain (non-AWQ) VL export produces 0 markers, as forecast.**
`OVModelForVisualCausalLM.from_pretrained(export=True).save_pretrained()`
lands NNCF int8_asym per-channel over 205 / 205 layers. On the exported
`openvino_language_model.xml`, `grep -c "_openvino_orig_weight" …` =
**0** markers over 2584 total `Const` nodes; the projection weights are
named `<...>.weight` with FQ companions `<...>.weight/scale`,
`<...>.weight/zero_point` (both `Const`), `<...>.weight/zero_point/subtract`
(`Subtract` op), `<...>.weight/fq_weights_1` (`Multiply` op — not a
Constant despite the name).

**Option (c) landed (`src/exec/gguf_graph.cpp`, 2026-09-10).**
`kPlainWeightSuffix = ".weight"`: the weight-Constant filter admits
`.weight`-suffixed Constants under `kLayerPrefix` when `matmul_of()`
succeeds. The FQ-chain-to-MatMul walk itself disambiguates a projection
weight from a norm weight (a norm's `.weight` feeds a fan-out Multiply,
`targets.size() != 1` → `matmul_of` returns nullptr → silent skip). The
27B AWQ path (`_openvino_orig_weight`) is unchanged and still throws on
a no-MatMul chain. A companion per-head-scalar shape admission
(`n == g.linear_v_heads` case) unblocks `linear_attn.in_proj_a` /
`in_proj_b`, which on `qwen3_5` are per-head scalars (`ssm_alpha` /
`ssm_beta`, one row per v-head) rather than full projections — the
RowsV reorder used to throw `expected 0 q/k rows and V value rows`
against them.

Synthetic-IR test in the ovsrc-m18 tradition:
`tests/test_gguf_graph.cpp::gguf_pass_matches_plain_export_names_and_repacks_the_same_projections`
builds a toy with the plain naming and confirms `replaced.size() == 4`
against the fixture; the prior gguf_pass suite (12 cases) still passes
in the same run. Full arcint-test invocation and output are in
`HANDOFF-0.5.0.local.md`.

**First served 2B on a 16 GiB card.** The first `qwen3_5` 2B dense IR
loaded on the 16 GiB card against `Qwen3.5-2B-Q4_K_M.gguf` (Unsloth
export) with `--paged-kv f16` and `n_ctx 4096`, one lane. The load
report: **186 projection(s) from the file (mixed: 98 repacked, 88
native rows), Q8_0 x36 (1 MiB), Q5_K x36 (198 MiB), Q4_K x98 (551
MiB), Q6_K x16 (129 MiB)** — 186 = 18 linear-attention layers × 8
projections + 6 full-attention layers × 7 projections, every layer
projection recovered from the file, no throws. Paged model ready 7.4 s,
device-resident 1.34 GiB, activation 0.19 GiB at chunk 2048; analytic
max_ctx per lane 1,079,520 on 15.11 GiB usable of the 16 GiB card.
`/v1/completions` at temperature 0.0 returned deterministic output for
the greedy prompt. Full command line, log tail and /props body are in
`HANDOFF-0.5.0.local.md`.

**Limitation: `lm_head` not admitted for tied-embedding checkpoints.**
`self.model.lm_head._openvino_orig_weight` (the AWQ head constant this
file's `kHeadConst` matches) has no plain-export analogue on the 2B:
its `config.json` sets `tie_word_embeddings: true`, and the exporter
inlines the head as `__module.model.lm_head/ov_ext::linear/{Convert,
MatMul}` against the shared `embed_tokens.weight` Const rather than
emitting a separate `self.model.lm_head.weight`. Option (c)'s
`kLayerPrefix` filter is therefore correct-by-construction on `qwen3_5`
2B: the 186-projection count accounts for every layer weight and the
head shares `embed_tokens.weight`, which the loader already replaces
through its own embedding-from-file path (dequantised on the host per
token, not part of `gguf_apply_to_template`'s projection loop). A
non-tied plain-export checkpoint would need `kPlainHeadConst`; there
is no such checkpoint on the served allowlist today.

**Host feed measurement in the same window.** Single-thread DDR4 read
bandwidth: **44.4 GiB/s** on the dev host's Zen 3 SoC (single-threaded
`memcpy` in read+write mode measured 19.1 GiB/s in the same run — half
the raw traffic per byte moved, matching the shared-bus arithmetic).
The probe source, exact byte-counts and both timings are in
`HANDOFF-0.5.0.local.md`. This is the ceiling the FIX E consequence
sentence's "≈45 GB/s" was rounded from and its "~11 %" versus the
414–418 GB/s VRAM ceiling holds under the measured figure.

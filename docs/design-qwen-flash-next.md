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
| 0010 | Per-side u8:i4 decode-read/write kernel | `num_kv_heads`/`head_size`/`num_blocks` all read as runtime parameters through the kernel's own port shapes | fits | none observed in the diff; **narrower than "fits" deserves** -- every measured cell for 0008-0010 (171,312 vs. 133,456 tokens, the +28% context gain) was taken at the served `(16, 2)`-class and `(24, 4)`-class shapes, never at `(24, 2)`. Flagged as untested-not-broken, see RED-C-03 below (shared with 0020/0032-0034, all of which have the same "parameterized correctly, only ever run at two of the three now-relevant shapes" property). |
| 0011-0012 | MoE host CPU tier kernel (AVX2/scalar) + decode-split wiring | `num_expert`, `per_expert_bytes` runtime params; LRU probe geometry-agnostic | fits mechanically, **same admission-ratio caveat as 0005-0007** | See RED-C-02 -- 0011-0012's own measured cells (15.0-15.5 t/s at ratio 50/75) are the SAME two ratios 0005-0007 were measured at; Flash-Next's arithmetic (Fit section) needs a ratio in the high 80s-low 90s, unmeasured for the host tier's own readback/overlap timing (0017's `usm_host` hoist was tuned at ratio 50/75 too). |
| 0013 | MoE routing histogram | instrumentation only | fits | none |
| 0014 | `assign_impl` output-layout adoption (DFlash2 K/V state chain) | DFlash2 is a separate drafter lineage (`--dflash`), not part of Flash-Next's own MTP head | N/A | out of scope -- Flash-Next serves through its OWN 1-layer MTP head (`mtp_num_hidden_layers`, `text_config.mtp`), not the DFlash2 draft head this patch fixes. |
| 0015-0016 | Paged-attention bounded partials, intermediate sizing | `PAGED_ATTENTION_MAX_PARTITIONS`, `get_internal_buffer_descs` sized from `num_of_partitions`/output dtype -- no head/expert-count literal | fits | none in the diff; the buffer terms scale from `n_ctx`/chunk, which are runtime values regardless of model architecture. |
| 0017-0019 | MoE host-tier readback decomposition, static partition, prefill-fallback tristate | `splitmix64(seed, layer_key, expert)` ranking is `expert`-count-agnostic (any `expert` value ranks); `slots` from the same `expert_slot_bytes` formula as 0005-0007 | fits mechanically, same ratio caveat as 0005-0007/0011-0012 | See RED-C-02. Additionally: 0018's own header records "five load-time bugs" surfaced only once the offload ratio was pushed hard enough for a 100%-pinned pool to be reachable at all -- exactly the regime Flash-Next's arithmetic requires by default, not as an edge case. A fresh load-time bug hunt at Flash-Next's admission ratio should be budgeted as likely, not assumed clean by analogy to the 256-expert result. |
| 0020 | Micro-SDPA admits u8 keys / i4 values on prefill (was OCL-fallback only) | Selector gates on precision, not head count; measured at `(16, 2, ~128)`-class (coder) and `(24, 4, 256)`-class (agent/drafter) shapes only | fits, **untested at `(24, 2, 256)`** | RED-C-03 (below): Flash-Next's full-attention layers are `(24 query heads, 2 KV heads, head_dim 256)` -- the query-head/head_dim pair (24, 256) matches the dense-drafter shape already exercised by 0032-0034's regression suite, but the KV-head count (2) matches the MoE/coder shape, not the dense/agent shape those same tests used at head_dim 256. No test in the plugin suite (0020, 0032-0035's own headers) combines `head_dim 256` with `num_kv_heads 2` -- a GQA group size of 12 query heads per KV head, versus 6 (dense, tested) or the MoE family's own group size at head_dim ~128 (tested, but at a different head_dim). Nothing in the read code (`num_kv_heads` is a plain runtime multiplier throughout `paged_attention_opt.cpp`/`.cl`) suggests group size interacts with the u8:i4 packing math, but 0033's own bug (V-operand alignment keyed to head_dim, not caught by the existing test suite's uniform-page fill) shows this exact kernel has shipped a real correctness bug that a "the parameters are generic" read did not predict. **What the test must assert**: the plugin's paged-attention/micro-SDPA regression harness (the one 0032-0034 extended) run at `(24 heads, 2 KV heads, head_dim 256, u8:i4)` under the same three page orders 0033 added, byte-exact against the float reference within the suite's existing 1e-2 tolerance -- red if it is not, which today it cannot be, because no plugin build or test binary exercises this exact triple yet. |
| 0021-0030 | GGUF K-quant `FullyConnectedKQuant` decode/tiled-prefill kernels | N/A -- GGUF-open serves the dense `qwen35` template only (FIX B §"What the loader actually reads") | N/A | Flash-Next is MoE; it does not reach `apply_gguf_weights` under any configuration this repository drives today. Listed per the assignment's own patch range, not because these kernels touch GDN/attention geometry. |
| 0031 | `fc-deterministic-gemm` (oneDNN split-K determinism) | Applies to the GGUF-open f16-activation compressed FC form only | N/A | same reason as 0021-0030. |
| 0032 | Micro-SDPA K-tile prefetch bounds fix (the CAT-error fix) | Regression test at `(24, 4, 256)` only | fits, **same untested-shape gap as 0020** | Covered by RED-C-03 -- this is the patch whose whole point was an out-of-bounds read keyed to `d` (head_dim) and remaining-key count, not `num_kv_heads`; the fix (`ldk`, row length, row count for both calls) is written in terms of `d` and key count, so it should generalize to `num_kv_heads=2` correctly by inspection, but "should generalize by inspection" is exactly the confidence level the pre-fix code also had. |
| 0033 | Micro-SDPA V-operand alignment fix (`alignment_for_ld` keyed to `head*2`, capped 128) | `head` here is `head_dim` (256 for both the tested dense shape and Flash-Next); `num_kv_heads` does not enter `alignment_for_ld` | fits | Confirmed from the patch diff: `alignment_for_ld` is a pure function of `head` (== head_dim) and the value precision, with `num_kv_heads` used only for buffer indexing (`num_kv_heads * block_size * head_size` strides) elsewhere in the same file -- these two roles are independent. Flash-Next's head_dim (256) is the exact value the fix was written and tested against, so this specific bug class is closed for Flash-Next's shape. Still covered by RED-C-03 for the combination as a whole (indexing correctness at `num_kv_heads=2`, not the alignment constant). |
| 0034 | Micro-SDPA tail test (reads past `seq_len`) + by-token reproducer (disabled) | Tail test run at `(24, 4, 256)`; by-token reproducer at `(32, 2, 128)` default and `(24, 4, 256)` served | fits (tail test), **open, disabled finding (by-token)** | The by-token pairing itself is not reached by Flash-Next (arcint never selects BY_TOKEN keys per the patch's own note), so this finding does not block Flash-Next. Carried in the table for completeness since the assignment asked for every kernel in the series; not a Flash-Next-specific concern. |
| 0035 | BY_TOKEN test harness key-fill fix | Test-only, no served kernel change | N/A to Flash-Next serving path | none |

**RED-C-03 (the one gap this audit actually found, shared by 0010/0020/0032/0033)**: every measured or tested cell for the u8:i4 paged-attention/micro-SDPA path uses one of two head shapes -- `(16 or 24 query heads, 2 or 4 KV heads, head_dim ~128 or 256)` -- drawn from the two served checkpoints. Flash-Next's full-attention layers are a THIRD combination, `(24, 2, 256)`, that shares its query-head count and head_dim with the dense/agent shape and its KV-head count with the MoE/coder shape, but has never itself been run through the suite. Every kernel this audit read is parameterized generically over `num_kv_heads` (a plain multiplier in strides and loop bounds, never a literal), so there is no code-reading basis to predict a break -- but 0033's own history (a real alignment bug that a geometry-blind test suite did not catch until the fill was redesigned) is the standing counter-example to "parameterized code is safe by inspection" in this exact kernel family. This is an honest "fits, unverified" verdict, not a "needs patch" one: nothing read here shows a mechanism that would misbehave at `(24, 2, 256)`; nothing measured proves it will not. The red case is the regression run named above -- it does not exist yet because no Flash-Next IR export exists to drive it (same caveat FIX B recorded for the whole delta table).

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

## FIX A — Export harness and upstream blocker

**The blocker.** `Qwen/Qwen3.8-Flash-Next` reports `model_type: qwen4_exp`,
`architectures[0]: Qwen4ExpForConditionalGeneration`. No transformers
release carries a `qwen4_exp` modeling module — `import
transformers.models.qwen4_exp` fails with `ModuleNotFoundError` against
every pinned version checked so far. optimum-intel's export path
(`OVModelForCausalLM.from_pretrained(..., export=True)`) resolves the
architecture through transformers' model-type registry, so it fails the
same way, one layer up. There is no export path for this checkpoint today.
This is upstream's gap: nothing in arcint's own code is implicated, and
nothing here is fixable by editing this repository.

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

**Current status (2026-09-09):** harness exists and its red case is
verified (both failure modes — transformers absent, and transformers
present without `qwen4_exp` — were exercised directly; the exact-reason
assertion fires and the script exits 1). Watcher is registered
(`tools/watch_flash_next_export.py`, log path `tools/flash-next-watch.log`,
suggested weekly cadence in its docstring) but not yet added to any
crontab — that step is operator-side, tracked in `CLAUDE.local.md`.
Neither of the two options above has been chosen; none of this repository's
other Flash-Next work (FIX B's delta table above, FIX C's kernel audit,
FIX D's n-gram offload) waited on this decision.

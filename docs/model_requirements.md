# Model requirements

What arcint loads, and at what precision it has been measured serving.
Sourced from `src/core/artifact.{h,cpp}`, `src/core/model_registry.cpp`,
`src/config.cpp`, `tools/export_*.py`, `contrib/packaging/`, the `patches/`
headers, DESIGN.md and CHANGELOG.md. Numbers carry the card, depth and
precision they were measured at.

## 1. Artifact format

Three kinds of input, all validated against the built-in allowlist
(`src/core/model_registry.cpp`: an entry's id, its artifact directory aliases,
`model_type`, `architectures[0]`, arch / template / tokenizer hashes and
weight bytes). A directory whose basename is not an allowlisted alias is
refused at load.

- **An OpenVINO IR directory** (optimum-intel export plus NNCF weight
  compression). `load_artifact` requires `openvino_language_model.{xml,bin}`,
  `openvino_text_embeddings_model.xml`, `openvino_tokenizer.xml`,
  `openvino_detokenizer.xml`, `config.json`, `chat_template.jinja` and
  `tokenizer.json`; `generation_config.json` and `tokenizer_config.json` are
  read when present. From `config.json` (or its `text_config`): `model_type`,
  `architectures[0]`, `num_hidden_layers`, `hidden_size`,
  `max_position_embeddings`, `num_experts` (> 0 sets `moe`),
  `full_attention_interval`, `layer_types[]` (or the interval derives the
  GDN / attention counts), `eos_token_id`. A `*ForConditionalGeneration`
  export's vision IRs are stat'd and reported at load, never loaded;
  `--vision` is reserved and refused.
- **A GGUF opened on such a directory** (`--gguf FILE --model DIR`): the
  served IR of the same architecture is the topology template and the file's
  Q4_K / Q5_K / Q6_K / Q8_0 projections replace its own
  (`docs/design-gguf-native.md`). The template must carry the
  `_openvino_orig_weight` markers of optimum-intel's weight-compression
  export; dense `qwen35` files only.
- **A serving-shape artifact** (`tools/export_serving_artifact.py`, from the
  checkpoint's GGUF shards): the same file set plus `serving-shape.json`, with
  the MoE expert bodies in the checkpoint's own block formats and, for
  Flash-Next, the n-gram table as `ngram_table.K` ports.
  `arcint --inspect-artifact --model DIR` prints the contract device-free.

Allowlisted families, all hybrid GatedDeltaNet + full attention at
`full_attention_interval = 4` with one shared tokenizer (`87a7830d63fcf43b`):

| id(s) | `model_type` | shape | artifact |
|---|---|---|---|
| `qwen3.6-27b-a3b-coder` | `qwen3_5_moe` | 40 layers, 184 experts (pruned from 256) | int4 IR `qwen36-coder-b5-ov` |
| `qwen3.6-35b-a3b`, `-mtp` | `qwen3_5_moe` | 40 layers, 256 experts, top-8 | Intel's int4 IR; the `-mtp` directory adds the reconstructed head |
| `qwen3.6-35b-a3b-native-*` | `qwen3_5_moe` | 40 layers | serving-shape, IQ2_S / IQ3_XXS / IQ4_XS expert bodies (packed and u8 variants) |
| `qwen3.8-27b`, `-intel-int4` | `qwen3_5` | dense, 64 layers | int4 IR `qwen38-b7c1-ov`; Intel's int4 IR |
| `qwen3.8-flash-next-*` | `qwen4_exp` | 48 layers, 512 experts, top-10 + 1 shared, n-gram table at layer 2, 12 full-attention layers (QSA variant: `-d48q8qsa`) | serving-shape, IQ3_XXS / IQ4_XS / IQ4_NL / Q8_0 expert bodies; `-d48q8` carries the dense projections as Q8_0 / Q6_K |
| `qwen3.5-2b` | `qwen3_5` | dense | provisional, served through `--gguf` |

## 1a. The Flash-Next n-gram table

`load_artifact` reads the n-gram keys from `text_config`: `ngram_size`,
`ngram_vocab_size_base`, `heads_per_ngram`, `ple_embed_dim`, `ple_layer_ids`,
plus `vocab_size` and `eos_token_id`; all zero means no table. The table is a
hashed-vocab store: `row_ids` (`src/exec/ngram_row_ids.h`) mixes the last n
token ids per n-gram order, reduces them per head by a prime vocabulary size
and offsets into a concatenated row space (16 heads x 160 on Flash-Next), as
FreeToken's `PLETableBackend.lookup` does (`code`, `docs/research-freetoken.md`).
The row ids are computed on the host and fed as `ngram_chunk_ids` /
`ngram_local_ids`.

`--ngram-gguf FILE` binds the table from the GGUF shard holding
`per_layer_token_embd.weight`. When the artifact's port is the whole table it
is pinned in USM host memory (26.82 GiB); when the port is a smaller window
(the `-d48s*` / `-d48q8` exports) it is a per-forward staging buffer filled by
`pread` of only the rows the forward names (2.884 MiB at the served geometry,
`measured-here`; `src/exec/ngram_staging.h`, `docs/design-ple-disk-backend.md`).
`--flash-next-ngram PATH` admits a standalone `ARCINGRM` table (Q4_0 / Q4_1 /
Q8_0, 160 columns) with a host-RAM fit and row-count check.

## 2. Weight precisions

| artifact | form | acceptance |
|---|---|---|
| coder int4 (b5) | AWQ + scale estimation, code corpus | **10/10** greedy |
| dense Qwen3.8-27B int4 (b7c1) | AWQ only | **10/10** greedy (paged + MTP, B60) |
| Intel's 35B int4 IR | Intel's export | **10/10** greedy, 3/3 tool calls |
| Intel's Qwen3.8-27B int4 IR | Intel's export | **10/10** greedy (paged + MTP, B60) |
| dense Qwen3.8-27B Q4_K_M through `--gguf` | repack / native | **10/10** (B60) |
| Flash-Next serving-shape | the checkpoint's native expert blocks, decoded in the plugin (patches 0043, 0045, 0069) | Paris at depth 48; quality judged by KL against the model's own f32 forward (`docs/campaigns/serving-shape-logits.md`) |
| Qwen3.6-35B-A3B serving-shape | native IQ2_S / IQ3_XXS / IQ4_XS expert blocks (patch 0050; packed variants), dense projections f16 or u8 | 10/10 all-resident on the A770, full depth (DESIGN §7.0.2cs) |

The dense Qwen3.8-27B with scale-estimation calibration degenerates under
greedy (0/10, `measured-here`) and is not used. `--quant q8` is accepted by the
flag; no q8-weight acceptance run is recorded.

## 3. KV-cache precision (paged path)

`--paged-kv KEY[:VALUE]` over `{f16, u8, i8, u4, i4}`. Asymmetric pairs need
patches 0008–0010; the u8:i4 mixed prefill stage runs on micro-SDPA from patch
0020 (`+p6`).

| precision | KiB/token | measured where |
|---|---|---|
| f16 | 20.0 | coder, B60 |
| u8 (default) | 11.3 | coder, B60 |
| u4 | 6.3 | 35B; `PagedAttentionExtension` +63 % at 32k (coder, B60) |
| u8:i4 | 8.8 | 35B; the dense agent serves `i8:u8` |

u8:i4 scores 10/10 on the coder; auto-fit at u8:i4 is 171,392 tokens on the
16 GiB card, and a 118k-token prefill runs at every chunk from 128 to 2,048
there (`measured-here`, DESIGN §7.0.2at/§7.0.2av). The micro-SDPA path's chunk
cap is 2,048; the generic kernel's is 128. `--paged-attention-max-partitions`
(patch 0015) bounds the generic path's partition count. Owed: cold/warm
prefix-cache byte-identity at u8:i4.

## 4. Drafters

**MTP head** (`has_mtp_head` in `artifact.cpp`): `openvino_mtp_lm_head.xml`
plus `openvino_mtp_layer.xml` (reconstructed, `tools/export_mtp.py`) or
`openvino_mtp_model.xml` (optimum-intel's export); `--mtp-layer` picks between
them. Model cards: `docs/mtp-head-card.md`.

**DFlash2 head** (`--dflash DIR`): `openvino_dflash_draft_stateful.xml`,
`config.json` (`dflash_config`: `block_size`, `mask_token_id`,
`selector_top_k`), `dflash_hidden_projection.f32.bin`,
`dflash_predecessor_codebook.f16.bin`, `dflash_successor_codebook.f16.bin`;
exported by `tools/export_dflash.py --compress int4`. Its K/V state window is
2,048 rows (plugin patch 0014 and `src/core/dflash_window.h` keep it drafting
past the edge).

At 77,134 tokens on the dense agent (B60, u8 KV, `measured-here`, DESIGN
§7.0.2ag): plain 15.3 t/s, DFlash2 18.8 t/s (40.6 % accepted), MTP 4.9 t/s at
90.8 % acceptance; the MTP cycle is the `mtp-cycle-wall` campaign.

## 5. Runtime stack

`marfrit-openvino`: a source build of OpenVINO at the pinned upstream commit
`71640275` (the 2026.4.0 nightly of 2026-08-21) with the patch series in
`contrib/packaging/marfrit-openvino/patches/` applied; the patch level is part
of the package version. arcint 0.5.4 is built against `+p25` (patches
0003–0074, CHANGELOG). Compute-runtime 26.27; the xe kernel driver.

## 6. Not supported

- safetensors (GPTQ, NVFP4): OpenVINO does not read them.
- `--gguf` for MoE files and for sub-4-bit types; those formats serve through
  serving-shape artifacts (§1).
- A plain-cast `q8` KV without scales.
- Vision IRs: `--vision` is refused.
- GDN context shift or server-side truncation: an overflow is an HTTP 400 with
  the numbers (DESIGN §3.8).

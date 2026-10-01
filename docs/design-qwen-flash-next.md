# Qwen3.8-Flash-Next — the served model and its open levers

Qwen3.8-Flash-Next (`Qwen/Qwen3.8-Flash-Next`, `qwen4_exp`): 48 layers (36 GDN,
12 sparse-attention), hidden 2560, 512 experts of which 10 routed + 1 shared
are active per token (`moe_intermediate_size` 640), a hashed n-gram embedding
table at layer 2, hyper-connections, a 1-layer MTP head. It is allowlisted
(`src/core/model_registry.cpp`, ids `qwen3.8-flash-next*`) and served from the
serving-shape IR built straight from the GGUF shards
(`tools/q4e/serving_shape.py`, `tools/q4e/gguf_feed.py`,
`tools/export_serving_artifact.py`; GGUF conventions in DESIGN §7.0.2bz). The
current artifact is `qwen3.8-flash-next-d48q8`: experts in the checkpoint's own
IQ3_XXS/IQ4_NL/IQ4_XS/Q8_0 blocks (`--expert-format native`), dense projections
as Q8_0/Q6_K (`--dense-q8`). On the B60 with the CPU expert tier it serves at
6.6 t/s decode and ~61–65 t/s prefill at 20–27k tokens (`measured-here`,
`docs/campaigns/research-reference-audit.md`). Expert placement and prefill are
campaigns: `docs/campaigns/expert-hot-set-lru.md`,
`docs/campaigns/prefill-expert-streaming.md`, `docs/campaigns/qsa.md`.

## FIX D — the n-gram table

`per_layer_token_embd` is a hashed-vocab table (16 heads, row ids by
FreeToken's `NGramEmbedding.row_ids`, transcribed in `src/exec/ngram_row_ids.h`;
`code`, `python/freetoken/models/qwen4_exp/ple.py`). It is not loaded whole:
the host computes each forward's row ids, `pread`s only those rows into a
bounded staging buffer (`src/exec/ngram_staging.h`, `src/exec/ngram_table.h`)
and binds it to the IR's `ngram_table` port, the shape of FreeToken's default
disk backend (`code`, `python/freetoken/models/qwen4_exp/ple_disk.py`). The
table enters through `--ngram-gguf FILE` (the GGUF shard) or
`--flash-next-ngram PATH` (a block-quantised ARCINGRM file); the gather
dequantises Q4_0/Q4_1/Q8_0 32-element blocks (`src/exec/ngram_gather.h`).
Campaign: `docs/campaigns/ple-disk-backend.md`.

## MTP head (open)

The GGUF ships no MTP block; the BF16 checkpoint does (31 `mtp.*` tensors,
range-fetchable from its safetensors index; arcint's
`tools/fetch_safetensors_tensors.py` has fetched them once). Mechanism to
follow, Strata's (`code`, `~/src/Strata-ref`):

- `tools/mtp_fetch.py` pulls the 31 `mtp.*` tensors out of the BF16 shards
  without downloading them whole; `tools/mtp_pack.py` packs them for the
  engine; `tools/draft_vocab.py` builds a reduced draft vocabulary.
- `src/core/mtp.cpp` runs the draft layer (its 512 routed experts loaded as
  per-expert blobs); `src/program/generate.cpp` drafts up to 3 tokens per pass,
  a draft entering the verify window only while its probability under the
  draft layer stays at or above `spec_min_p`, and one pass over all 48 layers
  verifies them; `src/spec/draft_policy.cpp` chooses per round between the MTP
  window and a prompt-lookup window by measured committed tokens per
  millisecond.
- Effect: 2.4–3.2 tokens per verify pass (`paper`, Strata `docs/DETAILS.md`),
  1.6–1.8× decode with CPU-resident experts (`paper`, Strata paper, via
  `docs/campaigns/research-reference-audit.md` §1).

In arcint the head lands as a third drafter beside `--mtp` and `--dflash`
(MTP IR export: `tools/export_mtp.py`, serving: `src/exec/backend_ov.cpp`),
verified at the answer-level bar in `CLAUDE.md`.

Full history: `git show b0447b8:docs/design-qwen-flash-next.md`.

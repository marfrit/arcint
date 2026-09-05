# Model requirements

What arcint will load, and at what precision it has actually been measured
serving. Sourced from README.md, llm.txt, DESIGN.md, CHANGELOG.md,
`src/core/artifact.{h,cpp}`, `src/config.cpp`, `tools/export_dflash.py`,
`contrib/packaging/`, and `patches/` headers. Numbers carry the card, depth
and precision they were measured at; where the repository is silent, that
is said explicitly.

## 1. Artifact format

An **OpenVINO IR directory**, nothing else — no GGUF, no safetensors (GPTQ,
NVFP4; llama.cpp/vLLM formats OpenVINO does not read).
`load_artifact` (`src/core/artifact.cpp`) requires, in a directory whose
basename matches an allowlist alias: `openvino_language_model.{xml,bin}`,
`openvino_text_embeddings_model.xml`, `openvino_tokenizer.xml`,
`openvino_detokenizer.xml`, `config.json`, `chat_template.jinja`,
`tokenizer.json`; `generation_config.json`/`tokenizer_config.json` are read
if present but not required. Read from `config.json` (or its `text_config`
for a VLM export): `model_type`, `architectures[0]`, `num_hidden_layers`,
`hidden_size`, `max_position_embeddings`, `num_experts` (>0 sets `moe`),
`full_attention_interval`, `layer_types[]` (or the interval derives
GDN/attention counts when absent), `eos_token_id`. Every served checkpoint
is a `*ForConditionalGeneration` VLM export; its vision tower/projector
(`openvino_vision_embeddings_model`, `..._pos_model`, `..._merger_model`,
each `.xml`/`.bin`) are stat'd and reported at load, never compiled or
loaded — `--vision` is reserved and refused (M13; coder: 6 files, 428.3 MiB
on disk, no VRAM touched).

The allowlist names **three models across five directory entries** (the 35B
also has an MTP-bearing directory, the dense model also has Intel's own
export). All are one **hybrid GatedDeltaNet (linear attention) +
full-attention** family: `full_attention_interval = 4`, 262144 trained
context, one shared tokenizer (`87a7830d63fcf43b`). Two are **MoE** (Qwen3.6-35B-A3B,
256 experts; Qwen3.6-27B-A3B-Coder, 184 pruned from 256), one is **dense**
(Qwen3.8-27B).

## 2. Weight precisions

`--quant q4|q8` is the accepted format pair (DESIGN §2); every artifact
actually measured is **int4**:

| artifact | recipe | acceptance |
|---|---|---|
| the coder int4 artifact (b5, 184/256 experts) | AWQ + scale estimation, code corpus | **10/10** greedy, 24 GB card |
| the dense int4 artifact (Qwen3.8-27B) | AWQ-only | **7/10** greedy, provisional |
| the dense int4 artifact, SE calibration | AWQ + scale estimation | **0/10**, degenerates — do not use |
| Intel's 35B int4 export, as-is | Intel's own; recipe not recorded | **10/10** greedy, 3/3 tool calls, 62.7 t/s |

`q8` weights are accepted by the flag; no q8-weight acceptance run is
recorded. **INT3-class expert weights are a study owed, not shipped** —
NNCF 3.3.0 lists `INT3_SYM`/`INT2_SYM` undocumented in its public docs
(DESIGN §7.0.2y); no published Intel/OpenVINO IR below int4 exists for
this class, and a 3-bit kernel is scoped as an 800–1,500-line divergence,
not built (`docs/milestone-0.3.0.md` M10).

## 3. KV-cache precision (paged path)

`--paged-kv KEY[:VALUE]` over `{f16, u8, i8, u4, i4}`; `i8` aliases the
plugin's u8-stored-as-signed-i8 convention. Asymmetric KEY:VALUE needs
patches **0008** (`VALUE_CACHE_PRECISION`), **0009** (per-side kernel
plan), **0010** (per-side decode kernels) — served: **u8:i4**.

| precision | KiB/token | measured where |
|---|---|---|
| f16 | 20.0 | coder, 24 GB card, §7.0.3 |
| u8 (default) | 11.3 | coder, same card |
| u4 (symmetric — "a tax") | 6.3 | **35B**, §7.0.2w; the +63% on `PagedAttentionExtension` at 32k is the coder, 24 GB card, §7.0.3 |
| u8:i4 | 8.8 | **35B**, §7.0.2w; the coder's 16 GiB auto-fit gain matches this cost model to 0.1 pp, §7.0.2y |

u8 leads decode by +2.5% at 32k but f16 leads by 7.8% at 53.5k (crossover
not located); u8 costs up to 22% of prefill at 115k. u8:i4 scores 10/10 and,
at first measurement, auto-fit +28% context (133,456→171,312 tokens on the
16 GiB card) — but that predates an owed fix: the u8:i4 **prefill scratch
buffer** (that micro-SDPA declines the mismatched packing, so prefill takes
the opt-kernel path, is read off the plugin source, not measured; the buffer
does not exist at the past-0 point the activation probe runs) grows with the
past until the card is full, and the driver's rebind worker then fails on its
own page tables — `-12` in the kernel log, `CL_OUT_OF_RESOURCES` at the
runtime, root-caused with a free-VRAM sampler (§7.0.2ab). The edge sits
between 71.7k and 119k tokens on that card at chunk 128 and moves with the
prefill chunk and the pool depth, so it is not a token threshold; a first
reading that named the buffer alone as the mechanism was retracted. The **belt** (`--prefill-chunk` capped under 4-bit
values) and an honest reservation term now price it; **once charged, u8:i4
auto-fit lands at 101,824 — below plain u8's 133,456** on that shape
(101,984 before the acceptance ceiling was narrowed by the same term), so
u8:i4 is a decode-only saving there until the buffer stops scaling with
depth. `--paged-attention-max-partitions` (engine side in this tree,
unreleased; plugin patch **0015** is written but in no built package, and
no plugin carrying its key exists yet, so its effect is priced by the
engine's own arithmetic and not measured on a card) bounds the partition
count to flatten the term past a fixed depth. Still owed: the full-resolution u8:i4 prefill
price, and cold/warm prefix-cache byte-exactness at u8:i4.

## 4. Drafters

**MTP head** (`has_mtp_head` in `artifact.cpp`; served for the dense 3.8, and
reconstructed for the 35B MoE, where it *loses* 30% — 48–53 t/s against 71.5
plain, §7.0.2o): needs
`openvino_mtp_lm_head.xml` plus one of `openvino_mtp_layer.xml`
(reconstructed, `tools/export_mtp.py`) or `openvino_mtp_model.xml`
(optimum-intel's export); `--mtp-layer` picks between them.

**DFlash2 head** (`--dflash DIR`): `openvino_dflash_draft_stateful.xml`,
`config.json` (`dflash_config`: `block_size`, `mask_token_id`,
`selector_top_k`), `dflash_hidden_projection.f32.bin`,
`dflash_predecessor_codebook.f16.bin`, `dflash_successor_codebook.f16.bin`.
`tools/export_dflash.py --compress int4`: data-free NNCF RTN, `INT4_ASYM`,
`group_size=64`, `ratio=1.0`, `all_layers=True`, 56 MatMul weights —
byte-identical to the served head. Its K/V state window is fixed at
**2,048 rows**; past it an `Assign` layout mismatch silently disabled the
drafter process-wide (measured: the failing case is a first draft that
concatenates *exactly* the 2,048 rows; shorter prompts draft), fixed by
plugin patch **0014** (released in `+p4`).

At depth (dense 27B agent, 24 GB card, u8 KV, §7.0.2ag): the zero
acceptance §7.0.2aa saw at 76.4k tokens was an f16 position overflow at
65,504 in the drafters' rotary subgraphs, now kept f32, and the MTP layer's
KV state is now charged against the reservation. At 77,134 tokens MTP
accepts 90.8% again but still loses to plain (4.9 against 15.3 t/s): its
cycle wall is about 390 ms against its own break-even near 130 ms, so even
full acceptance cannot outrun it on this artifact. DFlash2 beats plain
there (18.8 against 15.3 t/s, 40.6% accepted). Guidance: at depth serve
with DFlash2 or plain, MTP off; the MTP cycle is the `mtp-cycle-wall`
campaign (`docs/campaigns/`).

## 5. Runtime stack

`marfrit-openvino`: source build pinned at upstream commit `71640275` (the
2026.4.0 nightly of 2026-08-21), patch series **0003–0019** applied
(`contrib/packaging/marfrit-openvino/patches/`). Package version carries
the patch level: **`+p4` is the 0003–0018 level** and the one 0.3.0
requires (CHANGELOG); **`+p5` adds 0019** (the prefill fallback's
three-way answer, DESIGN §7.0.2ap; recipe bumped 2026-09-05, package not
yet built); the arcint package depends on `+p4` as a floor within the
pinned nightly, since nothing arcint drives reaches 0019's branch.
At the 0.3.0 tag the dev host's production units still serve `+p3` —
deployment is a separate decision (DESIGN §7.0.2ai). Compute-runtime
**26.27** (past the fix window for USM-pool issue 916). Kernel driver:
**xe KMD**; no version recorded.

## 6. Not supported / not measured

- GGUF, safetensors (GPTQ, NVFP4): wrong format, not loaded.
- A plain-cast `q8` KV (no scales): refused as "quietly worse."
- INT3/INT2 expert weights: study owed, no kernel, no allowlist entry.
- `q8` weight format: accepted by the flag, no acceptance run found.
- Vision IRs: reserved, `--vision` refused.
- Full-resolution u8:i4 prefill price and prefix-cache byte-exactness at
  u8:i4: named, not yet measured.
- GDN context shift / server-side truncation: refused by policy — an
  overflow is an HTTP 400 with the numbers, and a shift is not honestly
  implementable on a recurrent state (§3.8).

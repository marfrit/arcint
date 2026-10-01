# arcint — Design

Full history of this document: `git show b0447b8:DESIGN.md`.

State as of 2026-10-01: arcint **0.5.4**, runtime `marfrit-openvino` **+p25**
(plugin patches 0003–0074 on the pinned OpenVINO nightly `71640275`). Two
units serve in production: the MoE coder on the 16 GiB Arc A770 and the dense
Qwen3.8-27B agent on the 24 GB Arc Pro B60. The development target is
Qwen3.8-Flash-Next (125B-A6B) on the B60 plus host RAM.

Every disposition below carries its evidence class: `paper` (a published
text), `code` (read in source), `measured-here` (run on this project's cards),
or `decision` (the operator's, dated). Measurements name card, depth, KV
precision and configuration. Retractions, rejected alternatives and the
debugging narrative of every section live in git history, not here.

How to read it: §1–§6 are the architecture, the rules and the serving
surface; §7 is the state of play and an index of the dated records that
other files cite; §8 is the open work, each item against the reference
engine that already does it.

## 1. The one idea

OpenVINO is the compiler and kernel library; arcint is everything else:
scheduling, the paged KV cache and recurrent-state ledger, prefix caching,
speculative decoding, the MoE expert placement, sampling and the HTTP
surface. OpenVINO's GPU plugin emits good Xe code (3.4× llama.cpp-SYCL and
7.6× llama.cpp-Vulkan on Battlemage for the same checkpoint at the project's
start, `measured-here`); OpenVINO GenAI's pipeline layer makes choices arcint
does not accept (§6).

**References first.** Inference-engine expertise lives in the reference
engines, and their source is the design reference (operator rule,
2026-10-01; `CLAUDE.md`, `AGENTS.md` rule 8):

- **Strata** (`~/src/Strata-ref`), written for Qwen3.8-Flash-Next on one
  12 GB GPU plus RAM. Its README reports, on an RTX 5070 + Ryzen 5 7600 +
  64 GB DDR5, IQ3_XXS: 62 t/s decode on a short chat, 49 t/s at 128K, 1,750
  t/s prompt reading at 32K (`paper`: `README.md`). Paper:
  `docs/paper/Strata-Paper.pdf`.
- **FreeToken** (`~/src/FreeToken-ref`), an edge-MoE engine: CPU-resident
  experts, one GPU LRU expert pool, bandwidth-adaptive split, double-buffered
  prefill streaming (`code`; paper arXiv 2608.16157).
- **llama.cpp** (ggml's CPU and GPU kernels) and **NInfer** (`~/src/ninfer`).

An arcint measurement counts against a reference technique only if it
tested the reference's mechanism; a conflict between a reference mechanism
and an invariant here goes to the operator. The audit that applies this rule
to the record is `docs/campaigns/research-reference-audit.md`.

### 1.1 When a kernel has to change: smallest sufficient divergence

In order: (1) an upstream PR written to upstream's conventions; (2) a
numbered patch against the pinned version, carried in
`contrib/packaging/marfrit-openvino/patches/` and applied at build time,
each patch PR-shaped and documented in that directory's `README.md`; (3) a
maintained fork, only with a stated reason and an exit condition (none
exists). Whatever is carried is published; a number that depends on a patch
nobody else has is reproducible from that directory. The patch level is
part of the package version (`+pN`) and every release names it
(`CHANGELOG.md`). Upstream reports filed from this work include
openvino.genai#4367, openvino#37607, openvino#38099 (a sibling comment) and
compute-runtime#948 (a comment).

## 2. Target constraints

All models are hybrid GatedDeltaNet / attention transformers: most layers
carry a fixed-size recurrent state (conv + delta-rule state), a minority are
attention layers with a KV cache. One tokenizer (hash `87a7830d63fcf43b`) is
shared by all of them (`code`: `src/core/model_registry.cpp`).

| model | layers (GDN + attn) | experts | hidden | served form | card |
|---|---|---|---|---|---|
| Qwen3.6-27B-A3B-Coder | 40 (30 + 10) | 184 (the 35B pruned) | 2048 | int4 AWQ IR, 12.8 GiB | A770, production |
| Qwen3.6-35B-A3B | 40 (30 + 10) | 256, top-8 | 2048 | int4 IR 17.4 GiB (offload + tier); native packed u8 IR all-resident 13.11 GiB | A770, development |
| Qwen3.8-27B | 64 (48 + 16) | dense | 5120 | int4 AWQ IR 13.4 GiB, MTP head reconstructed; or its GGUF via `--gguf` | B60, production |
| Qwen3.8-Flash-Next | 48 (36 + 12 sparse attention) | 512, top-10, + shared | 2560 | serving-shape IR from the GGUF, native expert formats, Q8_0 dense (`d48q8`, 60.7 GiB `.bin`) | B60 + host RAM, development |

Geometry is `code` (the registry and the artifacts' configs); Flash-Next's
36 + 12 split and its sparse attention (at most ~2,048 selected positions per
query) are also Strata's `paper` §2.

The cards, measured here unless marked:

| | Arc A770 (16 GiB, 15.11 usable) | Arc Pro B60 (24 GB, 22.71 usable) |
|---|---|---|
| silicon | ACM-G10 (DG2-512), PCI `8086:56a0`, Xe-HPG: subgroup 8 or 16 by kernel, 128 GRF, no 2D block loads | Xe2 (BMG), PCI `8086:e211`: subgroup 16 only, 2D block loads |
| OpenVINO device | `GPU.1` | `GPU.0` |
| host link | PCIe 3.0 x4 behind the chipset, 1.8 GB/s H2D | PCIe 4.0 x8, 14.3 GB/s |
| random-read ceiling | 414–418 GB/s (nominal 560, `paper`) | 453 GB/s (nominal 456, `paper`) |
| run-to-run determinism of the served Flash-Next path | bit-identical | not bit-identical (GDN state output, §7.0.2cb) |

- Xe KMD only. SYCL runtimes abort on memcpy under it and Vulkan is slow on
  BMG; OpenVINO's OpenCL path is the only proven fast route on both cards
  (`measured-here`).
- The dev host is an 8-core / 16-thread AVX2 CPU (Zen 3, no AVX-512) with
  52 GiB of RAM in the development container and an NVMe store
  (`measured-here`). Host RAM is the binding resource for Flash-Next: its
  expert constants are 56.4 GiB (`measured-here`).

## 3. Architecture

```
             ┌────────────────────────────────────────────────┐
             │ HTTP server (one thread pool, no framework)    │
             │  /v1/chat/completions  /v1/completions         │
             │  /v1/models  /health  /props                   │
             └───────────────┬────────────────────────────────┘
                             │ request objects
             ┌───────────────▼───────────────┐
             │ SlotPool: admission by memory │  N lanes = N InferRequests
             │ reservation (§4.3)            │  on one CompiledModel
             └───────────────┬───────────────┘
                             │ Turnstile (FIFO ticket lock, §4.1)
     ┌───────────────────────▼─────────────────────────┐
     │ Executor (paged by default; stateful reference) │
     │  ┌───────────────┐   ┌──────────────────────┐   │
     │  │ Cache manager │   │ OV compiled models   │   │
     │  │  paged KV pool│◄─►│  language model      │   │
     │  │  GDN rows     │   │  embeddings gather   │   │
     │  │  prefix cache │   │  MTP layer + lm_head │   │
     │  │  host tier    │   │  DFlash2 drafter     │   │
     │  └───────────────┘   └──────────────────────┘   │
     │  sampling, drafting/verify, detokenise, emit    │
     └─────────────────────────────────────────────────┘
       MoE experts: GPU plugin OTD pool + CPU tier (§3.9)
```

### 3.1 Model artifacts

- **Allowlist.** arcint validates every artifact against a compiled-in
  registry (`src/core/model_registry.cpp`, transcribed from
  `models/allowlist-raw.json`): architecture hash (`lm_xml_sha`), chat
  template and tokenizer hashes, layer geometry, trained context, weight
  bytes. Anything else is refused. Provenance is part of the contract:
  scale-estimation calibration degenerated greedy decoding on the dense 3.8
  (0/10) where AWQ-only held (7/10) (`measured-here`). Unpinned fields are
  reported as `null` on `/props`, not invented.
- **OpenVINO IR** (optimum-intel export), one directory per model. The
  exports are VLMs; the vision IRs are never loaded and `--vision` is refused
  (`code`, §7.0.2y).
- **GGUF on a template IR.** `--gguf FILE --model TEMPLATE_IR` opens a GGUF
  in process: the file's Q4_K/Q5_K/Q6_K/Q8_0 rows replace the template's
  projections (`src/exec/gguf_graph.*`, `src/core/gguf*.{h,cpp}`); the
  template's AWQ activation multipliers are set to one; the value-head
  reorder is inverted by a gather. Forms (`--gguf-mode`): `mixed` (default:
  Q4_K repacked into the runtime's compressed form, Q5_K/Q6_K native rows in
  the K-quant kernel, patches 0021–0030), `repack`, `native`. Mins
  (`--gguf-mins exact|split|shared|nibble`; exact default, refuses any weight
  over a per-type deviation bound of 1/64–1/16 of a quantisation step).
  `--gguf-q6k aligned` (224-byte blocks, default), `--gguf-embed file`,
  `--gguf-check once` (verdict cache). Paged path only. Dense 27B Q4_K_M on
  the B60, u8 KV: warm prefill 1,001 t/s at 856 tokens and 464 t/s at 71.7k,
  decode step 54.8 / 73.7 ms, 16.54 GiB, Prüfstand 10/10; the native form
  51.3 ms at 15.22 GiB (`measured-here`, §7.0.2bn–bp). The same file through
  llama.cpp SYCL on the same card: 249 t/s prefill, 14.2 t/s decode at 1k
  (`measured-here`, §7.0.2bf).
- **Serving-shape IR** (Flash-Next, the 35B native forms). Emitted from the
  GGUF by `tools/q4e/serving_shape.py` and
  `tools/export_serving_artifact.py`. Fill conventions that only an outside
  reference shows (`measured-here` against llama.cpp tensors, `code` in
  llama.cpp's consumer): the GGUF stores plain-RMSNorm gammas folded as
  `(1 + w)` and `ssm_a` as `−exp(A_log)`, both undone at the feed
  (`q4e.gguf_feed`); the GDN output gate is sigmoid
  (`output_gate_type: sigmoid`); key heads pair with value heads as
  `h % 16` (`gdn_key_head_map: tiled`). BF16 tensors are read through
  `gguf.quants` (the raw-byte cast was a defect until 2026-09-28).
  Expert blocks ship in their native formats (IQ3_XXS / IQ4_XS gate-up,
  IQ4_NL / Q8_0 down; IQ2_S-packed for the 35B) through the fused op's
  `weight_format` (patches 0043, 0045, 0050–0052, 0054–0057); dense
  projections in the checkpoint's Q8_0/Q6_K form (the `--dense-q8`,
  `--dense-u8` and `--dense-fp16` export options; also `--expert-format`,
  `--native-packed`, `--qsa`). The Flash-Next n-gram table is either bound
  as pinned USM-host ports from the shard (`--ngram-gguf`, 26.82 GiB) or
  **staged per forward** (`--ngram-staging-rows N` at export: one
  `[N, 90 B]` buffer, 2.884 MiB, filled by `pread` of the named rows), as
  FreeToken's disk backend does (`python/freetoken/models/qwen4_exp/ple_disk.py`).
- **Blob cache.** Opt-in (`--cache-dir`); a cached blob is proven by a real
  forward at load and discarded on failure, because the shipped 2026.4
  runtime imports blobs whose MoE weight provider is uninitialised
  (`measured-here`, openvino#37607).

### 3.2 Graph strategy

- **The paged executor is the served path.** OpenVINO's
  `SDPAToPagedAttention` turns the exported stateful graph into a paged one
  (`key_cache.N`, `value_cache.N`, the `la.*` linear-attention block tables);
  arcint owns the page tables, the GDN rows and their checkpoint interval.
  Against the stateful graph on the same IR the paged decode step is 1.68×
  faster (11.3 vs 19.0 ms, coder, B60): the GDN transposes vanish and the
  paged GDN kernel is the optimised one (`measured-here`, §7.0). `--no-paged`
  keeps the stateful executor as the reference implementation the suites
  compare against.
- **Logits slice.** The LM-head input is sliced to the rows anything samples
  (the last row, plus `1 + draft` rows under speculation); the token axis is
  read from the head's declared shape (0 for `[tokens, 1, hidden]`, 1 for a
  serving-shape `[1, tokens, hidden]`), and a probe forward at load verifies
  it or the load refuses (`code`: `slice_logits_to_last_token`,
  `paged_logits_token_axis`). `--no-logits-slice` exists for measurement.
  The slice works for a K-quant head too (`measured-here`, §7.0.2bg).
- **Prefill chunking on an absolute grid.** Chunks are multiples of
  `--prefill-chunk` counted from position 0, and prefix-cache checkpoints
  fall only on that grid, so a warm run presents the model the same
  boundaries a cold run did; `--prefill-chunk` must be a multiple of
  `--kv-block-size` when the cache is on (`code`: `src/config.cpp`).
  Different chunkings of the same tokens give different (equally valid)
  floating-point results; the default chunk is 2,048, capped by the fit.
  `--cache-grid N` snapshots on a finer grid by cutting a chunk (exact on
  the paged path; ~0.45 s fixed per continuation, `measured-here`).
- **The prefill is chunked to bound activations** and the activation peak is
  probed upward at load (§4.3); a chunk that does not fit is lowered, never
  silently raised (`code`: `src/exec/fit.h`).
- **Static shapes.** Constant dedup in the GPU plugin is per compile
  (`code`: `ops/constant.cpp`; `measured-here`: two compiles of one model
  double its residency), so arcint compiles one language model and runs N
  requests on it.
- Main-model inference precision is the plugin's default f16. The drafters'
  rotary subgraphs are marked `disable_conversion` and stay f32 (an f16
  absolute position overflows at 65,504, `measured-here`).

### 3.3 Memory: paged KV + GDN ledger

- **KV.** The plugin's paged cache: 16-token pages; precision per side
  (`--paged-kv KEY[:VALUE]` over f16/u8/i8/u4/i4). **u8 is the default**
  (11.3 KiB/token on the coder against f16's 20.0); `u8:i4` (u8 keys,
  4-bit values; 8.8 KiB/token on the 35B) buys +28 % auto-fit context on
  both served configurations and since `+p6` prefills on micro-SDPA at u8's
  rate (patches 0008–0010, 0020; `measured-here`, §7.0.2w, §7.0.2y,
  §7.0.2as). Precision choice and its measured price: §7.0.3.
  `ARCINT_PAGED_KV` overrides the flag for A/B runs.
- **Pool.** Sized at startup from the measured reservation (§4.3), refcounted
  (`src/core/block_pool.h`): a page is live in a sequence, held by a prefix
  cache entry, or both; a hit only ever maps complete pages, which are never
  written again. When a lane needs pages, cached prefixes are dropped (or
  demoted to the host tier, §4.4) first; a live sequence's pages are never
  taken (`code`).
- **GDN ledger.** One fixed-size state row per lane plus checkpoint rows;
  fresh rows are zeroed on the device from a resident zero row
  (`measured-here`, §7.0.2bt); `--gdn-checkpoint-budget` caps checkpoint
  rows. A prefix hit restores KV pages and the GDN checkpoint at the same
  block boundary, both or neither.
- **Stateful reference path.** KV state retyped to fp16 by graph surgery
  (`--kv-dtype`, half of fp32's 40,960 B/token); a plain int8 cast is
  refused because it has no scales (`measured-here`: it produced a fluent,
  worse answer).

### 3.4 Prefix caching, and the correctness rule

**Mechanism.** A hash chain over token blocks (keyed 128-bit content hash,
token identity verified on hit). A hit restores KV pages by reference and the
GDN checkpoint at the same boundary; the paged cache entry holds page
references plus a host GDN blob (~32 MiB per row on the coder).
`--prefix-cache-mib`, `--kv-block-size`, `--prefix-cache-reserve PCT`. Real
agent sessions replayed against this policy hit on 97 % of turns and serve
96 % of prompt tokens from cache (`measured-here`, §7.0.2j).

**The correctness rule** (`decision`, operator, 2026-10-01; this replaces
the earlier byte-identity invariant and its two same-day amendments, all in
git history; precedent, `paper`: Strata's expert cache changes the top-1
pick at 2–5 % of positions at equal perplexity,
`~/src/Strata-ref/bench/results/2026-09-27-cache-parity/README.md`):

1. **Correctness is judged at the answer.** The bar for any equivalence or
   quality gate is the answer-level bar in `CLAUDE.md`: the answers stay
   right (facts, the needle, the task battery); the candidate's mean KL
   against the reference is at most **0.03 nats** worse than the baseline
   arm's on the same card and window; argmax agreement drops by at most
   **1 point**. The tolerances are the operator's to change. Bit-identical
   *output* across configurations, placements, timing or request history is
   not required anywhere.
2. **Integrity checks on copied data stay exact**: expert stores, slots,
   staged tables, restored pages and rows.
3. **Deterministic replay is a cheap default, not a law.** The same request
   sequence giving the same output is kept where it costs nothing (it makes
   measurements comparable) and may be dropped by a mechanism that buys
   measured speed (non-blocking cache admission, a timing-probed miss split).
4. **The prefix cache may load together with an adaptive expert tier**, as
   in both references. A restored continuation is held to the answer-level
   bar. `tier_prefix_cache_decision` (`code`: `src/config.cpp`) still refuses
   `--moe-cpu-tier` + `--prefix-cache-mib > 0` unless the plugin reports a
   static partition; changing that refusal is owed (§8).
5. The byte-equality cells in §5 stay as **tripwires**: a red one on a
   change that passes the answer-level bar is reported, not a veto.

What holds today, measured: on the dense and non-tier configurations a warm
run is byte-identical to a cold one, per lane, with the other lane active
(`measured-here`, the equivalence and concurrency suites).

### 3.5 Speculative decoding

- **Machinery** (`code`; `measured-here`). Greedy-only. Acceptance is the
  sampler's own decision (penalties applied first), and a drafted token clears
  the same gates as a sampled one (EOS, `max_tokens`, `n_ctx`). On the paged
  path a verify pass checkpoints the GDN state after every token into scratch
  rows (`la.cache_interval = 1`) and rollback is choosing which row to keep:
  zero bytes copied; attention KV rolls back by `past_lens`. The verify pass
  is a multi-token forward, so a near-tie can resolve differently from a
  one-token step; that is admissible under §3.4 and reported by the suite.
- **MTP head, Qwen3.8 dense** (`--mtp on`). No public implementation
  consumed the checkpoint's `mtp.*` tensors when it was built (2026-08-28,
  `code`: transformers 5.16.1, optimum-intel), so `tools/export_mtp.py`
  reconstructs the layer and extracts the base lm_head; every forward-pass
  choice was set by acceptance (zero-centred `(1 + w)` norms, per-head
  q/gate interleave, sigmoid gate; `measured-here`). Acceptance 93 % on our
  export; Intel's public `Qwen3.8-27B-int4-ov` pairs with our lm_head
  (`--mtp-layer exported`), 90.8 % acceptance, 10/10, +36–48 % decode at
  short depth (`measured-here`, §7.0.2n). The MTP layer's own KV is charged in
  the reservation at 8 KiB/token (`measured-here`, §7.0.2ag). The head and
  the embeddings gather can run on the other card (`--mtp-device`,
  `--emb-device`).
- **MTP head, Qwen3.6 MoE** (`qwen3.6-35b-a3b-mtp`): exported with the MoE
  geometry; 93.9 % / 75.4 % acceptance (code / prose); with patch 0003 and an
  int4 head +17 % on code (`measured-here`, §7.0.2o–p).
- **DFlash2 drafter** (`--dflash DIR`, block-diffusion head for the 3.8):
  44.8 t/s against 24.0 plain on the B60 at short depth, 3.13 accepted per
  verify cycle; at 77k 18.8 t/s against 15.3 plain (`measured-here`,
  §7.0.2r, §7.0.2ag). Block and selector options, `ARCINT_PROFILE_CYCLE`.
- **Prompt lookup** (`--draft N --draft-ngram K`), off by default (`code`).
- Speculative decoding on Flash-Next, multi-token drafting and MTP at depth
  are open items (§8.3).

### 3.6 Sampling

Greedy, temperature, top-k, top-p, repetition, presence and frequency
penalties, host-side (`src/core/sampler.cpp`). Penalties apply before the
greedy decision. Seeded; an unseeded request gets a logged seed. Defaults
chain request > operator flags (`--temp`, `--top-p`, `--top-k`,
`--repetition-penalty`, `--presence-penalty`,
`--chat-template-kwarg enable_thinking=BOOL`) > artifact
`generation_config.json` > model-card defaults; `/props` reports values and
provenance. No `min_p` (not implemented, so no flag). `code`.

### 3.7 Tokenizer, templates, tool calls

- Tokenizer and chat template come from the artifact; the template hash is
  pinned. Rendered prompts are byte-identical to reference jinja2
  (`measured-here`).
- Incremental detokenisation never splits a UTF-8 code point across SSE
  chunks; stop strings and token ids; `usage` in every response (`code`,
  unit-tested).
- **Reasoning.** When the rendered prompt ends inside `<think>`, the server
  splits everything before the first `</think>` into `reasoning_content`
  (`split_reasoning`, `ReasoningStreamer`); `enable_thinking` and
  `reasoning_effort` are honoured (`code`).
- **Tool calls.** The models' native format is parsed into OpenAI
  `tool_calls`; requests without `tools` get raw text. A template that
  iterates arguments as a mapping is handed the parsed object
  (`tool_call_arguments_for_template`); minja polyfills stay off (`code`).
- **Cancellation.** A dropped connection aborts at the next scheduler
  boundary and frees the lane's pages (`code`; gated by the concurrency
  suite).

### 3.8 Context-overflow policy

A prompt or continuation that exceeds the context is **rejected with HTTP
400** and a JSON body with `prompt_tokens`, `n_ctx` and `overflow`. No
server-side truncation, context shift or sliding window: a server-side
history edit changes what the model saw and breaks prefix-cache identity,
and a GDN state cannot un-see a token, so any shift is an approximation.
History management is the client's job. This invariant is unamended
(`decision`; `code`: the 400 and its numbers are unit-tested).

### 3.9 MoE expert execution

The GPU plugin's fused MoE op owns routing; arcint configures it through
compile properties and patches. Pieces in service, with where they live:

| piece | what it does | where | evidence |
|---|---|---|---|
| OTD offload | `--offload-ratio R`: R % of experts not resident; resident slots in a pool, misses loaded on demand from the weight file | stock plugin `OFFLOAD_RATIO`; pool size `floor(E·(100−R)/100)` per layer is the served truth (the fit ledger's `ceil` is a ledger ceiling only) | `code`, `measured-here` |
| device slot pool | slots in VRAM up to a byte budget, async batched uploads | patches 0004–0007; budget `ARCINT_MOE_DEVICE_POOL_BYTES` → `MOE_OTD_DEVICE_POOL_BYTES`; 0070 extends it to the per-expert route | `measured-here` (35B on the A770: 0.4 → 9.1 t/s, §7.0.2v) |
| CPU tier | non-resident experts computed on the host instead of uploaded | `--moe-cpu-tier` (patches 0011/0012, 0017); `--moe-cpu-tier-threads` | `measured-here` (§7.0.2x) |
| static partition | placement a pure function of (layer key, expert, configuration), pinned at bind; the default tier mode | patch 0018, property `MOE_CPU_TIER_STATIC_PARTITION`; the LRU mode stays available (`MOE_CPU_TIER_PARTITION=lru`) | `code` |
| census seed | the resident set chosen from a recorded routing census instead of a hash rank | patch 0046, `MOE_CPU_TIER_SEED=<file>` (`# space=layer_key`); tools `tools/expert_policy_compare.py`, hot-set census tools | `measured-here` (§7.0.2cc) |
| hybrid prefill | grouped GEMM over the resident subset, host tier for the rest | patches 0037, 0042 | `measured-here` |
| per-expert dispatch | experts computed by a per-expert GPU kernel with in-kernel dequant, bypassing the fused GEMM; misses on the host | `--moe-per-expert-dispatch`; patches 0038–0041, 0045, 0047, 0059–0061, 0064, 0069 | `measured-here` |
| all-resident native pool | every expert in a slot, filled at bind; decode routing on the device | patches 0051, 0058, 0062, 0067 | `measured-here` (35B, A770: decode 19.3 → 28.1 t/s, §7.0.2cs) |
| CPU tier kernels | native formats decoded once per call, AVX2 row decode, jobs per lane, two-phase split over the pool, quantised-domain dot for decode-shaped calls (≤ 8 jobs) | patches 0043, 0065, 0066, 0068, 0071, 0074 (`MOE_CPU_TIER_Q8_DOT=0` reverts 0074) | `measured-here` (§7.0.2cp–cq, §7.0.2cx; 0074: decode +10 %, KL +0.018 nats) |
| host expert bank | tier experts read from a RAM bank filled at load by sequential O_DIRECT reads instead of page faults | patch 0072, `MOE_CPU_BANK_BYTES`, `MOE_CPU_BANK_SEED`, `MOE_CPU_BANK_FILL_PER_LAYER` (anonymous `MAP_NORESERVE`; a budget above free RAM ends in an OOM kill) | `measured-here` (first answer −18…−21 %, §7.0.2da) |
| NVMe pinned fill | the pinned expert set DMA'd from an NVMe expert store into VRAM at load (arcwell, B60 only) | patches 0048–0049 | `measured-here` (cold TTFT 92.5 vs 99.7 s at depth 4, §7.0.2ch) |
| perf counters | `MOE_OTD_PERF_LOG=1`, routing histogram/trace | patches 0004, 0013, 0044 | `code` |

Two facts every MoE measurement must state:
- **The partition's layer key is the `.bin` offset of the layer's
  `weight_0` constant** (patch 0013), so any export that shifts the file
  layout moves every static resident set (`measured-here`, §7.0.2cz). Control
  it with a census seed until the key is fixed (§8).
- **Residency can move arithmetic.** Under per-expert dispatch a resident
  expert runs on the GPU kernel and a miss on the host tier; native-format
  GPU and host paths are not bit-identical (affine ones are). Under §3.4 that
  is admissible; it means outputs depend on the resident set (`measured-here`,
  §7.0.2cf).

The reference development configuration for Flash-Next (B60, `d48q8`,
`--offload-ratio 75 --moe-cpu-tier --moe-per-expert-dispatch`, census seed
of 128 experts per layer, 30 GiB host bank, u8 KV, chunk 2,048, plugin with
0074): 20,085-token needle answered, prefill 63.1 t/s, decode 6.5 t/s; a
27,603-token needle at `n_ctx` 32,768 answered, prefill 61.2 t/s, decode
6.6 t/s. Per decode token 308 experts run on the CPU tier against 170 GPU
hits (36 %), and the GPU idles ~46–50 % of decode wall waiting on the tier
(`measured-here`, `docs/campaigns/host-expert-bank.md` 2026-10-01). The
references' levers for exactly this are §8.1–§8.6.

## 4. Serving surface

| endpoint | contract |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; SSE and non-stream; `chat_template_kwargs.enable_thinking`, `reasoning_effort`; `tools` |
| `POST /v1/completions` | raw completion; accepts token-id prompts |
| `GET /v1/models` | the served model with the context it is **running with** (§4.2) |
| `GET /health` | model, loaded, slots free/total, queue depth, prefix-cache counters (`cache.{entries,tiered_entries,host_mib,hits,demotions,promotions}`) |
| `GET /props` | model identity and hashes, context, quant, the served path and KV precision (`cache.path`, `kv_dtype`, `kv_block_tokens`, `kv_block_size`, `prefix_cache`, `prefix_cache_mib`), MTP state, sampler defaults and provenance, build info |

`usage.completion_tokens_details` carries `accepted_prediction_tokens` /
`rejected_prediction_tokens`. Console output goes to stderr, one greppable
line per event (`lgc  load:`, `mem:`, `http:`, `slot N: prefill … | decode …`
with graph / embed / sample / emit / wait terms and a stall p95); `-v`, `-vv`.
The stub backend (`--stub`) serves the whole surface without a model for the
device-free suites.

### 4.1 Two lanes (M6)

`--parallel N` (gated at 2). A lane is one sequence's mutable state: its own
`InferRequest`s (language model, embeddings gather, MTP layer and head), GDN
rows, KV block table and output buffers. Compiled models and the page pool
are shared. Sequences are never batched into one graph execution.

The **turnstile** (`src/core/turnstile.h`) orders graph executions FIFO. It
is a correctness mechanism: the GPU plugin pools intermediates per compiled
model, so a request's output tensor is valid only until the next execution
on that model by anyone, and each lane copies its outputs inside its turn
(`measured-here`: a second request adds 0.00 GiB). A decode step waits for
at most one other execution, so the worst stall is one prefill chunk; the
console reports the wait as its own term. The prefill grid is configuration,
never a scheduling variable. Measured on the B60 coder: a second lane costs
1.7 % single-stream decode; agent session + subagent split the card evenly,
stall p95 17 ms, max one chunk; Prüfstand 10/10 on each lane concurrently
(`measured-here`, §7.2).

### 4.2 The name, and the context, are contracts with a proxy

- `--served-model-name NAME` sets the id on `/v1/models`, `/props` and in
  responses; `--model-id` stays the artifact assertion against the
  allowlist. Both names are accepted in a request's `model` field;
  `/props` publishes `model.canonical_id`, `answers_to` and
  `enforces_model_field: false`.
- The `/v1/models` entry carries `n_ctx` (what this process runs with),
  `n_ctx_train`, `quant` and `lanes`, because a discovering proxy reads
  context only from there (`code`; both learned by breaking a proxy).

### 4.3 Admission: a lane is a memory reservation

At startup the fit (`src/exec/fit.h`) reserves, per card and measured where
it can be: weights + graph (`GPU_MEMORY_STATISTICS` after compile, device
and host-mapped ledgers kept apart), drafter residency, MTP state
(8 KiB/token), the activation peak at the chosen chunk (probed upward by
doubling; the plugin's intermediate pool never shrinks, so an over-large
probe is a permanent tax), GDN rows per lane, the expert slot pool
(plateau probe, `source: probe` / `probe-static`), KV, and a margin
(`--fit-margin-mib`, 256 default). With `--n-ctx` omitted it adopts the
largest admissible depth and corrects by a residency audit; an explicit
`--n-ctx` is verify-only and never lowered (`code`, `measured-here`). An inadmissible request is
refused at startup with every term; an N+1-th concurrent sequence gets a
503 with the same terms unless `--queue-timeout S` lets it wait (the
production unit sets 30). `--fit-ledger-dir` persists the probes so a second
matching load skips them (`measured-here`: 14–27 min saved on per-expert
loads). The context ceiling is the depth at which the prefix-cache reserve
reaches zero (§7.0.3).

### 4.4 A host tier for evicted prefixes

`--cache-host-mib N` (0 = off): an evicted prefix entry is demoted (its KV
pages copied to host memory by page runs through `RemoteTensor` ROI views,
references released) and promoted on a hit; one LRU order across both
tiers. Pages come back byte-exact. Measured on the B60: a 4,096-token hit
from the host tier in 0.02 s against a 1.79 s cold prefill
(`measured-here`). Production: `--cache-host-mib 4096` on the agent unit,
sized from the replay's 37 % of prefill spent re-prefilling evicted sessions
(`measured-here`, §7.0.2j).

## 5. Testing and acceptance

**The bar** is the answer-level bar of §3.4 / `CLAUDE.md`, per phase: a
change that measurably improves prefill or decode and does not regress the
other beyond the run-to-run spread is adopted if it passes the answer-level
bar (`decision`, operator, 2026-10-01). The byte-equality cells listed below
(cold/warm, chunked/unchunked, restored continuation, MTP identity,
`tier-reference-cell`) are tripwires: reported, not a veto. The cells' own
text and the generated manifest still say "byte-equality" and are owed an
update to say so (§8).

- **Prüfstand** (the acceptance task: a Lua CSV parser to RFC 4180, ten
  cases, scored by executing the candidate). 10/10 is the bar for the
  production coder artifact, through the deployed package
  (`tests/acceptance/cells.json`, cell `pruefstand`; harness path from the
  run manifest, `ARCINT_ACCEPTANCE_PRUEFSTAND`). The score is a floor, not a
  fingerprint: two different programs can both score 10/10.
- **Equivalence suite** (`tests/equivalence/run.sh`, on a card):
  determinism, warm vs cold, restored continuation, logits slice, drafter
  determinism and acceptance, MTP; `ARCINT_EXTRA_ARGS="--parallel 2"` runs it
  on two lanes; `ARCINT_SKIP_STATEFUL=1` for paged-only models.
- **Concurrency suite** (`tests/concurrency/run.py`): no cross-lane bleed in
  both start orders, both lanes used, cold/warm per lane, the cache holds
  pages, cancellation, admission 503 with numbers, the stall printed.
  Verified red before green.
- **Unit tests** (`ctest -L unit`, device-free, stub build): config and the
  refusal ladder, fit arithmetic, decode accounting, HTTP round-trip
  (`tests/roundtrip.sh`), lane stress, acceptance enumeration consistency.
  Warning-clean under `-Wall -Wextra -Wpedantic -Werror`.
- **Sanitizers**: ASan + UBSan, no-recover, on x86_64 (ASan aborts at
  startup on the aarch64 build container).
- **Plugin unit tests** each patch carries (`ov_gpu_unit_tests` filters; the
  tier cells build standalone without `ENABLE_TESTS`), red-first.
- **KLD against the model's own f32 forward** (Flash-Next):
  `tools/ref_forward_stream.py` captures; `tools/kld_served.py --replay` plus
  `ARCINT_LOGITS_DUMP` and `--no-logits-slice` on the served side;
  `kld_vs_capture.py`. Rows at or above position 2,051 of every capture
  before 2026-09-28 are void (indexer fed garbage); re-capture owed.
- **Performance bars**: B60 coder warm decode ≥ 60 t/s (gated, reference
  66.5); tier cell warm ON decode ≥ 14.8 t/s and ON/OFF ratio ≥ 1.17
  (gated). Prefill is a first-class metric on every card.
- **A test must be able to fail**: the red case runs first.

### 5.1 The test ladder: what runs when

| class | cadence | card time | members |
|---|---|---|---|
| Unit tests | every commit | seconds to minutes | `ctest -L unit` in the stub build, plus the plugin unit tests of any patch that changes |
| Milestone gates | once per milestone increment | one card window | the equivalence and concurrency suites on the configuration the milestone changes, plus its own measurement cell, one process per configuration |
| Acceptance | once per release, before the tag | hours | <!-- BEGIN GENERATED by tools/acceptance_manifest.py; verify with its check mode -->`coder-offload-1lane` gates byte-equality: cold vs warm cache, chunked vs unchunked prefill, one chunk size vs another (reports chunk sweep) (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-offload-2lane` gates the same byte-equality claims as coder-offload-1lane, held at --parallel 2 (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-offload-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `coder-served-large` gates byte-equality: two greedy runs, cold vs warm cache, a restored continuation vs cold, and (draft 4) a copy-the-input prompt (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-served-large-decode` gates byte-identity of the two requests' outputs; the warm second-request decode rate against its reference once filled (reports until then, §8.8) (reports 0.3.0 record (DESIGN §7.0.2ai): 53.4 t/s cold / 69.2 t/s warm decode) (references: decode-warm-2nd 66.5 t/s (gate lower-is-worse at 60.0 t/s); decode-cold-1st 66.6 t/s (report only, not gated); prefill-warm-2nd 2820.8 t/s (report only, not gated); prefill-cold-1st 2812.3 t/s (report only, not gated)); `coder-served-small` gates byte-equality (as coder-served-large), on the 16 GiB card (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-served-small-decode` gates byte-identity of the two requests' outputs; the warm second-request decode rate against its reference once filled (reports until then, §8.8) (reports 0.3.0 record (DESIGN §7.0.2ai): 48.0/49.5 t/s decode) (references: decode-warm-2nd 47.6 t/s (report only, not gated); decode-cold-1st 46.0 t/s (report only, not gated); prefill-warm-2nd 508.9 t/s (report only, not gated); prefill-cold-1st 510.5 t/s (report only, not gated)); `coder-served-small-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `agent-dense` gates MTP identity (equivalence's MTP section) and MTP acceptance > 10%; `agent-dense-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `tier-reference-cell` gates tier ON byte-identical to itself across processes and requests, tier OFF likewise, and E2 (reports ON vs OFF identity and the first divergence; every output's hash; the first process's rates beside the second's) (references: decode-warm-2nd-on 18.2 t/s (gate lower-is-worse at 14.8 t/s); decode-ratio-on-off 1.46 ratio (gate lower-is-worse at 1.17 ratio); decode-warm-2nd-off 12.5 t/s (report only, not gated); prefill-warm-2nd-on 27.9 t/s (report only, not gated); prefill-warm-2nd-off 87.2 t/s (report only, not gated); grouped-fallbacks-on 0 count (gate higher-is-worse at 0 count); decode-cold-1st-off 10.2 t/s (report only, not gated); prefill-cold-1st-off 80.5 t/s (report only, not gated); decode-cold-1st-on 13.0 t/s (report only, not gated); prefill-cold-1st-on 24.8 t/s (report only, not gated); decode-cold-warm-ratio-on 1.4 ratio (report only, not gated)); `ngram-determinism-repeat` gates six fresh processes produce identical output; `depth-ladder` gates the load completes at both KV precisions on both cards (reports t/s per card and precision) (references: prefill-large-u8 1025.5 t/s (report only, not gated); decode-large-u8 20.8 t/s (report only, not gated); prefill-large-u8i4 120.9 t/s (report only, not gated); decode-large-u8i4 45.9 t/s (report only, not gated); prefill-small-u8 621.0 t/s (report only, not gated); decode-small-u8 29.2 t/s (report only, not gated); prefill-small-u8i4 170.3 t/s (report only, not gated); decode-small-u8i4 26.1 t/s (report only, not gated)); `sanitizers` gates zero sanitizer reports (reports device-free build only (ARCINT_OPENVINO=OFF): the 4 rope-precision cases gated on ARCINT_OPENVINO are not instrumented by this cell); `package-build` gates the package build succeeds, is version-stamped, and its RPATH probe passes; a post-deploy smoke run follows the install, outside this cell; `pruefstand` gates 10/10 on the production coder artifact through the deployed package (references: score 10 points (gate lower-is-worse at 10 points)) <!-- END GENERATED by tools/acceptance_manifest.py --> |

Section numbers inside the generated cell list (§8.8 and the like) point
into `docs/design-0.3.1-test-ladder.md`, not into this document. Long
prefills are acceptance work. Within one arm, repeated decode samples
may reuse a warm prefix; with an adaptive tier, state the tier's history.
The arcint unit ladder runs on the exact tree before every tag.

### 5.2 Operating facts that prevent damage

- **Card identity by PCI id**, never by DRM number: DRM numbering is inverted
  against OpenVINO's (`docs/sop-card-window.md`). `GPU.0` = B60, `GPU.1` =
  A770.
- **The cards are full while the production units run.** Stop the owning
  unit for a window and restore it after; two arcint processes on one card
  under xe produce GPU faults (job timeouts, coredumps, resets), not slow
  sharing. Test zombies ignore SIGTERM: sweep by pid with SIGKILL and verify
  with a real completion.
- **A wedged xe card** (GuC job timeouts, an oops in the xe scheduler, a
  process stuck in `dma_fence_default_wait`) has only been recovered by a
  host reboot, and **rebooting the dev host is the operator's call**, every
  time (`decision`). Read the host's captured kernel log first so the
  request carries the kernel's own signature.
- **Forecast host memory before any launch**: visible + driver-side
  (`drm-resident-*` in fdinfo) + page cache / ZFS ARC (absent from
  `MemAvailable`) + the fence. The GPU plugin stages constants in host
  memory at compile: compare constant bytes with host RAM before compiling
  (the first full-depth Flash-Next compile was refused by the host with
  `CL_OUT_OF_HOST_MEMORY`, `measured-here`). A
  `MAP_NORESERVE` bank above free memory ends in an OOM kill.
- **VRAM exhaustion under xe** shows as `VM worker error: -12` and an exec
  queue reset; the reservation is the defence. The runtime's
  direct-submission semaphore can be evicted under VRAM pressure with
  concurrent load (a CAT error at a fixed address); fallback
  `NEOReadDebugKeys=1 EnableDirectSubmission=0` (`measured-here`, §7.0.2ad).
- **The A770's GT clock is pinned at 2000 MHz** (operator decision,
  2026-09-27; host-side oneshot service): 35B decode +14 %, prefill −9 %, the
  coder −5 % decode (§7.0.2cw). Any clock A/B states the pin.
- **One fresh process per measurement cell**: a GPU fault or failed compile
  poisons later cells in the same process. A fresh process's first request
  compiles kernels for its row count (~120–320 ms at 856 tokens); warm
  figures come from request 2.
- **Measure floors on the A770**: the B60's served Flash-Next path is not
  run-to-run deterministic (KL 0.136 nats mean between identical forwards;
  `measured-here`, §7.0.2cb).
- **Verify the artifact carries the change**: `dpkg -V marfrit-openvino`, the
  plugin's sha in every leg header, `LD_LIBRARY_PATH` exported to the
  intended plugin; drop `__pycache__` on every source bend.
- **Clear the GPU model cache** when upgrading across a plugin level that
  changes the option schema (0015 did; `measured-here`, §7.0.2ac).
- **One measurement: at most 2 hours**, forecast from a bounded leg; above
  it, the row is "to be determined" (`decision`, operator).
- Profiles: `PERF_COUNT` reports ~55 % of a kernel's device time and no
  transfers; use the OpenCL intercept device timeline for shares; a past-0
  chunk is the cheapest chunk and overstates every node share.
- Plugin timing tests must allocate operands as `usm_device`
  (`allocate_memory(layout)` is host memory) and warm three executions.

## 6. Why the pipeline layer is rewritten

OpenVINO GenAI's continuous-batching path diverged from its stateful path
under greedy on this hardware (−2/10 on the code task; openvino.genai#4367),
its CB-vs-stateful equality test is skipped upstream, and its hybrid-state
prefix cache checkpoints at memory-tuned intervals; its CB admission refuses
the coder on the A770 where arcint's measured reservation serves it
(`measured-here`). The kernels are OpenVINO's; the state, scheduling and
admission are arcint's.

## 7. State of play

### 7.0 Milestones

| line | content | state |
|---|---|---|
| M0–M6 | skeleton, executor, paged KV + GDN ledger, prefix cache, MTP, all models, two lanes | done (paged path served since 2026-08-29) |
| M7–M14 (0.3.0) | fit pass, u8:i4 KV, offload v2, M10 re-scoped (§7.0.2ah), drafting II, exporter lowering, vision reserved, CPU tier | done; 0.3.1 tagged 2026-09-05 (`+p6` floor) |
| 0.4.0–0.4.7 | GGUF checkpoints on a template IR, the K-quant kernel series, host prefix tier | done (`+p7`…`+p15`) |
| 0.5.0 | Flash-Next serving shape at full depth | released (0.5.0, 0.5.0.1) |
| 0.5.1 BERLIN, 0.5.2 VENICE, 0.5.3 LISBON, 0.5.4 LYON | Flash-Next fill and quality, expert residency, NVMe/host tiers, stateful prefill and QSA | acceptance closed 2026-10-01 under the operator's re-gating, on readable rows (`docs/window-051.md`…`054.md`); 0.5.4 shipped on `+p25` |
| 0.5.5 ROMA | speculation for Flash-Next | next (§8.3) |

Charters live in `docs/milestone-0.3.0.md`, `0.4.0`, `0.4.1`, `0.5.0`;
every open defect or lever since 0.3.1 is a campaign in `docs/campaigns/`.

Current served and development numbers (`measured-here`):

| configuration | prefill | decode | quality |
|---|---|---|---|
| coder, A770 (pinned 2000 MHz), u8, prefill/decode after 4,096 tokens | 1,351 t/s | 44.0–44.3 t/s | Prüfstand 10/10 |
| coder, B60, u8, ~1k prompt, warm | 2,821 t/s | 66.5 t/s | 10/10 |
| dense agent, B60, `u8:i4`, MTP on, 151,552 ctx, 850-token prompt | 1,436 t/s | 23.5 t/s | 10/10 |
| dense agent, same unit, 71.7k-token prompt | 377.5 t/s | 2.2 t/s (MTP at depth, §8.3) | — |
| 35B native all-resident + dispatch, A770, 4,096 tokens | 961 t/s | 28.1 t/s (32.1 pinned) | 10/10 |
| Flash-Next `d48q8`, B60, 20k / 27.6k tokens (§3.9) | 63.1 / 61.2 t/s | 6.5 / 6.6 t/s | needle answered; KL below 2,051 0.319 vs base 0.300 |
| Strata, IQ3_XXS, RTX 5070 + 64 GB (`paper`, README) | 1,750 t/s at 32K | 62 t/s short, 49 at 128K | KL 0.022 against llama.cpp (paper finding 3) |

### 7.0.1 Retractions

A mechanism that was narrated and not measured is retracted on the record
when it falls, rather than edited away (the house rule `CLAUDE.md` cites);
since 2026-10-01 the retracted text lives in git history and this document
carries only the corrected fact. The first such correction, kept as a fact:
decode of a resident model is kernel-bound, and PCIe carries tokens, not
weights (0.3–1.7 % of a decode step; `measured-here`). The bus is the cost
only for state snapshots, host round-trips of experts, and offload streaming.

### 7.0.2 Record index

Dated records that other files cite, one line each: what it established that
is still true. Ids not listed recorded work since superseded or rejected;
their text is in `git show b0447b8:DESIGN.md`.

| id | established | class |
|---|---|---|
| 7.0 | paged decode step 11.3 vs 19.0 ms stateful on the same IR and card (1.68×): the basis for serving the paged path | measured-here |
| 7.0.0a | at 32k the stateful engine beat GenAI CB 2.8× prefill, 1.5× decode (B60 coder) | measured-here |
| 7.0.0b | rooflines are quoted from parameter arithmetic (active bytes / bandwidth), never as measured traffic | decision (method) |
| 7.0.2 | the two cards select different kernels for one graph; a kernel conclusion on one does not transfer | measured-here |
| 7.0.2a | admission by measured reservation; `GPU_MEMORY_STATISTICS` lumps `usm_host` with device memory if summed blindly | measured-here, code |
| 7.0.2b | prefill attention runs micro-SDPA with `dpas` (IGC shader dump, u8 and f16); `exec_type` cannot tell micro from opt; head size 256 sits exactly at the micro-SDPA ceiling | measured-here, code |
| 7.0.2c | served prefill is >99 % graph time; the shared-expert gate at M ≥ 128 is a oneDNN catalog miss (ref kernel) | measured-here |
| 7.0.2d | the OpenCL device timeline is the share instrument; PERF_COUNT sees ~55 % of kernel time and no transfers | measured-here |
| 7.0.2e | logits slice on the paged layout, verified by a probe forward at load: prefill +27 %, 2.09 GiB of activation reservation back | measured-here |
| 7.0.2f | coder decode on its IR (B60, 2026-08-30) was launch-bound in the plugin's per-node host path (~1,130 launches, ~12 µs each; device ~6.5 ms of a ~15 ms step) | measured-here |
| 7.0.2g, 7.0.2i | `--gate-pad N`: the gate on the jit path, prefill −13 % wall, decode −5 %; off by default | measured-here |
| 7.0.2h | decode histogram: 2,315 primitives walked, 1,171 launched; ~4 µs per walked, ~3.5 µs more per launched | measured-here |
| 7.0.2j | prefix-cache replay of real sessions: 97 % of turns hit; 37 % of agent prefill re-prefills evicted sessions | measured-here |
| 7.0.2k | tool-call arguments handed to the template as an object when it requires one | code |
| 7.0.2l | `--cache-grid N`: exact on the paged path, ~0.45 s fixed per continuation | measured-here |
| 7.0.2m | reasoning split into `reasoning_content`; `reasoning_effort` accepted | code |
| 7.0.2n | the reconstructed MTP head pairs with Intel's public Qwen3.8 IR: 90.8–96.3 % acceptance, +36–48 % decode, 10/10 | measured-here |
| 7.0.2o, 7.0.2p | Qwen3.6 MoE MTP head (93.9 / 75.4 %); patch 0003 removes 20,480 subbuffer creations per verify | measured-here |
| 7.0.2q | operator sampler flags and per-response prediction counts | code |
| 7.0.2r | DFlash2 drafter: 44.8 vs 24.0 t/s plain (dense, B60) | measured-here |
| 7.0.2t | two-ledger reservation (device vs host-mapped); auto-fit within 0.02 % of the hand-tuned depth | measured-here |
| 7.0.2u | the served MoE fusion entry is the tiled block (`ConvertTiledMoeBlockToGatherMatmuls`); exporter `--moe-lowering tiled`; `--pin-dispatch` is an opt-in instrument | code, measured-here |
| 7.0.2v | patches 0004–0007: device slot pool, async uploads; 35B on the A770 0.4 → 9.1 t/s | measured-here |
| 7.0.2w | `--paged-kv u8:i4` (patches 0008–0010): 8.8 vs 11.3 KiB/token | measured-here |
| 7.0.2x | CPU tier (0011/0012): 35B A770 ratio 50, 15.0/15.5 vs 10.4/10.6 t/s; the CPU kernel needs a per-source `-O3` | measured-here |
| 7.0.2y | u8:i4 +28 % auto-fit context on both served configurations; vision IRs never loaded; routing histogram (0013) | measured-here |
| 7.0.2z | DFlash block/selector options, the cycle dump and the offline oracle (re-rank headroom ≥ +0.744 accepted/cycle) | measured-here |
| 7.0.2aa | depth table, dense agent, B60, u8: plain decode 22.3/19.9/16.3/12.9 t/s at 9k/38k/76k/143k; DFlash2 window fix (0014) | measured-here |
| 7.0.2ab | VRAM exhaustion signature under xe; the prefill-chunk belt for 4-bit values on the generic kernel | measured-here, code |
| 7.0.2ac | patch 0015: `--paged-attention-max-partitions`; clear the model cache across it | measured-here |
| 7.0.2ad | deep-prompt crash = direct-submission semaphore evicted under VRAM pressure + concurrent load; the reservation is the defence | measured-here |
| 7.0.2ae | patch 0018 static partition (the default tier mode) and its load-time fixes; the LRU mode stays available; adaptive placement is admissible under §3.4 | measured-here, code |
| 7.0.2af | patch 0017: the tier's 283 µs "readback" was the host waiting for the GPU to reach the layer's router; export `ARCINT_MOE_DEVICE_POOL_BYTES` | measured-here |
| 7.0.2ag | drafters' rotary kept f32 (overflow at 65,504 zeroed acceptance); MTP state charged 8 KiB/token; at 77k DFlash 18.8, plain 15.3, MTP 4.9 t/s | measured-here |
| 7.0.2ah | M10 re-scoped: context via u8:i4; the per-expert kernel is `sub4bit-vram-kernel` | decision 2026-09-05 |
| 7.0.2ai | 0.3.0 gate: plateau probe under the static partition; tier ON warm decode 16.4 vs 11.3 OFF | measured-here |
| 7.0.2aj, 7.0.2ak, 7.0.2al | acceptance runner corrections and the filled references (tier 16.4 / ratio 1.31; coder B60 66.5) | measured-here, code |
| 7.0.2am, 7.0.2an | turnstile tests synchronise on `issued()`; roundtrip servers probe their own ports (TIME-WAIT collisions) | measured-here |
| 7.0.2ao | the Prüfstand cell runs through the run manifest and gates the score at 10 | code |
| 7.0.2ap | patch 0019: the per-expert prefill weight answer is three-way | measured-here |
| 7.0.2aq | cold start: ~170 s of load-time probe forwards at tier speed, plus the disk; kernel JIT 7 s; lever landed as `--fit-ledger-dir` | measured-here |
| 7.0.2ar, 7.0.2as | patch 0020: u8:i4 mixed stage on micro-SDPA; prefill parity with u8 at a held chunk | measured-here |
| 7.0.2at, 7.0.2au, 7.0.2av | the micro path allocates no generic partials; its chunk cap is 2,048; 118k tokens served at every chunk to 2,048 on both cards; package floor `+p6` | measured-here |
| 7.0.2aw | 0.3.1 deployed; an explicit `--n-ctx` is verify-only | measured-here |
| 7.0.2ax | agent unit: u8:i4, chunk 512, MTP on, 151,552 tokens | measured-here |
| 7.0.2ay | GGUF stage 1 (patch 0021, `--gguf`), 10/10 | measured-here |
| 7.0.2az | patch 0022: K-quant decode split by architecture | measured-here |
| 7.0.2ba | repack at load into the runtime's compressed form, mins as exact columns, per-type deviation bound | measured-here |
| 7.0.2bc | patch 0023: decode kernel in llama.cpp's mmvq shape, +20 % served decode at 1k | measured-here |
| 7.0.2bd | timing method (warm three executions; cold-kernel-cache stall with direct submission); depth captures walk in served chunks | measured-here |
| 7.0.2be | `--gguf-mode mixed` default, file embedding, `--gguf-check once`, `+p8` | measured-here |
| 7.0.2bf | same GGUF on the B60: arcint mixed 309 / 13.4, llama.cpp SYCL 249 / 14.2, Vulkan 126 / 7.8 t/s at 1k | measured-here |
| 7.0.2bg | decode device timeline: fully-connected kernels at 86–89 % of the 453 GB/s ceiling; logits slice for a K-quant head | measured-here |
| 7.0.2bh, 7.0.2bi, 7.0.2bj | Q6_K decode: read probe, patches 0024–0026 (no shuffles, 224-byte blocks): 400 → 204 µs on the down projection | measured-here |
| 7.0.2bk | patch 0027: the plugin's runtime fusion check accepts the K-quant kernel; 79 queue drains per step gone | measured-here, code |
| 7.0.2bl | `--gguf-mins shared|nibble` (inexact, flags) | measured-here |
| 7.0.2bm, 7.0.2bn | patches 0028–0029: tile in matrix-unit layout, 2D block loads, 64-row tile at 256 GRF on Xe2 | measured-here |
| 7.0.2bo | `--gguf-mins split`; the runtime's int4 gemm on the repacked set runs at 88 % of the f16 matrix roof | measured-here |
| 7.0.2bp | patch 0030: 32-row activation reads; warm 1k prefill 1,001 t/s | measured-here |
| 7.0.2bq | Q5_K served decode row 279 GB/s; open lever a re-blocked Q5_K layout | measured-here |
| 7.0.2br | A770 read ceiling 414–418 GB/s; parallel repack check; patch 0031 (oneDNN deterministic attribute on f16-activation compressed FCs) | measured-here |
| 7.0.2bs | patch 0032: micro-SDPA next-K-tile prefetch out of bounds (upstream PR #37878) | measured-here |
| 7.0.2bt | kernel census of both forms; recurrent-state rows zeroed on the device | measured-here |
| 7.0.2bu | patch 0033: u8:i4 micro-SDPA V alignment from the value precision; page-order test fills | measured-here |
| 7.0.2bv | `/props` reports the served path; patch 0035 test fill; patched vs stock runtime identical output and rate on a matched configuration | measured-here |
| 7.0.2bw | the hidden-state tap knows the K-quant head; MTP on a GGUF-opened model accepts 73 % | measured-here |
| 7.0.2bx, 7.0.2by | patches 0037/0042: hybrid grouped prefill; gather over the filled count | measured-here |
| 7.0.2bz | Flash-Next fill: folds undone at the feed, sigmoid gate, tiled key-head pairing; Paris served | measured-here, code |
| 7.0.2ca | patch 0043: native expert formats; layer 0 corr 0.99991 against llama.cpp | measured-here |
| 7.0.2cb | B60 GDN state output nondeterministic run to run (0.136 nats); A770 bit-identical; reported upstream as a #38099 sibling | measured-here |
| 7.0.2cc | patch 0046: census seed; the served pool is `floor(512·(100−R)/100)` slots | measured-here, code |
| 7.0.2cd | patch 0047: resident-sized slot pool on the per-expert route | measured-here |
| 7.0.2ce, 7.0.2cf | per-expert route rate and card-vs-host arithmetic, measured before patch 0056; re-measurement owed | measured-here |
| 7.0.2cg | staged n-gram table: 2.884 MiB instead of 26.82 GiB, depth-4 output byte-identical | measured-here |
| 7.0.2ch | patch 0049: NVMe → VRAM pinned fill at load (arcwell, B60) | measured-here |
| 7.0.2ci | 35B full depth all-resident on the A770 (0054–0058, `--dense-u8`); q, k and v all compressed are miscomputed by the plugin's horizontal fusion, so k/v and the shared expert stay plain | measured-here |
| 7.0.2cj, 7.0.2ck, 7.0.2cl, 7.0.2cm, 7.0.2cn, 7.0.2co | 35B native prefill @4096: 12.5 → 143.9 (0059) → 222.9 (0060) → 349.9 (logits axis, CPU embeddings in RAM) → 625.7 (0061) → 653.7 (0062) → 952.1 t/s (0064) | measured-here |
| 7.0.2cp, 7.0.2cq | patches 0065/0066: Flash-Next prefill 1.04 → 6.4 t/s (A770) | measured-here |
| 7.0.2cr, 7.0.2cs | 40 per-layer routing waits; patch 0067 routes decode on the device: 19.3 → 28.1 t/s | measured-here |
| 7.0.2ct, 7.0.2cv | A770: served dense decode GEMMs run up to 3.7× their isolated time; after a ≥ 20 MiB read-only kernel the next GEMM is slower; open | measured-here |
| 7.0.2cu | the tier-off offload rate depends on the host page cache; a ctest allow-list fix | measured-here |
| 7.0.2cw | A770 clock pin 2000 MHz: frequency transitions removed | measured-here |
| 7.0.2cx | patch 0068: Flash-Next decode 1.8–2.2× (A770) | measured-here |
| 7.0.2cy | one Flash-Next compile in 40.1 s, under the 45.2 s bound | measured-here |
| 7.0.2cz | the partition keys on `weight_0`'s `.bin` offset; equal resident sets give equal bits; the in-graph IQ4_NL scale decode made f16-exact | measured-here, code |
| 7.0.2da | patch 0072: host expert bank, first answer −18…−21 % (B60) | measured-here |
| 7.0.3 | KV precision, below | measured-here |
| 7.1 | the 35B runs on the A770 through `OFFLOAD_RATIO`; superseded in rate by the native all-resident route | measured-here |
| 7.2 | two lanes, below | measured-here |

### 7.0.3 KV precision on the paged path

`measured-here`, coder unless named:
- u8 costs 11.3 KiB/token against f16's 20.0 (0.565×: per-block scales); at
  262,144 tokens 2.83 against 5.00 GiB.
- Decode: u8 +2.5 % at 32k, f16 +7.8 % at 53.5k (the crossover between them
  is not located). Prefill: u8 +3.1 % at 14k, −16.5 % at 58k, −22 % at 115k.
- Symmetric u4 quarters the memory and costs 6 % decode at 32k (the attention
  kernel +63 %).
- u8:i4 on `+p6`: prefill at u8's rate at a held chunk; context +28 %.
- **Default u8**, chosen on what fits rather than what benchmarks: it is the
  setting that does not refuse (B60, two lanes at 262,144: u8 22.04 of
  22.71 GiB, f16 refused). `--paged-kv f16` suits a one-lane deep-context
  endpoint with room.
- The pool is sized in bytes and live pages are a count
  (`n_ctx / 16 + 2` per lane), so everything that costs bytes comes out of
  the prefix-cache reserve; the context ceiling is the depth where that
  reserve reaches zero. Auto-fit leaves no reserve unless
  `--prefix-cache-reserve PCT` asks for one.

### 7.1 The 35B on the A770

`--offload-ratio` (the plugin's `OFFLOAD_RATIO`) first made the 17.4 GiB
int4 35B load on the 15.1 GiB card; with the device slot pool and the CPU
tier it serves at 15–18 t/s decode (§7.0.2v, §7.0.2x, §7.0.2ai); the native
packed-u8 artifact serves it all-resident at 28.1 t/s decode and 961 t/s
prefill (§7.0.2cs). `measured-here`.

### 7.2 Two lanes, measured (M6)

B60 coder, u8, `--parallel 2` (`measured-here`): the second lane's
activations cost 0.001–0.003 GiB (pooled per compiled model); single-stream
decode 67.6 vs 68.8 t/s (−1.7 %); an agent session plus a subagent: 32.5 and
31.1 t/s, stall p95 17 ms, max 516 ms (one 309-token chunk); Prüfstand 10/10
on each lane concurrently. Admission on the A770 at two lanes: 44,608
tokens per lane at `n_ctx` 8,192 / chunk 256, refusal with the terms at
65,536.

## 8. Open items, against the references

Each item names the reference implementation to follow (paths verified in
`~/src/Strata-ref` at `c499bd1` and `~/src/FreeToken-ref`), the reference's
measured effect, and arcint's measured starting point. Where arcint built
something differently before, the item says what to match; it does not
carry a verdict.

### 8.1 Let the GPU expert cache learn the conversation

- **Reference.** Strata: routing census per layer, every 4 rounds up to 96
  gain-ranked swaps, the old expert evicted at once (the CPU computes it
  meanwhile) and the new one admitted when its copy lands
  (`src/program/generate.cpp`: `adapt_every` :358, `adapt_swaps` :380,
  `apply_pending` :4414–4478); slot storage, residency table and the
  byte-compare `verify_slot` in `src/core/expert_cache.cpp`
  (`include/strata/core/expert_cache.hpp`: `open_sized` for per-layer slot
  bytes, `set_per_layer_admission`). FreeToken: one slot pool shared by all
  layers, flat id `layer · E + expert`, slots rewritten on the GPU
  (`python/freetoken/moe/offload_cache.py:169–184`, `ensure_experts` :843).
- **Effect** (`paper`, Strata §6): profile fill 0.50 → adaptive 0.72 hit
  rate (finding 4); no-wait admission 91.7 → 94.4 t/s (finding 10); slots
  sized in bytes per layer +13 % decode (finding 8).
- **arcint now**: static partition + census seed; 36 % GPU hits, 308 tier
  experts per decode token on Flash-Next `d48q8`, B60 (`measured-here`). An
  offline replay of arcint's own routing: per-layer LRU 55.6 / 69.2 % against
  the census partition's 21.2 / 38.5 % at 32 / 64 slots per layer; one pool
  shared across layers 93.8 % at 16 GiB (`measured-here`,
  `docs/campaigns/expert-hot-set-lru.md`). In progress (campaign
  `expert-hot-set-lru`).
- **Prerequisites owed**: a partition layer key independent of the file
  layout (decoder index; §7.0.2cz), and the `tier_prefix_cache_decision`
  change (§3.4).

### 8.2 Read prompts on the GPU, streaming the missing experts

- **Reference.** Strata: from chunk 1,024 every non-resident expert of every
  layer streams in a fixed order through a ring of slots borrowed from the
  expert cache, copies overlapping compute, quantised (MMQ) grouped kernels,
  chunks up to 8,192 (`src/prefill/prefill.cpp:71–102`,
  `include/strata/prefill/moe_mmq.hpp`; the comment there records 384 ring
  slots at 8,192-token chunks at 1,294 t/s). FreeToken: two whole-layer
  buffers, the next layer copied on `prefill_copy_stream` with ready/release
  events while the current one computes
  (`python/freetoken/moe/offload_cache.py:606–691`,
  `python/freetoken/layers/moe.py:344–370`).
- **Effect**: Strata IQ3_XXS 1,750 t/s at 32K (`paper`, README); prompt
  speed flat from 4K to 262K, 539 → 496 t/s on Q2_0 (`paper`, finding 11).
- **arcint now**: 61–68 t/s at 20–27k on Flash-Next `d48q8`, B60; the CPU
  tier takes ~138–152 ms per MoE layer call at chunk 512 (`measured-here`).
  The campaign's first build used CPU-thread copies into separate pinned
  staging beside the bank, one queue, no prefetch and chunk 512 (2,048
  crashed, uninvestigated); match the references instead: a batch-sized ring
  borrowed from the slot pool so the bank keeps the whole host tier, a second
  queue with cross-queue events, the next layer prefetched, chunk ≥ 2,048
  (`docs/campaigns/prefill-expert-streaming.md`). Depends on §8.4.

### 8.3 Draft several tokens with the model's own MTP layer

- **Reference.** Strata fetches the 31 `mtp.*` tensors from the BF16
  checkpoint (`tools/mtp_fetch.py`), drafts up to 3 tokens kept while the
  MTP layer is ≥ 50 % confident, verifies them in one window through all 48
  layers, and lets the MTP layer attend only to the last 32,768 positions
  (`include/strata/core/mtp.hpp`, `src/core/mtp.cpp`; `paper` §3.3, §3.7).
- **Effect**: 1.6–1.8× decode with CPU-held experts, 47–57 → 82–92 t/s at
  4K (`paper`, finding 2); speculative output token-for-token equal to
  greedy (finding 3).
- **arcint now**: no Flash-Next MTP head is exported (ROMA R1; the head is
  in the original checkpoint's index). On the dense agent arcint drafts one
  token per cycle and the MTP layer keeps f32 state over the whole prompt
  (8 KiB/token): at 77k the cycle is 390 ms against a 130 ms break-even
  (4.9 t/s against plain 15.3), and the served unit decodes 2.2 t/s at 71.7k
  (`measured-here`; campaign `mtp-cycle-wall`). Build multi-draft MTP with a
  bounded MTP attention window; use DFlash2 at depth meanwhile.

### 8.4 A pinned host bank

- **Reference.** FreeToken fills host banks by chunked O_DIRECT and then
  pins them (`cudaHostRegister`), and only pinned banks feed the GPU
  movement paths (`python/freetoken/moe/host_banks.py`); Strata keeps every
  expert in one pinned RAM arena (`include/strata/core/pinned.hpp`
  `PinnedArena`, `src/core/pinned.cu`).
- **arcint now**: the bank (patch 0072) is anonymous pageable memory; the
  host's TTM `pages_limit` is 8,220,668 pages = 31.4 GiB against a 44–46 GiB
  bank (`measured-here`). Raising the TTM limit is a host setting: the
  operator's decision. §8.1, §8.2 and §8.6 depend on it.

### 8.5 Take the host out of the per-layer hand-off

- **Reference.** Strata's doorbell: a kernel writes x, the expert ids and the
  weights into mapped pinned memory with a system fence, the CPU spins on it,
  a GPU wait kernel polls a host flag, and the whole pass is a recorded graph
  replayed per window (`src/core/layer.cpp`: `doorbell_init`, `moe_route`;
  `src/kernels/cuda/elementwise.cu:206–311`;
  `src/kernels/cuda/verify_kernels.cu:426`; capture and replay in
  `src/core/verify.cpp:889–1058`). FreeToken's GPU-slot decode is device-side
  with fixed shapes and graph-capturable
  (`python/freetoken/layers/moe.py:255–268`). Intel equivalents:
  `cl_khr_command_buffer`, Level Zero command lists (the pinned plugin's L0
  runtime creates immediate lists only, `code`, so a replayable pass is
  plugin work).
- **arcint now**: on the Flash-Next tier route the GPU idles ~46–50 % of
  decode wall; per token 47.2 ms waits on the tier's writeback and 31.3 ms
  is readback/staging round trips (`measured-here`). Device-side routing on
  the all-resident 35B gave +45 % decode (§7.0.2cs). Patch 0075 (not in the
  series) removed only the writeback copies; the x/id readback hops and the
  per-layer host block remain. Build the doorbell shape on the tier route.

### 8.6 Choose CPU or a copy for each miss from a measured speed comparison

- **Reference.** FreeToken measures CPU-MoE kernel bandwidth against PCIe
  gather bandwidth and picks `hybrid` only when the CPU is > 2× the link,
  otherwise `offload`, with a per-step fetch fraction
  (`python/freetoken/moe/benchbw.py:1–19`); fetched experts stay cached.
  Strata issues miss DMAs from the CPU thread at plan time from the pinned
  arena (`include/strata/core/verify.hpp`: `set_pcie_mode` :154,
  `fetch_dma` :228).
- **Effect** (`paper`, Strata finding 9): 55 % of misses over PCIe for
  i-quants, IQ3_XXS 56 → 65 t/s; 20 % for the RAM-bound Q2_0.
- **arcint now**: on the B60 the link moves an expert in ~0.19 ms and the
  tier computes one in ~0.15 ms (`measured-here`), so FreeToken's rule
  selects offload on this host. The campaign's build copied from pageable
  memory into transient slots overwritten every step with a fixed K = 3;
  match the references: pinned source (§8.4), fetched experts cached in the
  pool (§8.1), the share set from a benchmark
  (`docs/campaigns/hybrid-expert-fetch.md`).

### 8.7 Keep conversation state between requests

- **Reference.** Strata parks conversations (GDN state, PLE tails, KV) and
  reuses them on token equality, beside its adaptive expert cache
  (`include/strata/core/conversation_cache.hpp`, `conversation_snapshot.hpp`;
  README: follow-ups read only what is new). FreeToken snapshots recurrent
  state at tool-call anchors (`python/freetoken/scheduler/cache.py:151`
  `snapshot_toolcall_anchor`).
- **Effect**: the largest single improvement for agents (`paper`, Strata §7:
  a 30-second re-read of a long chat becomes a fraction of a second).
- **arcint now**: the prefix cache and its host tier exist (§3.4, §4.4); with
  the CPU tier the code still refuses the pair; restart persistence is the
  `kv-checkpoint-restore` campaign (backlog).

### 8.8 The CPU tier's arithmetic

- **Reference.** ggml dots in the quantised domain against an activation
  quantised once per token (`vec_dot_iq3_xxs_q8_K`, `vec_dot_iq4_nl_q8_0`,
  read in llama.cpp's source for `docs/campaigns/host-expert-bank.md`;
  llama.cpp is not checked out under `~/src`), with a batched path for
  prefill; Strata: decoding once for several tokens 2.0–2.4×, i-quants
  ~5 GB/s per core (`paper`, finding 7).
- **arcint now**: patch 0074 brings the decode-shaped call to 651 µs per
  layer against llama.cpp's `mul_mat_id` at ~540 µs (1.21×); prefill calls
  keep the f32 path at ~152 ms against 116.5 ms per chunk-512 call (1.30×)
  (`measured-here`). Port ggml's batched prefill dot. Worker spin before
  sleeping measured +6.4 % prefill with decode unchanged (0075's spin arm,
  `measured-here`) and passes the per-phase gate: re-land it on its own.
  Per-core pinning: llama.cpp at cold DRAM is 10–30 % faster on 8 physical
  cores than 16 SMT threads (`measured-here`).

### 8.9 Sparse attention served as the reference serves it

- **Reference.** Strata serves Qwen Sparse Attention natively: indexer,
  selection and sparse decode/prompt attention kernels
  (`src/kernels/cuda/native_qsa_indexer.cu`, `qsa_select.cu`,
  `qsa_decode_attn.cu`, `qsa_prompt_attn.cu`), the indexer appended at
  commit so speculation works with it (`src/core/verify.cpp:990`).
- **Effect**: prompt speed flat to 262K (finding 11); long context costs
  VRAM, not compute (finding 12) (`paper`).
- **arcint now**: QSA serves correctly but non-default through patch 0073
  (`d48q8qsa`): decode 0.87× dense, prefill 1.27× slower, `n_ctx` ≤ 32,768
  under a fixed block cap, speculation and the prefix cache refused with it
  (`measured-here`; `docs/campaigns/qsa.md`).

### 8.10 Other open items

- **Partition layer key** independent of the `.bin` layout (plugin; moves
  every recorded static-partition digest and the seed key space;
  `measured-here`, §7.0.2cz).
- **The KLD reference re-capture** with the fixed BF16 feed; until then rows
  ≥ 2,051 and the `d48q8` served KL/argmax row are owed (`measured-here`;
  `qsa` T8).
- **§5's cells and generated manifest** restated to the answer-level bar
  (`decision`, 2026-10-01).
- **B60 GDN run-to-run nondeterminism**: mechanism open at execution level in
  the GDN state output (`measured-here`;
  `docs/campaigns/served-prefill-determinism.md`).
- **A770 dense GEMM slowdown in the served graph** and the post-footprint
  penalty (`measured-here`, §7.0.2ct, §7.0.2cv); per-kernel hardware
  counters not yet obtained.
- **Horizontal fusion of compressed q/k/v** in the plugin miscomputes; k/v
  stay plain until its mechanism is found (`measured-here`, §7.0.2ci).
- **Per-expert dispatch readings before patch 0056** to re-measure
  (`measured-here`, §7.0.2ce–cf).
- **Hybrid prefill descriptor M** keyed on the routing-dependent filled count
  can rebuild grouped primitives per prompt; bucket it (`code`, §7.0.2by).
- **Plugin framework**: `primitive_inst::realloc_intermediates` changes an
  intermediate's identity without forcing an argument rebind (`code`; patch
  0015 carries a local fix); upstream.
- **GGUF path**: a re-blocked Q5_K layout (§7.0.2bq); an 85-token prompt
  alternates between two texts on the GGUF forms (unattributed;
  `measured-here`).
- **KV codec beyond u8/i4**: NInfer's int8 group-64 codec with a fused
  256-wide Hadamard pre-rotation (`code`:
  `~/src/ninfer/src/ops/kv_cache/int8_g64_codec.cuh`, `hadamard_d256.cuh`);
  a custom kernel here ships with a fusion-impact profile of the surrounding
  graph.
- **Direct-submission fault**: the kernel-side ring-ordering fix
  (`drm/xe` "Order ring writes before ring tail updates", in 7.1.y stable;
  `code`) on the dev host is the operator's decision.
- **Multimodal**, an Anthropic Messages adapter and a local CLI: not
  committed (`decision`); the request parser rejects non-text content parts.

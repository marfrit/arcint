# arcint

A deliberately narrow LLM inference engine for Intel Arc GPUs.

arcint runs exactly three model families on exactly two cards, and tries to do
that better than the general-purpose engines do. The inspiration is
[NInfer](https://github.com/Neroued/ninfer), a from-scratch engine that
supports two checkpoints on one GPU and beats every generalist on that pair.
arcint translates the idea to Intel: kernel work is delegated to OpenVINO's
compiler stack, which already emits good Xe code, and arcint owns everything
around the compute graph — the serving loop, the scheduler, the KV and
recurrent-state memory, prefix caching, and speculative decoding.

All three target models are hybrids: most layers use linear attention
(GatedDeltaNet), a minority use full attention. Most of the design follows
from that; see [DESIGN.md](DESIGN.md). [llm.txt](llm.txt) is the
machine-readable summary.

## Building

C++20, CMake, no network at build time. `third_party/` holds the two vendored
single headers (cpp-httplib, nlohmann/json) and
[models/allowlist-raw.json](models/allowlist-raw.json) holds the IR metadata
the allowlist is pinned against.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure

    ./build/arcint --stub --port 8090 -v

This builds the **stub**, which serves the HTTP surface without a model and is
what the test suite runs against. In this configuration `--model` is refused
at startup instead of starting something that cannot run.

A build that serves a model needs OpenVINO:

    cmake -S . -B build-ov -DARCINT_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ov -j"$(nproc)"
    cmake --install build-ov --prefix ~/.local

`-DARCINT_WERROR=ON` gives the warning-clean build CI should use. Pass
`-DARCINT_GIT_SHA` whenever the build tree has no `.git`; without it
`--version` and `/props` report `unknown`.

**Packaging.** `contrib/packaging/` holds the Debian recipes this project is
deployed with, including `build-openvino.sh`, which builds the pinned OpenVINO
with the patch series the measurements below depend on. No `.deb` is published
anywhere; the directory contains everything needed to build the same thing,
and it is the shortest path to reproducing a number.

## Measured

One task, one model, four engines. The acceptance task is a Lua CSV parser to
RFC 4180: ten named cases (CRLF and bare LF, a missing final terminator,
quoted fields, an embedded comma, a doubled quote, an embedded newline, empty
and empty-quoted fields, no trimming). The candidate code is **executed**, not
read; one point per case. Greedy where the model tolerates it, otherwise the
model card's own sampling defaults.

Same artifact (the Qwen3.6-27B-A3B-Coder), same card, only the serving
pipeline changes:

| engine | card | weights | decode | task |
|---|---|---|---|---|
| OpenVINO GenAI, stateful pipeline | B60 | int4 AWQ (expert-pruned 184/256) | ≈43 t/s | 10/10 |
| arcint, stateful executor (`--no-paged`) | B60 | same IR | 51.4 t/s | 10/10 |
| **arcint, paged executor, u8 KV** | B60 | same IR | **71.3 t/s** (68.6 at f16 KV) | 10/10 |
| arcint, paged, at ~30k context | B60 | same IR | 70.1 t/s | 10/10 |

For scale, the same model on the other card and the other engine. Different
quantisation and different card, so read it as a rough bound rather than a row
of the series above:

| engine | card | weights | decode | task |
|---|---|---|---|---|
| llama.cpp, SYCL | A770 | GGUF Q4_K_M | 14.4 t/s | 10/10 |

The 70.1 t/s row at ~30k context matters as much as the peak: the usual
throughput collapse with depth is absent from the served path, a property of
the paged block tables. Every arcint row is additionally gated by
byte-equality tests: warm cache against cold, one lane against two, paged
against stateful.

**Speculative decoding on the dense Qwen3.8** (B60, int4, u8 KV, greedy,
`--repetition-penalty 1.0`, 400 tokens, 32768 context unless stated). Same
prompt, same server binary, only the drafter changes:

| drafter | decode | acceptance | max context (reservation) |
|---|---|---|---|
| none | 24.0 t/s | — | 199,712 |
| MTP head (`--mtp on`) | 33.0 t/s | 76.7%, 1 draft per pass | 155,680 |
| **DFlash2 int4 (`--dflash`)** | **44.8 t/s** | 3.13 tokens per verify cycle | 136,640 |
| DFlash2 int4, draft on the A770 | 39.8 t/s | 3.13 tokens per verify cycle | 171,904 |

The DFlash2 drafter is the public block-diffusion head
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
exported with `tools/export_dflash.py` and drafting seven tokens per verify
pass through the same checkpoint-row verify the MTP head uses. On one card it
costs ~63k tokens of context headroom; parking the draft on the A770
(`--dflash-device`) buys 35k of that back for 5 t/s of PCIe round-trips, with
output byte-identical to the same-card run. The DFlash2 rows were taken with prompts under 2,048 tokens: the exported
head's state variable is fixed at 2,048 rows and the drafter disables itself
on longer prompts (known defect, 2026-09-03; the drafter-on acceptance task
scores 10/10 at that depth). At production context (155648) the MTP
head serves at 36.2 t/s and 93.2% acceptance, 10/10 greedy, on short
prompts; with prompts of 76k tokens and more its acceptance falls to zero and
decode to 1 t/s (known defect, 2026-09-03; the cause is not yet measured —
by code reading the reconstructed MTP layer keeps an unpaged state and a
dense mask over the whole context). Prefill on the
coder reaches ~1970 t/s.

**Long context against the short-prompt numbers** (2026-09-03, `DESIGN.md`
§7.0.2aa; a real document truncated to depth, 400 greedy tokens, prefix cache
off, one process per depth). Dense 27B agent on the 24 GB card, u8, decode
t/s: plain 22.3 / 19.9 / 16.3 at 8.9k / 37.7k / 76.4k tokens; MTP 25.7 /
12.6 / 1.0 (acceptance 89% / 79% / 0%); DFlash2 19.8 / 11.0 / 5.3 (1.69 /
1.43 / 1.00 tokens per cycle, measured on the unreleased build that carries
plugin patch 0014 and the capped, recoverable drafter). Both drafters are
below plain decoding at every one of these depths and accept nothing at
76k; the short-prompt rows above are short-prompt rows. Coder on the 16 GiB
card: u8 and u8:i4 decode at parity to 72k tokens (40.3 vs 39.2 t/s) while
the u8:i4 prefill costs +7% / +25% / +72% at 8.9k / 37.7k / 71.7k.

**Expert offload and the host tier on the 16 GiB card** (35B int4, u8 KV,
one lane, n_ctx 65,536, 64 greedy tokens). The host compute tier
(`--moe-cpu-tier`) computes capacity-miss experts on the CPU instead of
uploading them over PCIe:

| device pool | upload path | host tier |
|---|---|---|
| ratio 50 / 8 GiB | 10.4/10.6 t/s | **15.0/15.5 t/s** |
| ratio 75 / 5 GiB | 7.4/7.5 t/s | **14.1/14.8 t/s** |

Text output byte-identical over 64 greedy tokens between the two paths at both ratios; acceptance
task 10/10 at ratio 50 / 8 GiB (the ratio-75 cell has the text identity, no
acceptance run).

**Context by KV precision.** (Known defect, 2026-09-03: a 141,902-token prefill at u8:i4 on the 16 GiB card ran the GPU out of resources; prompts to 8,909 tokens prefill at every depth setting tried, the failing depth between the two is being bracketed; an earlier "over 2,048 tokens" statement here was a window-design artifact and is retracted.) `--paged-kv u8:i4` (asymmetric key/value
precision) against the `u8` default, same artifact, same card:

| model | card | u8 auto-fit | u8:i4 auto-fit | gain |
|---|---|---|---|---|
| coder | 16 GiB card | 133,456 | 171,312 | +28% |
| dense 27B agent, MTP on | 24 GB card | 155,376 | 199,424 | +28% |

u8:i4 costs 8.8 KiB/token against u8's 11.3, and scores 10/10 on the
acceptance task. Owed before it becomes the default: the prefill price, and
prefix byte-exactness.

## Scope

**Models** (these, and nothing else). Geometry read off the IRs on 2026-08-28
and kept in [models/allowlist-raw.json](models/allowlist-raw.json); all three
are `full_attention_interval = 4`, 262144 context, and share one tokenizer:

| model | architecture | layers (GDN + attn) | experts | weights | quant |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | hybrid GDN + attention, MoE | 40 (30 + 10) | 256 | 17.4 GiB | q4, q8 |
| Qwen3.6-27B-A3B-Coder | hybrid GDN + attention, MoE | 40 (30 + 10) | 184, pruned from 256 | 12.8 GiB | q4, q8 |
| Qwen3.8-27B | hybrid GDN + attention, dense | 64 (48 + 16) | dense | 13.4 GiB | q4, q8 |

**Hardware** (these, and nothing else):

| card | VRAM | notes |
|---|---|---|
| Intel Arc A770 (DG2/Alchemist) | 16 GB | ~560 GB/s; the 35B needs `--offload-ratio` here |
| Intel Arc Pro B60 (BMG G21) | 24 GB (~22.7 usable) | ~456 GB/s |

## Where the artifacts come from

**arcint loads an OpenVINO IR directory, and nothing else.** Concretely:
`openvino_language_model.{xml,bin}`, `openvino_text_embeddings_model.{xml,bin}`,
the tokenizer and detokenizer IRs, `config.json` and the chat template. A GGUF
will not load. Neither will GPTQ or NVFP4 safetensors — those are llama.cpp
and vLLM formats, and OpenVINO does not read them.

All three models are downloadable, and every number in the *Measured* section
was taken on the published copy. Two are Intel's own exports and need no work;
the coder had to be built because its checkpoint is a community fine-tune with
no official IR.

| model | where to get it | note |
|---|---|---|
| Qwen3.6-35B-A3B | [`OpenVINO/Qwen3.6-35B-A3B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov) | Intel's export, used as-is. Verified byte-identical to the copy these measurements ran on. |
| Qwen3.6-27B-A3B-Coder | [`marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov`](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov) | Apache-2.0. No official IR exists for this checkpoint; the calibration section below records what it cost. |
| Qwen3.8-27B | [`OpenVINO/Qwen3.8-27B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov) | Intel's export. **The MTP head is not in it**: reconstruct it with `tools/export_mtp.py`, or run without `--mtp`. |

**The allowlist keys on the artifact's directory name**, so a download has to
land in the directory the entry names — `qwen36-35b-a3b-int4-ov`,
`qwen36-coder-b5-ov`, `qwen38-b7c1-ov` respectively:

    hf download OpenVINO/Qwen3.6-35B-A3B-int4-ov \
        --local-dir /models/ov/qwen36-35b-a3b-int4-ov

An entry asserts geometry, quantisation and a measured status for one
artifact, and the directory name is the handle it is asserted through. A
refusal at load time means the artifact is not the one the entry describes.

**`OpenVINO/Qwen3.8-27B-int4-ov` is not allowlisted.** The dense entry
describes this project's own AWQ-only export, whose greedy behaviour was
measured. Intel's export has a different calibration and no measurement behind
it; aliasing it onto an entry that claims 10/10 would make the allowlist
assert something nobody checked. Serving it needs its own entry, which means
measuring it first.

Two findings decide whether you need the rest of this section at all:

- **Try the official IR before exporting anything.** The 35B above sat unused
  for weeks on the assumption that a stock export would not clear the quality
  bar. When it was measured it scored 10/10 on the acceptance task greedy, 3/3
  on tool calls, clean German, at 62.7 t/s — better than the artifact it was
  being compared against. Export only when no official IR exists.
- **The dense model's MTP head is in every checkpoint and in no published
  IR.** optimum-intel drops it on export, so neither Intel's IR nor yours
  will have it (llama.cpp's GGUFs keep it, as block 64).
  `tools/export_mtp.py` reconstructs it from the checkpoint's own weights;
  it is worth 1.35× on the dense model and is the only reason to touch the
  export pipeline for that model.

If you only want to run arcint, take the table and stop reading here. The rest
of this section is for anyone who needs a different calibration than the
published ones provide.

**There is no shortcut from GGUF.** llama.cpp gained an OpenVINO backend in
early 2026, and it was built and tested against these models: the production
GDN hybrid aborts during scheduling because the recurrent DeltaNet states are
not mapped in the GGML frontend (`pre-allocated tensor (cache_r_l0) in buffer
(OPENVINO0) that cannot run CPY`), and a classic MoE falls to a CPU path and
asserts in `get_rows`. Its validation list is dense models only. The path
starts at the Hugging Face checkpoint, not at a quantised file you already
have.

    pip install "optimum[openvino]" nncf accelerate pillow "huggingface_hub[cli]"
    pip install "transformers==5.2.0" "openvino==2026.3.0" torchvision

    optimum-cli export openvino --model <checkpoint> \
      --task image-text-to-text --weight-format int4 --group-size 64 \
      --awq --scale-estimation --dataset <corpus> --num-samples 32 out/

### Choosing the calibration

The calibration decides whether the artifact is usable, and it has to be
measured against the task you actually serve. Same model, same bit width, only
the calibration changing, scored on the acceptance task (greedy, and three
sampled runs at the model card's own settings):

| export | greedy | sampled |
|---|---|---|
| naive int4, group 64, data-free | 0/10 | 8, 8, 8 |
| AWQ + scale estimation, image dataset | 7/10 | 10, 7, 0 |
| AWQ + scale estimation, code corpus | **10/10** | 10, 8, 8 |

Two findings from that series generalise beyond the recipe:

- **Scale estimation is model-class dependent. Never set it blanket.** It is
  part of the 10/10 recipe for the MoE coder, and it destroys the greedy path
  of the dense 27B: 0/10 with two entirely different corpora, degenerating
  into repetition loops, while AWQ-only on the same model is a healthy 7/10.
  Measure AWQ+SE against AWQ-only per model before believing either.
- **Calibration cuts both ways.** The code-and-English corpus that bought
  10/10 on code produced token salad in German prose (invented compounds, CJK
  characters mid-word) where the GGUF baseline was clean. What is not in the
  corpus is what you lose. Pick the corpus to match the distribution the
  endpoint will actually see, and say so on the artifact.

### Traps that cost a day each

- **`--task text-generation` does not export this architecture.** `qwen3_5_moe`
  exports only as `image-text-to-text`, the same shape Intel's own IRs use.
- **`--ratio` must stay 1.0.** Mixed precision makes group sizes non-uniform
  and the GPU's MoE fusion refuses to load the result. Every published Intel
  IR for these models is ratio 1.0 for the same reason.
- **Export on a stable OpenVINO, not a nightly.** An IR produced by a nightly
  segfaults on a stable runtime; a stable IR runs fine on a newer runtime.
- **Scale estimation with code samples is memory-hungry** in a way image
  datasets do not prepare you for: it OOM'd on a 121 GB machine, thrashed on
  247 GB (RSS 244 GB, 99.7% iowait), and completed on 494 GB with a ~152 GB
  peak. `TMPDIR` needs ~110 GB — the fp16 intermediate and the final save both
  land there, and running out produces `basic_ios::clear: iostream error`
  after the entire compression has already finished.
- **Do not use `ov::cache_dir` with these MoE IRs.** The first run writes the
  blob, the second imports it without expert weights and fails at inference
  ("expert weight provider not initialized"). Tracked upstream as
  openvinotoolkit/openvino#37607.

### Registering it

arcint refuses artifacts outside its allowlist at load time, so a new export
needs an entry in [models/allowlist-raw.json](models/allowlist-raw.json):
geometry, quantisation, and the tokenizer and chat-template hashes. This is
the mechanism behind "no model zoo" in the non-goals: the allowlist asserts
that the process is serving what it claims to serve.

## Features

Marked with the milestone that delivered them. **[M0]** means it works against
the stub backend; **[M1]**–**[M6]** mean it runs against the real models on a
real card.

- **[M0]** OpenAI-compatible **`/v1/chat/completions`** and
  **`/v1/completions`** over HTTP.
- **[M0]** **`/health`** (liveness, model-loaded, queue depth) and **`/props`**
  (model metadata, context length, quant, cache configuration, build info,
  sampler defaults with their provenance).
- **[M3]** **Prefix caching** for the full hybrid state — attention KV pages
  and GDN recurrent state — with a hard invariant: greedy output with a warm
  cache is byte-identical to greedy output with a cold one. Cache reuse that
  changes the answer is treated as a bug.
- **[M2]** **Block-aligned GDN state checkpoints** for the linear-attention
  layers, so a cache hit restores both halves of the hybrid state exactly.
- **[M4]** **Speculative decoding.** Verification is exact: a drafted token is
  accepted only when it equals what the sampler would have picked anyway. The
  answer can still differ from plain greedy, because a multi-token verify pass
  and a single-token pass differ by up to 0.013 in the logits on this backend,
  enough to flip a near-tie — measured at about one token in 64. When such a
  flip lands early, everything after it is a different draw. On one of three
  probe tasks the speculative answer was a different and worse program (3/8
  against 5/8 on an executed harness). Gate it on your own task, and serve
  without a drafter if you need bit-exact reproducibility.

  Three drafters, one per server: the **native MTP head** for Qwen3.8
  (`--mtp on`, one draft per pass), the **DFlash2 block-diffusion head**
  (`--dflash DIR`, seven drafts per pass, exported with
  `tools/export_dflash.py`; `--dflash-device` parks it on the other card with
  byte-identical output), and a prompt-lookup drafter (`--draft N`). All
  engage only under greedy, since acceptance is defined as equality with the
  sampler's pick; a sampled request runs at the plain rate. Measured numbers
  are in the *Measured* section.

  All drafters are off by default. On a **stock** OpenVINO build the MoE
  models lose with a drafter on, and the cause is in the plugin rather than in
  the rollback this README once blamed: at `token_num > 1` the plugin's MoE
  implementation rebuilt its per-expert mask subbuffers on every inference —
  20,480 `create_subbuffer` calls per two-token forward — for a prefill
  fallback that the batched-GEMV path never reads. `patches/0003` skips them;
  the verify forward drops from 27.3 ms to 18.1 ms with byte-identical
  output. With that patch and an int4 head, the 35B measured 72.9 t/s greedy
  against ~62 plain on a code prompt. Single-prompt and not yet through the
  acceptance harness. See DESIGN §3.2 and §3.5.1–3.5.2.
- **[M1]** **Fused SDPA** on the full-attention layers: OpenVINO selects
  `ocl::sdpa::opt` for them on both cards (confirmed in a decode profile).
- **[M0]** **Tool-call parsing**: native Qwen tool-call output is returned as
  structured `tool_calls` (OpenAI-compatible). Parsed, never executed;
  requests without declared tools get raw text untouched.
- **[M1]** **Tokenizer and chat template come from the model artifact**, never
  from the server; the template hash is part of the model allowlist.
- **[M0]** **Serving defaults in four layers**: request fields win over
  operator flags (`--temp`, `--top-p`, `--top-k`, `--repetition-penalty`,
  `--presence-penalty`, `--chat-template-kwarg enable_thinking=BOOL`), which
  win over the artifact's `generation_config.json`, which wins over the model
  card. `/props` reports the resulting defaults and where they came from, and
  `usage.completion_tokens_details` reports per response whether speculation
  engaged.
- **[M0]** **Hard context-overflow rejection**: HTTP 400 with the numbers. No
  silent truncation, no context shift — on hybrid GDN models a shift is not
  honestly implementable, because the recurrent state cannot un-see past
  tokens. History management belongs to the client.
- **[M0]** **Streaming that does not corrupt anything**: a multi-byte code
  point is never split across two SSE chunks, a stop sequence never leaks out
  one fragment at a time, and tool-call syntax never reaches a content delta.
- **[M6]** **Two client sessions per service**, in one process: `--parallel 2`
  gives each lane its own `InferRequest`s (language model, embeddings, both
  MTP requests), its own GDN checkpoint rows and its own KV block table out of
  a shared, refcounted page pool. Gated: two prompts interleaved are
  **byte-identical** to their solo runs in both start orders, and the
  acceptance task scores **10/10 on each lane while both run it at once**. The
  measured cost of sharing: an agent session at 30k context and a subagent
  burst get 32.5 and 31.2 t/s against 68.7 t/s alone, with the session's
  inter-token stall at **p95 17 ms**. Sequences are never batched into one
  graph call, which is what keeps the equivalence invariants intact.
- **[M6]** **Admission with the numbers**: a lane is a memory reservation, so
  a request beyond the reserved lanes is a **503 carrying the arithmetic**
  rather than an unbounded wait (`--queue-timeout S` restores queueing). A
  context that does not fit is refused at startup the same way — on the A770
  two lanes of 40960 serve, two of 65536 are refused with every term spelled
  out.
- **[M0]** **Cancellation**: a dropped client aborts the request at the next
  scheduler boundary and frees its slot — and with it the lane and its KV
  pages, without touching the other lane's stream.
- **[M0]** **Console state output** in the llama.cpp tradition: per-request
  timing lines (prefill/decode token counts and rates), slot states,
  memory-map printouts at startup, cache hit statistics. stderr is the
  dashboard.

## Non-goals

- No web UI, no GUI, no gradio, no metrics dashboard. Console and HTTP JSON only.
- No model zoo. A checkpoint outside the table above is rejected at load time.
- No multi-GPU, no pipeline or tensor parallelism. One process, one card
  (several *sequences* per process is M6; several *cards* is not on the list).
- No batching of two sequences into one graph call. Lanes interleave at
  execution granularity instead; batching is an optimisation with its own
  equivalence burden, and it has not been paid.
- No training, no LoRA, no quantization tooling (artifacts are produced offline).
- No Windows.

## Status

Serving all three models on both cards, on the paged executor: arcint-owned
block tables, measured-reservation admission, speculative decoding whose
rollback moves zero bytes. u8 KV is the default, at half the memory and
bit-identical output. The stateful executor stays behind `--no-paged` as the
reference implementation the equivalence suite compares against.

| milestone | state |
|---|---|
| M0 skeleton | done |
| M1 executor, greedy, 27B coder on B60 | done — 51.4 t/s, 10/10 |
| M2 chunked prefill, cache ledger, long context | done — 257,167 tokens loaded |
| M3 prefix caching | done — warm/cold byte-identical, hit stats on console |
| M4 MTP | done and served — 36.2 t/s at 93.2% acceptance, 10/10 greedy |
| M5 all three models | done — the 35B fits the A770 too, via `--offload-ratio` |
| M6 two lanes per service | done — byte-identical under interleaving on both cards, 10/10 on each lane concurrently |
| M7 fit pass | done — two-ledger reservation measured; an explicit `--n-ctx` is verify-only, never lowered, and its refusal trims the prefix-cache reserve first |
| M8 asymmetric KV (`--paged-kv u8:i4`) | done — serves, 10/10; +28% context (8.8 vs 11.3 KiB/token); owed: prefill price, prefix byte-exactness |
| M9 device-tier expert slot pool | done — 35B on the 16 GiB card 0.4 t/s (ratio 25, unpatched) → 9.1 (ratio 50 / 8 GiB pool, 16-token probe) and 10.4 (64-token probe) |
| M11 drafting II | measured — free levers on the DFlash chain: Viterbi negative, blocks 12/16 trade throughput for tokens per cycle, ngram below plain decode; the oracle floors the re-rank headroom at +0.74 per cycle; adapter is a decision, tree not pursued |
| M12 dispatch pin, tiled exporter | done — `--pin-dispatch` measured null on a quiet host, opt-in; exporter `--moe-lowering tiled` |
| M13 vision reserved | done — `--vision` refused; vision IRs reported at load and never loaded (coder: 6 files, 428.3 MiB on disk) |
| M14 host compute tier (`--moe-cpu-tier`) | done — 35B on the 16 GiB card 15.0/15.5 t/s vs 10.4/10.6 at ratio 50 / 8 GiB; 14.1/14.8 vs 7.4/7.5 at ratio 75 / 5 GiB; text byte-identical, 10/10 |

Verification: 306 unit cases, a 61-check curl round-trip, a lane-accounting
stress (200 requests, 24-way, 8 lanes, queueing and refusing), an equivalence
suite and a concurrency suite that run where the card is. Clean under ASan and
UBSan on x86_64.

[CHANGELOG.md](CHANGELOG.md) lists what each release changed and the
runtime it depends on. [DESIGN.md](DESIGN.md) records how these numbers were taken, including the
negative results and the explanations that were retracted when a measurement
contradicted them.

## Deploying it

The installed binary is self-contained: it resolves the OpenVINO runtime and
the tokenizers extension through RPATH, both found at configure time, so a
unit file carries flags and not a library search path. With `LD_LIBRARY_PATH`
unset there are no unresolved objects, and it serves `/health` under `env -i`.

Two lanes are a flag and a memory claim: `--parallel 2` reserves activations,
GDN checkpoint rows and KV for two concurrent sequences at the configured
`--n-ctx`, and refuses at startup with the arithmetic if the card cannot hold
them. If a deployment would rather queue than be refused at request time, set
`--queue-timeout S`.

`--n-ctx` decides who owns the prefix-cache reserve pages. Omitted, the M7
fit pass adopts the maximum admissible depth itself, and the prefix cache
gets no reserve pages of its own unless
`--prefix-cache-reserve PCT` holds that share of the affordable pages spare
(the dense agent at 25%: 116,528 adopted with 2,428 pages spare, against
155,376 with none). Given explicitly, it is verify-only —
never lowered on its own — and a refusal at that depth first trims the
prefix-cache reserve before it is itemized; the pool is sized in bytes, and
`--paged-kv u8:i4` (asymmetric key/value precision, +28% context on the
measurements above) changes what fits in it. Ship a unit's `--n-ctx` as a
multiple of 4096 below the admissible depth actually measured for its card,
precision and prefix-cache budget, not the model's own trained maximum.

`packaging/arcint.service` is a systemd **user** unit. Its flags are literal
rather than `${ARCINT_PORT}` out of an environment file: the fleet's unit
manager reads the port, the served name and the context out of `ExecStart`,
and an env-file indirection hides all three from the port board and from an
automated edit. `ExecStart` points at `/usr/bin/arcint`, which is where the
package puts the binary; building into `~/.local` instead means changing that
one line.

The name the endpoint answers to is `--served-model-name`, and it is separate
from `--model-id` on purpose: the first is what `/v1/models` reports and what
a discovering proxy pins its roster to, the second is the allowlist assertion
about which artifact this process will accept. Renaming an endpoint is a
one-flag edit and never weakens the assertion.

One process holds its model's VRAM for its whole lifetime. Whatever else was
serving that card has to stop first.

## Why not just use …

- **OpenVINO GenAI**: its continuous-batching path (required for prefix
  caching) produces measurably different greedy output than its stateful path,
  the equivalence test in its own CI has been skipped since 2025-02, and
  hybrid state is checkpointed at coarse intervals sized for memory rather
  than correctness. arcint keeps the OV *compiler* and replaces the pipeline
  layer.
- **vLLM**: no usable Intel path for these hybrids (the XPU build lags, and
  Arc is not a first-class target). The paged-KV and prefix-cache ideas are
  taken; the engine is not.
- **llama.cpp**: excellent console ergonomics and a sane server, but Vulkan on
  Battlemage is unoptimized (measured 7.9 tok/s dense where OpenVINO does 60)
  and SYCL is broken under the xe KMD. The ergonomics are taken; the backend
  situation is why arcint exists.

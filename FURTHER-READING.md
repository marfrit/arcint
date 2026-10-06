# arcint — further reading

The material behind the README's short tables: the fuller measured record,
the scope, where the served artifacts come from and how they were calibrated,
the features, the non-goals, the status of each milestone, deployment notes,
and why arcint exists instead of an established engine. Every number here was
measured on this project's own hardware and names its card and configuration;
[DESIGN.md](DESIGN.md) §7 gives each one's standing and its evidence class.
The dated record behind them — the measurement diary, the campaigns, the
design notes — lives on the development branch, `qfndev`.

## Measured

One task throughout: the Lua CSV parser to RFC 4180, ten named cases,
**executed**, one point per case. Greedy where the model tolerates it,
otherwise the model card's own sampling defaults.

**The pipeline is the variable.** Same artifact (the Qwen3.6-27B-A3B-Coder
int4), same card (the 24 GB Arc Pro B60), same OpenVINO compiler:

| engine | decode | task |
|---|---|---|
| OpenVINO GenAI, stateful pipeline | ≈43 t/s | 10/10 |
| arcint, stateful executor (`--no-paged`) | 51.4 t/s | 10/10 |
| **arcint, paged executor, u8 KV** | **71.3 t/s** (68.6 at f16 KV) | 10/10 |
| arcint, paged, at ~30k context | 70.1 t/s | 10/10 |

The ~30k row matters as much as the peak: the usual throughput collapse with
depth is absent from the served path, a property of the paged block tables.
Every arcint row is additionally gated by byte-equality tests: warm cache
against cold, one lane against two, paged against stateful where both exist.
For scale, llama.cpp SYCL on the A770 with a GGUF Q4_K_M of the same model
decodes at 14.4 t/s (a different quantisation and card: a rough bound, not a
row of the series).

**The coder at depth on the 16 GiB card** (u8 KV, 400 greedy tokens, one
process per arm): decode 45.1 / 44.5 / 40.3 t/s and prefill 525 / 462 / 403
t/s at 8.9k / 37.7k / 71.7k tokens; at a 98k-token prompt prefill 621 t/s.
`--paged-kv u8:i4` prefills at u8's rate from `marfrit-openvino +p6` on (459
against 457 t/s at 37.7k and 401 against 398 at 71.7k, chunk held at 128).

**Context by KV precision.** `--paged-kv u8:i4` (u8 keys, i4 values) against
the `u8` default, auto-fit, same artifact and card:

| model | card | u8 | u8:i4 | gain |
|---|---|---|---|---|
| coder | 16 GiB | 133,456 | 171,392 | +28% |
| dense 27B agent, MTP on | 24 GB | 155,376 | 199,424 | +28% |

u8:i4 costs 8.8 KiB/token against u8's 11.3 on the coder and scores 10/10.
f16 costs 20.0 KiB/token and is faster at depth (u8's prefill 16.5% below it
at 57.8k, f16's decode 7.8% ahead at 53.5k); u8 is the default because it is
the setting that does not refuse — two lanes at 262k on the 24 GB card fit at
u8 and not at f16.

**Speculative decoding on the dense Qwen3.8-27B** (24 GB card, int4, u8 KV,
greedy, repetition penalty 1.0, 400 tokens, 32,768 context):

| drafter | decode | acceptance | max context (reservation) |
|---|---|---|---|
| none | 24.0 t/s | — | 199,712 |
| MTP head (`--mtp on`) | 33.0 t/s | 76.7%, one draft per pass | 155,680 |
| **DFlash2 int4 (`--dflash`)** | **44.8 t/s** | 3.13 tokens per verify cycle | 136,640 |
| DFlash2 int4, draft on the A770 | 39.8 t/s | 3.13 tokens per verify cycle | 171,904 |

The DFlash2 drafter is the public block-diffusion head
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
exported with `tools/export_dflash.py`, drafting seven tokens per verify pass
through the same checkpoint-row verify the MTP head uses. Parking the draft on
the other card (`--dflash-device`) buys back 35k tokens of context for 5 t/s,
with output byte-identical to the same-card run. **At depth** (77k tokens):
plain 15.3 t/s, DFlash2 18.8 t/s (40.6% accepted, about 4.4 tokens per cycle),
MTP 4.9 t/s at 90.8% acceptance — MTP, capped at two tokens per cycle, does
not beat plain decoding at depth on this artifact; DFlash2 does. The drafters'
rotary subgraphs are kept at f32 and the MTP layer's state is charged against
the reservation; without those two, acceptance collapsed past 65,504 tokens
and the card was overcommitted.

**The MTP head on Intel's public Qwen3.8 IR** (24 GB card, greedy, thinking
off, 320 tokens): plain 25.0 t/s; with the reconstructed head 36.9–37.3 t/s at
96.3% acceptance on code; with Intel's own exported MTP layer and this
repository's LM-head graph 37.7–38.1 t/s at 93.9%. The acceptance task scores
10/10 with the head.

**Qwen3.6-35B-A3B on the 16 GiB card.** The stock int4 IR is 17.4 GiB. Two
routes serve it on the A770:

| route | configuration | prefill | decode |
|---|---|---|---|
| fused int4 + expert offload + host tier | ratio 50, 8 GiB device pool, `--moe-cpu-tier` | 27.9 t/s | 18.2 t/s (12.5 without the tier) |
| **native experts, all-resident** | `qwen3.6-35b-a3b-native-d40packed-u8`, `--offload-ratio 0 --moe-per-expert-dispatch`, u8 KV | **~960 t/s at 4,096; 779 at 32,768** | **28.1 t/s after 4,096** |

The native artifact keeps the checkpoint's own IQ2_S / IQ3_XXS expert blocks
and computes only the routed experts, each by a per-expert kernel on the card;
it loads at 13.11 GiB, admits 112,288 tokens per lane at u8 KV, and scores
10/10. On the 24 GB card the stock IR serves resident (62.7 t/s decode at a
53.5k-token context, 1,584 t/s prefill, 10/10 greedy).

**The same GGUF on the same card through three stacks** (24 GB card, the dense
Qwen3.8-27B as Unsloth's Q4_K_M, 64 generated tokens per cell, one fresh
process per cell):

| stack | prefill 1k / 10k | decode at 1k / 10k | resident |
|---|---|---|---|
| Intel int4 IR through arcint (different bytes, the reference) | 1,598 / 1,434 t/s | 23.4 / 23.4 t/s | 13.06 GiB |
| arcint, GGUF mixed form (the default) | ~1,000 t/s at 856 | 18.5 t/s at 856 | 16.54 GiB |
| llama.cpp SYCL | 249 / 206 t/s | 14.2 / 12.4 t/s | — |
| llama.cpp Vulkan | 126 / 108 t/s | 7.8 / 7.0 t/s | — |

At 71.7k tokens the mixed form prefills at 464 t/s and decodes at 13.7 t/s
against the IR's 552 / 16.5; the native-rows form (`--gguf-mode native`,
15.22 GiB) decodes faster (19.5 / 14.2 t/s at 856 / 71.7k) and prefills slower
(662 / 395). Vulkan is the slowest stack measured on this card; SYCL decodes
at about the GGUF form's level from the same bytes.

**Qwen3.8 Flash-Next** (512 experts, top-10) serves at full depth on one
16 GiB card with a quarter of its experts on the CPU tier (ratio 75, the
checkpoint's own IQ3_XXS / IQ4_NL expert formats): prefill 15.1 t/s over a
32,768-token prompt, decode 0.5 t/s. Against the model's own f32 forward its
short-prompt logits sit at KL 0.017 nats (llama.cpp on the same GGUF: 0.053).
It is a capability, not yet a recommendation: the rate is set by the host
tier.

## Scope

**Models**: the allowlist in `src/core/model_registry.cpp` (provenance in
[models/allowlist-raw.json](models/allowlist-raw.json)). All share one
tokenizer, and every served family puts full attention on one layer in four:

| model | architecture | layers (GDN + attn) | experts | weights |
|---|---|---|---|---|
| Qwen3.6-27B-A3B-Coder | hybrid GDN + attention, MoE (`qwen3_5_moe`) | 40 (30 + 10) | 184, pruned from 256 | 12.8 GiB int4 |
| Qwen3.6-35B-A3B | hybrid GDN + attention, MoE (`qwen3_5_moe`) | 40 (30 + 10) | 256 | 17.4 GiB int4; 14.1 GB native-format artifact |
| Qwen3.8-27B | hybrid GDN + attention, dense (`qwen3_5`) | 64 (48 + 16) | dense | 13.4 GiB int4; 15.3 GB as GGUF Q4_K_M |
| Qwen3.8 Flash-Next | hybrid, MoE, hyper-connections, n-gram table (`qwen4_exp`) | 48 | 512 | 77.5 GB native-format artifact |

**Hardware**:

| card | VRAM | notes |
|---|---|---|
| Intel Arc A770 (Alchemist, Xe-HPG) | 16 GiB (15.11 usable) | 414–418 GB/s measured random read |
| Intel Arc Pro B60 (Battlemage, Xe2) | 24 GB (22.71 usable) | 453 GB/s measured random read |

Both behind the xe kernel driver. Byte-identity and numeric floors are read on
the A770: the B60's GDN state output differs by one f16 ulp run to run on the
served Flash-Next path, an open per-card defect reported upstream.

## Where the artifacts come from

**arcint loads an OpenVINO IR directory.** Concretely:
`openvino_language_model.{xml,bin}`, `openvino_text_embeddings_model.{xml,bin}`,
the tokenizer and detokenizer IRs, `config.json` and the chat template. GPTQ
or NVFP4 safetensors will not load — those are vLLM formats, and OpenVINO does
not read them.

**A GGUF opens on top of such a directory**: `--gguf FILE --model DIR` takes
the served IR of the same architecture as the topology template and replaces
its projections with the file's own K-quant rows (Q4_K, Q5_K, Q6_K, Q8_0).
`--gguf-mode mixed` (the default) repacks Q4_K at load into the runtime's own
compressed form and keeps Q5_K and Q6_K as the file's rows, decoded in a
kernel the patch series carries; `repack` and `native` are the other forms.
`--gguf-embed file|template` takes the token embedding rows from the file,
dequantised on the host per token (`file`, the default), or keeps the
template's embedding model; `--gguf-check once|always` keeps each repacked
projection's deviation verdict between loads of the same file (`once`, the
default) or re-checks it. The file's geometry is checked against the
template's; the template's tokenizer, chat template, GDN state tensors and MTP
layer are served. Dense models of the allowlisted families.

**Serving-shape artifacts** — the Flash-Next ones and the native-format
Qwen3.6-35B ones — are built from the GGUF shards by
`tools/export_serving_artifact.py` (`--expert-format native` keeps the
checkpoint's expert blocks; `--native-packed` carries IQ2_S verbatim;
`--dense-u8` stores the dense projections in the runtime's u8 group-16 form).
No PyTorch checkpoint is needed.

The IR models are downloadable, and every number above was taken on the
published copy where one exists:

| model | where to get it | note |
|---|---|---|
| Qwen3.6-35B-A3B | [`OpenVINO/Qwen3.6-35B-A3B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov) | Intel's export, used as-is; verified byte-identical to the copy the measurements ran on |
| Qwen3.6-27B-A3B-Coder | [`marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov`](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov) | Apache-2.0; no official IR exists for this community fine-tune; the calibration below records what it cost |
| Qwen3.8-27B | [`OpenVINO/Qwen3.8-27B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov) | Intel's export, allowlisted as its own entry (`qwen3.8-27b-intel-int4`); it carries an MTP layer but no MTP LM head — `tools/export_mtp.py` builds the head, or run without `--mtp` |

**The allowlist keys on the artifact's directory name**, so a download has to
land in the directory the entry names — `qwen36-35b-a3b-int4-ov`,
`qwen36-coder-b5-ov`, `qwen38-intel-int4-ov` respectively:

    hf download OpenVINO/Qwen3.6-35B-A3B-int4-ov \
        --local-dir <model root>/qwen36-35b-a3b-int4-ov

An entry asserts geometry, quantisation, hashes, the weight byte count and a
measured status for one artifact, and the directory name is the handle it is
asserted through. A refusal at load time means the artifact is not the one
the entry describes. Intel's Qwen3.8 export is its own entry rather than an
alias of this project's AWQ export (`qwen38-b7c1-ov`), because an alias would
assert a measurement nobody took on that file.

Two findings decide whether you need the rest of this section at all:

- **Try the official IR before exporting anything.** The 35B sat unused for
  weeks on the assumption that a stock export would not clear the quality
  bar. Measured, it scored 10/10 greedy, 3/3 on tool calls, clean German, at
  62.7 t/s. Export only when no official IR exists.
- **The dense model's MTP head is in every checkpoint and incomplete in every
  published IR.** optimum-intel drops it (Intel's own export carries the layer
  but not the LM head the draft needs to become a token).
  `tools/export_mtp.py` reconstructs it from the checkpoint's own weights.

If you only want to run arcint, take the table and stop reading here. The rest
of this section is for anyone who needs a different calibration than the
published ones provide.

**The GGUF shortcut still needs the IR** for the dense and coder families: the
template supplies the topology, the tokenizer, the chat template and the
tensors the file's rows do not replace. llama.cpp's own OpenVINO backend was
built and tested against these models and does not run them (the GDN hybrid
aborts in scheduling because the recurrent states are not mapped in the GGML
frontend; a classic MoE falls to a CPU path). A new calibration starts at the
Hugging Face checkpoint:

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
  into repetition loops, while AWQ-only on the same model is healthy (10/10
  greedy on the paged path). Measure AWQ+SE against AWQ-only per model before
  believing either.
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
- **Scale estimation with code samples is memory-hungry**: it OOM'd on a
  121 GB machine, thrashed on 247 GB, and completed on 494 GB with a ~152 GB
  peak. `TMPDIR` needs ~110 GB — the fp16 intermediate and the final save both
  land there, and running out produces `basic_ios::clear: iostream error`
  after the entire compression has already finished.
- **Do not trust `ov::cache_dir` with these MoE IRs.** The first run writes the
  blob, the second imports it without expert weights and fails at inference
  ("expert weight provider not initialized"; openvinotoolkit/openvino#37607).
  arcint's `--cache-dir` proves every imported blob with a real forward before
  serving and recompiles on failure.

### Registering it

arcint refuses artifacts outside its allowlist at load time, so a new export
needs an entry: geometry, quantisation, the architecture, tokenizer and
chat-template hashes and the weight byte count. `arcint --model DIR
--inspect-artifact` prints the contract a directory implies, device-free,
before any entry exists. This is the mechanism behind "no model zoo" in the
non-goals: the allowlist asserts that the process is serving what it claims to
serve.

## Features

- OpenAI-compatible **`/v1/chat/completions`** and **`/v1/completions`** over
  HTTP (SSE streaming; token-id prompts on `/v1/completions`), **`/health`**,
  **`/props`** (the model, the reservation terms, the served cache block,
  build info, sampler defaults with provenance) and **`/v1/models`** (the name
  and the context the process is running with).
- **Prefix caching** for the full hybrid state — attention KV pages and GDN
  recurrent state — with a hard invariant: greedy output with a warm cache is
  byte-identical to greedy output with a cold one, and output never depends on
  the process's own request history. Cache reuse that changes the answer is a
  bug. Evicted entries can park in host RAM (`--cache-host-mib`) and come back
  byte-exact.
- **Block-aligned GDN state checkpoints** for the linear-attention layers, on
  the same absolute grid as prefill chunking, so a hit restores both halves of
  the hybrid state exactly.
- **Speculative decoding** with three drafters, one per server: the native MTP
  head (`--mtp on`), the DFlash2 block-diffusion head (`--dflash DIR`), and a
  prompt-lookup drafter (`--draft N`). Verification is exact, rollback moves
  zero bytes on the paged path, and drafters can live on the other card with
  byte-identical output. All engage only under greedy; a multi-token verify
  pass differs from a single-token step at a near-tie, so the answer can
  differ from plain greedy — gate it on your own task.
- **Asymmetric KV** (`--paged-kv u8:i4`): +28% context at u8's prefill rate.
- **MoE routes**: fused int4 resident; expert offload with a device-resident
  slot pool (`--offload-ratio`); a host CPU compute tier for capacity misses
  (`--moe-cpu-tier`) under a static, history-independent residency partition;
  per-expert dispatch computing only the routed experts in the checkpoint's own
  block formats (`--moe-per-expert-dispatch`), all-resident at
  `--offload-ratio 0`.
- **GGUF opened in process** (`--gguf`): the file's K-quant rows on the
  template's topology, repacked within a measured bound or decoded as stored.
- **Serving defaults in four layers**: request fields over operator flags
  (`--temp`, `--top-p`, `--top-k`, `--repetition-penalty`,
  `--presence-penalty`, `--chat-template-kwarg enable_thinking=BOOL`) over the
  artifact's `generation_config.json` over the model card.
- **Tool-call parsing** into OpenAI `tool_calls` (both Qwen wire forms), never
  executed; requests without declared tools get raw text untouched. Reasoning
  split into `reasoning_content` when the template opens a think block.
- **Hard context-overflow rejection**: HTTP 400 with the numbers. No silent
  truncation, no context shift — a GDN recurrent state cannot un-see past
  tokens. History management belongs to the client.
- **Streaming that does not corrupt anything**: a multi-byte code point is
  never split across two SSE chunks, a stop sequence never leaks out one
  fragment at a time, and tool-call syntax never reaches a content delta.
- **Two client sessions per service** in one process (`--parallel 2`), gated
  byte-identical under interleaving in both start orders, 10/10 on each lane
  concurrently; an agent session at 30k context beside a subagent burst gets
  32.5 and 31.2 t/s against 68.7 alone, with a p95 inter-token stall of 17 ms.
- **Admission with the numbers**: a lane is a memory reservation in which every
  term is measured; what does not fit is refused at startup, and a request
  beyond the reserved lanes is a 503 carrying the arithmetic
  (`--queue-timeout S` restores queueing). `--fit-ledger-dir` persists the
  probed terms so later starts skip the probes.
- **Cancellation**: a dropped client aborts the request at the next scheduler
  boundary and frees its lane and pages without touching the other lane.
- **Console state output** in the llama.cpp tradition: per-request timing
  lines with where the time went, the reservation at startup, cache hit
  statistics. stderr is the dashboard.

## Non-goals

- No web UI, no GUI, no metrics dashboard. Console and HTTP JSON only.
- No model zoo. A checkpoint outside the allowlist is rejected at load time.
- No multi-GPU, no pipeline or tensor parallelism. One process, one card
  (several *sequences* per process, yes; several *cards* per model, no —
  a drafter or the embeddings may sit on the other card).
- No batching of two sequences into one graph call. Lanes interleave at
  execution granularity; batching would change every sequence's arithmetic.
- No training, no LoRA, no quantization tooling beyond the offline exporters.
- No Windows.

## Status

Every milestone and its standing result is in DESIGN.md §7; in short:

| milestone | state |
|---|---|
| M0–M3 skeleton, executor, paged KV, prefix caching | done — the paged executor is the served path; warm/cold byte-identical |
| M4 MTP | done — served; not byte-identical to plain greedy by design (§3.5) |
| M5 all target models | done |
| M6 two lanes per service | done |
| M7 fit pass | done — two-ledger measured reservation; an explicit `--n-ctx` is verify-only |
| M8 asymmetric KV | done — `u8:i4`, +28% context, u8's prefill rate from `+p6` |
| M9 / M14 expert offload, host tier | done — static residency partition closes the history invariant with the tier on |
| M10 sub-4-bit experts | superseded by the native expert formats |
| M11 drafting at depth | done — drafter fixes landed; DFlash2 wins at 77k |
| M12 / M13 dispatch pin, tiled exporter, vision reserved | done |
| 0.4.x GGUF in process | done — 10/10; the mixed form meets the 460 t/s depth prefill bar |
| 0.5.x Flash-Next, native experts, all-resident 35B | Flash-Next serves at full depth (slowly); the 35B serves all-resident on the 16 GiB card at ~960 / 28 t/s |

Verification: a device-free unit ladder (what bare `ctest` runs: the unit
harness, a curl round trip against the stub, a lane-accounting stress), clean
under ASan and UBSan on x86_64; and an acceptance target of enumerated cells
(`tests/acceptance/`) that runs where the cards are, fails on any skip or
regression not named on its command line, and gates against references filled
from its own runners' samples. The equivalence and concurrency suites are its
runners, the acceptance task its external cell.

[CHANGELOG.md](CHANGELOG.md) lists what each release changed and the runtime
it depends on.

## Deploying it

The installed binary is self-contained: it resolves the OpenVINO runtime and
the tokenizers extension through RPATH, both found at configure time, so a
unit file carries flags and not a library search path. With `LD_LIBRARY_PATH`
unset there are no unresolved objects, and it serves `/health` under `env -i`.
The runtime is the `marfrit-openvino` package (`contrib/packaging/`); arcint's
package depends on it at `+p20` or later within the pinned nightly.

Two lanes are a flag and a memory claim: `--parallel 2` reserves activations,
GDN checkpoint rows and KV for two concurrent sequences at the configured
`--n-ctx`, and refuses at startup with the arithmetic if the card cannot hold
them. If a deployment would rather queue than be refused at request time, set
`--queue-timeout S`.

`--n-ctx` decides who owns the prefix-cache reserve pages. Omitted, the fit
adopts the maximum admissible depth itself, and the prefix cache gets no
reserve pages of its own unless `--prefix-cache-reserve PCT` holds that share
of the affordable pages spare. Given explicitly, it is verify-only — never
lowered on its own — and a refusal at that depth first trims the prefix-cache
reserve before it is itemized. Ship a unit's `--n-ctx` below the admissible
depth actually measured for its card, precision, drafter and prefix-cache
budget, not the model's own trained maximum.

`packaging/arcint.service` is a systemd **user** unit, and `contrib/systemd/`
carries the two units this project deploys as examples. Flags are literal
rather than read from an environment file, so a unit manager and a journal can
both read the port, the served name and the context out of `ExecStart`.
`ExecStart` points at `/usr/bin/arcint`, where the package puts the binary.

The name the endpoint answers to is `--served-model-name`, and it is separate
from `--model-id` on purpose: the first is what `/v1/models` reports and what
a discovering proxy pins its roster to, the second is the allowlist assertion
about which artifact this process will accept.

One process holds its model's VRAM for its whole lifetime, and two processes
on one card fault it rather than share it. Whatever else was serving that card
has to stop first.

## Why not just use …

- **OpenVINO GenAI**: its continuous-batching path (required for prefix
  caching) produces measurably different greedy output than its stateful path,
  the equivalence test in its own CI has been skipped since 2025-02, and
  hybrid state is checkpointed at coarse intervals sized for memory rather
  than correctness. arcint keeps the OV *compiler* and replaces the pipeline
  layer.
- **vLLM**: no usable Intel path for these hybrids at the time this started
  (the XPU build lags, and Arc is not a first-class target). The paged-KV and
  prefix-cache ideas are taken; the engine is not.
- **llama.cpp**: excellent console ergonomics and a sane server, but Vulkan on
  Battlemage is the slowest stack measured on the 24 GB card, and SYCL is
  broken under the xe KMD on older runtimes. The ergonomics are taken; the
  backend situation is why arcint exists.

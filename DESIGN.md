# arcint — Design

State as of 2026-10-15: arcint **0.8.0** on llama.cpp **`3e4dfa8`** with the
patch series `contrib/llama.cpp/patches` 0001-0056. One engine: libllama with
ggml's OpenCL backend and arcint's Intel kernels. Three services run in
production on two Intel Arc cards: the dense Qwen3.8-27B agent and
Qwen3.8-Flash-Next (125B-A6B, on demand) on the 24 GB Arc Pro B60, the MoE
coder on the 16 GiB Arc A770.

**History.** Until 0.7.12 arcint also had an OpenVINO engine: OpenVINO IRs on
a patched GPU plugin (`marfrit-openvino`, plugin patches 0003-0077), with
arcint's own paged KV, prefix cache, host prefix tier, measured admission and
a Flash-Next expert tier between card and RAM. It was retired by the
operator's decision of 2026-10-15 (charter item A6,
`docs/milestone-1.0.0.md`). Its design, its record index (§7.0.2a-§7.0.2da)
and every document of that era are kept at the tag **`openvino-history`**:
`git show openvino-history:DESIGN.md`, or
<https://github.com/marfrit/arcint/tree/openvino-history>.
`docs/history.md` lists the retired documents. A citation of the form "DESIGN
§7.0.2x" anywhere in the record points into that tagged DESIGN.md.

Every disposition below carries its evidence class: `paper` (a published
text), `code` (read in source), `measured-here` (run on this project's
cards), or `decision` (the operator's, dated). Measurements name card,
depth, KV precision and configuration. Debugging narratives and superseded
numbers live in the campaign documents and in git history, not here.

How to read it: §1-§6 are the architecture, the rules and the serving
surface; §7 is the state of play; §8 is the open work, each item against the
reference that already does it.

## 1. The one idea

llama.cpp is the model runtime and ggml's OpenCL backend the kernel library;
arcint is the serving layer and the Intel work below it. arcint owns the HTTP
surface, chat templates, tool calls and reasoning, the sampler, named lanes
and their admission, speculative decoding with the model's own MTP head, the
context checkpoints a lane resumes from, the expert-cache policy and the
CPU's share of its misses, and the kernels that make ggml fast on Arc. The
GGUF is served as published, with no export step.

The kernels are the reason the engine is usable here at all. Stock ggml
OpenCL is tuned for Adreno: on the B60 the dense 27B prefilled 71 t/s and
decoded 10.3 t/s, and the coder's K-quant experts ran on the CPU
(`measured-here`, llama-bench, 2026-10-03). With the series the same files
serve at 695 t/s prefill at 55k tokens, the coder decodes 44-51 t/s and
Flash-Next 43 t/s (§7).

**References first.** Inference-engine expertise lives in the reference
engines, and their source is the design reference (operator rule,
2026-10-01; `CLAUDE.md`, `AGENTS.md`):

- **Strata** (`~/src/Strata-ref`), written for Qwen3.8-Flash-Next on one
  12 GB GPU plus RAM: expert cache with an adapt rule, CPU-computed misses
  through a doorbell, multi-token MTP drafting, prefill streaming of the
  non-resident experts (`code`; paper `docs/paper/Strata-Paper.pdf`). Its
  README reports, on an RTX 5070 + Ryzen 5 7600 + 64 GB DDR5, IQ3_XXS: 62
  t/s decode on a short chat, 49 t/s at 128K, 1,750 t/s prompt reading at
  32K (`paper`).
- **FreeToken** (`~/src/FreeToken-ref`), an edge-MoE engine: CPU-resident
  experts, one GPU LRU expert pool, bandwidth-adaptive split, double-buffered
  prefill streaming (`code`; paper arXiv 2608.16157).
- **Infernix** (`~/src/Infernix-ref`), CUDA Flash-Next on one GPU: a global
  LFRU expert cache, layer-walk prefill, a shared prefix cache
  (`docs/campaigns/research-infernix.md`).
- **llama.cpp** itself (CUDA and Vulkan backends for kernel shapes, its
  server for checkpoints and slot batching), **NInfer** (`~/src/ninfer`;
  dense models fully resident on one GPU), **vllm-xpu-kernels** and
  **sycl-tla** (Xe2 kernels).

An arcint measurement counts against a reference technique only if it tested
the reference's mechanism; a conflict between a reference mechanism and an
invariant here goes to the operator, priced. The audit that applies this
rule to the record is `docs/campaigns/research-reference-audit.md`.

### 1.1 When llama.cpp has to change: smallest sufficient divergence

In order: (1) an upstream PR written to upstream's conventions; (2) a
numbered patch against the pinned commit, carried in
`contrib/llama.cpp/patches/` and applied at build time, each patch
PR-shaped and documented in that directory's `README.md` with its
measurement and its red case; (3) a maintained fork, only with a stated
reason and an exit condition (none exists). Whatever is carried is
published; a number that depends on a patch nobody else has is reproducible
from that directory. The pin and the patch count are part of every release
note (`CHANGELOG.md`).

Moving the pin is a procedure, not an event: survey upstream, port the
series one commit at a time, run the op suites on both cards, the patches'
own red cases, KL as served and the three services
(`contrib/llama.cpp/README.md`, "Moving the pin"; the move of 2026-10-15 is
`docs/campaigns/research-upstream-llama-2026-10.md`).

## 2. Target constraints

All served Qwen models are hybrid GatedDeltaNet / attention transformers:
most layers carry a fixed-size recurrent state (conv + delta-rule state), a
minority are attention layers with a KV cache. The engine admits a GGUF by
its own `general.architecture`: `qwen35`, `qwen35moe`, `qwen4exp`, and
`llama` at the Mistral Small 24B geometry only (`code`:
`src/exec/backend_llama.cpp`); anything else is refused at load.

| service | model | architecture | served file | card |
|---|---|---|---|---|
| agent | Qwen3.8-27B | `qwen35`, 64 layers (48 GDN + 16 attention), dense, MTP layer | Q4_K_M GGUF | B60, production |
| coder | Qwen3.6-27B-A3B-Coder | `qwen35moe`, 40 layers (30 + 10), 184 experts (the 35B pruned), MTP layer | a searched per-layer quant mix (`c5f495ac-shq8`, 14.6 GB, `docs/llama-engine.md`) | A770, production |
| Flash-Next | Qwen3.8-Flash-Next | `qwen4exp`, 48 layers (36 GDN + 12 sparse attention), 512 experts, top-10 + shared, n-gram embedding table | ISTA-DASLab's IQ3_XXS GGUF (GSQ-RCO; IQ2_XS as the faster option), the MTP layer from a separate GGUF built from the checkpoint | B60 + host RAM, on demand |
| writer | Cydonia 24B v4.3 (Mistral Small 3.2) | `llama`, 40 attention layers | imatrix Q4_K_M | B60, on demand |

Geometry is `code` (the GGUFs' metadata); Flash-Next's 36 + 12 split and its
sparse attention are also Strata's `paper` §2.

The cards (`measured-here` unless marked):

| | Arc A770 (16 GiB) | Arc Pro B60 (24,480 MiB) |
|---|---|---|
| silicon | ACM-G10 (DG2-512), PCI `8086:56a0`, Xe-HPG: subgroup 8 or 16 by kernel, 128 GRF, no 2D block loads | Xe2 (BMG), PCI `8086:e211`: subgroup 16, 2D block loads, 160 compute units |
| arcint device | `GPU.1` | `GPU.0` |
| OpenCL platform (`GGML_OPENCL_DEVICES`) | `1:0` | `0:0` |
| host link | PCIe 3.0 x4 behind the chipset, 1.8 GB/s H2D | PCIe 4.0 x8, 13.9-14.3 GB/s H2D from pinned memory |
| random-read ceiling | 414-418 GB/s (nominal 560, `paper`) | 453 GB/s (nominal 456, `paper`) |

- The xe kernel driver only; OpenCL through Intel's compute runtime is the
  route on both cards. SYCL aborts under xe (a memcpy abort) and Vulkan is
  slow on Battlemage (126 / 7.8 t/s against llama.cpp SYCL's 249 / 14.2 on
  the dense 27B at 1k, 2026-09, `measured-here`).
- The two generations are two OpenCL platforms in Intel's runtime; a
  two-card process gives each its own context (`contrib/llama.cpp` 0023).
- The dev host is an 8-core / 16-thread AVX2 CPU (Zen 3, no AVX-512) with
  52 GiB of RAM in the serving container and an NVMe store
  (`measured-here`); the host's RAM goes to 96 GB (operator, planned
  2026-10-15).
  Host RAM bounds Flash-Next: its non-resident experts sit in a host bank
  (28.72 GiB for IQ3_XXS at the unit's slots, 20.32 GiB for IQ2_XS, ~39 GiB
  for UD-Q3_K_XL).

## 3. Architecture

```
         ┌──────────────────────────────────────────────────┐
         │ HTTP server (cpp-httplib thread pool)            │
         │  /v1/chat/completions  /v1/completions           │
         │  /v1/models  /health  /props                     │
         └────────────────────────┬─────────────────────────┘
                                  │ request objects, routed by `model`
         ┌────────────────────────▼─────────────────────────┐
         │ lanes: one slot pool per named lane, its context │
         │ cap, --queue-timeout, the §3.8 400               │
         └────────────────────────┬─────────────────────────┘
                                  │ step combiner (both lanes in one decode)
   ┌──────────────────────────────▼─────────────────────────────────┐
   │ LlamaBackend (src/exec/backend_llama.cpp)                      │
   │  chat template (minja) · tool calls · reasoning split          │
   │  sampler (src/core/sampler.cpp) · verify walk (verify_walk.h)  │
   │  MTP drafter (llama_spec.cpp) · context checkpoints per lane   │
   └──────────────────────────────┬─────────────────────────────────┘
                                  │ libllama C API
   ┌──────────────────────────────▼─────────────────────────────────┐
   │ llama.cpp 3e4dfa8 + contrib/llama.cpp/patches 0001-0056        │
   │  model, tokenizer, unified KV with a window per lane,          │
   │  recurrent state with per-token snapshots,                     │
   │  expert cache: VRAM slots + USM host bank + CPU misses         │
   │  ggml-opencl: Intel K-quant/IQ/attention/GDN kernels           │
   └────────────────────────────────────────────────────────────────┘
```

### 3.1 Model files

- **The GGUF as published**, one file or the first shard of a split file
  (`--gguf FILE`). Tokenizer and chat template come from the GGUF; arcint
  renders the template with minja and derives the user-message delimiter by
  rendering it twice (`docs/llama-engine.md`, checkpoints).
- **No allowlist** (`decision`, operator, 2026-10-15): the OpenVINO era's
  compiled-in registry of artifact hashes is retired with that engine. What
  is admitted is decided by the GGUF's architecture (§2); what is served is
  stated on `/props` (the file, its architecture, the context, the KV types,
  MTP, the build).
- **The MTP layer.** The dense 27B and the coder carry it in their GGUF.
  Flash-Next's GGUFs do not: `tools/flash_next_mtp_gguf.sh` builds an
  MTP-only GGUF from the checkpoint's 31 `mtp.*` tensors (the head at Q4_K
  since 0.7.12: +1.5 % decode, 156 MiB less VRAM, `measured-here`), loaded
  with `--llama-mtp-gguf` as a second model on the card.
- **The expert profile** that fills Flash-Next's cache at load is Strata's
  decode profile (`--llama-expert-profile`, Strata's `data/expert-profile.bin`,
  MIT).

### 3.2 Kernels: the patch series

`contrib/llama.cpp/patches` holds 56 patches against the pin, each with its
measurement, KL, red case and review in `contrib/llama.cpp/README.md`
(`measured-here` throughout). By what they change:

| area | patches | what |
|---|---|---|
| K-quant products | 0001, 0009, 0011-0014 | integer-dot matvec for 1-4 columns, few-column XMX for verify rows, the fp16/int8 DPAS GEMMs with 2D block loads, tiles from a genetic search |
| low-bit and IQ types | 0002, 0022, 0030, 0034, 0035, 0053, 0054 | IQ2_XS/IQ2_S/IQ2_XXS/IQ3/IQ4/Q2_0 in planes, cheaper sign and codebook decode |
| gated delta-net | 0003, 0050, 0051 | the GDN kernel, rollback snapshots written into the recurrent cache, gating in the kernel |
| attention | 0007, 0008, 0015-0020, 0028, 0037-0041, 0045, 0048 | split decode attention, XMX prompt attention, quantized KV (q8_0, q8_0:q4_0), GQA verify attention, head size 128, the key split by wave |
| MoE | 0006, 0033, 0043 | tokens as the DPAS A operand, the B60 tile, the cache's GEMM threshold |
| expert cache | 0021, 0024, 0042, 0046, 0055, 0056 | VRAM slots and a USM bank, per device, non-blocking admission on its own queue, Infernix's LFRU (opt-in), the misses on the CPU |
| prefill bank stream | 0032, 0044 | each bank tensor copied whole into a VRAM ring ahead of its GEMM, from 256 tokens |
| decode launches and inputs | 0004, 0005, 0027, 0031, 0047, 0049 | small ops, few-row F32, fewer launches, inputs staged in a USM ring with one wait a split |
| lanes and devices | 0023, 0025, 0026, 0029 | one OpenCL context per platform, a KV cell window per lane, per-sequence attention windows |
| graph and buffers | 0010, 0036, 0052 | MTP masked rows, Q8_0 GET_ROWS, compute buffers reserved for N cells |

Every switch the series reads is classified in `docs/switches.md`
(operational, tuning, opt-in alternative, off switch, red case, diagnostic).

### 3.3 Memory

- **Attention KV** is llama.cpp's cache. Types per side with
  `--llama-kv K[:V]`: f16 (default), q8_0, or q8_0:q4_0, on the Intel
  attention kernels of 0015; any other pair is refused, because it has no
  kernel and would dequantize K and V on every call, and 4:4 is refused
  because it moves the dense 27B's top-1 agreement by 1.2 points, past the
  answer-level bar (`measured-here`). A quantized V forces flash attention
  on.
- **Recurrent state.** One GDN state per lane, with per-token snapshots for
  the verify (`n_rs_seq` = the draft count): a rejected draft is rolled back
  by choosing the snapshot, and since 0050 the snapshots are written into
  the recurrent cache, with no copy back (`code`, `measured-here`).
- **Compute buffers.** `--llama-reserve-cells N` (0052) reserves them for N
  cache cells instead of the whole context and cuts deeper prompt chunks to
  fit; on Flash-Next at 131,072 tokens it buys +10-13 % decode
  (`measured-here`).
- **VRAM is sized by measurement, not by an admission model.** Each unit's
  context, KV types and expert-slot budget were set from the peak VRAM and
  GTT at the deepest prompt the unit admits, needle answered (§7). A thin
  post-load margin is not enough: under xe a request that allocates past the
  card evicts the process's buffers to host memory and they stay there
  (Flash-Next 24 tokens in 49 s, `measured-here`, 2026-10-11). Peaks: the
  agent 24,311 MiB of 24,480 at 138k, Flash-Next 23,787, the coder 15,971 of
  16,384 (`measured-here`, 0.7.12).

### 3.4 Reuse, and the correctness rule

**Mechanism.** A lane reuses the prefix it holds. A hybrid model's recurrent
state cannot be cut back to an earlier token, so each lane keeps **context
checkpoints** of that state in host memory, llama.cpp server's mechanism
(`tools/server/server-context.cpp` at the pin, `code`): at the start of the
batch that begins the last user message, at a user message more than
`--llama-checkpoint-step` (8,192) tokens past the newest checkpoint, and
4 + n_ubatch and 4 tokens before the prompt's end; at most `--llama-checkpoints` (32) a lane; restore from the
newest at or below the shared prefix, the attention KV trimmed there. A
thinking turn's follow-up or an edited message resumes instead of
re-prefilling: 38.75 -> 0.61 s on a 24.3k-token conversation (dense 27B,
B60, MTP 5, `measured-here`). There is no prefix cache shared between lanes
(`decision`, operator, 2026-10-15) and none kept across a restart (§8).

**The correctness rule** (`decision`, operator, 2026-10-01; this replaced an
earlier byte-identity invariant; precedent, `paper`: Strata's expert cache
changes the top-1 pick at 2-5 % of positions at equal perplexity,
`~/src/Strata-ref/bench/results/2026-09-27-cache-parity/README.md`):

1. **Correctness is judged at the answer.** The bar for any equivalence or
   quality gate is the answer-level bar in `CLAUDE.md`: the answers stay
   right (facts, the needle, the task battery); the candidate's mean KL
   against the reference is at most **0.03 nats** worse than the baseline
   arm's on the same card and window; argmax agreement drops by at most
   **1 point**. The tolerances are the operator's to change. Bit-identical
   *output* across configurations, placements, timing or request history is
   not required anywhere.
2. **Integrity checks on copied data stay exact**: expert slots and bank
   entries, restored checkpoints and states.
3. **Deterministic replay is a cheap default, not a law.** The same request
   sequence giving the same output is kept where it costs nothing (it makes
   measurements comparable) and may be dropped by a mechanism that buys
   measured speed (non-blocking cache admission, the CPU's share of the
   misses).
4. **Checkpoints may load together with an adaptive expert cache**, as in
   the references: Flash-Next resumes from a checkpoint while its cache
   follows the conversation. A restored continuation is held to the
   answer-level bar.
5. The byte-equality cells in §5 stay as **tripwires**: a red one on a
   change that passes the answer-level bar is reported, not a veto.

What holds today, measured: a resumed turn against a full re-prefill with
the same batch breaks gives the same text (193/193 and 209/209 characters);
against a re-prefill with other batch breaks the text moves and the answers
stay right (`measured-here`, 2026-10-05, `docs/llama-engine.md`). Both
mutants (state not loaded; an older checkpoint's state under a newer one's
count) fail that test.

### 3.5 Speculative decoding

- **The model's own MTP head** (`--llama-mtp N`) drafts up to N tokens; the
  target verifies them in one decode, and arcint's sampler walks the verified
  rows in order, keeping the draft while it agrees, so the emitted tokens are
  what the plain loop would sample, greedy or seeded sampled
  (`src/exec/verify_walk.h`, a pure function with its own tests). The drafter
  is the single-head case of llama.cpp's `draft-mtp` rebuilt on libllama
  (`src/exec/llama_spec.cpp`): llama.cpp's common library would compile a
  second cpp-httplib into the binary (`code`).
- Draft 0 comes from the catch-up batch, as Strata's does (0010).
  `--llama-mtp-min-p P` stops drafting at a token the head gives less than P
  (Strata's `--spec-min-p`); `--llama-mtp-vocab FILE` drafts from a subset of
  the vocabulary; the logits rows are bounded to 1 + N per lane.
- Served: the agent 5 drafts, the coder 4, Flash-Next 3 with the stop at 0.5
  (Strata's setting: 37.9 against Strata's 37.8 t/s on Strata's request,
  2026-10-14, `measured-here`).
- DFlash2 drafting exists in llama.cpp at the pin (`draft-dflash`, `code`)
  and is not wired here (§8).

### 3.6 Sampling

Greedy, temperature, top-k, top-p, min-p, repetition, presence and frequency
penalties, host-side (`src/core/sampler.cpp`). Penalties apply before the
greedy decision. min-p follows llama.cpp (`llama_sampler_min_p_apply`,
`code`): it keeps p >= min_p · p_max on the logits before temperature, after
top-p. Seeded; an unseeded request gets a logged seed. Defaults chain request
> operator flags (`--temp`, `--top-p`, `--top-k`, `--min-p`,
`--repetition-penalty`, `--presence-penalty`, `--chat-template-kwarg
enable_thinking=BOOL`) > the GGUF's defaults; `/props` reports values and
provenance (`code`).

### 3.7 Tokenizer, templates, tool calls

- Tokenizer and chat template come from the GGUF.
- Incremental detokenisation never splits a UTF-8 code point across SSE
  chunks; stop strings and token ids; `usage` in every response (`code`,
  unit-tested).
- **Reasoning.** When the rendered prompt ends inside `<think>`, everything
  before the first `</think>` goes to `reasoning_content`; `enable_thinking`
  and `reasoning_effort` are honoured (`code`).
- **Tool calls.** The models' native format is parsed into OpenAI
  `tool_calls`; requests without `tools` get raw text. A template that
  iterates arguments as a mapping is handed the parsed object; minja
  polyfills stay off (`code`).
- **Cancellation.** A dropped connection aborts at the next step boundary
  and frees the lane (`code`; gated by the concurrency suite, §5).

### 3.8 Context-overflow policy

A prompt or continuation that exceeds the context is **rejected with HTTP
400** and a JSON body with `prompt_tokens`, `n_ctx` and `overflow`. No
server-side truncation, context shift or sliding window: a server-side
history edit changes what the model saw and breaks prefix reuse, and a GDN
state cannot un-see a token, so any shift is an approximation. History
management is the client's job. With named lanes the 400 quotes the lane's
own context. This invariant is unamended (`decision`, operator; `code`: the
400 and its numbers are unit-tested).

### 3.9 MoE expert execution

The coder's 184 experts are all resident on the A770. Flash-Next's 512
experts a layer are not: with `--llama-cpu-moe 48` their weights load into
host memory and the **expert cache** (0021, after Strata) serves them:

| piece | what it does | where | evidence |
|---|---|---|---|
| slots | the hot experts in VRAM, sized in bytes (`--llama-expert-cache MIB`), filled in the profile's rank order | 0021, 0024 (one cache per card) | `measured-here` |
| bank | every other expert in USM host memory the card reads by pointer: a decode miss is read over the link in place, as Strata's kernels read its pinned mirror | 0021 | `measured-here` |
| adapt | swaps from decayed usage counts on Strata's constants (`LLAMA_EXPERT_CACHE_{ADAPT_EVERY,MAX_SWAPS,MIN_GAIN,MIN_USAGE,DECAY}`), on their own queue right behind a graph, the victim to a spare bank entry (non-blocking admission) | 0042, 0046 | `measured-here` (verify input copies 2.6-6.0 -> 0.94 ms) |
| policy | Strata's rule by default; Infernix's LFRU opt-in (`LLAMA_EXPERT_CACHE_POLICY=lfru`): +1.6 % on the gate set, -2.1 % on six other prompts, not a general gain | 0055 | `measured-here` |
| CPU misses | up to 16 missed (token, expert) pairs of a decode layer computed on the CPU from the bank with Strata's multi-token AVX2 rows while the card computes the hits and reads the other misses (`LLAMA_EXPERT_CACHE_CPU_MISSES=1`, 8 threads) | 0056 | `measured-here` (decode +15.2 % on six answers, KL +0.0028) |
| prefill | the bank stream: from 256 tokens a ubatch each bank tensor is copied whole on a copy queue into a VRAM ring ahead of its GEMM (`GGML_OPENCL_BANK_STREAM=1`); below, a GEMM gathers its routed bank experts; below 64 tokens the cache's matvec path | 0032, 0043, 0044 | `measured-here` (the 20k needle 408 -> 645 t/s in 0.7.3; 690 in 0.7.12) |
| n-gram table | the 27 GiB embedding table read lazily (`MADV_RANDOM`, a graph's rows prefetched together) | arcint, 0.7.5 | `measured-here` (2.0-7.7 -> 0.3-0.6 ms a verify) |

Two facts every expert-cache measurement must state: the slot budget and
the profile, and whether the bank's pages were warm (a repeated prompt hides
the page cache: an IQ3_XXS "28 t/s" was warm pages, 4-5 t/s from ZFS,
`measured-here`). Serve Flash-Next from the NVMe, not from the ZFS pool.

### 3.10 Lanes and two cards

- **Named lanes** (`--served-model-name A,B --lane-ctx CA,CB`): one process,
  one copy of the weights, a context per lane, one unified KV pool in which
  each lane attends over its own window of cells (0026, 0029). With both
  lanes decoding, a step combiner drafts for both in one MTP loop and
  verifies both in one `llama_decode` (0028): 21.6 + 21.7 t/s against 15.9 +
  15.8 taking turns (dense agent, MTP 5, `measured-here`). Recurrent state
  and checkpoints stay per lane. Flash-Next's indexed memory refuses
  windows. The production agent runs one lane: the two-lane test drive of
  2026-10-08 ended because a lane slowed with the other's pool fill before
  0026 (`docs/campaigns/lanes-agent-subagent.md`).
- **Two cards** (`--device GPU.0,GPU.1 --llama-layer-split K`): layers
  [0, K) on the first card, [K, n_layer) and the output head on the second,
  each card with its own KV, recurrent state and expert cache
  (`--llama-expert-cache A,B`), after Strata's layer split. Flash-Next at
  131,072 tokens: 21.7 t/s split at layer 36 against 19.8 on the B60 alone
  (`measured-here`, 0.7.0; `docs/campaigns/layer-split-two-cards.md`). Not
  a production configuration.

## 4. Serving surface

| endpoint | contract |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; SSE and non-stream; `chat_template_kwargs.enable_thinking`, `reasoning_effort`; `tools` |
| `POST /v1/completions` | raw completion; accepts token-id prompts |
| `GET /v1/models` | one entry per served name with the context it is **running with** (§4.2) |
| `GET /health` | model, loaded, slots free/total, queue depth; per lane with named lanes |
| `GET /props` | the served file and architecture, context, KV types, MTP state, sampler defaults and provenance, build info |

`usage.completion_tokens_details` carries `accepted_prediction_tokens` /
`rejected_prediction_tokens`. Console output goes to stderr, one greppable
line per event (`lgc  load:`, `llama:`, `http:`, `slot N: prefill … |
decode …`, `expert cache:` per request); `-v`, `-vv`. The stub backend
(`--stub`) serves the whole surface without a model for the device-free
suites.

### 4.1 Lanes

`--parallel N` equal lanes, or named lanes (§3.10), up to 4. A lane is one
sequence's state: its KV cells, its recurrent state, its checkpoints, its MTP
draft state. Each lane has its own slot pool: a request for a busy lane
waits up to `--queue-timeout S` (the units set 30) and then gets a 503 with
the lane's free count; it never takes another lane.

### 4.2 The name, and the context, are contracts with a proxy

- `--served-model-name NAME` sets the id on `/v1/models`, `/props` and in
  responses. With one name, any `model` value is served and `/props`
  publishes `enforces_model_field: false`.
- The `/v1/models` entry carries `n_ctx` (what this process runs with),
  `n_ctx_train`, `quant` and `lanes`, because a discovering proxy reads
  context only from there (`code`; learned by breaking a proxy).
- [Amended 2026-10-07, operator.] Named lanes (`--served-model-name A,B
  --lane-ctx CA,CB`) make the `model` field binding: a lane's name picks
  that lane. An empty name and the GGUF's canonical id go to the first lane,
  and an unknown name gets a 404. `/health` adds a `lanes` array with each
  lane's free count; `/v1/models` carries one entry per lane name with that
  lane's `n_ctx`; `/props` reports `enforces_model_field: true`.

### 4.3 Admission

The context is configuration: `--n-ctx` (default the smaller of 32,768 and the trained context), or `--lane-ctx`
per lane. Whether it fits is decided once, by measurement at the deepest
admitted prompt (§3.3), not per request. A request is refused only by §3.8
(400) or by a full lane after `--queue-timeout` (503).

### 4.4 The command line

`--engine llama --gguf FILE` selects the engine and the file (`--engine` is
kept for the units' compatibility; `llama` is its only value, and `ov` is
refused with a pointer to the `openvino-history` tag). The engine's flags
are `--llama-*`; `arcint --help` and `llm.txt` list them. The units in
`contrib/systemd/` carry the production flags.

## 5. Testing and acceptance

**The bar** is the answer-level bar of §3.4 / `CLAUDE.md`, per phase: a
change that measurably improves prefill or decode and does not regress the
other beyond the run-to-run spread is adopted if it passes the answer-level
bar (`decision`, operator, 2026-10-01). Byte-equality cells are tripwires:
reported, not a veto.

- **The acceptance task (Prüfstand)**: a Lua CSV parser to RFC 4180, ten
  cases, scored by executing the candidate. Through the deployed package, at
  temperature 0: **≥ 9/10 is the bar, 10/10 the target**, for the coder and
  the agent (`decision`, operator, 2026-10-15). The score is a floor, not a
  fingerprint: two different programs can both score 10/10, and a greedy
  score is one trajectory (a batch layout can flip one case; the sampled
  task separates configurations, `docs/llama-engine.md`).
- **Facts and needles**: every served check asks the capital of France and
  finds a needle at depth (20k for Flash-Next, 55k and 138k for the agent,
  41k for the coder).
- **KL as served** against a reference on the CPU or a higher quant
  (`llama-perplexity -lm mmap --no-repack`, `-ub 4` where the expert cache
  is on; the batch must exercise the changed path), baseline arm and
  candidate on the same card and window.
- **Performance floors**: per service, the last release's pre-tag served
  numbers minus their run-to-run spread (`decision`, operator, 2026-10-15).
  The `*-served` cells carry 0.8.0's (§7.1): two processes a service, the
  floor the lower run minus their spread, which back to back was 0.0-0.4 %.
- **Equivalence suite** (`tests/equivalence/run.sh`, on a card, `--engine
  llama --gguf`): two greedy runs, a lane's reused prefix against a fresh
  process (warm against cold), a checkpoint resume against a full
  re-prefill, MTP on against off, one lane against two. Answers right in
  every arm is the gate; byte equality is the tripwire.
- **Concurrency suite** (`tests/concurrency/run.py`): no cross-lane bleed
  in both start orders, both lanes used, the 404 for an unknown lane name,
  cancellation, the 503 with the lane's numbers after the queue timeout.
  Verified red before green.
- **Op suites**: `test-backend-ops` on both cards for each op family the
  series touches (MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET,
  SSM_CONV, GET_ROWS, CPY, SET_ROWS, TOP_K, FILL, RMS_NORM, SOFT_MAX), and
  each patch's red case (`contrib/llama.cpp/README.md`). A wrong backend
  name skips every case and prints OK: check the case count.
- **Unit tests** (`ctest -L unit`, device-free, stub build): config and the
  refusals, the verify walk, lanes and the step combiner's rules, chat
  rendering and tool calls, the HTTP round trip (`tests/roundtrip.sh`), lane
  stress, the acceptance enumeration's consistency. Warning-clean under
  `-Wall -Wextra -Wpedantic -Werror`.
- **Sanitizers**: ASan + UBSan, no-recover, on x86_64.
- **A test must be able to fail**: the red case runs first.

### 5.1 The test ladder: what runs when

| class | cadence | card time | members |
|---|---|---|---|
| Unit tests | every commit | seconds to minutes | `ctest -L unit` in the stub build; the op suites and red case of any patch that changes |
| Milestone gates | once per increment | one card window | KL as served, the op suites, and the served check of the service the change touches, one process per configuration |
| Acceptance | once per release, before the tag | about an hour per card | <!-- BEGIN GENERATED by tools/acceptance_manifest.py; verify with its check mode -->`agent-equivalence` gates the answers right in every arm: two greedy runs, a lane's reused prefix against a fresh process, a checkpoint resume against a full re-prefill, MTP on against off, one lane against two (reports byte equality of each pair (tripwire, §3.4 point 5)); `coder-equivalence` gates the answers right in every arm: two greedy runs, a lane's reused prefix against a fresh process, a checkpoint resume against a full re-prefill, MTP on against off, one lane against two (reports byte equality of each pair (tripwire, §3.4 point 5)); `agent-concurrency` gates no cross-lane bleed in both start orders, both lanes used, the 404 for an unknown lane name, cancellation, the 503 with the lane's numbers after the queue timeout (references: not yet filled); `coder-concurrency` gates no cross-lane bleed in both start orders, both lanes used, the 404 for an unknown lane name, cancellation, the 503 with the lane's numbers after the queue timeout (references: not yet filled); `agent-served` gates Paris and the needle right; prefill at both depths at or above the release floor (the last release's pre-tag number minus its run-to-run spread, §5) (reports peak VRAM and GTT) (references: prefill-55k 695.2 t/s (gate lower-is-worse at 694.9 t/s); prefill-138k 491.75 t/s (gate lower-is-worse at 491.3 t/s)); `coder-served` gates Paris and the needle right; decode on each answer at or above the release floor (§5) (reports peak VRAM and GTT) (references: decode-a 51.1 t/s (gate lower-is-worse at 51.1 t/s); decode-b 47.5 t/s (gate lower-is-worse at 47.5 t/s); decode-c 44.05 t/s (gate lower-is-worse at 43.9 t/s); prefill-41k 725.45 t/s (gate lower-is-worse at 725.3 t/s)); `flashnext-served` gates Paris and the needle right; prefill at 20k and decode at or above the release floor (§5) (reports peak VRAM and GTT) (references: prefill-20k 595.1 t/s (gate lower-is-worse at 594.8 t/s); decode 35.05 t/s (gate lower-is-worse at 33.4 t/s)); `sanitizers` gates zero sanitizer reports (reports the stub build (no card): unit, roundtrip, stress); `package-build` gates the package build succeeds, is version-stamped, and its ggml-cpu is native (AVX2); a post-deploy smoke run and a fresh container follow the install, outside this cell; `pruefstand` gates at least 9/10 on the coder through the deployed package (10/10 the target, §5) (references: score 9 points (gate lower-is-worse at 9 points)); `pruefstand-agent` gates at least 9/10 on the agent through the deployed package (10/10 the target, §5) (references: score 10 points (gate lower-is-worse at 9 points)) <!-- END GENERATED by tools/acceptance_manifest.py --> |

The release procedure around it (pre-tag served check, second-model review,
the package built and checked, a fresh container) is
`docs/release-checklist.md`. The unit ladder runs on the exact tree before
every tag.

### 5.2 Operating facts that prevent damage

- **Card identity by PCI id**, never by DRM number: DRM numbering is
  inverted against arcint's `GPU.N` (`docs/sop-card-window.md`). `GPU.0` =
  B60, `GPU.1` = A770.
- **The cards are full while the production units run.** Stop the owning
  unit for a window and restore it after; two processes on one card under
  xe produce GPU faults (job timeouts, resets), not slow sharing. Wedged
  tests ignore SIGTERM: kill by pid with SIGKILL and verify the card with a
  real completion.
- **A wedged xe card** (GuC job timeouts, an oops in the xe scheduler, a
  process stuck in `dma_fence_default_wait`) has only been recovered by a
  host reboot, and **rebooting the dev host is the operator's call**, every
  time (`decision`). Read the host's captured kernel log first. A
  copy-engine CAT error after an out-of-bounds probe is recovered by killing
  the stuck process (`measured-here`).
- **The B60's first job after a long idle can wedge**: run legs back to
  back, or hold the device's runtime power on.
- **Forecast host memory before any launch**: visible + driver-side
  (`drm-resident-*` in fdinfo) + page cache / ZFS ARC (absent from
  `MemAvailable`) + pinned memory (the expert bank is pinned USM) + the
  fence.
- **VRAM**: size the peak at the deepest prompt, not the post-load free
  (§3.3); sample fdinfo's `resident-vram0` against `resident-gtt` during a
  leg.
- **The A770's GT clock is pinned at 2000 MHz** (operator, 2026-09-27):
  any clock A/B states the pin.
- **One fresh process per measurement cell**; a fresh process's first
  request compiles kernels: label first against warm.
- **Different prompts per repeated answer** (a repeated prompt hides page
  cache and expert-cache warmth), and the depth, card, KV types and flags of
  every number.
- **Verify the artifact carries the change**: the binary's sha in every leg
  header; `dpkg -V arcint` for a package leg; the package's ggml-cpu must
  carry AVX2 (`build-deb.sh` checks `ggml_vec_dot_q4_K_q8_K` for `%ymm`).
- **One measurement: at most 2 hours**, forecast from a bounded leg; above
  it, the row is "to be determined" (`decision`, operator).
- Profiles taken at a past-0 chunk overstate every node share.

## 6. Why one engine, and why this one

The OpenVINO engine came first (0.1-0.5.4): OpenVINO's GPU plugin emitted
good Xe code and arcint built the serving layer around it. The libllama
engine (0.5.5, 2026-10-03) replaced it service by service, on measurement
(`measured-here`):

- **The dense agent** (0.5.6): decode with MTP about twice the OpenVINO
  service's (47 against 24.6 t/s on the acceptance prompt); prefill at depth
  since caught up and passed it (695 t/s at 55k, 492 at 138k in 0.7.12,
  against the OpenVINO service's 852 at 16k and 377.5 at 71.7k).
- **The coder** (2026-10-05): a searched GGUF that fits 98,304 tokens with
  MTP on the 16 GiB card and passes the answer-level bar.
- **Flash-Next** (0.6.0): the OpenVINO engine's best full-depth decode was
  6.5 t/s with 63 t/s prefill at 20k (the CPU tier computing 64 % of the
  routed experts); the libllama engine with the expert cache decodes 43.4 and
  prefills 690 t/s at 20k (0.7.12).
- **No export step**: the GGUF as published, against a per-model IR export,
  a template IR for GGUF weights, and the plugin patch series 0003-0077 on
  a pinned OpenVINO nightly.

From 2026-10-05 nothing in production ran OpenVINO, and the operator retired
it on 2026-10-15 after this document was rewritten (charter item A6). What
the OpenVINO era measured and established (the paged executor, the exact
prefix cache, admission by measured reservation, the plugin's MoE tier, the
GGUF repack) is at the tag `openvino-history`.

## 7. State of play

### 7.0 Milestones

| line | content | state |
|---|---|---|
| 0.1-0.5.4 | the OpenVINO engine: paged KV, prefix cache, MTP, two lanes, the K-quant plugin series, Flash-Next's serving shape and expert tier (shipped beside the libllama engine until 0.7.12) | retired; `openvino-history` |
| 0.5.5-0.5.12 | the libllama engine: Intel kernels 0001-0020, MTP drafting, quantized KV, context checkpoints; the agent (0.5.6), the writer (0.5.8) and the coder (2026-10-05) on it | done |
| 0.6.0-0.6.x | Flash-Next on the libllama engine: the expert cache, the IQ kernels, its MTP layer | done |
| 0.7.0-0.7.12 | named lanes and the step combiner, two cards by layers, the bank stream, decode launches, GDN snapshots, the CPU misses, the pin moved to `3e4dfa8` | done; 0.7.12 released 2026-10-15 |
| 0.8.0 | the OpenVINO engine and `marfrit-openvino` retired; this document; the suites on the libllama engine; IQ3_XXS as a Flash-Next option | released 2026-10-15 |
| 1.0.0 | feature complete, almost no gains by obvious changes | charter `docs/milestone-1.0.0.md` |

Every open lever or defect is a campaign in `docs/campaigns/`.

### 7.1 Current numbers

0.8.0's pre-tag check: the release package, the units' flags, the served
check of each service (`tests/acceptance/cells/served_check.py`) in two fresh
processes, the model on the NVMe, `measured-here`, 2026-10-15. Decode is a
300-token answer; prefill excludes nothing (cold lanes). Flash-Next asks the
needle right after the capital, with fewer warm-up requests than 0.7.12's
check had (690.3 t/s and 43.4 there); the 0.8.0 dev build under that older
order decoded the same answer at 43.4.

| service | card | configuration | prefill | decode | answers |
|---|---|---|---|---|---|
| agent | B60 | Q4_K_M, 245,760 ctx, q8_0:q4_0 KV, MTP 5 | 695.3 / 695.1 t/s at 54,895 tokens, 491.9 / 491.6 at 138,028 | not in the pre-tag check | Paris, the needle at 138k, the task 10/10 |
| coder | A770 | `c5f495ac-shq8`, 98,304 ctx, q8_0:q4_0 KV, MTP 4 | 725.5 / 725.4 t/s at 40,841 tokens | 51.1 / 47.5 / 44.0-44.1 t/s on three answers | Paris, the needle, the task 9/10 |
| Flash-Next | B60 + host | IQ3_XXS, 131,072 ctx, q8_0 KV, 12,000 MiB of slots (7,208 slots), bank 28.72 GiB, MTP 3 with min-p 0.5, Q4_K draft head, the bank stream, the CPU misses | 595.2 / 595.0 t/s at 20,045 tokens | 35.6 / 34.5 t/s | Paris, the needle |
| Flash-Next, IQ2_XS option | B60 + host | IQ2_XS, 13,400 MiB of slots (9,724), bank 20.32 GiB, the rest as above | 665.8 / 665.4 t/s | 42.2 / 42.3 t/s | Paris, the needle |

The agent's decode in the same release: 36.6 / 32.3 / 19.3 t/s at 4k / 29k /
130k prompt tokens (2026-10-08 protocol; since then 0016-0051 moved it,
not re-measured as a table). The writer: 25.2 t/s short, 18.2 at 16k
(0.5.8).

**Against the reference**: Strata's SYCL port on the same B60 decodes 37.8
t/s on its own request and prefills 620 t/s at the 20k needle
(`measured-here`, 2026-10-14); arcint 43.8-44.0 and 690 on the same
request and needle with the CPU misses (charter B3).

**Quality**: Flash-Next's default IQ3_XXS has KL 0.218 nats and top-1
89.7 % against unsloth's Q8_0 (16 x 512), IQ2_XS 0.294 / 87.1 %; both are
outside the answer-level bar, a shortfall the operator accepted (A2,
2026-10-15), and IQ3_XXS became the default for its accuracy the same day.
The other quants are options (§7.2). The dense agent's q8_0:q4_0 KV: KL 0.005652 /
97.30 % against the CPU reference; the coder's searched quant: 0.02367 /
95.43 % against Q8_0 (`measured-here`, `docs/llama-engine.md`).

### 7.2 Flash-Next quant options

IQ3_XXS is the default (operator, 2026-10-15, for its accuracy; IQ2_XS was
the shape until then); the others stay available as on-demand configurations
on the same endpoint (`decision`, operator, 2026-10-15; `docs/llama-engine.md`,
"Flash-Next quant options"):

| GGUF | KL / top-1 against unsloth's Q8_0 on the CPU (16 x 512) | host bank | state |
|---|---|---|---|
| ISTA-DASLab IQ2_XS | 0.294 / 87.1 % | 20.32 GiB at 13,400 MiB of slots | the faster option, from the Flash-Next unit with its own `--gguf` and slot budget: 665.6 t/s at the 20k needle, 42.3 t/s |
| ISTA-DASLab IQ3_XXS | 0.218 / 89.7 % (served) | 28.72 GiB at 12,000 MiB of slots | the default (`arcint-flashnext`): 595.1 t/s at the 20k needle, 34.5-35.6 t/s on the hash-table answer (0.8.0), 30.6-37.5 t/s on six answers with the CPU misses; their KL cost +0.0044 nats against the misses off on the served reference capture (8 x 512, ubatch 4: 0.2910 / 0.2866), a different window from this column's (`measured-here`, 2026-10-15) |
| unsloth UD-Q3_K_XL | 0.100 / 93.3 % | ~39 GiB | inside the bar; needs the planned 96 GB host |

### 7.3 Retractions

A mechanism that was narrated and not measured is retracted on the record
when it falls, rather than edited away (`CLAUDE.md`). The retracted text
lives in git history and in the campaign documents; this document carries
the corrected fact. Two that shaped the work: a repeated prompt's warm page
cache was once read as a 28 t/s IQ3_XXS decode (4-5 t/s from ZFS); and every
package from 0.5.5 to 0.7.11 shipped an SSE-only ggml-cpu (the recipe's
`SOURCE_DATE_EPOCH` turned `GGML_NATIVE` off), found in 0.7.12 when the
CPU misses made it cost 7.7 %.

## 8. Open items, against the references

Each item names the reference to follow and the measured starting point.

### 8.1 The soak (charter C2)

Several days of the three services under real use on both cards, logged:
no wedge, no eviction, no restart. The record has the B60's wedge after a
long idle, xe eviction at thin VRAM margins and copy-engine resets, each
recovered, none soaked.

### 8.2 Owed from the pin move to `3e4dfa8`

The patches' own red cases and the four-sequence KL arms on the new base
(`contrib/llama.cpp/README.md`, "Moving the pin", step 3). Upstream's open
work that meets the series (`docs/campaigns/research-upstream-llama-2026-10.md`):
the OpenCL backend split into per-op files (#29354, the natural shape for
upstreaming), #30345 (supersedes 0010), #26631 (MUL_MAT_ID id -1), #30539
(OpenCL declining tensors above the device's largest allocation: check
before a move past it).

### 8.3 Flash-Next's quant options

- **IQ3_XXS**: the default since 2026-10-15 (§7.2). Its 18 IQ4_NL down
  projections made the CPU misses fault before 0.8.0 (the slow path read the
  split IQ4_NL bank as GGUF blocks; fixed in 0056, `measured-here`). Owed:
  its slot budget searched finer than the two points measured (12,000 MiB
  serves, 13,400 thrashed).
- **UD-Q3_K_XL**: served at 235 t/s prefill and 12-13 t/s decode without
  MTP before the bank stream and the CPU misses (2026-10-06); re-measure on
  the 96 GB host.

### 8.4 Conversation state across a restart

- **Reference.** Strata parks conversations (GDN state, PLE tails, KV) and
  reuses them on token equality (`include/strata/core/conversation_cache.hpp`,
  `conversation_snapshot.hpp`); FreeToken snapshots recurrent state at
  tool-call anchors (`python/freetoken/scheduler/cache.py:151`).
- **arcint now**: checkpoints in process only (§3.4); the campaign is
  `kv-checkpoint-restore` (backlog).

### 8.5 DFlash2 on the agent

llama.cpp at the pin carries DFlash and DFlash2 drafting (`draft-dflash`,
`src/models/dflash.cpp`, `code`). On the OpenVINO engine DFlash2 decoded
44.8 against MTP's 33.0 t/s at short depth (`measured-here`, 2026-09). Not
wired on the libllama engine, unmeasured.

### 8.6 Levers priced below 2 % (charter B1)

Priced by a ceiling and not built (`measured-here`, 2026-10-15): the GDN
conv stage fused (vllm-xpu-kernels; ~1.1 % of the agent's decode, ~1.4 % of
the coder's); the activation quantization folded into its producers
(OpenVINO's post-op fusion; at most ~1.4 % of a round); Infernix's global
expert pool (+0.4 points of hits, ~0.5 %); the rest of the decode kernel
count (~13 % of a Flash-Next round, many small fusions, none obvious alone).

### 8.7 Interface (charters A7, C4)

Which switches and flags stay stable within 1.x is the operator's
(`docs/switches.md`). 0.8.0 changes the surface: the OpenVINO engine's
flags are refused by name, `/props` and `/health` lose its fields (the
reservation, the prefix cache, the paged-KV block), `/props` gains
`model.architecture`, `quant` names the GGUF's file type, `kv_dtype`
reports `--llama-kv` (it read "f16" whatever the flag said before), and
the 503 bodies lose `reservation` and `kv_blocks_free`.

### 8.8 Other open items

- **The B60's run-to-run determinism** on this engine is not measured; the
  OpenVINO path's GDN state output was not bit-identical on the B60 (0.136
  nats mean between identical forwards) and was on the A770.
- **xe page faults** with compute-engine resets logged in windows where
  `test-backend-ops` ran (2026-10-03); a full `MUL_MAT`/`MUL_MAT_ID` run
  reproduced none; not attributed.
- **The direct-submission fault**: the kernel-side ring-ordering fix
  (`drm/xe` "Order ring writes before ring tail updates") on the dev host is
  the operator's decision.
- **Multimodal**, an Anthropic Messages adapter and a local CLI: not
  committed (`decision`, operator); the request parser rejects non-text
  content parts.

# arcint

A deliberately narrow LLM inference engine for Intel Arc GPUs.

arcint serves one model family, the hybrid GatedDeltaNet/attention Qwen
models, on two cards: the Arc A770 (16 GiB) and the Arc Pro B60 (24 GB). The
inspiration is [NInfer](https://github.com/Neroued/ninfer), a from-scratch
engine for a closed set of checkpoints fully resident on one GPU (the dense
27Bs and the Qwen3.6-35B-A3B) that beats every generalist on them. arcint translates the idea to Intel: kernel work goes through
OpenVINO's compiler stack plus a published GPU-plugin patch series, and arcint
owns everything around the compute graph: the serving loop, the scheduler,
the KV and recurrent-state memory, prefix caching, admission by measured
reservation, expert placement for large mixture-of-experts models, and
speculative decoding.

Most layers of every target model use linear attention (GatedDeltaNet), a
minority full attention. The design follows from that; see
[DESIGN.md](DESIGN.md). [llm.txt](llm.txt) is the machine-readable summary.
For mixture-of-experts serving beyond the card's memory, arcint follows the
expert engines [Strata](https://github.com/Niko1221/Strata) and
[FreeToken](https://github.com/FlashML-org/FreeToken)
([docs/campaigns/research-reference-audit.md](docs/campaigns/research-reference-audit.md)).

## Models

| model | architecture | served form | card |
|---|---|---|---|
| Qwen3.6-27B-A3B-Coder | `qwen3_5_moe`, 40 layers, 184 experts (pruned from 256) | int4 AWQ IR | A770 (production) |
| Qwen3.8-27B | `qwen3_5` dense, 64 layers, MTP head | Q4_K_M GGUF on the libllama engine (production since 0.5.6); int4 IR, or a GGUF on that IR | B60 (production) |
| Qwen3.6-35B-A3B | `qwen3_5_moe`, 40 layers, 256 experts | the checkpoint's own IQ2_S/IQ3_XXS expert blocks, all resident | A770 |
| Cydonia 24B v4.3 (Mistral Small 3.2 finetune) | `llama`, 40 layers, all full attention, 32 query heads on 8, head size 128 | imatrix Q4_K_M GGUF on the libllama engine, q8_0 KV, 98,304 ctx | B60, on demand (creative writing, since 0.5.8) |
| Qwen3.8-Flash-Next | `qwen4_exp`, 48 layers, 512 experts (10 routed + 1 shared), n-gram embedding table | ISTA-DASLab's IQ2_XS GGUF on the libllama engine, every expert on the card through an expert cache, MTP (since 0.6.0); or the checkpoint's own expert blocks on the OpenVINO engine, experts split between card and a host RAM bank | B60 |

## Two engines

arcint has two inference backends behind the same HTTP surface, sampler,
lanes and chat templates. Both services run on the second: the agent from
0.5.6, the coder from 2026-10-05, and Flash-Next from 0.6.0. The first
carries Flash-Next's full-depth checkpoint blocks and stays as the coder's
way back.

**OpenVINO (the default).** The model is an OpenVINO IR, or a GGUF fed
through it. Runs on the patched `marfrit-openvino` runtime. arcint owns the
whole memory picture here:
- paged KV in u8 or i8:u8;
- the exact prefix cache, in VRAM and in host RAM;
- admission by measured reservation;
- the Flash-Next expert tier between card and RAM;
- DFlash and MTP drafting.

That is what carried the coder service's 98,304 tokens until 2026-10-05
(and the agent's 122,880 before 0.5.6), and it carries Flash-Next at depth.

**libllama (`--engine llama --gguf FILE`).** llama.cpp at a pinned commit
with ggml's OpenCL backend and arcint's Intel kernels
(`contrib/llama.cpp/patches`: K-quant matvec and XMX GEMMs, int8 DPAS,
gated delta-net, decode and prompt attention). It takes the GGUF as it is,
with no export step, and drafts with the GGUF's own MTP head (`--llama-mtp
N`, `--llama-mtp-vocab`). Its numbers (`measured-here`, 2026-10-04; the
acceptance task 10/10 at temperature 0 on both):

| | coder, A770 | dense 27B, B60 |
|---|---|---|
| decode, MTP | 79.3 t/s (4 drafts) | 52.6 t/s (5 drafts) |
| prefill, 4,096 tokens | 1,431 t/s | 935 t/s |
| OpenVINO service, decode / prefill at 4k (table below) | 43.9 / 1,379 t/s | 24.6 / 1,141 t/s |

The two engines run different weights: the services' int4 IRs are ~18 %
fewer bytes than the GGUFs' Q4_K_M. The llama-engine figures are
llama-bench prefill and served decode on the acceptance prompt, so they are
not the services' long-context conditions.

Context on the llama engine (`measured-here`, 2026-10-04). The KV cache is
f16 by default; `--llama-kv q8_0` (or `q8_0:q4_0`) quantizes it, on
arcint's attention kernels (`contrib/llama.cpp` 0015). `--n-ctx` defaults to
32,768.
- **Dense 27B, B60:** with f16 KV, 122,880 tokens without MTP: 24.4 of
  25.7 GB of VRAM in use. A 30,065-token prompt prefills at 528 t/s at
  `--n-ctx 122880` and at `--n-ctx 32768` alike, so nothing pages. A
  120,945-token prompt took 579 s (209 t/s on average). With MTP (5 drafts)
  and `--llama-kv q8_0`, 131,072 tokens:
  - peak VRAM 23.06 GB;
  - a 128,133-token prompt in 772 s (166 t/s), then decode at 7.5 t/s at
    that depth;
  - the acceptance task 8/10 at temperature 0 as deployed; sampled, q8_0
    within the noise of f16 (`docs/llama-engine.md`).

  f16 KV with MTP at 131,072 overcommits the card.
- **Coder, A770:** bound by its weights, not its KV. The Q4_K_M GGUF is
  14.9 GiB on the 16 GB card, while 98,304 tokens of its KV would be
  1.9 GiB. With MTP, VRAM pages over the card's x4 link beyond ~16,384
  tokens (prefill 585 t/s at 24,576).

Production (operator, 2026-10-04):
- **The agent (dense 27B, B60) runs the libllama engine from 0.5.6**, at
  131,072 tokens with MTP and `--llama-kv q8_0`. Measured on llama-bench
  against the OpenVINO service's extension prefill:
  - decode with MTP is about twice the OpenVINO service's (47 against 24.6
    t/s on the acceptance prompt);
  - prefill is slower: 896 t/s at 4k (OpenVINO 1,141) and 435 at 16k
    depth (852).
- **The coder moved to libllama on 2026-10-05**, with a searched GGUF.
  - Context: the Q4_K_M GGUF leaves ~16k tokens with MTP on the A770. An
    evolutionary search over per-layer expert types found one that serves
    98,304 tokens and passes the answer-level bar (top-1 -0.10 points
    against Q4_K_M, KL lower). It scores 10/10 on the acceptance task
    (`docs/llama-engine.md`). The OpenVINO service stays as the way back.
  - Prefix reuse: the libllama engine has no shared prefix cache. A lane
    reuses the prefix it holds, and llama.cpp's context checkpoints let a
    hybrid model resume a follow-up or an edited message
    (`--llama-checkpoints`).
- Flash-Next (since 0.6.0): an expert cache keeps the hot experts in VRAM
  slots and the rest in a USM bank in host memory, every expert computed on
  the card (`--llama-expert-cache`, `--llama-expert-profile`); the
  checkpoint's MTP layer drafts from a separate GGUF (`--llama-mtp-gguf`).

`docs/llama-engine.md` has the details.

## Current numbers

Every number names the card and configuration; all are `measured-here`.
"task" is the acceptance task: a Lua CSV parser to RFC 4180, ten named cases,
the candidate code executed, one point per case.

**Production services** (2026-09-24, one fresh process per arm, 32 greedy
tokens after the prompt; extension prefill excludes prefix-cache hits):

| service | card | configuration | prefill @ 4k / 16k | decode @ 4k / 16k |
|---|---|---|---|---|
| coder, until 2026-10-05 | A770 | u8 KV, 98,304 ctx, 2 GiB prefix cache | 1,379 / 1,209 t/s | 43.9 / 42.4 t/s |
| dense agent, until 0.5.5 | B60 | `i8:u8` KV, 122,880 ctx, MTP on, `--gate-pad 16`, prefill chunk 512 | 1,141 / 852 t/s | 24.6 / 20.2 t/s |

Both artifacts score 10/10 on the task (greedy; the dense model with MTP
drafting).

From 0.5.6 the agent runs the libllama engine (2026-10-04, a different
protocol: llama-bench and the served acceptance prompt):
- Q4_K_M GGUF, `--llama-kv q8_0`, 131,072 ctx, MTP 5 drafts;
- prefill 896 t/s at 4k and 435 at 16k depth (llama-bench, no MTP);
- decode 47.1 t/s on the acceptance prompt, 7.5 t/s at 128k depth;
- the acceptance task 8/10 at temperature 0 on the deployed unit; sampled,
  30 runs, mean 7.4 against f16's 8.1, within the noise
  (`docs/llama-engine.md`).

From 2026-10-05 the coder runs it too (the served acceptance prompt and long
prompts through arcint):
- the searched `c5f495ac shq8` GGUF (14.6 GB), `--llama-kv q8_0:q4_0`,
  98,304 ctx, MTP 4 drafts;
- prefill 285 t/s on a 94,926-token prompt;
- decode 57.5 t/s on the acceptance prompt, 19.5 t/s at 95k depth;
- the acceptance task 10/10 at temperature 0, 10 of 10 sampled runs at
  10/10.

**Qwen3.6-35B-A3B**, full depth, native expert blocks with u8 dense
projections, all resident on the A770, u8 KV: prefill about 960 t/s at 4,096
tokens, decode about 28 t/s (`qwen3.6-35b-a3b-native-d40packed-u8`, plugin
patches 0059–0067).

**Qwen3.8-Flash-Next**, IQ2_XS (ISTA-DASLab GSQ-RCO) on the libllama engine,
B60, 14,500 MiB of expert slots, MTP 2: the 20,045-token needle prefills at
422 t/s, the 500-token long answer decodes at 33.0-34.8 t/s, the needle
answered (0.6.0; Strata's own engine: 620 / 37.2-37.8 t/s on the same card).
The KL of IQ2_XS against a reference is owed.

**Qwen3.8-Flash-Next**, full depth (`qwen3.8-flash-next-d48q8`) on the B60:
`--offload-ratio 75 --moe-cpu-tier`, a 128-expert-per-layer census seed, a
30 GiB host expert bank, u8 KV, chunk 2048; at a 20,085-token prompt, prefill
63.1 t/s and decode 6.5 t/s, the needle answered (`+p25`, patches
0003–0074). Levers, in build order: an adaptive GPU expert cache that
follows the conversation, with a link-probed share of each layer's misses
read from pinned host memory (after Strata); the doorbell hand-off between
GPU and CPU tier; multi-token drafting with the checkpoint's MTP head; and
prompt processing on the GPU with the missing experts streamed from pinned
memory.

**Speculative decoding on the dense Qwen3.8-27B** (B60, int4, u8 KV,
greedy, 400 tokens, prompts under 2,048 tokens, 32,768 context):

| drafter | decode | acceptance | max context (reservation) |
|---|---|---|---|
| none | 24.0 t/s | — | 199,712 |
| MTP head (`--mtp on`) | 33.0 t/s | 76.7 %, 1 draft per pass | 155,680 |
| **DFlash2 int4 (`--dflash`)** | **44.8 t/s** | 3.13 tokens per verify cycle | 136,640 |
| DFlash2 int4, draft on the A770 (`--dflash-device`) | 39.8 t/s | 3.13 tokens per verify cycle | 171,904 |

The DFlash2 drafter is the public block-diffusion head
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
exported with `tools/export_dflash.py`, drafting seven tokens per verify
pass. At 77k tokens DFlash2 decodes at 18.8 t/s against plain decoding's
15.3.

**GGUF opened on the dense IR** (Unsloth's Q4_K_M of Qwen3.8-27B; B60, u8
KV, one lane, MTP off, warm prefill, decode from the step time):

| weights | prompt | prefill | decode | task |
|---|---|---|---|---|
| GGUF Q4_K_M, `--gguf-mode mixed` (the default) | 856 | 1,001 t/s | 18.2 t/s (54.8 ms) | 10/10 |
| GGUF Q4_K_M, `--gguf-mode mixed` | 71,727 | 464 t/s | 13.6 t/s (73.7 ms) | — |
| GGUF Q4_K_M, native rows (`--gguf-native`) | 856 | 213 t/s | 9.9 t/s | 10/10 |
| Intel int4 IR | 856 | 1,609 t/s | 23.1 t/s | — |
| Intel int4 IR | 71,727 | 552 t/s | 16.5 t/s | — |

## Building

C++20, CMake, no network at build time. `third_party/` holds the two vendored
single headers (cpp-httplib, nlohmann/json) and
[models/allowlist-raw.json](models/allowlist-raw.json) holds the IR metadata
the allowlist is pinned against.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure -L unit

    ./build/arcint --stub --port 8090 -v

This builds the **stub**, which serves the HTTP surface without a model and is
what the test suite runs against; `--model` is refused in this
configuration. `-L unit` selects the device-free gate (`arcint-test`, the
HTTP round trip, the stub-only concurrency stress test, and the acceptance
enumeration's own consistency checks). The card-requiring acceptance cells
are a separate, enumerated target
([docs/design-0.3.1-test-ladder.md](docs/design-0.3.1-test-ladder.md)); see
[DEVELOPMENT.md](DEVELOPMENT.md) for how to build and run them.

A build that serves a model needs OpenVINO:

    cmake -S . -B build-ov -DARCINT_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ov -j"$(nproc)"
    cmake --install build-ov --prefix ~/.local

The libllama engine needs a llama.cpp tree at the pinned commit (`bed0a85`)
with `contrib/llama.cpp/patches` applied, plus the OpenCL headers and ICD
loader:

    git -C <llama.cpp> apply <arcint>/contrib/llama.cpp/patches/*.patch
    cmake -S . -B build-ov -DARCINT_OPENVINO=ON -DARCINT_LLAMA=ON \
          -DARCINT_LLAMA_DIR=<llama.cpp> -DCMAKE_BUILD_TYPE=Release

The Debian recipe does exactly this (`contrib/packaging/arcint/build-deb.sh`).

`-DARCINT_WERROR=ON` gives the warning-clean build CI should use. Pass
`-DARCINT_GIT_SHA` whenever the build tree has no `.git`; without it
`--version` and `/props` report `unknown`.

**Packaging.** `contrib/packaging/` holds the Debian recipes this project is
deployed with, including `marfrit-openvino/build-openvino.sh`, which builds
the pinned OpenVINO nightly with the patch series the measurements depend on
(`+p25`, patches 0003–0074; `+p27` adds 0076-0077 for the Flash-Next tier's
opt-in switches), and `arcint/build-deb.sh`, which builds both engines
from the release tarball and the llama.cpp pin. No `.deb` is published
anywhere; the directory
contains everything needed to build the same thing, and it is the shortest
path to reproducing a number.

## Supported model formats

- **An OpenVINO IR directory**: `openvino_language_model.{xml,bin}`,
  `openvino_text_embeddings_model.{xml,bin}`, the tokenizer and detokenizer
  IRs, `config.json` and the chat template, validated against the allowlist
  (`src/core/model_registry.cpp`). Safetensors formats OpenVINO does not
  read (GPTQ, NVFP4) do not load.
- **A GGUF on top of such a directory**: `--gguf FILE --model DIR` takes the
  served IR of the same architecture as the topology template and replaces
  its projections with the file's K-quant rows (Q4_K, Q5_K, Q6_K, Q8_0).
  `--gguf-mode mixed` (the default) repacks Q4_K into the runtime's own
  compressed form and keeps every other type as the file's rows, decoded in
  the plugin's kernel; `--gguf-mode repack|native` choose one form for all.
  Design: [docs/design-gguf-native.md](docs/design-gguf-native.md).
- **Serving-shape artifacts exported from a GGUF**
  (`tools/export_serving_artifact.py`): the Flash-Next and Qwen3.6-35B-A3B
  graphs with expert bodies in the checkpoint's own block formats (IQ2_S,
  IQ3_XXS, IQ4_NL, IQ4_XS, Q8_0), decoded by the patched plugin on the card
  and on the host expert tier. Flash-Next's n-gram table is read from the
  GGUF per forward (`--ngram-gguf`).

Expert placement for models larger than the card: `--offload-ratio N` keeps
`100 - N` percent of each layer's experts on the card, `--moe-cpu-tier`
computes the rest on the host CPU, `--moe-per-expert-dispatch` runs the
resident ones as per-expert GPU kernels, and the plugin's
`MOE_CPU_BANK_BYTES` holds the host tier's experts in a RAM bank filled at
load.

If you're interested in the details, there's more in
[FURTHER-READING.md](FURTHER-READING.md).

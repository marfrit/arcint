# arcint

A deliberately narrow LLM inference engine for Intel Arc GPUs.

arcint serves one model family, the hybrid GatedDeltaNet/attention Qwen
models, on two cards: the Arc A770 (16 GiB) and the Arc Pro B60 (24 GB). The
inspiration is [NInfer](https://github.com/Neroued/ninfer), a from-scratch
engine that supports two checkpoints on one GPU and beats every generalist on
that pair. arcint translates the idea to Intel: kernel work goes through
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
| Qwen3.8-27B | `qwen3_5` dense, 64 layers, MTP head | int4 IR, or a GGUF on that IR | B60 (production) |
| Qwen3.6-35B-A3B | `qwen3_5_moe`, 40 layers, 256 experts | the checkpoint's own IQ2_S/IQ3_XXS expert blocks, all resident | A770 |
| Qwen3.8-Flash-Next | `qwen4_exp`, 48 layers, 512 experts (10 routed + 1 shared), n-gram embedding table | the checkpoint's own expert blocks, Q8_0 dense projections, experts split between card and a host RAM bank | B60 |

## Current numbers

Every number names the card and configuration; all are `measured-here`.
"task" is the acceptance task: a Lua CSV parser to RFC 4180, ten named cases,
the candidate code executed, one point per case.

**Production services** (2026-09-24, one fresh process per arm, 32 greedy
tokens after the prompt; extension prefill excludes prefix-cache hits):

| service | card | configuration | prefill @ 4k / 16k | decode @ 4k / 16k |
|---|---|---|---|---|
| coder | A770 | u8 KV, 98,304 ctx, 2 GiB prefix cache | 1,379 / 1,209 t/s | 43.9 / 42.4 t/s |
| dense agent | B60 | `i8:u8` KV, 122,880 ctx, MTP on, `--gate-pad 16`, prefill chunk 512 | 1,141 / 852 t/s | 24.6 / 20.2 t/s |

Both artifacts score 10/10 on the task (greedy; the dense model with MTP
drafting).

**Qwen3.6-35B-A3B**, full depth, native expert blocks with u8 dense
projections, all resident on the A770, u8 KV: prefill about 960 t/s at 4,096
tokens, decode about 28 t/s (`qwen3.6-35b-a3b-native-d40packed-u8`, plugin
patches 0059–0067).

**Qwen3.8-Flash-Next**, full depth (`qwen3.8-flash-next-d48q8`) on the B60:
`--offload-ratio 75 --moe-cpu-tier`, a 128-expert-per-layer census seed, a
30 GiB host expert bank, u8 KV, chunk 2048; at a 20,085-token prompt, prefill
63.1 t/s and decode 6.5 t/s, the needle answered (`+p25`, patches
0003–0074). Levers in progress, in the references' order: an adaptive GPU
expert cache shared across layers, prompt processing on the GPU with the
missing experts streamed from pinned memory, and multi-token drafting with
the checkpoint's MTP head.

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

`-DARCINT_WERROR=ON` gives the warning-clean build CI should use. Pass
`-DARCINT_GIT_SHA` whenever the build tree has no `.git`; without it
`--version` and `/props` report `unknown`.

**Packaging.** `contrib/packaging/` holds the Debian recipes this project is
deployed with, including `marfrit-openvino/build-openvino.sh`, which builds
the pinned OpenVINO nightly with the patch series the measurements depend on
(`+p25`, patches 0003–0074). No `.deb` is published anywhere; the directory
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

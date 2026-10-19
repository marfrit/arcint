# arcint

An inference server for Intel Arc: OpenAI-compatible endpoints for a short
list of models on the Arc A770 and the Arc Pro B60, from the dense
Qwen3.8-27B to the 512-expert Qwen3.8-Flash-Next on one card plus host RAM,
and since 1.1.0 text-to-image: Z-Image Turbo, Qwen-Image (lettering, German
included), FLUX.2 klein with image editing, and an ESRGAN upscaler.

arcint runs GGUF files through libllama (llama.cpp at a pinned commit) with
ggml's OpenCL backend and a published series of Intel kernels
([contrib/llama.cpp/patches](contrib/llama.cpp/patches)): K-quant and
i-quant matrix products on the XMX units, the gated delta-net, decode and
prompt attention, quantized KV, an expert cache for mixture-of-experts models
larger than the card. Around it arcint owns the serving layer: the HTTP
surface, chat templates, tool calls and reasoning, the sampler, named lanes,
speculative decoding with the model's own MTP head, and context checkpoints
so a follow-up or an edited message resumes instead of re-reading the
conversation.

Images come from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s
own `sd-server`, built against the same patched ggml tree, so the diffusion
transformer runs on the same Intel kernels; two patches of the series were
written for it (0057, a fused RoPE; 0058, the q8_0 GEMM on the XMX units),
two more for the 1.1.1 models (0059, the 3D convolution of Qwen-Image's
VAE; 0060, the upscaler's leaky ReLU).
It is its own package and unit, `arcint-imggen` ([contrib/sdcpp](contrib/sdcpp),
[docs/imggen.md](docs/imggen.md)).

The inspiration is [NInfer](https://github.com/Neroued/ninfer), a
from-scratch engine for a closed set of checkpoints that beats every
generalist on them. For mixture-of-experts serving beyond the card's memory
arcint follows the expert engines [Strata](https://github.com/Niko1221/Strata)
and [FreeToken](https://github.com/FlashML-org/FreeToken)
([docs/campaigns/research-reference-audit.md](https://github.com/marfrit/arcint/blob/qfndev/docs/campaigns/research-reference-audit.md)).

[DESIGN.md](DESIGN.md) is the architecture, the rules and the state of play;
[llm.txt](llm.txt) the machine-readable summary;
[docs/llama-engine.md](docs/llama-engine.md) the engine's record, mechanism by
mechanism.

**History.** Until 0.7.12 arcint also had an OpenVINO engine, on a patched
OpenVINO runtime. It was retired in 0.8.0; its design and record are at the
tag [`openvino-history`](https://github.com/marfrit/arcint/tree/openvino-history)
([docs/history.md](docs/history.md)).

## Models

| service | model | served file | card | context |
|---|---|---|---|---|
| agent | Qwen3.8-27B (`qwen35`, dense, 64 layers, MTP layer) | Q4_K_M GGUF | Arc Pro B60 | 245,760 tokens, q8_0:q4_0 KV, MTP 5 |
| coder | Qwen3.6-27B-A3B-Coder (`qwen35moe`, 184 experts, MTP layer) | a searched per-layer quant mix (14.6 GB) | Arc A770 | 98,304 tokens, q8_0:q4_0 KV, MTP 4 |
| Flash-Next | Qwen3.8-Flash-Next (`qwen4exp`, 125B-A6B, 512 experts, n-gram table) | ISTA-DASLab's IQ3_XXS GGUF (IQ2_XS as the faster option) through an expert cache: the hot experts in VRAM, the rest in host RAM, read over the link or computed on the CPU; its MTP layer from the checkpoint | Arc Pro B60 + host RAM, on demand | 131,072 tokens, q8_0 KV, MTP 3 |
| writer | Cydonia 24B v4.3 (Mistral Small 3.2, `llama`) | imatrix Q4_K_M | Arc Pro B60, on demand | 98,304 tokens, q8_0 KV |
| images | Z-Image Turbo (6B DiT) with the Qwen3-4B-Instruct-2507 encoder and the FLUX autoencoder | Q8_0 GGUFs, `ae.safetensors` | Arc Pro B60, on demand | 1024x1024, 8 steps |
| images, lettering | Qwen-Image (20B MMDiT) with the Qwen2.5-VL-7B encoder and the Wan VAE | Q4_K_M GGUFs | Arc Pro B60, on demand | 1024x1024, 20 steps |
| images, editing | FLUX.2 klein 4B with the Qwen3-4B encoder and its own VAE; ESRGAN 4x on every image unit | Q8_0 GGUFs | Arc Pro B60, on demand | 1024x1024, 4 steps; `/v1/images/edits` |

The engine admits these architectures (`qwen35`, `qwen35moe`, `qwen4exp`, and
`llama` at the Mistral Small 24B geometry) and refuses others at load.
Flash-Next's other files are on-demand options on the same endpoint: IQ2_XS,
faster and less exact, and UD-Q3_K_XL, inside the KL bar once the host has
the RAM for its bank (DESIGN §7.2).

## Current numbers

1.1.0's pre-tag check (2026-10-18): the release package with each unit's
flags, alternated with 1.0.0's binary in one window (two runs each,
Flash-Next three), the model on the NVMe, `measured-here` on the named card;
the IQ2_XS row is 0.8.0's. "task" is the acceptance task: a
Lua CSV parser to RFC 4180, ten named cases, the candidate code executed, one
point per case, at temperature 0.

| service | card | prefill | decode | answers |
|---|---|---|---|---|
| agent | B60 | 708.7 t/s at 54,895 tokens, 498.5 at 138,028 | not in this check (38.9 / 25.6 / 16.6 / 13.5 t/s at 4k / 55k / 130k / 200k, 2026-10-16, the DFlash2 campaign's MTP 5 arm at 212,992 ctx) | the needle at 138k; task 10/10 |
| coder | A770 | 724.3 t/s at 40,841 tokens | 51.1 / 47.5 / 44.0 t/s on three 300-token answers | the needle; task 9/10 |
| Flash-Next, IQ3_XXS | B60 + 28.7 GiB host bank | 617.2 t/s at 20,045 tokens (asked right after the capital) | 35.2-35.4 t/s | the needle |
| images, Z-Image Turbo | B60 | | 1024x1024, 8 steps: 2.14 s a step, the VAE 6.3 s, ~23.5 s a request; 512x512: 0.48 s a step | the image gate against a CPU reference: 42.6 dB |
| Flash-Next, IQ2_XS option | B60 + 20.3 GiB host bank | 665.6 t/s, the same protocol | 42.3 t/s | the needle |

Every speed is equal to 1.0.0's in the same window or faster (the release
gate, `tests/acceptance/ab.py`): the agent's and Flash-Next's prefill gained
1-4 % from patch 0058, which runs their MTP heads' Q8_0 weights on the XMX
units in the prompt pass. sd.cpp's own Vulkan backend on the same card takes
7.64 s a 1024² step; the dev host drifts by up to 0.3 % on prefill and
1 % on decode over a day, so numbers from different windows differ by that
much.

Strata's own engine (its SYCL port) on the same B60: 620 t/s at the 20k
needle and 37.8 t/s decode on its own request, where arcint decodes
43.8-44.0 (`measured-here`, 2026-10-14/15).

Quality: KL against a Q8_0 or CPU reference, 16 x 512 tokens. The agent's
q8_0:q4_0 KV 0.0057 nats; the coder's searched quant 0.0237, inside the
answer-level bar against its Q4_K_M; Flash-Next IQ3_XXS 0.218 nats, top-1
89.7 % (IQ2_XS 0.294, 87.1 %), an accepted shortfall (UD-Q3_K_XL: 0.100, 93.3 %).

## Building

C++20, CMake >= 3.20, no network at build time (`third_party/` holds
cpp-httplib, nlohmann/json and minja).

The device-free build serves the HTTP surface from a stub and is what the
unit tests run against:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure -L unit
    ./build/arcint --stub --port 8090 -v

A serving build needs a llama.cpp tree at the pinned commit (`3e4dfa8`) with
`contrib/llama.cpp/patches` applied, and the OpenCL headers and ICD loader:

    git -C <llama.cpp> apply <arcint>/contrib/llama.cpp/patches/*.patch
    cmake -S . -B build-ll -DARCINT_LLAMA=ON -DARCINT_LLAMA_DIR=<llama.cpp> \
          -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ll -j"$(nproc)"

Pass `-DARCINT_GIT_SHA` whenever the tree has no `.git`; without it
`--version` and `/props` report `unknown`. `-DARCINT_WERROR=ON` is the
warning-clean build CI uses. [DEVELOPMENT.md](DEVELOPMENT.md) has the test
ladder.

**The GPU runtime on a fresh host.** Debian trixie ships no Intel compute
runtime in any component; install Intel's release `.deb`s from
[intel/compute-runtime](https://github.com/intel/compute-runtime/releases)
with the IGC and gmmlib versions its release notes name, plus `ocl-icd`.
Every number here was measured on compute-runtime 26.27.39122.11 with IGC
2.38.2, under the xe kernel driver.

**Packaging.** `contrib/packaging/arcint/build-deb.sh` builds the Debian
package from the release tarball and the llama.cpp pin, with ggml-cpu built
native (and checked for AVX2); `contrib/packaging/arcint-imggen/build-deb.sh`
builds the image service's package (`arcint-sd-server`, `arcint-sd-cli`)
from the same tarball, the pin and stable-diffusion.cpp's tarball at
`f0aac52`. **Containers:** `docker/Dockerfile.arcint`
and `.github/workflows/build-toolbox.yml`; see [docs/docker.md](docs/docker.md).

## Running

    arcint --engine llama --gguf Qwen3.8-27B-Q4_K_M.gguf --device GPU.0 \
        --host 0.0.0.0 --port 8087 --served-model-name qwen3.8-agent \
        --n-ctx 245760 --llama-kv q8_0:q4_0 --llama-mtp 5 --queue-timeout 30

`--device GPU.N` is the N-th GPU in OpenCL's enumeration, or a substring of
the device name. The units in [contrib/systemd](contrib/systemd) carry each
service's production flags, Flash-Next's expert cache included
([docs/llama-engine.md](docs/llama-engine.md), "Flash-Next").

- A prompt longer than the context is refused with HTTP 400 and the
  numbers; arcint never truncates or shifts a conversation (DESIGN §3.8).
- Named lanes (`--served-model-name A,B --lane-ctx CA,CB`) serve two names
  with their own contexts from one copy of the weights; the request's
  `model` field picks the lane.
- Speculative decoding drafts with the model's own MTP head. A DFlash2
  drafter is opt-in (`--llama-dflash-gguf`, [docs/llama-engine.md](docs/llama-engine.md)):
  on the dense agent it is faster on short code answers, slower on prose
  and at every measured depth (-8 % at 4k to -36 % at 200k), it costs VRAM, and at 212,992 tokens its
  acceptance task scored 0/10 (MTP 10/10, not attributed), so the agent's
  unit keeps MTP.
- One process holds its card's VRAM for its whole lifetime: whatever else
  was serving that card has to stop first.
- **No authentication, permissive CORS, no TLS.** arcint is built for a LAN
  or a reverse proxy that provides all three.

## Scope

- **Models**: the table above. Other architectures are refused at load.
- **Cards**: the Intel Arc A770 (16 GiB, Xe-HPG) and the Arc Pro B60 (24 GB,
  Xe2), under the xe kernel driver. One card per process, or two by layers
  for one model (`--llama-layer-split`, DESIGN §3.10).
- **Images**: text-to-image and editing through `arcint-imggen` (sd.cpp's
  `sd-server`, a separate binary and process): Z-Image Turbo, Qwen-Image and
  FLUX.2 klein, one unit each, one at a time on the card
  ([docs/imggen-service.md](docs/imggen-service.md)).
- **Not goals**: a web UI or dashboard (console and HTTP JSON only), a model
  zoo, training or quantization tooling, multimodal input (images in), Windows.

## Why not just use …

- **llama.cpp's own server** on Arc: its OpenCL backend is tuned for Adreno
  (the dense 27B prefilled 71 t/s and decoded 10.3 t/s on the B60, against
  695 and the MTP decode above with the series), SYCL aborts under the xe
  driver, and Vulkan is slow on Battlemage. arcint carries the kernels as
  patches against upstream and keeps llama.cpp's model code, its KV and its
  checkpoint mechanism.
- **OpenVINO**: arcint's first engine (0.1-0.7.12). Its GPU plugin emits good
  Xe code, but the services needed an IR export per model and a GPU-plugin
  patch series on a pinned nightly, and the libllama engine overtook it on
  every service (DESIGN §6).
- **vLLM**: no usable Intel path for these hybrid models on Arc.

[CHANGELOG.md](CHANGELOG.md) lists what each release changed and what it
depends on.

# arcint

An inference server for Intel Arc: OpenAI-compatible endpoints for a short
list of models on the Arc A770 and the Arc Pro B60, from the dense
Qwen3.8-27B to the 512-expert Qwen3.8-Flash-Next on one card plus host RAM,
since 1.1.0 text-to-image: Z-Image Turbo, Qwen-Image (lettering, German
included), FLUX.2 klein with image editing, and an ESRGAN upscaler; and since
1.2.0 voice and vision: image input to the chat endpoint, speech to text, and
German text to speech that clones enrolled voices.

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

Voice and vision (1.2.0, [docs/speech-service.md](docs/speech-service.md),
[docs/milestone-1.2.0.md](docs/milestone-1.2.0.md)) are three on-demand
units on the B60. Image input and speech synthesis run in arcint's own binary
through llama.cpp's `libmtmd`: OpenAI `image_url` parts on
`/v1/chat/completions` with a multimodal projector (`--mmproj`), and
`/v1/audio/speech` with Qwen3-TTS (`--tts`), whose voices are enrolled on the
server: the API names a voice and never takes a reference recording. Speech
to text is [whisper.cpp](https://github.com/ggml-org/whisper.cpp)'s own
server built against the same patched ggml, its own package `arcint-stt`
([contrib/whisper](contrib/whisper)).

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

## New: loop guard for small models

Small and mixture-of-experts coder models sometimes think themselves into a
loop: the same line, or the same check with one number changed, comes back
inside the think block until the token cap, and the turn ends without an
answer. 1.2.3 adds a loop guard, **off by default** (`--loop-guard
off|detect|nudge|slap|ladder`, the request field `loop_guard`; `/props`
reports it under `sampler_defaults.loop_guard`).

**The ladder.** The detector reads only the reasoning, one line at a time (a
line ends at a newline; it is stripped, and only lines longer than 20
characters count). Over the newest 200 such lines (`--loop-guard-window`) it
keeps the share of lines that repeat an earlier one, either word for word or
nearly: a character-3-gram Jaccard of at least 0.8 against one of the last 400
distinct lines (`--loop-guard-match near`, the default; `exact` counts
identical lines only). At a share of 0.8 (`--loop-guard-threshold`) the loop is
detected, once per crossing, with hysteresis at 0.7. Then, rung by rung:
- **nudge**: one fixed sentence ("I am repeating myself. I will decide now
  and write the final answer.") goes into the reasoning as if the model had
  written it, once;
- **slap**: if 40 more counted lines (`--loop-guard-grace`) still loop, the
  think block's closer is forced as the next token, by its token id, and the
  model answers;
- **abort**: if the answer itself loops after a slap, the turn ends with
  `finish_reason` `loop`.

The request field switches the guard for that request even where the
operator left the flag off, as `think_eog_mask` and `tool_grammar` do, so a
client can opt in. The nudge's tokens count as generated ones, so a presence
or repetition penalty treats them as the model's own words for the rest of
the turn. With speculative decoding a forced token ends the verify walk at
its position and the drafts behind it are dropped. The response carries a `loop_guard`
object (detections, nudges, slaps, aborts) when the ladder moved, and the
decode line in the log counts the same.

**How it was tested** (`measured-here`, October 2026). The coder (Qwen3.6-27B-A3B,
the A770) was given 30 competitive-programming tasks published after its
training cutoff, AtCoder Beginner Contest 475 to 480, problems A to E. Each
answer was compiled and judged in a sandbox without network against an
independent reference solution plus brute-force and generated tests. The
coder ran at its model card's thinking settings with a 64k-token cap. First a
screening without the guard, then the guard on the tasks that hit the cap,
then the guard on the tasks that did not (the false-positive run). The line
rule was chosen offline beforehand, over 42 recorded reasoning texts: exact
matching missed a loop that counts upward in words, and masking the numbers
instead put a clean enumeration over the threshold, so near matching it is.

- **Screening:** 19 tasks fully correct, 2 answered partly wrong, 9 ended at
  the cap without an answer: 8 of them in a loop, 1 in long but genuine
  thinking. Loops hit easy tasks as well as hard ones.
- **The guard on the 9 cap runs:** all 9 end fully correct. 4 loops closed
  after the nudge alone, 4 needed the slap, and 1 task did not loop on that
  run.
- **The false-positive run:** all 21 other tasks, 19 fully correct. One wrong
  answer had no guard event (the coder varies from run to run; it was correct
  in the screening). Two of these tasks looped this time, and the guard caught
  both: the nudge fixed one; on the other the answer after the slap looped as
  well (its code comments re-checked one contradiction again and again), and
  the abort ended the turn without a program, where the screening, without
  the guard, had answered partly wrong. An earlier build triggered twice on
  code drafts written out several times (short indented lines passed the
  length bar unstripped); stripping the lines first fixed both.

**What it does not heal.** The guard ends loops; it does not make a wrong
algorithm right. An answer given after a full reasoning is set by that
reasoning: re-sampling only the answers of the two partly wrong tasks (ten
seeds in each of three sampling rows) gave one of them right 30 of 30 times,
its reasoning was right and the screening's miss was bad luck in the answer,
and the other wrong 30 of 30 times in the same way, because the algorithm in
its reasoning was wrong. Of the two kinds of loop, "done but cannot stop" (the model has decided
and keeps confirming it) and "stuck" (one sub-question asked again and
again), the stuck kind gets an answer only because the close is forced, and
not always: when the answer loops too, the abort ends it cleanly with
`finish_reason` `loop`, but it does not produce a program. And
it is a small-model tool: the dense Qwen3.8-27B agent did not loop on any of
the 11 tasks the coder failed, and answered all 11 correctly.

One observation from the probes: in a done-but-cannot-stop loop the model
would close the think block with near certainty after a single newline, but
keeps choosing a double newline instead, and the loop goes on; in a stuck loop
it never comes close to closing. Prior art: vLLM's ThinkingLoopBreaker
(token-cycle detection, then the reasoning end forced) is an open pull
request; llama.cpp's issue for a line-based detector went stale.

## Models

| service | model | served file | card | context |
|---|---|---|---|---|
| agent | Qwen3.8-27B (`qwen35`, dense, 64 layers, MTP layer) | Q4_K_M GGUF | Arc Pro B60 | 245,760 tokens, q8_0:q4_0 KV, MTP 5 |
| coder | Qwen3.6-27B-A3B-Coder (`qwen35moe`, 184 experts, MTP layer) | a searched per-layer quant mix (14.6 GB) | Arc A770 | 98,304 tokens, q8_0:q4_0 KV, MTP 4 |
| Flash-Next | Qwen3.8-Flash-Next (`qwen4exp`, 125B-A6B, 512 experts, n-gram table) | ISTA-DASLab's IQ3_S GGUF since 2026-10-20 (IQ3_XXS and IQ2_XS as the faster options) through an expert cache: the hot experts in VRAM, the rest in host RAM, read over the link or computed on the CPU; its MTP layer from the checkpoint | Arc Pro B60 + host RAM, on demand | 131,072 tokens, q8_0 KV, MTP 3 |
| writer | Cydonia 24B v4.3 (Mistral Small 3.2, `llama`) | imatrix Q4_K_M | Arc Pro B60, on demand | 98,304 tokens, q8_0 KV |
| images | Z-Image Turbo (6B DiT) with the Qwen3-4B-Instruct-2507 encoder and the FLUX autoencoder | Q8_0 GGUFs, `ae.safetensors` | Arc Pro B60, on demand | 1024x1024, 8 steps |
| images, lettering | Qwen-Image (20B MMDiT) with the Qwen2.5-VL-7B encoder and the Wan VAE | Q4_K_M GGUFs | Arc Pro B60, on demand | 1024x1024, 20 steps |
| images, editing | FLUX.2 klein 4B with the Qwen3-4B encoder and its own VAE; ESRGAN 4x on every image unit | Q8_0 GGUFs | Arc Pro B60, on demand | 1024x1024, 4 steps; `/v1/images/edits` |
| vision | Qwen2.5-VL-7B-Instruct (`qwen2vl`) with its vision projector | Q4_K_M GGUF + Q8_0 mmproj | Arc Pro B60, on demand | 32,768 tokens, q8_0:q4_0 KV; `image_url` data URIs |
| speech out | Qwen3-TTS-12Hz-1.7B-Base (`qwen3tts`) with its audio-generation projector (speaker encoder, code predictor, acoustic decoder) | Q8_0 GGUF + Q8_0 mmproj with the code head and embedding at f16 | backbone on the Arc Pro B60, the projector on the CPU; on demand | German; enrolled voices only |
| speech in | Whisper large-v3-turbo (whisper.cpp) | whisper.cpp's `.bin` | Arc Pro B60, on demand | German first; `/v1/audio/transcriptions` |

The engine admits these architectures (`qwen35`, `qwen35moe`, `qwen4exp`,
`llama` at the Mistral Small 24B geometry, `qwen2vl` for vision, and under
`--tts` `qwen3tts`) and refuses others at load.
Flash-Next's other files are on-demand options on the same endpoint: IQ2_XS,
faster and less exact, and UD-Q3_K_XL, inside the KL bar once the host has
the RAM for its bank (DESIGN §7.2).

## Current numbers

1.2.3 (2026-10-24, `measured-here`, the pre-tag package): coder and agent
equal to 1.2.2 in the same-window A/B with the switches off; the coder with
its 1.2.3 flags (loop guard, non-thinking row) 5/5 at 10/10 greedy and 30/30
at 10/10 at temperature 0.7.

1.2.2 (2026-10-23, `measured-here`, the build of this release): coder, agent and
Flash-Next equal to 1.2.1 in the same-window A/B; the acceptance task 5/5 at
10/10 greedy on the coder and the agent; the two new guards are off by default.

1.2.1 (2026-10-20, `measured-here`, the pre-tag package, the B60): Flash-Next
IQ3_XXS decodes 38.1 t/s on six answers (1.2.0: 33.9), prefill 622 t/s at
20k, KL 0.217 against Q8_0, the answers right; the other services as below.

1.2.0's release check (2026-10-19, `measured-here`, the pre-tag packages,
`docs/milestone-1.2.0.md` §9.11): the agent and the coder alternated with
1.1.1's binary are equal to it (709.4 / 498.9 t/s at 55k / 138k on the B60;
51.1 / 47.6 / 44.1 t/s decode and 725 t/s at 41k on the A770), with the same
answers byte for byte and the task at 10/10 and 9/10. The new services on the
B60: vision 9/9 on the content check; speech out at 1.12× real time (182.96
s of audio in 163.84 s) with a round-trip word error rate of 0.026 on 46
German sentences; speech in at a word error rate of 0.055 on 12 FLEURS
German clips, equal to the CPU.

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
| images, Z-Image Turbo | B60 | | 1024x1024, 8 steps: 2.14 s a step, the VAE 6.12 s, ~23.4 s a request (1.1.1-1 package, 2026-10-23); was 6.3 s / ~23.5 s on 2026-10-18; 512x512: 0.48 s a step | the image gate against a CPU reference: 42.6 dB |
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

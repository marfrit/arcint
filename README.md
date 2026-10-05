# arcint

A deliberately narrow LLM inference engine for Intel Arc GPUs.

arcint runs a short allowlist of hybrid Qwen models on two Intel cards — the
Arc A770 (16 GiB) and the Arc Pro B60 (24 GB) — and tries to do that better
than the general-purpose engines do. The inspiration is
[NInfer](https://github.com/Neroued/ninfer), a from-scratch engine that
supports two checkpoints on one GPU and beats every generalist on that pair.
arcint translates the idea to Intel: kernel work is delegated to OpenVINO's
compiler stack, which already emits good Xe code, and arcint owns everything
around the compute graph — the serving loop, the scheduler, the KV and
recurrent-state memory, prefix caching, and speculative decoding. Where a
kernel has to change, the change is a small, published patch series against a
pinned OpenVINO build.

Every served model is a hybrid: most layers use linear attention
(GatedDeltaNet), one in four uses full attention. Most of the design follows
from that; see [DESIGN.md](DESIGN.md). [llm.txt](llm.txt) is the
machine-readable summary, [CHANGELOG.md](CHANGELOG.md) the per-release record
and the runtime each release depends on.

## Two engines

arcint has two inference backends behind the same HTTP surface, sampler,
lanes and chat templates. Both services run on the second: the agent from
0.5.6, the coder from 2026-10-05. The first carries Flash-Next and stays as
the coder's way back.

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
- Flash-Next at speed: experts not on the card run on llama.cpp's CPU
  backend (`--llama-cpu-moe`).

`docs/llama-engine.md` has the details.

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
what the device-free tests run against. In this configuration `--model` is
refused at startup instead of starting something that cannot run. `-L unit`
selects the device-free gate (`arcint-test`, the HTTP round trip, the
stub-only concurrency stress test, and the acceptance enumeration's own
consistency checks) — the only thing bare `ctest` ever runs. The
card-requiring acceptance cells are a separate, enumerated target
(`tests/acceptance/`, checklist in
[docs/release-checklist.md](docs/release-checklist.md)); see
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

**The runtime.** The OpenVINO engine's numbers below depend on a *patched*
OpenVINO: `marfrit-openvino +p25`, the 2026.4.0 nightly at upstream commit
`71640275` with the patch series 0003–0074 applied (`+p27` adds 0076-0077
for the Flash-Next tier's opt-in switches). `contrib/packaging/` holds the
Debian recipes that build it and arcint itself, both engines in one binary,
and
`contrib/packaging/marfrit-openvino/patches/README.md` says what each patch
does and what it measured. No `.deb` is published anywhere; the directory
contains everything needed to build the same thing, and it is the shortest
path to reproducing a number.

## Measured

The acceptance task is a Lua CSV parser to RFC 4180: ten named cases (CRLF
and bare LF, a missing final terminator, quoted fields, an embedded comma, a
doubled quote, an embedded newline, empty and empty-quoted fields, no
trimming). The candidate code is **executed**, not read; one point per case.
"task" below is the score out of 10. Every row was measured on this project's
own hardware; the card, the configuration and the runtime are named, and the
dated measurement is in DESIGN.md §7.

### The two served configurations

| endpoint | card | model | configuration | prefill | decode | task |
|---|---|---|---|---|---|---|
| coder | A770, 16 GiB | Qwen3.6-27B-A3B-Coder, int4 (MoE, 184 experts) | `u8` KV, prefix cache | ~510 t/s at ~1k tokens; 621 t/s at 98k | 47.6 t/s at ~1k; 40.3 t/s at 71.7k | 10/10 |
| agent | B60, 24 GB | Qwen3.8-27B dense, int4 AWQ, MTP head | `u8:i4` KV, MTP on, 151,552 tokens | 1,436 t/s at 850 tokens | 23.5 t/s | 10/10 (at `u8` KV) |

The coder on the 24 GB card decodes at 66.5 t/s and prefills at 2,821 t/s at
~1k tokens — above the 60 t/s OpenVINO GenAI baseline on the same card and
artifact that justifies the project.

### Qwen3.6-35B-A3B on the 16 GiB card, all-resident

The 35B's int4 IR does not fit the A770 resident. Exported from the
checkpoint's own GGUF with its expert blocks kept in their native formats
(IQ2_S gate/up, IQ3_XXS down) and computed per routed expert on the card
(`--offload-ratio 0 --moe-per-expert-dispatch`), it does, at 13.11 GiB:

| A770, all-resident, `u8` KV | value |
|---|---|
| prefill, 4,096 tokens | ~960 t/s |
| prefill, 32,768 tokens | 779 t/s |
| decode after 4,096 tokens | 28.1 t/s |
| max context per lane | 112,288 tokens |
| task | 10/10 |

### Speculative decoding on the dense Qwen3.8-27B (B60)

| drafter | short prompt | at 77k tokens |
|---|---|---|
| none | 24.0 t/s | 15.3 t/s |
| MTP head (`--mtp on`) | 33.0 t/s, 76.7% accepted | 4.9 t/s |
| DFlash2 int4 (`--dflash`) | **44.8 t/s**, 3.13 tokens per verify cycle | **18.8 t/s** |

The DFlash2 drafter is the public block-diffusion head
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
exported with `tools/export_dflash.py`. Drafters engage only under greedy, and
a drafted token is accepted only when it equals what the sampler would have
picked — but a multi-token verify pass and a single-token step are not
bit-identical on this backend, so the answer can differ from plain greedy at a
near-tie. Serve without a drafter where bit-exact reproducibility matters.

### A GGUF through arcint (B60)

The dense Qwen3.8-27B as Unsloth's Q4_K_M, opened with `--gguf` on the dense
IR as the topology template, `u8` KV, one lane, MTP off, against Intel's own
int4 IR of the same model:

| weights | resident | prefill 856 / 71.7k tokens | decode 856 / 71.7k tokens | task |
|---|---|---|---|---|
| **GGUF Q4_K_M, mixed form (the default)** | 16.54 GiB | **1,008 / 464 t/s** | **18.5 / 13.7 t/s** | 10/10 |
| GGUF Q4_K_M, native rows (`--gguf-mode native`) | 15.22 GiB | 662 / 395 t/s | 19.5 / 14.2 t/s | 10/10 |
| Intel int4 IR | 13.06 GiB | 1,609 / 552 t/s | 23.1 / 16.5 t/s | — |
| llama.cpp SYCL, the same file (1k / 10k) | — | 249 / 206 t/s | 14.2 / 12.4 t/s | — |
| llama.cpp Vulkan, the same file (1k / 10k) | — | 126 / 108 t/s | 7.8 / 7.0 t/s | — |

The mixed form repacks the file's Q4_K rows at load into the runtime's own
compressed form — every weight within a measured 1/64 of a quantisation step
of ggml's value, the mins as exact extra columns — and runs Q5_K and Q6_K as
the file's rows through a K-quant kernel carried in the patch series.

## Supported models and formats

**arcint loads an OpenVINO IR directory** — `openvino_language_model.{xml,bin}`,
`openvino_text_embeddings_model.{xml,bin}`, the tokenizer and detokenizer IRs,
`config.json` and the chat template — and only one whose hashes and byte count
match an allowlist entry. GPTQ or NVFP4 safetensors will not load.

| family | models | what serves |
|---|---|---|
| `qwen3_5_moe` | Qwen3.6-27B-A3B-Coder, Qwen3.6-35B-A3B | int4 IRs (the 35B via expert offload on the 16 GiB card); the 35B's native-format artifact all-resident |
| `qwen3_5` dense | Qwen3.8-27B | our AWQ export and Intel's int4 IR (MTP head reconstructed with `tools/export_mtp.py`); its GGUF through `--gguf` |
| `qwen4_exp` | Qwen3.8 Flash-Next (512 experts) | serving-shape artifacts built from the GGUF with `tools/export_serving_artifact.py`; full depth serves on one card with part of the experts on the CPU tier, slowly |

**A GGUF opens on top of an IR directory**: `--gguf FILE --model DIR` takes
the served IR of the same architecture as the topology template and replaces
its projections with the file's own K-quant rows (Q4_K, Q5_K, Q6_K, Q8_0).
`--gguf-mode repack|native|mixed` chooses the form (mixed, the default);
`--gguf-embed file|template` where the token embeddings come from;
`--gguf-check once|always` whether the repack's per-projection deviation check
is cached between loads. Dense models of the allowlisted families.

Where to get the artifacts, how the served ones were exported and calibrated,
the features, the non-goals and deployment notes are in
[FURTHER-READING.md](FURTHER-READING.md).

**This branch carries the current state.** The dated measurement record, the
campaign documents, design notes and acceptance windows behind every number
here live on the development branch, `qfndev`.

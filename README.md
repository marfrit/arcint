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
    ctest --test-dir build --output-on-failure -L unit

    ./build/arcint --stub --port 8090 -v

This builds the **stub**, which serves the HTTP surface without a model and is
what the test suite runs against. In this configuration `--model` is refused
at startup instead of starting something that cannot run. `-L unit` selects
the device-free gate (`arcint-test`, the HTTP round trip, the stub-only
concurrency stress test, and the acceptance enumeration's own consistency
checks) — the only thing bare `ctest` ever runs. The card-requiring
acceptance cells are a separate, enumerated target
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
deployed with, including `build-openvino.sh`, which builds the pinned OpenVINO
with the patch series the measurements below depend on. No `.deb` is published
anywhere; the directory contains everything needed to build the same thing,
and it is the shortest path to reproducing a number.

## Benchmark: prefill and decode

The acceptance task is a Lua CSV parser to RFC 4180: ten named cases (CRLF
and bare LF, a missing final terminator, quoted fields, an embedded comma, a
doubled quote, an embedded newline, empty and empty-quoted fields, no
trimming). The candidate code is **executed**, not read; one point per case.
Greedy where the model tolerates it, otherwise the model card's own sampling
defaults. "task" below is the score out of 10 on that task.

arcint serves two configurations in production: the mixture-of-experts coder
on the 16 GiB Arc A770, and the dense agent model on the 24 GB Arc Pro B60.

### Mixture-of-experts: Qwen3.6-27B-A3B-Coder (Arc A770, 16 GiB)

A comparison point on the same card, different quantisation than the AWQ IR
arcint serves and a different engine, so read it as a rough bound rather than
a directly comparable figure:

| engine | card | weights | decode | task |
|---|---|---|---|---|
| llama.cpp, SYCL | A770 | GGUF Q4_K_M | 14.4 t/s | 10/10 |

**Long context against the short-prompt numbers** (2026-09-03, `DESIGN.md`
§7.0.2aa; a real document truncated to depth, 400 greedy tokens, prefix cache
off, one process per arm, shallowest first). Coder on the 16 GiB card: u8 and
u8:i4 decode at parity to 72k tokens (40.3 vs 39.2 t/s) while the u8:i4
prefill costs +7% / +25% / +72% at 8.9k / 37.7k / 71.7k.

Prefill on the coder reaches ~1970 t/s (card, prompt depth and KV precision
not stated for this figure in the source measurement).

### Dense: Qwen3.8-27B (Arc Pro B60, 24 GB)

**The same file through arcint** (0.4.0 stage 1, DESIGN §7.0.2ay): the dense
Qwen3.8-27B as Unsloth's Q4_K_M, opened on the dense IR template with
`--gguf`, against Intel's own int4 IR export of the same model. B60, `u8` KV,
one lane, MTP off, prefix cache off, one fresh process per cell, a 64-token
decode after the prompt; the chunk is the fit's choice per arm:

| weights | prompt | chunk | prefill | decode | task |
|---|---|---|---|---|---|
| **GGUF Q4_K_M, repacked at load (0.4.1, the default)** | 856 | 2048 | **940 t/s** | **16.2 t/s** | 10/10 |
| GGUF Q4_K_M, repacked, `u8:i4` KV (the KV that fits 71.7k) | 71,727 | 512 | 420 t/s | 13.4 t/s | — |
| GGUF Q4_K_M, native rows, K-quant kernel (`--gguf-native`) | 856 | 256 | 213 t/s | 9.9 t/s | 10/10 |
| GGUF Q4_K_M, native rows | 71,727 | 256 | 174 t/s | 8.5 t/s | — |
| Intel int4 IR | 856 | 2048 | 1,609 t/s | 23.1 t/s | — |
| Intel int4 IR | 71,727 | 2048 | 552 t/s | 16.5 t/s | — |

The repack (DESIGN §7.0.2ba) puts the file's K-quant rows into the runtime's
own compressed form at load — the mins as exact extra columns, no zero
point — so the projection stays within a measured fraction of a
quantisation step of ggml's values (1/64 for Q4_K) and the greedy output is
the native path's byte for byte; it costs resident memory (18.7 against
14.9 GiB: Q6_K and Q5_K at u8) and with it context at `u8` KV (46k on this
card). The native path is the exact-bytes reference, kept behind a flag.

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
on longer prompts (fixed in 0.3.0 by plugin patch 0014 and a capped,
recoverable drafter; the drafter-on acceptance task scores 10/10 at that
depth). At production context (155648) the MTP
head serves at 36.2 t/s and 93.2% acceptance, 10/10 greedy, on short
prompts; with prompts of 76k tokens and more its acceptance fell to zero and
decode to 1 t/s — measured and fixed in 0.3.0 (DESIGN §7.0.2ag): an f16
position overflow at 65,504 tokens in the drafters' rotary subgraphs, now
kept at f32, and the MTP layer's unpaged KV state, now charged against the
reservation. Acceptance is back at depth (90.8% at 77k), but MTP's cycle
wall still loses to plain decoding there; DFlash wins at depth instead
(18.8 vs 15.3 t/s at 77k).

**Long context against the short-prompt numbers** (2026-09-03, `DESIGN.md`
§7.0.2aa; a real document truncated to depth, 400 greedy tokens, prefix cache
off, one process per arm, shallowest first). Dense 27B agent on the 24 GB card, u8, decode
t/s: plain 22.3 / 19.9 / 16.3 at 8.9k / 37.7k / 76.4k tokens; MTP 25.7 /
12.6 / 1.0 (acceptance 89% / 79% / 0%); DFlash2 19.8 / 11.0 / 5.3 (1.69 /
1.43 / 1.00 tokens per cycle, on the build that carries plugin patch 0014
and the capped, recoverable drafter). At that point DFlash was below plain
decoding at every one of these depths, the MTP head from 37.7k, and both
accepted nothing at 76k — the rotary overflow above, since fixed; after the
fix DFlash beats plain at 77k. The short-prompt rows above are short-prompt
rows.

## Supported model formats

**arcint loads an OpenVINO IR directory.** Concretely:
`openvino_language_model.{xml,bin}`, `openvino_text_embeddings_model.{xml,bin}`,
the tokenizer and detokenizer IRs, `config.json` and the chat template. GPTQ
or NVFP4 safetensors will not load — those are vLLM formats, and OpenVINO does
not read them.

**A GGUF opens on top of such a directory** since 0.4.0 stage 1:
`--gguf FILE --model DIR` takes the served IR of the same architecture as the
topology template and replaces its projections with the file's own K-quant
rows (Q4_K, Q5_K, Q6_K, Q8_0). Since 0.4.1 the rows
are repacked at load into the runtime's own compressed form (the default;
`--gguf-mode repack|native|mixed` chooses the form (0.4.1; `mixed`, the
default, repacks Q4_K and keeps every other type as the file's rows, which
is where the repack's residency went: 16.26 GiB resident against 18.73, so
a 71.7k context fits at `u8` KV); `--gguf-embed file|template` takes the
token embedding rows from the file, dequantised on the host per token
(`file`, the default), or keeps the template's embedding model;
`--gguf-check once|always` keeps each repacked projection's deviation
verdict between loads of the same file (`once`, the default) or re-checks
at every load. `--gguf-native` keeps 0.4.0's path, the rows decoded inside the patched
runtime's kernel (patch 0021 in `marfrit-openvino +p7`).

The file's geometry is checked against the template's; the template's
tokenizer and chat template are served; the embedding, the GDN state tensors
and the MTP layer stay the template's. GGUF support currently covers dense
models of the allowlisted families — stage 1 of the format
(`docs/design-gguf-native.md`, `docs/milestone-0.4.0.md`); the rates are in
*Benchmark: prefill and decode* above.

If you're interested in the details, there's more in
[FURTHER-READING.md](FURTHER-READING.md).

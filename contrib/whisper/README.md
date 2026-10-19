# whisper.cpp on the same ggml — speech to text for 1.2.0

The STT service (`docs/milestone-1.2.0.md` D3, A3, C3) is whisper.cpp at its
own pin, built **against a ggml installed from arcint's patched llama.cpp
tree**, so the whisper encoder and decoder run on the same OpenCL backend and
the same Intel kernels as the language services. whisper.cpp itself has no
external-ggml switch by accident: the pin carries `WHISPER_USE_SYSTEM_GGML`
(`CMakeLists.txt`, a `find_package(ggml)` instead of vendoring its own
`ggml/`), so the route exists and is the reason a *separate binary* exists —
llama.cpp's and whisper.cpp's `ggml_tensor` layouts have to agree at compile
time, and a vendored ggml would not.

    build-whisper.sh <whisper-src> <llama-tree> <build-dir>

- `<whisper-src>`: whisper.cpp at the pin. A git checkout is checked against
  the commit and built from a pristine `git archive` export, so a dirty tree
  cannot pass. The pin is named here; a tarball tree without git is patched in
  place.
- `<llama-tree>`: llama.cpp at arcint's pin `3e4dfa8` **with the series
  applied** (`contrib/llama.cpp/patches/*`). The recipe refuses one without
  arcint's kernels, one without 0058's q8_0 XMX GEMM (an older cut of the
  series), and a git tree that does not descend from the pin; a tarball tree
  has no history to check.
- `<build-dir>`: holds `llama/` (the ggml source build), `ggml-prefix/` (its
  install), `whisper-src/` (the pristine export) and `whisper/` (the built
  binary).

The pin of whisper.cpp, to move only as its own decision (`CLAUDE.md`: the
llama pin moves only per its own release procedure; the whisper pin likewise):

| field | value |
|---|---|
| repo | `ggml-org/whisper.cpp` |
| pin commit | `d1be6fde11ac6e0407606b4e42fe72d34add8037` (tag `v1.9.5`) |
| licence | MIT (`paper`, the repository); the whisper model weights `openai/whisper-large-v3-turbo` are MIT-tagged on their card |

Steps the recipe takes:

1. Configure the llama tree for ggml only (`GGML_OPENCL=ON`,
   `GGML_OPENCL_USE_ADRENO_KERNELS=OFF`, `LLAMA_BUILD_TESTS/EXAMPLES/TOOLS/
   COMMON=OFF`), the top-level `app/` executable OFF
   (`LLAMA_BUILD_APP=OFF`, `LLAMA_TOOLS_INSTALL=OFF`: it needs `llama-common`,
   which is off, and would otherwise fail the install), build `ggml` and
   `llama` (the install rule wants `libllama.a`), and `cmake --install` it
   into `ggml-prefix/`; the installed `ggml-config.cmake` is what whisper
   consumes. `GGML_MAX_NAME` stays the pin's 64 (the arcint tree's), so the
   two binaries can never share a process, as sd.cpp cannot.
2. Export the pinned whisper.cpp export and configure it with
   `-DWHISPER_USE_SYSTEM_GGML=ON -Dggml_DIR=<prefix>/lib/cmake/ggml`
   `-DWHISPER_BUILD_EXAMPLES=ON` for `whisper-cli` and the server.
3. Assert the whisper binary carries the OpenCL backend and the Intel kernels
   through the linked ggml: `strings` for a series kernel name
   (`kernel_mul_mv_kq_q8_1`: no vendored whisper.cpp ggml has it) and for a
   `ggml_opencl:` log string (there is no literal `Intel(R)` in the back-end
   source at this pin), and that the model loads with `--device` named (a
   leg, not this recipe).

The `/v1/audio/transcriptions` route is **built** (2026-10-18,
`measured-here`): whisper.cpp's own `whisper-server` serves it with
`--inference-path /v1/audio/transcriptions` (an existing flag, no patch), and a
plain `json` response_format returns `{"text":...}` in the OpenAI shape;
`language` and `response_format` match (`docs/speech-service.md`, milestone
§9.2).

**`server-props.patch`** (1.2.0, `examples/server/`) makes two changes.

`GET /props`, which whisper-server lacked (it served `/health` and `/load`
only). It reports the model's file name (the unit's `-m`), whether it is ready,
the language, the inference path, threads, beam size and best-of, the GPU and
flash-attention flags, ggml's GPU devices (name and description, the one in
use marked) and the build string (`ARCINT_BUILD`, set by the recipe).
`build-whisper.sh` applies it idempotently and refuses a server binary
without the build string. Red first (`measured-here`, a CPU-only build of the
pin with `ggml-tiny.bin`): the unpatched server answers `/props` with 404, the
patched one with 200 and the fields above.

`POST /load` is no longer served: it answers 404, like any unknown path
(the server's error handler maps every status but 400 and 500 to 404).
Upstream it loads any path on the host that a client names, leaves the state
at "loading" when the path is missing, and calls `exit(1)` when the file is
not a model, so one request from the network could end the unit. Red first
(`measured-here`, the same CPU build): on the unpatched server `/load` with
`model=/etc/passwd` ended the process. On the patched one it answers 404,
the process stays up, `/health` is ok and `/inference` still transcribes. The
unit names its model with `-m`; another model is another unit.

## The `.bin` loader path

The whisper model is whisper.cpp's own `.bin` container, not a GGUF
(milestone §2's risk). `whisper-server` loads it with `-m`/`--model FILE` via
whisper.cpp's own loader (`whisper_init_from_file_with_params`; never arcint's
GGUF engine), so the format check and the load error are whisper.cpp's. The
service stages the file at an operator-local path (the measured one:
`ggml-large-v3-turbo.bin`, 1.62 GiB, MIT-tagged `openai/whisper-large-v3-turbo`)
and the unit passes `--model`; the `.bin` never goes through arcint's model
gate or the mmproj path, which is a reason STT is a separate process
(charter §3). Staging: the arcint engine takes safetensors/GGUF only; the STT
model is the exception the charter names (`.bin`), and it stays
operator-local.

## Why not `libmtmd`'s ASR instead

The pin's `libmtmd` also carries audio *input* models (`Qwen3-ASR`,
`Voxtral`, `UltraVox`; `docs/multimodal.md`), and `mtmd-audio.cpp` is a port
of whisper's log-mel front-end. Qwen3-ASR would run in the same arcint binary
as the VLM with no second engine. The operator's mandate names whisper (D3),
so whisper is the primary and Qwen3-ASR is the surveyed alternative, recorded
in `docs/milestone-1.2.0.md` §2. The decision stays the operator's if the
whisper/OpenCL path fails; the two do not both get built in this release
unless that happens.

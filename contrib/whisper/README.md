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
  arcint's kernels or without `GGML_OPENCL`.
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

What is **not** here: the `/v1/audio/transcriptions` route. whisper.cpp's own
server examples serve `/inference`, not the OpenAI shape; a thin
arcint-shaped wrapper or an arcint route is the service phase's work
(`docs/speech-service.md`). The recipe only builds the engine.

## Why not `libmtmd`'s ASR instead

The pin's `libmtmd` also carries audio *input* models (`Qwen3-ASR`,
`Voxtral`, `UltraVox`; `docs/multimodal.md`), and `mtmd-audio.cpp` is a port
of whisper's log-mel front-end. Qwen3-ASR would run in the same arcint binary
as the VLM with no second engine. The operator's mandate names whisper (D3),
so whisper is the primary and Qwen3-ASR is the surveyed alternative, recorded
in `docs/milestone-1.2.0.md` §2. The decision stays the operator's if the
whisper/OpenCL path fails; the two do not both get built in this release
unless that happens.

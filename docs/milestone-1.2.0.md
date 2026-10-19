# 1.2.0 — voice and vision on the Arc cards

The 1.2.0 charter, drafted 2026-10-18 on the `v120` branch (from 1.1.1's
`imggen-111`). §1 is the operator's mandate as given to the session, §2 the
model set it implies (licences and download counts), §3 the architecture, §4
the enrolment rules, §5 the kernel gap, §6 the criteria with their gates, §7
the phases, §8 the risks. Every claim carries its evidence class: `paper`
(a published text, README, model card), `code` (read in source, file and
commit named), `measured-here` (run on this project's cards), `arithmetic`
(derived, not measured) or `decision` (the operator's, dated).

Goal (**decision**, operator, via the session's mandate, 2026-10-18): the
fleet's Arc cards get **voice and vision**. Three capabilities, each a
service like the image units, on the same ggml OpenCL base and with the same
Intel kernels:

1. **Vision input (VLM)** — image understanding through arcint's own
   `/v1/chat/completions`, in the OpenAI `image_url` content-part shape; a
   Qwen-VL or Gemma vision model that fits next to the card duties. It also
   closes 1.1.0's open content check of generated images (charter §3 there).
2. **Speech to text (STT)** — whisper (ggml) on the cards, German first,
   OpenAI `/v1/audio/transcriptions`; the gate is word error rate on a fixed
   German test set against the same run on the CPU.
3. **Text to speech with voice cloning (TTS)** — German first, OpenAI
   `/v1/audio/speech`; candidates evaluated at source and by licence, with
   the operator's enrolment rules (arbitrary reference audio is refused).
   The operator and a second household speaker will record their own
   reference samples later (about 10-30 s each); until then build and gate with a freely licensed sample
   voice and document exactly how a new voice is enrolled.

The 1.2.0 line is a **revision of DESIGN §8.8**: the "multimodal input is not
committed" sentence is void for 1.2.0 by the operator (`decision`,
2026-10-18). Image output was already in scope since 1.1.0; 1.2.0 takes in
image input, audio input and audio output and widens "inference server" to
"voice and vision server". Nothing else of §8.8 changes.

## 0. Starting point

### 0.1 What the plan assumes, and what of it holds

The starting thesis (external, 2026-10-18): llama.cpp's multimodal library
(`libmtmd`) is the reference for both vision input and audio generation, and
the pinned tree already carries it. Checked against the source at the pin
`3e4dfa8`:

| thesis | finding | class |
|---|---|---|
| The pin has `libmtmd` | yes: `tools/mtmd/` with `mtmd.cpp`, `mtmd-image.cpp`, `mtmd-audio.cpp`, `clip.cpp`, and the model dump `models/` | `code` |
| The pin carries audio generation (speaker reference) | yes: `tools/mtmd/mtmd-helper-gen.cpp` (`mtmd_gen_audio_pipeline`), types `MTMD_GEN_AUDIO_TYPE_QWEN3TTS` and `..._POCKETTTS` (`tools/mtmd/mtmd.h:371-373`), `--tts-speaker-file` encodes a reference bitmap (`mtmd-helper-gen.cpp:142-143`, `encode_speaker`) | `code` |
| The pin carries audio input | yes: `mtmd-audio.cpp` has a whisper-style log-mel preprocessor (`mtmd-audio.h:64`, `mtmd-audio.cpp:543`), and `models/whisper-enc.cpp`, `models/qwen3a.cpp`, `models/parakeet.cpp` | `code` |
| `libmtmd` is standalone-buildable without the rest of the tools | yes: `LLAMA_BUILD_MTMD` (root `CMakeLists.txt:266-268`) adds `tools/mtmd` when `LLAMA_BUILD_TOOLS` is OFF; `mtmd` links only `ggml` and `llama`, never `llama-common` (`tools/mtmd/CMakeLists.txt:88,191-194`) | `code` |
| The vision/audio encoder ops are on the OpenCL backend | mostly: the pin's `supports_op` takes `CONV_2D`, `RMS_NORM`, `NORM`, `GROUP_NORM`, `SOFT_MAX`, `GLU`, `MUL_MAT`, `IM2COL`, `FLASH_ATTN_EXT`, `CONCAT`, `PAD`, `UPSCALE`, `SUM_ROWS` (`ggml-opencl.cpp:10721-11060`, `code`). Not taken: `CONV_1D` as its own op, and whatever the TTS acoustic decoder and the whisper decoder need beyond these; the profile decides | `code` |
| arcint's chat endpoint accepts image input today | **no**: `src/core/chat.cpp:20-45` rejects every non-text part with "v1 is text-only" | `code` |
| whisper.cpp can be built against this patched ggml | yes: `WHISPER_USE_SYSTEM_GGML` (`CMakeLists.txt:88`) takes a `find_package(ggml)` instead of vendoring (`:160-167`); a `ggml` package installed from the patched tree carries the OpenCL backend and the series | `code` |
| The challenge is the ops, not the plumbing | the language kernels are built for the Qwen hybrid geometry; the vision tower, the codec decoder and the whisper encoder/decoder are new shapes and new op mixes | `arithmetic` |
| Qwen3-TTS exists, is open, German-capable, cloning-capable | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`: `license:apache-2.0`, 3,093,627 downloads, not gated (`measured-here`, HF API, 2026-10-18); the `tools/tts` README lists `--tts-lang` `de` and `--tts-speaker-file` (`code`) | `paper` / `measured-here` |
| OuteTTS is gated | `OuteAI/OuteTTS-1.0-1B`'s HF API answers 401 "Invalid username or password" without a token (`measured-here`, 2026-10-18); a token on a command line is forbidden here (`CLAUDE.md`, the 1.1.1 aria2c incident) | `measured-here` |

### 0.2 What the pin's `libmtmd` brings (`code`, `3e4dfa8`)

- **Two GGUF files a multimodal model**: the language/backbone model and the
  **`mmproj`** projector; `mtmd_init_from_file(mmproj, model, params)`
  (`mtmd.h:133`). `mtmd_context_params` carries `use_gpu`, `n_threads`, the
  media markers, image/audio max tokens, and `print_timings` (`mtmd.h:97`).
- **Encode → decode**: `mtmd_tokenize(ctx, chunks, text, bitmaps, n)` builds
  text and image chunks with the model's own chat template; the image chunk's
  embeddings are produced by a clip graph on the chosen backend, then fed to
  `llama_decode` as embedding tokens (`mtmd.h`'s chunk API).
- **Audio generation** (`mtmd-helper-gen.cpp`): a pipeline per model type.
  Qwen3TTS: the speaker reference is encoded by a speaker encoder, the
  language is a special token `<|codec_language_...|>`, the backbone samples
  codec tokens, and an acoustic decoder (the `mmproj`'s gen half) makes the
  waveform; `sample_rate` and WAV writing are in the helper (`:308,320`).
  PocketTTS: a speaker encoder plus a WavTokenizer-style decoder.
- **Vision models supported** in the tree's `models/`: `qwen2vl.cpp`,
  `qwen3vl.cpp`, `gemma4v.cpp`, `gemma4uv.cpp`, `internvl.cpp`, `pixtral.cpp`,
  `minicpmv.cpp` and more; the `docs/multimodal.md` pre-quantized list names
  Gemma 3 4B/12B/27B, Qwen2-VL 2B/7B, Qwen2.5-VL 3B/7B/32B/72B, Gemma 4 E2B/
  E4B/26B-A4B/31B (`code` + `paper`).
- **Audio-input models**: Ultravox, Voxtral-Mini-3B, Qwen3-ASR-0.6B/1.7B,
  and the mixed-modality Qwen2.5-Omni / Qwen3-Omni / Gemma 4 (`paper`).
- **TTS in `tools/tts`** (`tts.cpp`, `mtmd-helper-gen`): Qwen3-TTS and Pocket
  TTS, both with `--tts-speaker-file`; `--tts-lang` for Qwen3-TTS.

### 0.3 What arcint has today (`code`)

- **One binary, one ggml**: `ARCINT_LLAMA=ON` builds llama.cpp as a
  subproject at `3e4dfa8` with the series and the OpenCL backend
  (`CMakeLists.txt:86-131`), `GGML_MAX_NAME` at the pin's 64. The image
  service is a separate binary because sd.cpp needs 160 (charter D5 of
  1.1.0); the mtmd library needs the same layout as libllama, so **it can go
  into the arcint binary itself** with `LLAMA_BUILD_MTMD=ON`.
- The chat parser is text-only (`chat.cpp`); `--vision` is on the **retired**
  flag list (`config.cpp:111`, the OpenVINO era's). The OpenAI surface has no
  `/v1/audio/*` route (`src/http`, `src/api`).
- **The image units** (`arcint-imggen{,qwen,flux}`) are the form to copy:
  separate unit files, one package, `Conflicts=` with the card's language
  units, started by the front on demand, a contract doc.

## 1. The operator's mandate

| # | question | decided (2026-10-18, via the session) |
|---|---|---|
| D1 | Scope | **(a)** 1.2.0 takes vision input, STT and TTS-with-cloning in; DESIGN §8.8's multimodal-input sentence is void for 1.2.0 |
| D2 | VLM route | **arcint's own `/v1/chat/completions`** with OpenAI `image_url` content parts, `libmtmd` inside arcint's binary. Not llama.cpp's own `llama-server` as the serving surface (that would be a second surface) |
| D3 | STT engine | **whisper.cpp** on the same ggml base, German first; Qwen3-ASR through `libmtmd` is surveyed but not the primary (the mandate names whisper) |
| D4 | TTS | **evaluated by the session** at source and by licence; the primary is the one that runs in llama.cpp's own TTS path, is German-capable and clone-capable, with a permissive licence and high downloads |
| D5 | Voice cloning | **enrolled voices only**: a server-side list of speaker profiles; the API must **not** accept arbitrary reference audio; reference recordings and profiles stay on the card host **outside the repository** and never in commits, logs or docs |
| D6 | Card | **the A770 where it fits next to the coder, otherwise the B60 on demand** (the mandate) |
| D7 | Units | each capability its own **on-demand unit**, the image units' discipline: `Conflicts=`, contract doc, `/health`, `/props` |
| D8 | Gate tolerances | **set by the operator 2026-10-19** (operator-set, replacing the §6 draft): C3 corpus WER ≤ 0.08 on the FLEURS `de` set and card ≤ CPU + 0.005; C4 corpus round-trip WER ≤ 0.10 with no single sentence above 0.5 on the enlarged ~300-word set; C5 the Seed-TTS-eval WavLM-large SV cosine, calibrated against same-speaker vs cross-speaker recordings; B3 RTF ≥ 1.0 (batch reading); time-to-first-audio recorded, not gated; naturalness listened, no automatic gate. Full text and sources in §6/D8 |
| D9 | Repository | the rules stand: no host names or credentials in anything committed; do not push, tag, release or open package PRs from this session |

## 2. The model set

Rule (`decision`, operator, 2026-10-18/19): official or high-download
(>10k downloads) weights only, safetensors or GGUF (here: also whisper.cpp's
`.bin`) from known publishers; anything else only inside a container.
Licences checked for private and commercial use. Sources checked on HF's API
on 2026-10-18 (`measured-here` for downloads, licences are `paper` where the
tag names a licence the publisher sets).

| role | repo / file | licence | downloads | size | notes |
|---|---|---|---|---|---|
| **VLM** LLM | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`, `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | apache-2.0 | 71,076 | ~4.5 GB | the `qwen2vl` architecture; the mandate's "Qwen-VL" |
| **VLM** mmproj | same repo, `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` | apache-2.0 | (repo) | ~1.4 GB | `code`: pre-quantized for `libmtmd` |
| VLM fallback | `ggml-org/gemma-3-12b-it-GGUF` (+ `mmproj-model-f16.gguf`) | gemma | 11,330 | 7.3 GB | Gemma Terms, not OSI; used only if Qwen is not good enough |
| **STT** | `ggerganov/whisper.cpp`, `ggml-large-v3-turbo.bin` (or `ggml-large-v3.bin`) | MIT (repo) / MIT (`openai/whisper-large-v3-turbo` tag) | canonical | ~1.5 / ~3.0 GB | whisper.cpp's own format, not GGUF |
| STT alternative | `ggml-org/Qwen3-ASR-1.7B-GGUF` (+ mmproj) | (Qwen) | 33,955 | ~1.9 GB | in `libmtmd`; surveyed per D3 |
| **TTS** backbone | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` → `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` | **apache-2.0** | 3,093,627 (HF) / 32,169 (GGUF) | Q8_0 1.85 GB, bf16 3.47 GB | German (`de`, `code`), cloning, not gated; **Q8_0 is fine** (the campaign measured the backbone quantization irrelevant) |
| **TTS** mmproj | same GGUF repo: `mmproj-...-bf16.gguf` (the official fix, kept as fallback) or requantized to Q8_0 with `gen.code.head`/`gen.code.embd` at f16 (the production set) | apache-2.0 | (repo) | bf16 0.67 GB; production Q8_0-codeheadf16 0.45 GB | the speaker encoder **+ the code predictor and the code2wav acoustic decoder**; the **Q8_0 mmproj is defective**: its code predictor (`a.gen.code`) quantized at Q8_0 never samples `codec_eos` and makes German unintelligible (the campaign, `measured-here`). The whole-code-predictor bf16 fix (or the mixed `a.gen.code` BF16, 0.56 GB) passes; the production set is the cheaper `gen.code.head`/`gen.code.embd`-f16 recipe, which passes **C4 and B3** (§9.7, `measured-here`) |
| TTS alternative | `OuteAI/OuteTTS-1.0-1B` | ? | 401 on the API (gated) | ~2.4 GB Q8 | in llama.cpp's TTS lineage; not primary, gated |
| TTS alternative | `kyutai/pocket-tts` (language dirs) | cc-by-4.0 | 1,133 | ~ | gated `auto`; German among its languages |

Notes (`paper`/`code`):

- **Qwen3-TTS-12Hz-1.7B-Base** is the strongest of the three the pin's TTS
  tool supports: 10 languages incl. German, free-form voice cloning from a
  few seconds, Apache-2.0, a stable production publisher, tensor sizes that
  fit a 16 GiB card on their own, and no gate. Its licence permits private
  and commercial use (`paper`: the model card's `license:apache-2.0`).
- **OuteTTS-1.0-1B** is gated on the Hub; a download needs a token, and a
  token on a command line is a repository rule violation. It is surveyed and
  recorded as an alternative, not built.
- **Pocket TTS** is gated `auto` and cc-by-4.0 with attribution conditions;
  German is a property of the weights, not a `--tts-lang` flag.
- **Whisper**'s large-v3 family is Apache/MIT and the standard German ASR
  reference; large-v3-turbo is the cheaper serving form. The mandate names
  whisper, so it is the primary; Qwen3-ASR stays as the alternative if the
  whisper/OpenCL path fails.
- **Qwen2.5-VL-7B** is chosen over Gemma for the licence (Apache-2.0 against
  Gemma Terms) and because it reads image text (the German-lettering content
  check of 1.1.0 whose checkbox this closes); F16 mmproj is the accuracy
  option, Q8_0 the memory option.

Staging list (operator-local, card host): the VLM pair, Qwen3-ASR's pair (if
measured), the whisper `.bin`, the Qwen3-TTS backbone (Q8_0) and the **bf16**
mmproj (the fix of the campaign), and one freely licensed German sample voice
for the gate. Nothing goes into the fast-staging NVMe between legs (SOP).

## 3. Architecture

**One arcint binary serves vision input and audio output; a second binary
serves STT.** The reason for the split is the engines, not the surface:

- **VLM and TTS: arcint's own process.** `libmtmd` is built into `arcint` with
  `LLAMA_BUILD_MTMD=ON` (the library links `ggml` + `llama` only, so
  `GGML_MAX_NAME` stays the pin's 64 and one process is possible). The vision
  tower runs on the mmproj's own clip graph on the chosen backend; the
  language model is the ordinary libllama path. TTS uses `libmtmd`'s
  `gen_audio` pipeline, which samples codec tokens from the backbone and
  decodes the waveform with the mmproj's gen half.
- **STT: a separate binary.** whisper.cpp at its own pin, built with
  `WHISPER_USE_SYSTEM_GGML=ON` against a `ggml` **installed from arcint's
  patched tree** (`find_package(ggml)`), so it runs on the same OpenCL backend
  and the same Intel kernels. Its GGML `ggml_tensor` layout must match the
  installed one (both 64), so whisper.cpp never shares a process with arcint
  either, like sd.cpp.

Routes added to arcint (`src/http`, `src/api`):

| route | shape | notes |
|---|---|---|
| `POST /v1/chat/completions` (existing) | `image_url` content parts accepted when a `--mmproj` is loaded; `data:image/...;base64` URIs only (a file path or a remote URL is 400: release review, 1.2.0), at most 8 images of at most 4096×4096 a request; without an mmproj the text-only refusal stands | VLM |
| `POST /v1/audio/speech` | OpenAI: `{model, input (text), voice, response_format (wav), speed?}` → the WAV bytes | TTS |
| `POST /v1/audio/transcriptions` | OpenAI multipart: `file`, `model`, `language`, `response_format (json/text)` → `{"text":...}` | STT **in its own unit/binary**, exposed by whisper.cpp's own server shim or a thin arcint-shaped wrapper |
| `GET /v1/audio/voices` | the enrolled profile names (names only, never audio) | TTS; not an OpenAI route, a probe |
| `/health`, `/props` (existing) | add the mmproj/backbone/card and the enrolled-voice count; TTS and VLM report their own needs | all |

**Card and unit choice.** The A770 is permanently held by the coder (14.6 GB
of 16); a VLM pair, the whisper form or a TTS pair cannot fit beside it, so
per D6 they are **B60 on demand**: `arcint-vlm`, `arcint-tts` and
(separately) `arcint-stt`, `Conflicts=` with the B60's language and image
units, started by the front on the first request, 20 idle minutes give the
card back. Ports: the next free ones beside the language units, the image
units and the front (operator's call). One unit file each; one package for
the arcint binary, one for the whisper binary.

**Memory (`arithmetic`).** VLM: ~4.5 + ~1.4 GiB plus activations on the
24 GB card, room to spare beside nothing. TTS: ~1.9 + ~1.6 GiB plus the
acoustic decoder's activations. Whisper large-v3-turbo: ~1.5 GiB. Each is
comfortably inside one card's idle VRAM; the units still take the card
whole (`Conflicts=`), as the image units do.

## 4. Enrolment and the voice-cloning rules (D5)

- **Speaker profiles are server-side.** A profile is a directory on the card
  host **outside the repository**: `reference.wav` (a 10-30 s clean
  recording), its `reference.txt` transcript, and a `profile.json` with the
  name, language and provenance. The default list holds only the freely
  licensed gate voice until the two household speakers record theirs.
- **The API never takes a reference.** `/v1/audio/speech` accepts `voice` as
  a **name from the server's list**; any `reference_audio`, `speaker_file`,
  `voice_sample`, `audio` or equivalent field is refused with 400 and a
  message naming the enrolment route. The list is served by
  `GET /v1/audio/voices` as names only. A cloned voice can therefore only be
  produced by something that already sits on the card host.
- **No reference bytes in the repository.** No audio, no profile, no
  transcript of a real voice in a commit, a log, a doc or a `/props` body.
  The enrolment **procedure** and the tool are in the repo; the bytes are
  operator-local (`*.local.md`, the profiles directory).
- **Enrolment is a documented command**: the repo tool
  `tools/tts_enrol.py --name NAME --audio FILE --text FILE [--profiles DIR]`
  writes the profile into the card host's profile directory. It checks that
  the sample is a readable WAV and its length (a sample under 8 s warns, over
  35 s is refused); it does not check the sample rate or for silence, which
  the listener and the C4/C5 gates catch (`code`, release review). The service manual (a new
  `docs/speech-service.md`) carries the exact steps, including the "clone two
  voices: the two household speakers" end state.
- **The gate voice is freely licensed:** one short German sample from a
  public, licence-clear source (recorded provenance and licence in the
  service doc; the audio itself staged outside the repo). The gate runs the
  full clone path on it, not a built-in default voice, so the enrolment
  mechanism is proven before the real samples arrive.

## 5. The kernel gap

The series and the pin's `supports_op` already take most of what a vision
tower, a whisper encoder and the audio decoders need. Open and to be sized
by a profile (`arithmetic`, profile owed):

- **`CONV_1D`** — whisper's first encoder layers are 1D convolutions and
  ggml presents them as `CONV_1D`/`CONV_2D`; the pin's `supports_op` takes
  `CONV_2D` (`code`) but whether whisper.cpp's call lands there and how it
  performs on Arc is open.
- **The TTS gen half** — measured on the card (2026-10-19, `measured-here`):
  the `libmtmd` gen graph builds at **947 splits / 6492 nodes**, OpenCL
  805.23 MiB / CPU 278.40 MiB, with f32 ops unsupported on OpenCL: `ARANGE`
  ×354, `GET_ROWS` ×46, `SIN` ×29, `LOG` ×24, `PAD_REFLECT_1D` ×22, `SUM` ×15,
  `COL2IM_1D` ×6; the code2wav transposed conv (`COL2IM_1D`) is among them
  (the prediction above, now measured). The card arm's real-time factor is
  **0.258×** (no faster than the CPU loop's 0.233) — campaign
  `qwen3tts-card-real-time`.
- **Flash attention in the vision tower** — Qwen2.5-VL uses full attention;
  the pin's `FLASH_ATTN_EXT` takes head sizes 128/256 (`code`, 1.1.0 §0.3);
  the ViT's head/gqa geometry may take the generic path. Size it.
- **Whisper's own `ggml` vs the installed one** — the whisper encoder graph
  calls `ggml_*`; if a call is missing from the pin, the build fails and the
  resolution is a whisper-side port, not a pin move (`CLAUDE.md`: the pin
  moves only per its own procedure).
- Already closed by 1.1.1 and reused: `CONV_3D`/`IM2COL_3D` (0059) and
  `LEAKY_RELU` (0060) — not needed on these paths but in the tree.
- **The content check's op** is trivially the VLM itself (charter §6).

Each kernel change is a numbered patch of the series with its README section,
measurement and red case (DESIGN §1.1), as 0057-0060 were.

## 6. Criteria for 1.2.0 and their gates

The release measurements will be on the dev host, on the card named, with the
binary's sha and every flag, `measured-here` unless marked. Each capability
has its own on-demand unit with `/health` and `/props` (D7).

### A. It runs

| # | criterion | gate |
|---|---|---|
| A1 | `libmtmd` in arcint builds at the pin with `LLAMA_BUILD_MTMD=ON`, no `llama-common`, `GGML_MAX_NAME` 64, the series' kernels in the binary | the build succeeds and asserts the kernel presence and layout as `contrib/sdcpp/build-sdcpp.sh` does for sd.cpp |
| A2 | A VLM image request through `/v1/chat/completions` on the card, no op silently on the CPU for the encoder (profile/`GGML_SCHED_DEBUG` checked) | the request answers and a schedule dump names the card for the mmproj graph |
| A3 | The whisper binary on the same installed ggml, an `/v1/audio/transcriptions` route, a transcription on the card | the route answers and the transcript is non-empty for a fixed German clip |
| A4 | `/v1/audio/speech` on the card, the full clone path with an enrolled profile (D5), a WAV out | a German sentence synthesizes to a playable WAV, the refusals (arbitrary reference field, unknown `voice`) are 400 |
| A5 | The language services stay as they are: the A/B (`tests/acceptance/ab.py`) against the last release passes, the acceptance task ≥ 9/10 | as 1.1.0 A3: alternated same-window A/B, the answers right |
| A6 | One unit, one package per binary, `/health` and `/props` (D7) | the units start on demand, the front's `Conflicts=` bookkeeping works, the probes answer |

### B. It is fast

| # | criterion | gate |
|---|---|---|
| B1 | A profile per capability: the VLM's encoder vs decode share, the whisper encode/decode split, the TTS backbone vs acoustic decoder | a `GGML_OPENCL_PROFILING`/timing run, every node over 10 % set against a measured ceiling or opened as a campaign (charter B2 of 1.1.0) |
| B2 | STT: the card's transcription time against the same whisper on the CPU (identical inputs) | the card is faster than the CPU beyond the run-to-run spread; the WER gate is separate |
| B3 | TTS: real-time factor (audio seconds per process second) at the fixed sentence set | **RTF ≥ 1.0** on the sentence set (operator 2026-10-19; the 1.2.0 use case is reading files aloud, batch — real time is a later milestone). Time to first audio is recorded with every arm but not gated here |
| B4 | VLM: prefill/encode time per image at a fixed size and token budget | reported against the CPU encoder on the same window |

### C. It is right

| # | criterion | gate |
|---|---|---|
| C1 | The **VLM content check** on a fixed set of images with checkable content (an object, a count, a colour, a short lettering): answers match ground truth | a fixed question set per image; every answer checked as a whole word; the gate is the set's pass rate, **floor 1.0 on the fixed 9-question set** (operator 2026-10-19) |
| C2 | The VLM closes 1.1.0's open content check: it reads the generated Qwen-Image `WILLKOMMEN` and the Z-Image lettering | the lettering is reported correctly; this is the 1.1.0 charter §3 "content check" that was left unbuilt |
| C3 | STT **word error rate** on a fixed German test set against the **same model's CPU run** (identical inputs) and against a floor | WER = `(S+D+I)/N` per clip, corpus WER over the whole set; candidate ≤ CPU WER + 0.005, and **corpus WER ≤ 0.08** on the FLEURS `de` set (operator 2026-10-19) |
| C4 | TTS **intelligibility**: round-trip the synthesized German through the STT model, WER against the written sentences on the fixed ~300-word set | **corpus WER ≤ 0.10 AND no single sentence above 0.5** (operator 2026-10-19); the share of sentences with WER ≤ 0.10 is reported with every verdict |
| C5 | TTS **voice similarity**: the clone's embedding against the reference's, the gate voice and (later) the enrolled ones | the **Seed-TTS-eval** speaker model (fine-tuned WavLM-large SV), named on the record; the cosine is **calibrated**: the clone-vs-reference cosine must come near the cosine between two real recordings of the same person, and clearly above the cross-speaker cosine (operator 2026-10-19). Until the real recordings exist, prove the method with CC stand-ins: two clips of one FLEURS speaker against another speaker |
| C6 | Every new or changed kernel has its red case, which turns the capability's gate red | a deliberate mutation fails the capability gate and its op suite, as 0057-0060's did |
| C7 | The answer-level bar for the language path unchanged: the VLM/TTS unit must not perturb the language services' answers or KL | the A5 A/B and the existing KL arms |

The image gate of 1.1.0 (`tools/imggen_gate.py`) stays for the image units;
the new gates are new tools (`tools/vlm_gate.py`, `tools/stt_gate.py`,
`tools/tts_gate.py`), each printing its tolerances and refusing arms whose
inputs differ, as `imggen_gate.py`'s parity check does. Gate comparisons use
identical inputs (the rule learned in `imggen-attention-path`).

### D8 — the operator's floors (operator-set, 2026-10-19)

The operator set the gate numbers from the community consensus the session
researched. The §6 rows above carry them; the former "proposal, D8" labels in
this document and in `tools/tts_gate.py` are void. Evidence class: the floors
are `decision` (operator); the cited sources are `paper`.

| # | floor (operator-set) |
|---|---|
| C3 | corpus WER ≤ 0.08 on the FLEURS `de` set, **and** the card ≤ the CPU's WER + 0.005 (the comparison unchanged from §9.2) |
| C4 | corpus round-trip WER ≤ 0.10, **and no single sentence above 0.5**, on the enlarged sentence set (~300 words). One error in the old 66 words was 1.5 pp; ~300 words resolves it. The share of sentences with WER ≤ 0.10 is reported with every verdict |
| C5 | use the **Seed-TTS-eval** speaker model (a fine-tuned WavLM-large SV), named on the record. The floor is **calibrated**, not absolute: the clone-vs-reference cosine must come near the cosine between two REAL recordings of the same person, and clearly above the cross-speaker cosine (one household speaker's clone against the other's reference and the reverse). Until the real recordings exist, prove the method with CC stand-ins: two clips of one FLEURS speaker against another speaker |
| B3 | the 1.2.0 use case is reading files aloud (batch); real time comes later. Floor: **RTF ≥ 1.0** (audio seconds per process second) on the sentence set. Time to first audio is recorded but not gated in 1.2.0; it becomes the gate of a later real-time milestone (community consensus: 200-300 ms TTFA with streaming) |
| C1 | pass rate **1.0** on the fixed 9-question set (`tools/vlm_cases.json`) |
| TTS placement | the hybrid, `--tts-gen-device cpu` (the backbone on the card, the mmproj's graphs on the CPU), accepted for the production unit |
| naturalness | no automatic gate: UTMOS is English-trained. The two household speakers listen A/B against their own recordings |

Sources (operator-provided, 2026-10-19): `github.com/BytedanceSpeech/
seed-tts-eval` (the speaker model and its SIM metric); arXiv 2601.15621
(Qwen3-TTS: German WER 1.235 at the 1.7B-12Hz); arXiv 2606.03455 (ground
truth test-en WER 1.79, SIM-o 0.73 — the calibration anchor); arXiv
2608.10606 (round-trip WER is a screening); arXiv 2605.26978 (the share of
utterances with WER ≤ 10 %).

The C4 sentence set (licence on record, §2's rule): `tools/tts_sentences_de.json`,
46 unique German sentences of 5-10 words, **Tatoeba** (CC-BY 2.0 FR,
attribution to Tatoeba and its contributors), drawn deterministically from the
`deu_sentences.tsv.bz2` export (sha256 `ccd06372…`, 2026-10-17) with the
selection rule recorded in the manifest; 303 words. The synthesized WAVs and
the round-trip transcripts stay operator-local (the leg's record); the
sentences themselves are public text, licence-clear, and committed.

## 7. Phases

1. **Phase 0, build and inventory** (one card window per leg, results
   `measured-here`): `libmtmd` into arcint (A1); the VLM on the CPU as the
   reference and on the card, `GGML_SCHED_DEBUG` for islands, a profile per
   node (B1, B4); whisper.cpp against the installed ggml, CPU and card (B2);
   Qwen3-TTS on the CPU and the card, the clone path on the gate voice, a
   profile (B3). Outcome: the op list by share for each capability.
2. **Phase 1, the gates**: the fixed image set, the German STT set, the
   German TTS sentence set, the gate voice, and each capability's red case
   (C1, C3, C4, C5, C6). Before any kernel work, as 1.1.0's Phase 1.
3. **Phase 2, closing gaps**, by the profile, not by guess: `CONV_1D` if
   whisper needs it; the TTS acoustic decoder if it is on the CPU; the ViT
   attention if it is generic. Each as a numbered patch with its README
   section, measurement and red case.
4. **Phase 3, the VLM endpoint and the content check** (C2): the `image_url`
   parser, the chunk path, the 1.1.0 lettering check.
5. **Phase 4, the STT and TTS services** (D5, D7): the enrolment tool and
   doc, the voices route and refusals, the unit files, the packaging, the
   `/health`/`/props` extensions.
6. **Release 1.2.0** by the existing procedure (review, pre-tag check, fresh
   container, A/B against the last release). The tag is the release session's
   (D9).

## 8. Risks

- **`GGML_MAX_NAME` and `GGML_MAX_OP_PARAMS`** diffs between the arcint tree
  and whisper.cpp's vendored expectations: the installed-ggml route is the
  reason a separate whisper binary exists; a mismatch shows at build, not at
  runtime (`arithmetic`).
- **OpenCL program compiles at the first request** (the VLM's clip graph and
  the TTS decoder add programs): a first-request load/compile label, and the
  card's program cache is cold per fresh process (1.1.0's lesson).
- **The whisper model format** is `.bin`, not GGUF; the staging and the
  loader differ from every other unit's (`code`).
- **TTS stochastics**: a fixed seed is needed for a reproducible arm; the
  sampling path (temperature/seed) is printed with every verdict, as
  `imggen_gate.py` demands parity. Voice similarity is a stochastic proxy;
  the WER round-trip is the stable gate.
- **Speaker similarity on the CPU** needs a named embedding model on the
  record; if none is licence-clear, C5 falls back to the WER gate plus an
  operator ear check (stated, not hidden).
- **A gated alternative** (OuteTTS) cannot be downloaded without a token; per
  the repository rule it is not used. Qwen3-TTS being ungated is a reason it
  is the primary.
- **The B60's first job after a long idle can wedge** and thin VRAM margins
  evict (SOP, DESIGN §5.2); the units are on demand for exactly that reason.
- **Multimodal models are under heavy development** upstream (`tools/mtmd/
  README.md`, `paper`): a model that runs at the pin may need a different
  mmproj layout after a pin move. The VLM's model choice is recorded with its
  mmproj source for that reason.

## 9. State as measured (Phase 0, 2026-10-18)

The VLM leg, on the dev host's B60 (PCI `8086:e211`, OpenCL GPU.0),
the B60's language unit stopped for the window and the card host's sampler on
the physical host, a fresh process: the staged dev build (libmtmd in arcint,
`LLAMA_BUILD_MTMD`; the `mtmd_input_text` `text_len` fix), llama.cpp at the
pin + the series through 0060. Models `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
(4,683,072,032 bytes) and `mmproj-...-Q8_0.gguf` (853,119,712 bytes), sizes
equal to HF's LFS pointers (`measured-here`).

| item | value (`measured-here`) |
|---|---|
| A1, `libmtmd` in arcint | builds: the `mtmd` target and the `arcint` binary; `LLAMA_BUILD_MTMD=ON`, no `llama-common`, the pin's `GGML_MAX_NAME` 64, the series' kernels present in the tree |
| A2, a VLM request through the chat endpoint | yes: `/v1/chat/completions` with an `image_url` data URI answers; the OpenCL device is named for the mmproj graph, no CPU island reported (`-v` log) |
| load | `qwen2vl` Q4_K - Medium (4.36 GiB) + Q8_0 mmproj on the B60, in 46.2 s; mmproj `image_min_pixels` 6272, `image_max_pixels` 3,211,264 |
| C1/C2, content check | **4/4**: the generated Qwen-Image `WILLKOMMEN` imprinted on a white sign reads `Willkommen` (1024² and 512²); the FLUX.2 klein generations read `Cat` (the windowsill) and `Palm trees and beach.` (the edit's background). 1.1.0's open content check is closed for these images |
| B1/B4, 1024² encode | 1369 image tokens (1407 with text), image slice encoded 1716 ms, image batch decoded 314 ms, prefill 2.34 s (600.9 t/s), decode 67.1 t/s; the first request compiles the clip kernels and is the slow one |
| 512² encode | image slice 242-243 ms, image batch 7 ms, prefill 0.37 s (958-973 t/s) |
| memory | peak VRAM 6445 MiB, GTT 46 MiB, RSS 745 MiB, physical host MemAvailable ~55 GiB during the requests (sampler) |

**Phase-0 dispositions (`measured-here`, 2026-10-18).** A1: `libmtmd` builds
inside arcint (`code`). A2: an `image_url` data URI answers and the OpenCL
device is named for the mmproj graph, no CPU island reported (`measured-here`).
B1/B4: 1024² 1369 image tokens (1407 with text), image slice 1716 ms, image
batch 314 ms, prefill 2.34 s (600.9 t/s); 512² slice 242-243 ms, batch 7 ms,
prefill 0.37 s (`measured-here`). C1/C2: the content check **4/4**
(`measured-here`). The mmproj's own halves (vision/audio) are now published on
`/props` (§9.9). **Open (owed):** `tools/vlm_gate.py` run as a tool (the set was
driven by a scratch script), a CPU-side encoder B4 reference, the per-node op
profile, a red case, and the endpoint's package/unit/probe completion.
**Resolved by §9.10 (2026-10-19):** the gate tool run with a committed fixed
set (C1/C2 9/9), the per-node op profile (A2/B1, no CPU island), the CPU-side
B4 reference (the reference CLI on the card CPU, §9.10) and the one-time
compile record; the red case is **vacuous** because no VLM kernel was added or
changed (the profile found no CPU island and no patch was needed); the
endpoint's package/probe completion stays owed.

**RESOLVED (2026-10-18 evening, operator):** the operator power-cycled the card
host after the §9.1 `xe` oops; the host, the guests and the production units
are back (the B60's language unit active again, the A770's coder active, the
image front idle). The cause was three processes on the B60 at once: the dev
VLM server, a server respawned by a stale deleted-but-running leg driver, and
the production agent restarted on top. The hard rules in force from then on,
and the §9.2 window below ran under them: before ANY card window list every
process holding that card's render node and every leg/driver still running
(deleted-but-running included: `ls -l /proc/*/exe | grep deleted`); if any
non-service process holds it, stop and investigate; one non-service process
per card; stop a dev server by pid and verify the node is free before
restarting production; for the B60 POST `/release` to the image front first;
wrap every experiment in `timeout`; never put tokens on a command line.
Everything above is committed on the `v120` branch; nothing was pushed, tagged
or released (D9).

### 9.1 The card-window incident (measured-here, 2026-10-18, on the record)

This is the window's own lesson, written down before it is lost. While the
dev VLM server held the **B60**, the 1.1.1 **release session's** own card
window also started a capital-check `arcint` on the B60 at 18:16:10 UTC (that
session's report; the process hung mid-load; one of the three was not this
window's), and the production language unit was restarted on it — **three
processes on one card under xe**. Result:
an `xe_pt_zap_ptes` NULL-pointer oops in the page-fault workqueue
(`xe_po_move` → `xe_pt_zap_ptes`, CPU 5, `kworker/u65:3 exited with irqs
disabled`) and an RCU self-detected stall with `kswapd0` spinning; the
physical host stopped answering SSH. Per `docs/sop-card-window.md` and
`CLAUDE.local.md`, that state has only been recovered by a host reboot, and
the reboot is the operator's call. The lessons: **sweep stale leg drivers by
pid before restoring production units**, **never restore a unit onto a card a
dev leg holds**, and **read the owner of every card process from the process
tree, not by assumption**: a process on the card can belong to another
session (the release session uses short B60 gaps too), so the non-service
holder check runs before **every** card step; the SOP's “zombie sweep by pid”
already said it, and this window paid for skipping it.

### 9.2 STT measured (Phase 0, 2026-10-18 late)

The whisper leg, on the dev host's B60 (PCI `8086:e211`, OpenCL `0:0`), built
and measured in two windows after the host recovery:

| item | value (`measured-here`) |
|---|---|
| A1/A3, the whisper binary | `whisper.cpp` at the pin `d1be6fde` (v1.9.5) built with `WHISPER_USE_SYSTEM_GGML` against the ggml installed from the card host's `3e4dfa8` + series tree; `whisper-server` and `whisper-cli`; both carry `kernel_mul_mv_kq_q8_1` and `ggml_opencl:` (the recipe's two markers). The recipe needed two fixes: the top-level `app/` executable cannot build with `LLAMA_BUILD_COMMON=OFF`, so it is `LLAMA_BUILD_APP=OFF`/`LLAMA_TOOLS_INSTALL=OFF` and `ggml`+`llama` are built for the install; the second marker is not a literal `Intel(R)` at this pin |
| model | `ggml-large-v3-turbo.bin`, 1.62 GiB (MIT-tagged `openai/whisper-large-v3-turbo`); `.bin`, not GGUF, per the charter's risk |
| route | `/v1/audio/transcriptions` served by `whisper-server --inference-path /v1/audio/transcriptions` (an existing flag, no patch); `language` and `response_format` match OpenAI; non-empty transcripts |
| card named | `ggml_opencl: OpenCL device: Intel(R) Arc(TM) Pro B60 Graphics` (26.27.39122.11), `whisper_backend_init_gpu: using GPUOpenCL backend`; `fallbacks = 0 p / 0 h` over the set (whisper.cpp's decoder temperature fallbacks, i.e. no segment was re-decoded; it says nothing about op placement, which the card's encode time against the CPU's shows, B2) |
| the German set | FLEURS `de_de` dev (CC-BY-4.0, `measured-here` licence tag), 12 clips 2.5–13.75 s of the dev split, normalised from the split's float32 WAV to 16 kHz mono PCM s16 (`arithmetic`); reference text from the split's TSV; manifest sha256 recorded, audio stays on the card host outside the repo |
| B2, card vs CPU | encode 804.6 ms/run on the card vs 3,553.5 ms/run on the CPU (**4.4×**); decode 2.11 vs 11.48 ms/run (**5.4×**); mel (54.8 ms) and sample (132.9 ms) are the CPU-side front end; load 11.6 s card vs 8.95 s CPU, 16 threads both |
| C3, the WER gate | card corpus WER **0.0547** (11 errors / 201 ref words) = CPU 0.0547 exactly; `compare` PASS (card ≤ CPU + 0.005); total 27.5 s for the 12 clips on the card |
| window hygiene | two B60 windows: the first aborted on an over-strict render-node check (the A770's coder **enumerates both** render nodes incidentally; the check is now "non-service holders"); the CPU arm ran with the CPU `whisper-server` holding both render nodes **beside the active agent**, a rule breach with no B60 compute and no damage, recorded; card server stopped by pid, node free, unit restored |
| the gate's own defect | the first gate arm returned 400 from `whisper-server`: the multipart builder had no CRLF before the closing boundary (RFC 2046); fixed in `stt_gate.py` and `tts_gate.py` |

Open dispositions of the next window: B1's per-node profile for the whisper encoder
and decoder; the service phase (the `arcint-stt` unit, the package, `/props`, the
`.bin` loader path); and the operator's absolute C3 floor (D8: only the CPU
comparison gates so far; since set, corpus WER ≤ 0.08, which 0.0547 passes, §6).

### 9.3 TTS (Phase 0/4, 2026-10-18 late)

The TTS work: the code is built; the card arm at the fixed model ran 2026-10-19 (**C4 PASS, B3 FAIL**, §9.4).

The code (`measured-here` as "it compiles and it loaded on the card"):

- `--tts` enables the audio-generation path (`cp.embeddings = 1`, which
  `llama_get_embeddings_ith` needs; the logits reservation lifts to n_batch for
  the embedding batches). `--tts-profiles DIR` (or `$ARCINT_TTS_PROFILES`) and
  `--tts-lang` (default `de`). Without `--tts` the speech routes refuse 400
  clearly; without an audio-generation mmproj the boot fails.
- `Backend::generate_speech` follows the reference `tools/tts/tts.cpp`:
  `mtmd_helper::gen_audio` set_input/step_prompt loop, then per frame arcint's
  own `lgc::Sampler` over `llama_get_logits_ith`, `h_state` from
  `llama_get_embeddings_ith`, `step_gen(sampled, h, &h_next, &stop)`, and
  `get_output` for the WAV bytes. The lane's KV is cleared before/after and the
  synthesis holds the context mutex (the helper's internal decodes touch the
  shared context).
- `POST /v1/audio/speech` (WAV bytes), `GET /v1/audio/voices` (names only),
  the D5 refusals of any reference-audio field and of an unknown voice, and the
  `/props` `tts`/`voices` count/`multimodal` fields. The profile scan reads
  `<dir>/<name>/profile.json` + `reference.wav`, enabled entries only.
- Two load defects were found on the card and fixed: the `qwen3tts`
  architecture was refused by the engine's model gate (admitted under `--tts`),
  and the Qwen3-TTS backbone carries **no chat template** (a ChatML fallback
  under `--tts` so a stray chat request renders; the speech prompt comes from
  the gen pipeline, not the template).
- The gate voice is `gate-de`, enrolled from a FLEURS `de_de` dev utterance
  (CC-BY-4.0, pronunciation recorded in the profile's metadata); the sentence
  set is ten short German sentences (operator-local, outside the repo).

**Measured (the card window, FAIL; the branch carries the wiring `dd2cc59`
and both load fixes, and the window ran a build of it):**

| item | value (`measured-here`) |
|---|---|
| A4, it answers | `/v1/audio/speech` returned WAV bytes for the enrolled voices (`gate-de` and `gate-fleurs`, both from FLEURS `de_de` samples); the D5 refusals — a reference-audio field and an unknown voice — answered 400 (`measured-here`) |
| C4, intelligibility (the greedy arm) | **FAIL**: round-trip corpus WER **2.0909** over 8 sentences at floor 0.20 (voice `gate-fleurs`, seed 42); the waveform is not intelligible German — the transcripts are unrelated phrases and `* Musik *` labels, and one sentence's own words never come back |
| B3, real time (the greedy arm) | **FAIL**: generation hit the 512-frame cap (41.0 s of audio) at 140–180 s of process per sentence — a real-time factor of ~**0.29** (slower than real time); one sentence stopped at 48 frames, the rest at the cap |
| the arm's own defect | the gate defaulted to a **greedy temperature 0** (a `body` field added silently by the first gate), where the reference `tools/tts/tts.cpp` samples the backbone's codec tokens with `common_params`' temperature 0.8 / top_k 40 / top_p 0.95 / repetition 1.0. Greedy derails the codec stream past the first frames and never reaches end-of-speech (`measured-here`): the 512-frame cap and the 2.09 WER are the **gate arm's**, not a verdict on the implementation. `tools/tts_gate.py` now carries the reference defaults as flags and prints the sampling with every verdict |
| the corrected (stochastic) arm | **also FAIL**: corpus WER **2.1667** (floor 0.20) with the reference's sampling defaults and a fixed seed; every sentence still hit the 512-frame cap (41.0 s), so end-of-speech never arrives and the waveform is not intelligible German (`* Musik *` labels). The gate's greedy setting was one defect; the code path is the other |
| the window | driven by this same session's own remote leg script — **there was no second session**: `--sh` returned after a few seconds while the remote `tts-leg-card.sh` (started by the remote daemon at 21:25:03 UTC) kept running for 20+ minutes (the wiring is on the branch); the physical-host sampler, the one-process-per-card check and the pid stop/restore all held, the B60's language unit restored after 100 s |

**Open, and the next step is a campaign:** with both arms failing (WER 2.09
greedy, 2.1667 stochastic), the `qwen3tts` gen path is wrong on the model side,
not in the gate. The campaign `docs/campaigns/qwen3tts-german-genpath.md`
(opened 2026-10-19) resolved it (see the paragraph below): the defect is the
Q8_0 mmproj's code predictor, and the fix is a bf16 (or code-predictor-bf16
mixed) mmproj. A4's WAV and the refusals are real; the earlier C4 and B3
arms **fail on the record** and are not a passing gate. The card arm at the
fixed model ran 2026-10-19: **C4 PASS, B3 FAIL** (§9.4).

**The upstream reference tool reproduces the German failure
(measured-here, 2026-10-19):** `llama-tts`, built from the same pin + series
on the card host and run on the **CPU** (`-ngl 0`) on the same model pair:
the English "Hello world, this is a test." comes back as ` this is a test`
(intelligible), while the German "Der Zug nach Berlin ..." comes back as an
unrelated phrase (`Vielen Dank.`) both **with and without** the enrolled FLEURS
speaker reference. So the defect is the `qwen3tts` gen path's model side, not
arcint's loop — the loop is faithful to `tts.cpp`.

**Resolved by the campaign (2026-10-19, `measured-here`):**
`docs/campaigns/qwen3tts-german-genpath.md` isolates it with one variable per
CPU arm: the **backbone quantization is irrelevant**, the **`codec_language`
mapping is consistent**, **sampling is not the cause** (Qwen's own defaults on
Q8_0 still run to the cap), and the defect is the **Q8_0 `mmproj`'s code
predictor `a.gen.code`** — keeping only that subgroup at BF16 (or using the
official bf16 mmproj) makes the sentence transcribe exactly and sample
`codec_eos` at frame 44-50, while keeping the code2wav `a.gen.wav` at BF16 does
not. The fix is `Q8_0 backbone + bf16 mmproj` (2.52 GB), or a mixed Q8_0
mmproj with `a.gen.code` at BF16 (0.56 GB). The card arm through arcint's own
route ran 2026-10-19 (§9.4): **C4 PASS, B3 FAIL**.

### 9.4 The TTS card arm at the fixed model (Phase 3/4, 2026-10-19)

The milestone C4/B3 gate, on the B60 (PCI `8086:e211`, OpenCL GPU.0), the
B60's language unit stopped for the window, the physical-host sampler
recording, one non-service process per card, the final unit restored and
`:8087` verified (`measured-here`). The dev binary sha256 `0179adb5…`, `Q8_0`
backbone `ac7931ae…` + **bf16 mmproj** `b9503e95…`, 4,096 ctx, the full clone
path on `gate-fleurs` (FLEURS `de_de`, CC-BY-4.0) through
`POST /v1/audio/speech`, sampling t 0.8 / top_p 0.95 / top_k 40 / repetition
1.0, seed 42, fixed 8-sentence set.

| item | measured |
|---|---|
| A4/D5 | the route returns WAV; `reference_audio`, `speaker_file` and an unknown voice are all 400; `/v1/audio/voices` lists 4 names (record through the leg) |
| C4 (the gate) | **PASS**: round-trip corpus WER **0.0455** (3 errors / 66 ref words), floor 0.20; `codec_eos` sampled, 46–69 frames per sentence, no cap |
| B3 | **FAIL**: 36.48 s audio / 141.38 s process = **real-time factor 0.258**; the same loop on the CPU is 0.233, so the card is no faster — the gen graph's 947 splits and f32 CPU islands own it (campaign `qwen3tts-card-real-time`) |
| enrolment proof (D5 end to end) | `tools/tts_enrol.py` enrolled two FLEURS stand-ins (10.0 s and 10.8 s; profile names operator-local) into the operator-local profile dir; the server served them as names; one sentence each synthesized and round-tripped at WER **0.0000** |
| window hygiene | sampler on the physical host, leg aborts on a stale sampler log; non-service holder check before every card step; agent stopped by its unit and restored, `:8087` verified; one non-service process per card; dev servers killed by pid |

The C4 reversal against §9.3's earlier FAIL confirms the campaign fix; B3 was
later closed by lever (b) (§9.7). The VLM phase-0 dispositions are recorded in
§9.1; the `vlm_gate` tool run, the CPU encoder B4 reference, the op profile, the
red case and the endpoint's package/unit/probe completion stay open. The speech
units' files are written (`contrib/systemd/`, ports 8083/8084/8091 proposed);
`/props` carries the tts flag, the enrolled-voice count and, since §9.9, the
projector's own `vision`/`audio` halves. The packages remain owed: the 1.2.0
`arcint` deb after the feature freeze, and a new `arcint-stt` recipe for the
`arcint-whisper` binary (no package PR, D9).

### 9.5 The TTS real-time profile (Phase 1, 2026-10-19)

The B3 campaign `qwen3tts-card-real-time` measured its B1 profile on the B60
(PCI `8086:e211`, the fixed Q8_0 backbone + bf16 mmproj, dev binaries named in
the campaign, `gate-fleurs`, seed 42, reference sampling, the 8-sentence set):

| arm | RTF | per frame | predictor | code2wav | GEN_CODE splits |
|---|---|---|---|---|---|
| wall (instrumented) | 0.267 | 280.3 ms | 272.4 ms (97 %) | 14.5 s | 1007 |
| `0061` OpenCL ARANGE | 0.280 | 265.3 ms | 257.4 ms | 14.5 s | 299 |
| hybrid (mmproj on CPU, backbone on card) | **0.828** | 72.2 ms | 61.9 ms | 13.5 s | 1 |

`measured-here`:`GGML_OPENCL_PROFILING` puts OpenCL kernel execution at 18.6 s
against 145.1 s of clip compute; the trace's host enqueue is 1.9 s. The**split
count is not the cost** (`0061` cut 708 splits for ~15 ms), the **graph build is
not** (0.65 ms/call). A `GGML_OPENCL=OFF` single-backend `llama-tts` runs the
same 6792-node predictor in 59.9 ms/frame — the card is **4.5× slower** for the
same graph, ~6,600 tiny device kernels per frame (per-node overhead, not copies
or enqueue). The code2wav is ~1.8 s/flush on card and CPU alike.

**Disposition:** card-only real time on this upstream graph path is **out of
reach** without a node-count change to the code predictor or code2wav; the best
measured arm (hybrid 0.828, CPU codec graphs + card backbone) is 17 % short and
is an operator placement call, not a card-only B3. The earlier §9.4 "CPU
0.233" was the OpenCL-enabled `llama-tts` (its clip graph still split), not a
single-backend CPU arm; the corrected CPU predictor is 59.9 ms/frame.

**Placement and overlap (measured-here, 2026-10-19, second window):** a second
dev binary (`arcint-pipe` `fd0890de…`, the B1 scratch tree + a second GEN_WAV
clip context + an optional code2wav worker), the same B60, model pair,
`gate-fleurs`, seed 42, **C4 PASS (WER 0.0455) on all arms**: hybrid 0.828
(48.3 s), codec-on-card 0.751 (53.3 s), window-16 sequential 0.420 (95.2 s),
window-16 async worker 0.536 (74.6 s). Three measured dispositions: the codec
is **cheapest on the CPU** (1,692 ms/flush against 1,780-2,000 on the card); the
decoder's flush cost is **fixed per call** (256 codes 1.79 s against 992 codes
1.95 s — the 6,492-node graph and its carried state, not the frame count), so
chunking multiplies it; and window < 72 is **not seamless** (card-vs-CPU at
window 72 differs ≤483/32768 (0.0147), window 16 vs 72 differs up to 25,749/32768 at the
same sample count and WER). With `window_frames=72` a 51-69-frame sentence has
one flush, so there is nothing to overlap; the async worker also drifts the
sampled frames (66,50,45,72,48,57,60,54 against the deterministic
62,61,56,66,51,64,69,51) — concurrent card use is not adoptable. **Update:** the
hybrid remains the best arm and is the production placement **accepted by
the operator 2026-10-19**; pipelining is closed by the refutation; `0061`'s op suite
is green/red-green on the B60 but the patch is **dropped** (its only
beneficiary, the card-only predictor, is not adopted).

**The hybrid landed as a real option (2026-10-19, `code`).** The placement
switch is the documented `--tts-gen-device cpu|gpu` (`gpu` default; `cpu` the
hybrid), parsed and unit-tested with the other `--tts` flags, carried by
`arcint-tts.service`, and applied in `backend_llama.cpp`; the boot refuses a
`cpu` request that builds any `mmproj` clip context on the card (libmtmd's
`clip_init` backend line is captured; `ARCINT_TTS_GEN_FORCE_GPU=1` the red
switch, `tools/tts_placement_check.sh` the red-first gate). C4 is unchanged.
The gate was then run on the B60 (2026-10-19, `measured-here`): **red
FAIL-before** (`ARCINT_TTS_GEN_FORCE_GPU=1` refuses, naming the card backend),
**green PASS** (`--tts-gen-device cpu`: boot up, CPU clip contexts only), card
hygiene per the SOP, `arcint-agent` stopped and restored.

**Lever (a), the code2wav fixed per-call cost (measured-here, CPU,
2026-10-19):** it is the **padded 72-frame `wav_tfm_swa` window**. The graph
sets `n_frames = 72` and the caller pads the codes to 72, so the decoder
computes 72 frames for a 16- or 72-frame flush alike (1.766-1.78 s, 99.6 %
compute; build/alloc/input/output 0.3 %; thread A/B 1.763-1.766 s over t=16..1,
not threads; perf: bf16/f16 gemms ~28 %, im2col ~11 %). Computing only the valid frames makes the flush proportional
(44 frames 1.11 s ... 64 1.59 s; 8-sentence total 14.25 -> 10.60 s) with a
small CPU-vs-CPU sample diff (max 162/32767, one sentence identical, inside the
placement arm's already-shown 483/32768). **The full-route card gate ran
(measured-here, 2026-10-19, window 72): the padded production arm 0.729
(55.66 s / 40.56 s audio, 11.0 s first audio) against the valid-frames arm
0.802 (50.58 s, 9.52 s), both round-trip corpus WER 0.0455 PASS (`codec_eos`
sampled, no cap), the same-binary sample diff at most 191/32768 -- the lever
is real, +10 %, still short of real time.** It landed as the engine **default**:
patch `0062-mtmd-qwen3tts-gen-valid-frames` (`tools/mtmd/{clip.cpp,
models/models.h, models/qwen3tts-gen.cpp}`), no operational or config switch, no
arcint flag (`MTMD_TTS_WAV_PAD` the red-case switch); the
patch binary's arm is bit-identical on all eight sentences to the scratch-switch
arm, so the default engages the measured mechanism; the red
`MTMD_TTS_WAV_PAD=1` restores the padded waveform (identical to the padded arm),
so the test can fail. The `-1` default keeps the `wav_tfm_swa` fallback width
for a no-codes caller and the `out_audio` null guard is kept (the upstream
PocketTTS `GEN_WAV` path decodes the caller's latent frames, passes features, no
codes); the pin + series application was checked clean.
Sub-real-time still: lever (b), the code predictor, is the remaining wall
(campaign `qwen3tts-card-real-time`). C7 (the language path's answers and KL
must not move) is unaffected by construction: 0062 changes only the Qwen3-TTS
`GEN_WAV` graph and the generic gen-audio `out_audio` tail block, and no chat
path builds it (`code`); a served language arm was not re-run for the patch.

### 9.6 C4 with the D8 floor on the enlarged set (Phase 3/4, 2026-10-19)

The D8 floors are recorded in §6; the C4 re-run on the enlarged Tatoeba set and
the current default (patch 0062, `--tts-gen-device cpu`, window 72) measured
(`measured-here`): the B60 (PCI `8086:e211`), dev binary `arcint-def-exact`
sha256 `be754e8b…` (the source built against the pin + series with the 0062
default; the D8 commits touch docs and the gate only, no engine source), Q8_0 backbone + bf16 mmproj, the full clone path on `gate-fleurs`, seed
42, reference sampling, 46 sentences / 303 words (manifest
`tools/tts_sentences_de.json`, sha256 `e0592aab…`). The gate's speech phase
times the RTF; the round-trip runs the CPU whisper-server (`-ng`), so only the
hybrid TTS touches the card.

| item | measured |
|---|---|
| C4 (D8 floor 0.10 / 0.5) | **PASS**: corpus round-trip WER **0.0066** (2 errors / 303 words), worst sentence **0.333** (one sentence: `rechtzeitig fertigmachen` -> `rechtzeitig fertig machen`, a compound split counted as two), sentences with WER <= 0.10: **97.8 %** (45 of 46); `codec_eos` sampled, 55-57 frames at the top of the set, no cap |
| B3 (D8 floor RTF >= 1.0) | **FAIL**: 177.28 s audio / 221.745 s process = **RTF 0.7995** — the valid-frames default on the enlarged set, still 20 % short; the earlier 8-sentence arms (0.729 padded / 0.802 valid) were a shorter set, so the numbers are not comparable |
| the card window | physical-host sampler: peak resident vram0 **2,586 MiB**, min physical MemAvailable **52,603 MiB**, no watchdog kill; image front released (`released:false`, idle); `arcint-agent` stopped by its unit and restored after 80 s with `:8087` health ok; one non-service process per card; no stale leg driver |
| the set | `tools/tts_sentences_de.json`: Tatoeba (CC-BY 2.0 FR, attribution), 46 unique German sentences of 5-10 words, 303 words, deterministic selection from the `deu_sentences.tsv.bz2` export (sha256 `ccd06372…`); text is committed, the WAVs and transcripts stay operator-local (the leg's output) |

C4 passes the operator's floor on the enlarged set; B3 stayed FAIL and its
lever (b) (the code predictor, campaign `qwen3tts-card-real-time`) was the next
measurement — landed in §9.7. C7 unchanged (`code`): 0062's graph is untouched
by this arm.

### 9.7 Lever (b): the code-predictor quantization, B3 PASS on the D8 set (Phase 2, 2026-10-19)

The remaining wall after §9.6 was the code predictor (the mmproj graphs on the
CPU under the accepted hybrid placement). A one-variable mmproj family
requantized from the official bf16 (`llama-quantize --pure --allow-requantize`,
single CPU backend) measures the predictor's per-call time: bf16-all 105.4 ms,
f16-all 57.7 ms, f32-all 121.1 ms, Q8_0-all 30.2 ms (the code2wav flush stays
~1.76 s on all). The production recipe requantizes the whole mmproj to Q8_0
while keeping the two sampling tensors `gen.code.head` (sampling head) and
`gen.code.embd` (code embedding) at f16:
`llama-quantize --pure --allow-requantize --tensor-type gen\.code\.head=f16
--tensor-type gen\.code\.embd=f16 <official-bf16-mmproj> <out> Q8_0 8`.
`q8codeheadf16` measures 31.0 ms per call, against 57.6 ms when the whole
`gen.code` stays f16: the 35 `gen.code.blk` matrices are the bulk speed. Which
of head/embd is load-bearing was **not measured** (no head-only arm); the pair
is the measured recipe that passes.

The card arm, the full clone path through `/v1/audio/speech`, dev binary
`arcint-def-exact` sha256 `be754e8b…` (0062 default, `--tts-gen-device cpu`,
window 72), Q8_0 backbone `ac7931ae…` + the requantized mmproj sha256
`4d28d3af…` (`/models/tts-qwen3/
mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0-codeheadf16.gguf`), `gate-fleurs`, seed
42, the enlarged D8 set (46 sentences / 303 words, manifest `e0592aab…`):

| item | measured |
|---|---|
| q8codeheadf16, before the card host's RAM upgrade | **C4 PASS**: corpus WER **0.0264**, worst **0.333**, WER<=0.10 **89.1 %**; **B3 PASS**: **RTF 1.1291** (182.960 s audio / 162.041 s process) |
| the staged production mmproj, after the upgrade (more host RAM) | **C4 PASS** (identical WER 0.0264 / worst 0.333); **B3 PASS**: **RTF 1.1597** (182.960 / 157.767) |
| the cross-check f16-all | **B3 FAIL**: RTF **0.8051** (172.560 / 214.324) — raising the whole mmproj to f16 does not reach real time; not adoptable, not re-run |
| the cross-check f32-all | no output; dominated (121.1 ms predictor call, slowest of the family); not re-run |
| the card window (after the upgrade) | peak resident vram0 **2,586 MiB**, min physical MemAvailable **116,787 MiB**, max container **4,066 MiB**; one non-service process per card; `arcint-agent` stopped by its unit and restored after 80 s, `:8087` health ok |

**Disposition (`measured-here`, `code`):** the recipe is the production TTS
model set (`contrib/systemd/arcint-tts.service` points at the requantized file;
the official bf16 stays as fallback); **B3 PASS** closes the campaign
`qwen3tts-card-real-time`; the two-f16-tensor choice refines the closed
`qwen3tts-german-genpath` (which tested the whole `gen.code` at bf16) and is not
a new defect. **Its quality cost** (`measured-here`, the same binary
`be754e8b…`, seed, voice and 46-sentence set as §9.6): the bf16 mmproj's
round-trip WER 0.0066 (2 errors / 303 words, 97.8 % of sentences ≤ 0.10)
becomes 0.0264 (8 errors, 89.1 %) with the production mmproj; the worst
sentence stays 0.333. Both pass the D8 floor; the bf16 file stays the
fallback where that margin matters more than real time (RTF 0.7995). C7 unchanged: the recipe changes only the mmproj file, no engine
graph.

### 9.8 C5 voice similarity: the SIM method landed and measured (Phase 4, 2026-10-19)

C5's named model is the Seed-TTS-eval SIM model (WavLM-Large + ECAPA-TDNN, the
UniSpeech `wavlm_large_finetune.pth`, sha256 `51f07e3b…`, mirror
`bezzam/wavlm_large_finetune_seed_tts_eval`, CC-BY-SA-3.0). Per §2's rule and
the operator's step-3 condition it is run in a torch **container** (an LXC) and
the kept copy is converted to safetensors there
(`wavlm_large_finetune_seed_tts_eval.safetensors`, 1,301,746,084 bytes). The
backbone is an HF `WavLMModel` with the checkpoint's original-WavLM names
remapped onto it, the ECAPA-TDNN head transcribed from UniSpeech, one clip per
forward pass, cosine — the `huggingface/open-tts-leaderboard` recipe
(Apache-2.0), landed as `tools/tts_voice_sim.py` and driven by
`tools/tts_gate.py --embed-cmd`.

| stand-in | cosine |
|---|---|
| same audio vs itself (sanity) | 1.000000 |
| **the gate-voice clone vs its reference** (46 synthesized `gate-fleurs` sentences) | **mean 0.598038**, min 0.439814, max 0.751893 |
| two FLEURS files of one sentence, two different speakers (CC-BY-4.0, cross) | **0.262419** |
| published same-speaker VoxCeleb1 range (this model; `paper`, the leaderboard README's example clips) | 0.60 - 0.69 |
| published cross-speaker VoxCeleb1 range (this model; `paper`, same source) | -0.17 - 0.18 |
| published ground-truth SIM-o, test-en (D8's anchor; `paper`, arXiv 2606.03455) | 0.73 |

**Disposition (`measured-here` for the cosines, `paper` for the anchors):**
the clone-vs-reference mean 0.598 sits at the low edge of, just under, the
published same-speaker range (0.60-0.69) and below the D8 ground-truth anchor
0.73, and clearly above both the measured CC cross pair (0.262) and the
published cross range. The method separates same from cross speakers, which
is what METHOD PASS claims; whether the clone comes "near" a real same-person
pair is the final calibration, open until the real recordings exist. **A same-speaker FLEURS pair is not
obtainable** (measured-here on the local `de_de` dev TSV and the canonical
`google/fleurs`: neither carries a speaker field; the `id` column groups the
2-3 speakers per sentence), so the reference's published anchors stand as the
same-side calibration and the CC stand-ins are the measured cross pair.
Low-download CC alternatives with speaker labels were surveyed and not taken
(Common Voice gated or empty; VoxPopuli `de` ungated with `speaker_id` but its
smallest parquet 936 MB). The real same-person recordings remain the operator's
final C5.

### 9.9 `/props` projector halves: VLM vs TTS from the projector itself (Phase 4, 2026-10-19)

`ModelStatus` gains `mmproj_vision`/`mmproj_audio`, filled at the mmproj load
from libmtmd's `mtmd_get_cap_from_file` (`mtmd_caps.inp_vision`/`inp_audio`;
EXPERIMENTAL at the pin). `/props` `multimodal` publishes them beside
`backbone_loaded`/`mmproj_loaded`, so a client tells the image path from the
audio-generation path without reading the process's `--tts` flag. The stub
reports false/false. Unit tests: a synthetic-status cell covers vision-only,
audio-only and none, and a source tripwire pins the cap call in the load site
(`tests/test_props.cpp`); the bare `ctest -L unit` ladder is green (178 cases,
0 failed, the stub roundtrip among them), `measured-here`. The omission was
`code`: the field `mmproj_loaded` did not say which half the file carried.

### 9.10 VLM gate, profile and content check (Phase 0/1, 2026-10-19)

The card window: B60 (PCI `8086:e211`, OpenCL GPU.0), one non-service process
per card sequentially, the image front released, `arcint-agent` stopped by its
unit and restored after 85 s (`:8087` active), a physical-host sampler. The
dev binary is `arcint-241c171` sha256 `65a95529…`, built from the staged v120
HEAD (`241c171`) against the pin + series through 0062, `--device GPU.0` with
the VLM unit's flags (Qwen2.5-VL Q4_K_M + Q8_0 mmproj, `--n-ctx` 32768,
`q8_0:q4_0`, `-v`); a profiling sibling `arcint-241c171-prof` sha256
`0ff0d66b…` (`GGML_OPENCL_PROFILING=ON`) carries B1.

| item | value (`measured-here`) |
|---|---|
| C1/C2, the gate tool | `tools/vlm_gate.py` with the committed manifest `tools/vlm_cases.json` (sha256 `7e95d05e…`): **9/9 = 1.000**, floor 1.0. Answers WILLKOMMEN, Willkommen ×2, Cat, cat, Palm trees and beach., Red, cube, 1. The first case 8.42 s (the one-time clip-kernel compile), the 2048² upscale 8.77 s, the rest 0.41-1.81 s |
| the set | seven images from the 1.1.0/1.1.1 record, nine questions, kinds lettering/object/background/colour/count, each sha256-pinned; the images stay operator-local beside the committed manifest (charter §4) |
| B1/B4 card side | 1024² after the compile: image slice encoded 1280 ms, image batch decoded 317 ms (encoder 1.60 s), prompt prefill 1407 tokens in 1.73 s (813.4 t/s), completion 2 tokens 0.03 s; 512² slice 247 ms + 7 ms batch, prefill 362 tokens 0.39 s (938.7 t/s); 2048² (the upscaled cat) slice 7230 ms + batch 479/651 ms, prefill 4130 tokens 8.55 s (483.2 t/s). First 1024²: 5483 ms slice / 8.27 s prefill (kernel compile). Encoder share of the server's recorded phases ≈ 48 % at 1024², ≈ 46 % at 2048², < 10 % at 512² |
| B4 CPU encoder reference | the reference `llama-mtmd-cli` built from the same pin + series through 0062 (sha256 `9c66c8a2…`, the CPU build's own ggml) with `--no-mmproj-offload -ngl 0 -t 16`, on the card-host CPU (Ryzen 5700X, 16 CPUs; a separate CPU-only run, identical images): CPU encode **14334 ms** (1024²) / **2836** (512²) / **65587** (2048²) / **2822** (512²) against the card's **1597** / **254** / **8360** / **254** → the CPU encoder is **8-11×** the card's; the CPU end-to-end wall 64/16/145/14 s is 16-37× the card's; all four CPU answers are the correct content-check answers. The CLI prints one whole `mtmd batch encoding` time (no slice/batch split), so the CPU/ card columns compare encoder+batch |
| A2 | `GGML_SCHED_DEBUG=2` on the profiling build: every node `[OpenC ]` (OpenCL), the clip graph `splits = 1, nodes = 1090`; the profiling dump `cl_profiling.csv` holds 11,358 kernels / 1,753.07 ms device exec (includes the load warmup), **zero CPU island** |
| memory | physical-host sampler (`/models/.tts-prof/vlm-gate.log`, `measured-here`): peak resident vram0 **7,567 MiB** (the 2048² case), GTT 40 MiB; physical MemAvailable min **117,319 MiB**; container peak 5,197 MiB |
| hygiene | `released:false` (the front was idle); one non-service process per card each phase; `arcint-agent` restored after 85 s, `:8087` active; servers stopped by pid; no stale leg driver |

**Dispositions (`measured-here`, `code`).** C1 PASS at the target 1.0; the
operator's C1 floor is still unset and the pass rate is 1.000. C2: the 1.1.0
content-check checkbox is closed for the Qwen-Image WILLKOMMEN (three
generations) and the generated object/colour/count images. **No Z-Image
lettering image exists in the 1.1.0/1.1.1 record** — the Z-Image case is its
red cube (object, colour, count), read from the arms' logs; the charter's
"Z-Image lettering" has no source image and is recorded as such. A2 holds: no
op of the vision tower or the backbone fell to the CPU. B1: the vision slice
is the largest single phase at 1024²/2048² and is **on the device**; no node
over 10 % needs a ceiling or a campaign. B4: the CPU encoder is 8-11× the
card's and the CPU answers are the same, so the card's vision path is the
serving one and the CPU stays the correctness reference. C6 is **vacuous for
the VLM**: no kernel was added or changed (the profile found no CPU island).
**A trap:** the server's own phase log sums larger than the gate's endpoint
wall for the first/warmup-heavy requests — the gate's per-case seconds are
≈ prefill + decode, i.e. they exclude the logged `image slice` / `image
decoded`; the two were not reconciled in this window. Report both, do not mix
them.

The endpoint's package/unit/probe completion stays owed: the unit files
exist (`contrib/systemd/arcint-vlm.service`), `/props` publishes the
mmproj/backbone state and the projector's own halves (§9.9), and the 1.2.0
`arcint` package is the release phase's.

## 10. References

- llama.cpp `3e4dfa8`: `tools/mtmd/{mtmd.h,mtmd.cpp,mtmd-image.cpp,
  mtmd-audio.cpp,mtmd-helper-gen.cpp,clip.cpp,models/*}`, `tools/tts/tts.cpp`,
  `docs/multimodal.md`, `ggml/src/ggml-opencl/ggml-opencl.cpp`'s `supports_op`
  and `can_fuse` lists.
- whisper.cpp (`ggml-org/whisper.cpp`, pin to be named in
  `contrib/whisper/README.md`): its `whisper.cpp` encoder/decoder graph, the
  `WHISPER_USE_SYSTEM_GGML` build.
- Qwen3-TTS (`QwenLM/Qwen3-TTS`, the `Qwen/Qwen3-TTS-12Hz-1.7B-Base` card and
  its Apache-2.0 tag; the ggml-org GGUF mirror), OuteTTS and Pocket TTS cards
  for the disposition.
- Qwen2.5-VL (`QwenLM/Qwen2.5-VL`, the ggml-org GGUF and its `qwen2vl` model
  in `tools/mtmd/models/qwen2vl.cpp`).
- arcint: `src/core/chat.cpp` (the text-only refusal to lift),
  `src/exec/backend_llama.cpp` (the encode→decode plumbing),
  `contrib/sdcpp/build-sdcpp.sh` (the separate-binary recipe form),
  `contrib/systemd/arcint-imggen*.service` and `docs/imggen-service.md` (the
  unit and contract form), `contrib/llama.cpp/patches` (0059, 0060, 0062).
- The 1.1.0 gate tool `tools/imggen_gate.py` (its parity check is the
  pattern for the three new gates) and `docs/milestone-1.1.0.md` §3.

## 11. Release readiness (1.2.0, state 2026-10-19)

Every §6 criterion with its measured verdict or the reason it is open. This
is the checklist the release session works from; the release itself (push,
tag, packages, fresh-container install, A/B) is that session's (D9).

| # | verdict | evidence / reason |
|---|---|---|
| A1 `libmtmd` in arcint at the pin, kernels present, `GGML_MAX_NAME` 64 | **PASS** | `code`+`measured-here`: built (§9.1, §9.10, the `mtmd` target, no `llama-common`); the gate binary named the OpenCL device for the mmproj graph |
| A2 a VLM image request on the card, no op silently CPU | **PASS** | `measured-here` §9.10: 9/9; `GGML_SCHED_DEBUG=2` every node `[OpenC ]`, clip graph splits=1/1090, zero island |
| A3 the whisper binary on the installed ggml, `/v1/audio/transcriptions`, a transcription on the card | **PASS** | `measured-here` §9.2: built with `WHISPER_USE_SYSTEM_GGML`; the route via `--inference-path`; the card named; non-empty transcripts; no decoder temperature fallback (`fallbacks = 0 p / 0 h`) |
| A4 `/v1/audio/speech` on the card, the enrolled clone path, WAV out, the 400 refusals | **PASS** | `measured-here` §9.4, §9.7: WAV bytes; arbitrary-reference/unknown-voice 400; C4 PASS |
| A5 the language services unchanged: the A/B vs the last release, acceptance ≥ 9/10 | **OPEN** | not re-run for 1.2.0 (owed at the release phase). By construction (`code`): 0062 touches only the TTS `GEN_WAV` graph and the generic gen-audio tail, no chat path builds it; the VLM/STT/TTS routes are separate processes. C7 rides with this |
| A6 one unit, one package per binary, `/health`/`/props` | **PARTIAL, improved** | unit files exist (`contrib/systemd/arcint-{vlm,tts,stt}.service`), `/props` extended on the arcint binary (§9.9, §9.10); the new `arcint-stt` recipe is written and its pre-tag deb **builds** (1.2.0-1, `arcint-whisper` + `arcint-whisper-cli` + unit template + gate tool, kernel markers and ldd checked; not published, not installed); the 1.2.0 `arcint` deb is **owed** (its recipe needs the post-tag release-tarball sha); `arcint-whisper`'s `/props` is **owed** (it serves `/health`/`/load` only); no package PR (D9) |
| B1 a profile per capability | **PASS (phase split) / a finer STT node profile optional** | VLM §9.10 (phase shares + op profile + no island); TTS §9.5/§9.7 (campaign `qwen3tts-card-real-time` closed); STT's criterion is the encode/decode split and §9.2 has it (encode 804.6 vs 3,553.5 ms, decode 2.11 vs 11.48, card named). No `CONV_1D` op needed a patch or a profile row: the encoder ran on the device, shown by its 4.4× over the CPU encoder on identical clips (§9.2, `measured-here`); a per-node placement dump of whisper's graphs was not taken |
| B2 STT card faster than the CPU beyond the spread | **PASS** | `measured-here` §9.2: encode 4.4×, decode 5.4×, WER equal |
| B3 TTS RTF ≥ 1.0 on the sentence set | **PASS** | `measured-here` §9.7: 1.1291 (pre) / 1.1597 (post) on the enlarged D8 set |
| B4 VLM prefill/encode vs the CPU encoder | **PASS** | `measured-here` §9.10: the CPU encoder is 8-11× the card's on identical images; the CPU answers are the same; a separate CPU-only run |
| C1 the VLM content check; the operator's threshold | **PASS at 1.0** | `measured-here` §9.10: 9/9 = 1.000, floor 1.0; the operator's floor is still unset |
| C2 closes 1.1.0's content check (the WILLKOMMEN and the “Z-Image lettering”) | **PASS (WILLKOMMEN) / recorded gap** | `measured-here` §9.10: WILLKOMMEN (three generations) + object/colour/count right. The charter's “Z-Image lettering” has **no source image** in the 1.1.0/1.1.1 record (§9.10) — recorded, not hidden; the Z-Image case is its red cube |
| C3 STT WER on the FLEURS `de` set: corpus ≤ 0.08 and card ≤ CPU + 0.005 | **PASS** | `measured-here` §9.2: card = CPU **0.0547**, corpus ≤ 0.08 |
| C4 TTS round-trip WER ≤ 0.10 and no sentence > 0.5 | **PASS** | `measured-here` §9.7: corpus **0.0264**, worst **0.333**, share ≤ 0.10 **89.1 %** |
| C5 voice similarity, the Seed-TTS-eval WavLM cosine, calibrated | **METHOD PASS / FINAL OPEN** | `measured-here` §9.8: the gate-voice clone-vs-reference mean **0.598** sits at the published same-speaker anchors and above the measured cross pair **0.262**; the calibration floor is the operator's real recordings (his own and his wife's) and is **owed** (D8) |
| C6 a red case per new or changed kernel | **PASS for 0062 / vacuous for the VLM** | `measured-here` §9.5: the red `MTMD_TTS_WAV_PAD=1` restores the padded waveform (the test can fail); 0061 is dropped; the VLM added no kernel (§9.10); 0059/0060 were closed in 1.1.x |
| C7 the language path's answers/KL unchanged | **OPEN (rides A5)** | `code`+`measured-here`: 0062's graph is not on any chat path; the served A/B and KL arms are owed with A5 |
| naturalness, listened, no automatic gate | **OPEN** | operator ear against the real recordings, owed (D8) |
| release mechanics: push, tag, packages, fresh container, A/B | **DEFERRED** | the release session's (D9); `tools/tts_gate.py` already carries `--embed-cmd` on `run`/`transcribe`, so every C4 arm can report the cosine once the reference WAV is handed to it (operator-local) |

Open summary for the release phase: A5/C7 (the language A/B and KL), A6
(the 1.2.0 `arcint` deb and the `arcint-stt` recipe), B1-STT (per-node
profile), C5/naturalness (the operator's recordings), and the release
mechanics. Everything measured is on the record in §9; nothing is pushed,
tagged or packaged (D9).

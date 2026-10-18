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
   Markus and his wife will record their own reference samples later (about
   10-30 s each); until then build and gate with a freely licensed sample
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
`3e4dfa8` (`~/src/llama-port`):

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
| D8 | Gate tolerances | the operator sets the numbers when the first arms are on the record; until then the draft in §6 is a proposal, printed with every verdict |
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
| **TTS** backbone | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` → `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`, `...-Q8_0.gguf` | **apache-2.0** | 3,093,627 (HF) / 32,169 (GGUF) | ~1.9 GB | German (`de`, `code`), cloning, not gated |
| **TTS** mmproj | same GGUF repo, `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` | apache-2.0 | (repo) | ~1.6 GB | the speaker encoder + acoustic decoder |
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
measured), the whisper `.bin`, the Qwen3-TTS backbone and mmproj, and one
freely licensed German sample voice for the gate. Nothing goes into the
fast-staging NVMe between legs (SOP).

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
| `POST /v1/chat/completions` (existing) | `image_url` content parts accepted when a `--mmproj` is loaded; `data:` base64 and `http(s)` URLs; without an mmproj the text-only refusal stands | VLM |
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
  licensed gate voice until Markus and his wife record theirs.
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
- **Enrolment is a documented command**, e.g. a repo tool
  `tools/tts_enrol.py --name NAME --audio FILE --text FILE` that writes the
  profile into the card host's profile directory and validates the sample
  (length, sample rate, non-silence). The service manual (a new
  `docs/speech-service.md`) carries the exact steps, including the "clone two
  voices: Markus and his wife" end state.
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
- **The TTS acoustic decoder** — the Qwen3TTS gen half runs a codec decoder
  (transposed conv / upsampling / vocoder-style ops); `CONV_TRANSPOSE_1D`/
  `_2D` are not in the pin's case list (`code`), so those layers may fall to
  the CPU, which would put the waveform step on the CPU. Size it; if it is
  the whole cost, the acoustic decoder is the campaign.
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
| B3 | TTS: real-time factor (audio seconds per process second) at a fixed sentence set, first request separate | the card beats real time by a stated margin at the chosen quant |
| B4 | VLM: prefill/encode time per image at a fixed size and token budget | reported against the CPU encoder on the same window |

### C. It is right

| # | criterion | gate |
|---|---|---|
| C1 | The **VLM content check** on a fixed set of images with checkable content (an object, a count, a colour, a short lettering): answers match ground truth | a fixed question set per image; every answer checked; the gate is the set's pass rate, with the operator's threshold once measured |
| C2 | The VLM closes 1.1.0's open content check: it reads the generated Qwen-Image `WILLKOMMEN` and the Z-Image lettering | the lettering is reported correctly; this is the 1.1.0 charter §3 "content check" that was left unbuilt |
| C3 | STT **word error rate** on a fixed German test set against the **same model's CPU run** (identical inputs) and against a floor | WER = `(S+D+I)/N` per clip, corpus WER over the whole set; candidate ≤ CPU WER + tolerance, and under the operator's absolute floor |
| C4 | TTS **intelligibility**: round-trip the synthesized German through the STT model, WER against the written sentences | corpus WER under the operator's floor on the fixed sentence set |
| C5 | TTS **voice similarity**: the clone's embedding against the reference's, the gate voice and (later) the enrolled ones | a speaker-embedding cosine above the operator's floor; the embedding model named on the record |
| C6 | Every new or changed kernel has its red case, which turns the capability's gate red | a deliberate mutation fails the capability gate and its op suite, as 0057-0060's did |
| C7 | The answer-level bar for the language path unchanged: the VLM/TTS unit must not perturb the language services' answers or KL | the A5 A/B and the existing KL arms |

The image gate of 1.1.0 (`tools/imggen_gate.py`) stays for the image units;
the new gates are new tools (`tools/vlm_gate.py`, `tools/stt_gate.py`,
`tools/tts_gate.py`), each printing its tolerances and refusing arms whose
inputs differ, as `imggen_gate.py`'s parity check does. Gate comparisons use
identical inputs (the rule learned in `imggen-attention-path`).

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

| item | measured () |
|---|---|
| A1, `libmtmd` in arcint | builds: the `mtmd` target and the `arcint` binary; `LLAMA_BUILD_MTMD=ON`, no `llama-common`, the pin's `GGML_MAX_NAME` 64, the series' kernels present in the tree |
| A2, a VLM request through the chat endpoint | yes: `/v1/chat/completions` with an `image_url` data URI answers; the OpenCL device is named for the mmproj graph, no CPU island reported (`-v` log) |
| load | `qwen2vl` Q4_K - Medium (4.36 GiB) + Q8_0 mmproj on the B60, in 46.2 s; mmproj `image_min_pixels` 6272, `image_max_pixels` 3,211,264 |
| C1/C2, content check | **4/4**: the generated Qwen-Image `WILLKOMMEN` imprinted on a white sign reads `Willkommen` (1024² and 512²); the FLUX.2 klein generations read `Cat` (the windowsill) and `Palm trees and beach.` (the edit's background). 1.1.0's open content check is closed for these images |
| B1/B4, 1024² encode | 1369 image tokens (1407 with text), image slice encoded 1716 ms, image batch decoded 314 ms, prefill 2.34 s (600.9 t/s), decode 67.1 t/s; the first request compiles the clip kernels and is the slow one |
| 512² encode | image slice 242-243 ms, image batch 7 ms, prefill 0.37 s (958-973 t/s) |
| memory | peak VRAM 6445 MiB, GTT 46 MiB, RSS 745 MiB, physical host MemAvailable ~55 GiB during the requests (sampler) |

The gate tool `tools/vlm_gate.py` is built but **not yet run as a tool** (the
phase-0 set above was run by a scratch script); a CPU-side encoder reference
for B4 is not measured; the op profile per node is not taken; no red case
yet; the endpoint is not package/unit/probe-complete. These are the open
dispositions of the next card window.

### 9.1 The card-window incident (measured-here, 2026-10-18, on the record)

This is the window's own lesson, written down before it is lost. While the
dev VLM server held the **B60**, a stale leg driver from a finished session
(a deleted `` script still running under the physical host's init, spawning
a 4k-context B60 server) held the same card, and the production language
unit was restarted on it — **three processes on one card under xe**. Result:
an `xe_pt_zap_ptes` NULL-pointer oops in the page-fault workqueue
(`xe_po_move` → `xe_pt_zap_ptes`, CPU 5, `kworker/u65:3 exited with irqs
disabled`) and an RCU self-detected stall with `kswapd0` spinning; the
physical host stopped answering SSH. Per `docs/sop-card-window.md` and
`CLAUDE.local.md`, that state has only been recovered by a host reboot, and
the reboot is the operator's call. The lessons: **sweep stale leg drivers by
pid before restoring production units**, and **never restore a unit onto a
card a dev leg holds**; the SOP's “zombie sweep by pid” already said
it, and this window paid for skipping it.

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
  unit and contract form), `contrib/llama.cpp/patches` (0059, 0060).
- The 1.1.0 gate tool `tools/imggen_gate.py` (its parity check is the
  pattern for the three new gates) and `docs/milestone-1.1.0.md` §3.

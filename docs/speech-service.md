# The voice and vision service — units, routes, enrolment (1.2.0)

The 1.2.0 charter (`docs/milestone-1.2.0.md`) adds vision input, speech to
text and text to speech with voice cloning. This is the service contract:
units, routes, the enrolment rules and the gate-voice handling. The **VLM chat
path is the only route built so far** (phase 0, 2026-10-18); the audio routes
below are the service phase's work and are marked **planned** until built.
The front and the client side are the operator's; the units, packages and this
contract live here.

## Units

Each capability is its own **on-demand unit** on the B60, with the image
units' discipline: `Conflicts=` with the card's language and image units,
started by the front on the first request, the card back after 20 idle
minutes, `/health` and `/props`. Ports are the next free ones beside the
language units (8080/8087), the image units (8081/8082/8085) and the front;
the operator fixes them.

| unit | binary | model files | routes | state |
|---|---|---|---|---|
| `arcint-vlm` | `/usr/bin/arcint` (`arcint` package) | `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` + `mmproj-...-Q8_0.gguf` | `/v1/chat/completions` with `image_url` | **built** (chat path + unit file; package and the `/props` multimodal fields owed) |
| `arcint-tts` | `/usr/bin/arcint` (`arcint` package) | `Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf` + **`mmproj-Qwen3-TTS-...-bf16.gguf`** (the Q8_0 mmproj's code predictor is defective; the campaign) | `POST /v1/audio/speech`, `GET /v1/audio/voices` | **built** (the `libmtmd` `gen_audio` path + routes + unit file; the unit takes `--tts-gen-device cpu`, the hybrid placement: the best measured TTS arm at 0.828 real-time factor, campaign `qwen3tts-card-real-time`); the first card gate **FAIL**ed on the Q8_0 mmproj (greedy WER 2.09, stochastic WER 2.1667, a 512-frame cap and no end-of-speech); `docs/campaigns/qwen3tts-german-genpath.md` (closed) isolated it to the mmproj's code predictor `a.gen.code` at Q8_0 and fixed it with a bf16 mmproj. The card arm at the fixed model (2026-10-19, `measured-here`) is **C4 PASS** (round-trip corpus WER 0.0455, floor 0.20, `codec_eos` sampled at 46–69 frames) but **B3 FAIL** (real-time factor 0.258, no faster than the CPU's 0.233; `docs/campaigns/qwen3tts-card-real-time.md`) |
| `arcint-stt` | `/usr/bin/arcint-whisper` (`arcint-stt` package) | `ggml-large-v3-turbo.bin` (or `ggml-large-v3.bin`) | `POST /v1/audio/transcriptions` | **built** (whisper.cpp on the installed ggml, card gate PASS; unit file written; package and `/props` owed) |

Ports (proposed; the operator fixes them): `arcint-vlm` 8083, `arcint-tts` 8084,
`arcint-stt` 8091, beside 8080/8087 (language), 8081/8082/8085 (image),
8086 (the image front) and 8088/8089 (the on-demand language units).

The VLM and TTS share one binary (`libmtmd` links `ggml` + `llama` only, so one
process with the pin's `GGML_MAX_NAME` 64, `docs/milestone-1.2.0.md §3`); the
whisper binary is separate (`contrib/whisper/build-whisper.sh`). No card holds
two of these at once (`Conflicts=` covers the card, as the image units do).

**Card choice.** None of the three fits beside the A770's permanent coder
(14.6 of 16 GiB), so all three are B60 on demand (per the mandate: "prefer the
A770 where it fits next to the coder, otherwise the B60 on demand"). The B60
is also shared with the image service: before a window, release through the
front; the units carry the same `Conflicts=` as `arcint-imggen`.

**TTS placement (operator's call).** `--tts-gen-device` selects where the
audio-generation `mmproj`'s own clip graphs run: `gpu` (default, on the card
with the backbone) or `cpu` (the hybrid: the speaker encoder, the code
predictor and the code2wav decoder on the CPU, the backbone on the card). The
flag needs `--tts`, and the card-only TTS arm is 0.751 real-time factor against
the hybrid's measured **0.828** (`measured-here`, 2026-10-19, C4 PASS on both;
`docs/campaigns/qwen3tts-card-real-time.md`): the CPU codec flushes in 1,692 ms
against the card's 1,780-2,000, and the card-only predictor is 4.5x the single
CPU backend. **Accepted by the operator 2026-10-19:** `arcint-tts.service`
carries `--tts-gen-device cpu`. An `mmproj` context takes one backend for all its
halves, so the flag selects the whole `mmproj`; the TTS `mmproj` carries no
vision half. The boot refuses with a placement error if a `cpu` request builds
any clip context on the card (`backend_llama.cpp` captures libmtmd's
`clip_init` backend line; `tools/tts_placement_check.sh` is the red-first gate,
`ARCINT_TTS_GEN_FORCE_GPU=1` its red switch).

## The routes

- `POST /v1/chat/completions` — the ordinary OpenAI chat body. With an
  `image_url` content part (`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`)
  the answer comes from the VLM. The URL forms served today are a `data:` URI
  and a **local file path**; a remote `http(s)` URL is refused 400 for now
  (open). Without `--mmproj` the request is refused 400, the text-only rule
  unchanged. One self-contained lane path: the request clears its lane's KV
  and uses no prefix cache, checkpoint or drafter (an image is not tokenisable
  text); the language requests on the same unit are untouched.
- `POST /v1/audio/speech` — **built.** OpenAI shape: `{model, input, voice,
  response_format:"wav", speed?, temperature?, top_p?, top_k?, seed?}` → the WAV
  bytes. `voice` is a **name from the server's profile list**
  (`GET /v1/audio/voices`); the route refuses `reference_audio`, `speaker_file`,
  `sample_audio`, `audio`, `reference`, `voice_sample`, `speaker_ref` or
  `speaker_reference` with 400 and a pointer to the enrolment route (D5), as it
  refuses an unknown `voice` and a non-WAV `response_format`. The reference
  recording is read server-side from the profile only.
- `POST /v1/audio/transcriptions` — **built for STT** (whisper.cpp's own server served with `--inference-path /v1/audio/transcriptions`, so no route patch is needed: `file`, `language` and `response_format` match the OpenAI shape and a plain `json` returns `{"text":...}`). OpenAI multipart: `file`, `model`, `language`, `response_format` (`json`/`text`) → `{"text":...}`. German first (D3).
- `GET /v1/audio/voices` — **built.** The enrolled names only; never a
  reference, never a path, never audio bytes (D5).
- `GET /health` and `GET /props` — existing; the service phase adds the
  mmproj/backbone/card, the enrolled-voice **count** (names not required on
  `/props` by D5) and the whisper model's own fields.

## The enrolment (D5)

Three rules, and they are absolute:
1. **Only enrolled voices.** A profile is a server-side entry; the API's
   `voice` field picks a name. Arbitrary reference audio is refused.
2. **Profiles stay outside the repository.** `<profiles>/<name>/` holds
   `reference.wav`, `reference.txt` and `profile.json`; the directory is
   operator-local (`$ARCINT_TTS_PROFILES`, a `*.local.md` path). No recording,
   profile or transcript of a real voice in a commit, a log, a doc or a
   `/props` body.
3. **Enrolment is a documented command.**
   `python3 tools/tts_enrol.py --name NAME --audio ref.wav --text ref.txt
   --profiles <dir> [--language de --source ... --licence ...]` validates a
   WAV (10-30 s envelope, a shorter one warns), copies it into the profile
   directory and writes the transcript and the metadata. `--force` replaces an
   existing profile. The service scans the profile directory at its **start**;
   there is no reload route yet, so a new enrolment is heard from the next
   start of the `arcint-tts` unit (restart it). The flow was proven end to end
   on the card host with two FLEURS stand-ins, `markus-test` and `frau-test`
   (CC-BY-4.0, outside the repository): the server served them as names and
   each synthesized sentence round-tripped at WER 0.0000 (`measured-here`,
   milestone §9.4).
4. **Markus and his wife** record their own 10-30 s samples later; the
   profile names and the recordings are entered with this tool. Until then the
   **gate voice** is one freely licensed German sample, staged on the card
   host and enrolled under a neutral name — the gate runs the *full clone
   path* on it, so the mechanism is proven before the real samples arrive. The
   gate voice's provenance and licence are in the leg's record, never the
   audio bytes in the repo.

## The fixed German STT set

The C3 set is 12 utterances of the FLEURS `de_de` dev split (CC-BY-4.0,
attribution to Google FLEURS), 2.5–13.75 s of the split, normalised from its
float32 WAV to 16 kHz mono PCM s16; the reference text comes from the split's
TSV. Audio and manifest stay on the card host outside the repo (the manifest
lists absolute paths, so it is operator-local); the corpus WER and the
manifest sha256 are the record. The TTS gate voice is a separate freely
licensed German sample, enrolled under a neutral name; its provenance and
licence are in the leg's record, never the audio bytes in the repo.

## The fixed German TTS set (C4) and the floors

The C4 set is committed (`tools/tts_sentences_de.json`): 46 unique German
sentences of 5-10 words, 303 words, **Tatoeba** (CC-BY 2.0 FR; attribution to
Tatoeba and its contributors), drawn deterministically from the
`deu_sentences.tsv.bz2` export (sha256 `ccd06372…`) with the selection rule in
the manifest (no common names, no digits, no pre-1996 orthography). Text is
public and licence-clear, so it is in the repo; the synthesized WAVs and the
round-trip transcripts are operator-local, in the leg's record.

The operator's C4 floor (D8, 2026-10-19): corpus round-trip WER <= 0.10 and no
single sentence above 0.5, on this set; `tools/tts_gate.py` prints the corpus
WER, the worst sentence and the share of sentences with WER <= 0.10. B3's floor
is RTF >= 1.0 on the set (batch reading); time to first audio is recorded, not
gated. The C3 floor is corpus WER <= 0.08 on the FLEURS set plus the card's
WER <= the CPU's + 0.005. C5 uses the Seed-TTS-eval WavLM-large SV speaker
model, calibrated against same-speaker vs cross-speaker recordings
(`docs/milestone-1.2.0.md` §6/D8).

## After a card window

Before a B60 window: release through the front (`POST /release`, check
`/status`), stop the owning language unit, run the leg, restore the units by
name. Sweep **stale leg drivers by pid** before restoring anything: a finished
session's deleted-but-running leg driver on the card host spawned a server
onto the card and a production unit restart on top of it produced an xe
page-fault oops and a hung host (`docs/milestone-1.2.0.md §9.1`,
`measured-here`, 2026-10-18). The exact commands are operator-local.

## Where it lives

- The binary and its `libmtmd` build: `CMakeLists.txt`
  (`LLAMA_BUILD_MTMD`), `src/exec/backend_llama.cpp`'s multimodal branch.
- The chat parser accepting `image_url`: `src/core/chat.cpp`, `src/api/
  handlers.cpp`.
- The whisper recipe: `contrib/whisper/`.
- The gates: `tools/vlm_gate.py`, `tools/stt_gate.py`, `tools/tts_gate.py`,
  `tools/tts_placement_check.sh` (the hybrid placement: red `ARCINT_TTS_GEN_FORCE_GPU=1`,
  green `--tts-gen-device cpu`);
  the enrolment: `tools/tts_enrol.py`.
- The models, licences and staging: `docs/milestone-1.2.0.md` §2.

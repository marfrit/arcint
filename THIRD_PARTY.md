# Third-party components

Vendored, single-header, licenses unchanged:

| component | version | license | file |
|---|---|---|---|
| cpp-httplib | 0.18.3 | MIT | `third_party/httplib.h` |
| nlohmann/json | 3.11.3 | MIT | `third_party/nlohmann/json.hpp` |
| minja (chat templating) | vendored | MIT | via `minja/chat-template.hpp` |

Adapted (not verbatim) in `tools/tts_voice_sim.py`, a measuring tool no
package ships: the speaker-similarity recipe from
`huggingface/open-tts-leaderboard`'s `transformers/score_similarity.py`
(Apache-2.0), and the ECAPA-TDNN head, **modified**, from:

- work: UniSpeech, `downstreams/speaker_verification/models/ecapa_tdnn.py`
- author / licensor: Microsoft (`microsoft/UniSpeech`); the file credits
  https://github.com/lawlict/ECAPA-TDNN for part of its code (kept verbatim
  in the header)
- source: https://github.com/microsoft/UniSpeech/blob/40ac183e772abdcbf3a318eccffaa70c0cfd3a74/downstreams/speaker_verification/models/ecapa_tdnn.py
  (the file's last commit, 2021-12-20)
- licence: CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/ (the
  repository's root LICENSE; no further copyright line, no NOTICE file)
- modified: transcribed without the torchaudio front end and the
  training-time loss, wired to an HF `WavLMModel`

The leaderboard script labels the head "MIT"; the UniSpeech LICENSE file does
not. `tools/tts_voice_sim.py` is therefore under CC BY-SA 3.0, share-alike,
and NOT under the repository's Apache-2.0; the rest of the repository stays
Apache-2.0. No weights are committed: the SIM model is staged operator-locally
(CC BY-SA 3.0), not redistributed.

Committed data: `tools/tts_sentences_de.json` holds 46 German sentences from
**Tatoeba** (tatoeba.org), CC-BY 2.0 FR, attribution to Tatoeba and its
contributors; the source export, its sha256 and the selection rule are in the
file's manifest.

Packaged separately: `arcint-stt` ships whisper.cpp (MIT, the ggml authors)
at the pin `contrib/whisper/build-whisper.sh` names, with
`contrib/whisper/server-props.patch` applied, built on the same patched ggml;
`arcint-imggen` ships stable-diffusion.cpp (MIT) per `contrib/sdcpp/`.

Compiled in from a pinned source tree (not vendored): llama.cpp and ggml
(MIT) at the commit `contrib/packaging/arcint/build-deb.sh` names, with
`contrib/llama.cpp/patches` applied. Patch 0056's CPU rows follow Strata's
AVX2 kernels (MIT, github.com/Niko1221/Strata); the expert profile the
Flash-Next unit loads is Strata's `data/expert-profile.bin` (MIT).

Vendored from that same llama.cpp pin (MIT, the ggml authors):
`third_party/llama-peg/` — `common/`'s PEG parser and JSON-Schema-to-GBNF
closure (`peg-parser`, `json`, `json-schema`, `json-schema-to-grammar`,
`trie`, `unicode`), verbatim except that each file includes
`llama-peg-shim.h` instead of `common.h`/`log.h`/`ggml.h`/`llama.h`; the shim carries the three
string helpers copied from `common/common.cpp`. `src/core/toolgrammar.cpp`
copies `permute` from `common/chat-peg-parser.cpp` and mirrors the tool-call
block of `common/parsers/qwen3-coder.cpp`.

Runtime dependencies (not shipped): the OpenCL ICD loader and Intel's
compute runtime. Model files are not shipped; GGUFs of Qwen checkpoints
inherit the Qwen license, and each quantization its publisher's terms.

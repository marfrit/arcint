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
(Apache-2.0), and the ECAPA-TDNN head transcribed from `microsoft/UniSpeech`
`downstreams/speaker_verification/models/ecapa_tdnn.py` (Copyright (c)
Microsoft Corporation, **CC-BY-SA-3.0** by that repository's root LICENSE,
unchanged since 2021; upstream credits `lawlict/ECAPA-TDNN` for part of it).
The leaderboard script labels the head "MIT"; the UniSpeech LICENSE file
does not. `tools/tts_voice_sim.py` is therefore licensed CC-BY-SA-3.0
(https://creativecommons.org/licenses/by-sa/3.0/), with its changes named in
its header; the rest of the repository stays Apache-2.0. The SIM model
itself is not shipped (CC-BY-SA-3.0, operator-local).

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

Runtime dependencies (not shipped): the OpenCL ICD loader and Intel's
compute runtime. Model files are not shipped; GGUFs of Qwen checkpoints
inherit the Qwen license, and each quantization its publisher's terms.

# Third-party components

Vendored, single-header, licenses unchanged:

| component | version | license | file |
|---|---|---|---|
| cpp-httplib | 0.18.3 | MIT | `third_party/httplib.h` |
| nlohmann/json | 3.11.3 | MIT | `third_party/nlohmann/json.hpp` |
| minja (chat templating) | vendored | MIT | via `minja/chat-template.hpp` |

Adapted (not verbatim) in `tools/tts_voice_sim.py`: the speaker-similarity
recipe from `huggingface/open-tts-leaderboard`'s
`transformers/score_similarity.py` (Apache-2.0), whose ECAPA-TDNN head is
transcribed from `microsoft/UniSpeech` `downstreams/speaker_verification` (MIT).
The SIM model itself is not shipped (CC-BY-SA-3.0, operator-local).

Compiled in from a pinned source tree (not vendored): llama.cpp and ggml
(MIT) at the commit `contrib/packaging/arcint/build-deb.sh` names, with
`contrib/llama.cpp/patches` applied. Patch 0056's CPU rows follow Strata's
AVX2 kernels (MIT, github.com/Niko1221/Strata); the expert profile the
Flash-Next unit loads is Strata's `data/expert-profile.bin` (MIT).

Runtime dependencies (not shipped): the OpenCL ICD loader and Intel's
compute runtime. Model files are not shipped; GGUFs of Qwen checkpoints
inherit the Qwen license, and each quantization its publisher's terms.

# Third-party components

Vendored, single-header, licenses unchanged:

| component | version | license | file |
|---|---|---|---|
| cpp-httplib | 0.18.3 | MIT | `third_party/httplib.h` |
| nlohmann/json | 3.11.3 | MIT | `third_party/nlohmann/json.hpp` |
| minja (chat templating) | vendored | MIT | via `minja/chat-template.hpp` |

`third_party/httplib.h` carries one small local change beyond verbatim:
`Request` gets a `socket` field, filled in `process_request` from the server
stream, so a handler can ask `detail::is_socket_alive` whether the peer is
still connected without writing a byte (a non-streamed request's prefill
writes nothing for a long stretch). The build also defines
`CPPHTTPLIB_USE_POLL` (`CMakeLists.txt`): without poll, `process_request`
answers 500 to every connection above `FD_SETSIZE`, and `select()` returns -1
there, which that liveness probe would read as "gone".

Adapted (not verbatim) in `tools/tts_voice_sim.py`, a measuring tool no
package ships: the speaker-similarity recipe from
`huggingface/open-tts-leaderboard`'s `transformers/score_similarity.py`
(Apache-2.0). Its ECAPA-TDNN head, `tools/speaker_head.py` (Apache-2.0), is a
clean-room implementation from a behavioural specification and the paper
(Desplanques, Thienpondt, Demuynck, "ECAPA-TDNN", Interspeech 2020), written
without access to an existing implementation; it reproduces the checkpoint's
embeddings (max |diff| below 1e-6 against the earlier transcription,
`measured-here`). The speaker model's weights are not shipped: the
`bezzam/wavlm_large_finetune_seed_tts_eval` mirror of the UniSpeech
fine-tuned checkpoint, CC BY-SA 3.0, is staged operator-locally.

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

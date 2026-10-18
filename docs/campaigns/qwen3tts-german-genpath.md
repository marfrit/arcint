# qwen3tts-german-genpath — German TTS through the Qwen3-TTS generation path is unintelligible and never stops

**Open 2026-10-19.** The 1.2.0 TTS route is wired and loads; the speech it
produces is not intelligible German and every utterance runs to the frame cap
without sampling `codec_eos`. The defect is on the **model / language /
sampling side**, not arcint's loop: the upstream reference tool reproduces it.

## Charter

`docs/milestone-1.2.0.md` §9.3: `/v1/audio/speech` synthesizes from the
Qwen3-TTS `qwen3tts` gen path (`libmtmd` `gen_audio`). The card gate FAILed on
both arms (greedy corpus WER 2.0909, the reference-defaults stochastic arm
2.1667, at floor 0.20, every sentence at the 512-frame cap, `codec_eos` never
sampled). The upstream `tools/tts/tts.cpp` tool, built from the same pin + the
same patch series, run on the **CPU** (`-ngl 0`) on the same GGUF pair,
reproduces it exactly: English "Hello world, this is a test." comes back as
` this is a test` (intelligible), German comes back as an unrelated phrase
(`Vielen Dank.`) **with and without** the enrolled speaker reference. So the
loop in `backend_llama.cpp` is faithful to `tts.cpp`, and the campaign is the
gen path's language / model side.

Three candidates, one variable per experiment, in order:

1. **Quantization.** `docs/milestone-1.2.0.md` §2 names the bf16 pair as the
   accuracy option; the arms measured Q8_0. The 1.7B talker is small, and the
   Q8_0 codec/talker heads may not survive 8-bit rounding for the German codec
   distribution. First lever: the **bf16 (or f16) pair** on the CPU with the
   upstream tool, the German sentence.
2. **The `codec_language` path.** Compare how upstream maps `--tts-lang de`
   to the vocab's special token against Qwen's own reference (the talker
   `config.json` `codec_language_id`, the `<|codec_language_german|>` token).
   An off-by-one or a missing German entry would give exactly
   English-OK/German-runaway. **Closed 2026-10-19** (below): the GGUF places
   the talker codec rows at `151936 + config_id` by construction
   (`text_vocab_size` 151936 ++ `vocab_size` 3072), so the mapping is
   consistent and not the defect.
3. **Sampling.** Qwen ships `generation_config.json` do_sample true,
   temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05, the
   subtalker the same values, and the talker suppresses every codec-special
   token in `[vocab_size-1024, vocab_size)` except `codec_eos`. llama.cpp's
   `common_params` defaults (temp 0.8, top_k 40, top_p 0.95, repetition 1.05,
   no suppress list) are not the same. Its own experiment.

## Reference to follow

- **Upstream llama.cpp** `tools/tts/tts.cpp` and
  `tools/mtmd/mtmd-helper-gen.cpp` at the pin (`3e4dfa8`, `~/src/llama-port`):
  the loop and the Qwen3TTS pipeline (`set_input`, `step_prompt`, `step_gen`,
  `mtmd_gen_inp_default`). Evidence: `code`.
- **Qwen's own reference**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
  `generation_config.json` (sampling), `config.json` (talker
  `codec_language_id`, `codec_*` ids, `spk_id`), `qwen_tts/core/models/
  modeling_qwen3_tts.py` (`generate`: the codec-prefill/speaker/text prompt
  and the trailing-text streaming loop). Evidence: `code` / `paper` (the
  model card). Public, ungated, no token.
- Where paper and code disagree, the code wins (`CLAUDE.md`).

## Gate

Red-first. `tools/tts_gate.py`'s answer-level bar: the **round-trip corpus
WER** of synthesized German through the STT model against the written
sentences, under the operator's absolute floor (D8; proposal 0.20). Sampling
printed with every verdict, identical inputs, one fixed sentence (or the fixed
set) across arms.

- **Red case (measured, on the record):** the Q8_0 upstream-CPU arm and the
  card arms fail German (unintelligible, 512-frame cap, `codec_eos` never
  sampled). A candidate that does not move the German WER leaves the campaign
  open.
- The same arm on **English** is green, the control that localizes the defect
  to the German/language side rather than to the tool.
- A candidate is adopted only if it speaks German before the frame cap; a
  quant/patch that merely reduces the WER without reaching a stop does not
  close it.

## Current state (2026-10-19, before the experiments)

`measured-here` unless marked:

| disposition | evidence |
|---|---|
| upstream tool reproduces the German failure on CPU, Q8_0, with and without the speaker; English passes | `measured-here` (c334b5f, milestone §9.3) |
| every German sentence at the 512-frame cap; `codec_eos` never sampled; the reference tool's `-n` cap likewise | `measured-here` |
| **language mapping is consistent, not off-by-one**: the GGUF places `<|codec_language_german|>` at vocab id **153989**; Qwen's talker `codec_language_id["german"] = 2053`; **153989 − 2053 = 151936** — and every other language and `codec_*` id differs from its config id by exactly 151936, so the converter placed the talker codec rows at columns `151936 + config_id`, consistently. No missing German entry; upstream's `find_special_token` resolves the name to the right column | `code` (GGUF `tokenizer.ggml.tokens` read with `gguf-py`; HF `config.json` read) |
| the talker's **text tokens are unreachable**: `output.weight` is `[2048, 3072]` while `token_embd.weight` is `[2048, 155008]`; the `qwen3tts` graph (`src/models/qwen3vl.cpp`, shared) pads the output with `-inf` over the `n_vocab_in − n_vocab_out = 151936` text columns before concatenating the real 3072. So a semantic-code sample lands in the codec region and `sampled − codec_0` (= 151936) yields the config code id. Not the defect | `code` (`src/models/qwen3vl.cpp:173-189`, upstream at the pin) |
| the Qwen3TTS pipeline's own default sampling is **Qwen's**: `mtmd_gen_inp_default` sets top_k 50, top_p 1.0, temp 0.9 for the code predictor. But `tools/tts/tts.cpp` **overrides** top_k/top_p from `common_params` (40 / 0.95) and samples the semantic code with the `common_sampler` (temp 0.8); no repetition penalty is passed anywhere | `code` (`tools/mtmd/mtmd.cpp:2051-2068`, `tools/tts/tts.cpp`) |
| Qwen's own defaults: do_sample true, temp 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05, subtalker the same; the talker **suppresses** `[2048, 3072)` except `codec_eos` | `code` (`generation_config.json`; `modeling_qwen3_tts.py:2036-2062`) |
| Q8_0 pair and a Q8_0 `mmproj` measured on the card; a **bf16 pair** exists at the GGUF mirror (backbone 3,472,593,760 bytes, mmproj 669,081,472 bytes, public) | `measured-here` (HF sizes) |
| the Qwen `text_vocab_size` is **151936** and `vocab_size` **3072**: `token_embd.weight` is `[2048, 155008] = 151936 + 3072`, i.e. `text_embedding` ++ `codec_embedding`; the codec column of config id `c` is `151936 + c`. The constant-offset identity is therefore the converter's construction | `code` (HF `config.json`; GGUF tensor shapes) |

## Result of the 2026-10-19 CPU arms (upstream `llama-tts`, `-ngl 0`, German "Der Zug fährt nach Berlin.", no speaker, one sentence)

`measured-here`, single card-free CPU window on the card host, one variable
per arm. WER is the round-trip against the written sentence (5 words),
transcribed with the same whisper model as the STT gate.

| arm | backbone / mmproj | sampling | frames / audio | reported stop | hypothesis | WER | verdict |
|---|---|---|---|---|---|---|---|
| `q8-de-default` (red baseline) | Q8_0 | llama.cpp defaults (temp 0.8, top_k 40, top_p 0.95, repetition 1.05) | **512 cap** / 40.96 s, 150.3 s gen + 1.8 s vocoder | no `codec_eos` | "Ja." | 1.00 | **FAIL** |
| `bf16-de-default` | **bf16** | same as the red arm | **44** / 3.52 s, 15.1 s gen | **`codec_eos` sampled** | "Der Zug fährt nach Berlin." exact | **0.00** | **PASS** |
| `q8-de-qwen` (sampling lever alone) | Q8_0 | Qwen as shipped (temp 0.9, top_k 50, top_p 1.0, repetition 1.05, seed 42) | **512 cap** / 40.96 s | no `codec_eos` | "Ja, das ist so, wie gesagt." | 1.20 | **FAIL** |

**Disposition so far:** the defect is **quantization of the talker** — the
bf16 pair speaks the sentence exactly and reaches `codec_eos` at frame 44,
where the Q8_0 arm runs to the cap under both the llama.cpp defaults and
Qwen's own sampling values. Sampling is **not** the cause (the Qwen-sampling
Q8_0 arm still fails), and the language path is consistent (candidate 2
closed). The 1.2.0 model set's bf16 pair is the fix; the substitute for the
Q8_0 model (`docs/milestone-1.2.0.md` §2). Open: the **card arm** at bf16
through arcint's route (the milestone C4/B3 gate), and whether the mmproj
(the code2wav/speaker half) also needs bf16 rather than its Q8_0 form — the
arm above pairs bf16 backbone with bf16 mmproj, so that is a separate
variable still to test on the card. The bf16 pair is 4.14 GB total against
Q8_0's 2.30 GB (arithmetic from the HF sizes), inside one card's idle VRAM.

## Where it lives

- The arcint loop: `src/exec/backend_llama.cpp` (`generate_speech`).
- The upstream reference: `~/src/llama-port/tools/{tts,mtmd}` at the pin.
- The Qwen reference: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (config,
  generation_config, modeling code) and the `ggml-org` GGUF mirror.
- The gate: `tools/tts_gate.py`; the enrolment: `tools/tts_enrol.py`.
- The models, staging and licences: `docs/milestone-1.2.0.md` §2.
- Operator-local paths (card host, sentence set, gate voice): `*.local.md`.

# qwen3tts-german-genpath — German through the Qwen3-TTS gen path was unintelligible and never sampled `codec_eos`; the Q8_0 mmproj's code predictor owned it

**Closed 2026-10-19.** The defect was the **Q8_0 `mmproj`'s code predictor
`a.gen.code`**; the fix is the official **bf16 mmproj** with a Q8_0 backbone
(2.52 GB). The card arm at the fixed model through arcint's own
`/v1/audio/speech` is green (C4 PASS; the real-time lever B3 is the separate
open `qwen3tts-card-real-time`). Full history in git before this stub.

## What was measured (CPU, upstream `llama-tts`, `-ngl 0`, German "Der Zug fährt nach Berlin.", one sentence, whisper-large-v3-turbo round trip, `measured-here` 2026-10-19)

| backbone / mmproj | sampling | frames | WER |
|---|---|---|---|
| Q8_0 / Q8_0 (red) | llama.cpp defaults | 512 cap | 1.00 FAIL |
| bf16 / bf16 | defaults | 44, eos | 0.00 PASS |
| Q8_0 / Q8_0 | Qwen as shipped | 512 cap | 1.20 FAIL |
| bf16 / bf16 | Qwen as shipped | 36, eos | 0.00 PASS |
| **Q8_0 / bf16** | defaults | 37, eos | 0.00 PASS |
| bf16 / Q8_0 | defaults | 512 cap | 1.00 FAIL |
| bf16 / pure-Q8 | defaults | 512 cap | 1.00 FAIL |
| bf16 / Q8_0 with `a.gen.code` BF16 | defaults | 50, eos | 0.00 PASS |
| bf16 / Q8_0 with `a.gen.wav` BF16 | defaults | 512 cap | 1.00 FAIL |

Dispositions (`measured-here`/`code`): the **backbone quantization is
irrelevant**; **sampling is not the cause** (Qwen's own defaults on Q8_0 still
cap); the **`codec_language` mapping is consistent** (GGUF codec column
`151936 + config_id`; `<|codec_language_german|>` 153989 = 151936+2053);
keeping only `a.gen.code` at BF16 fixes it and `a.gen.wav` at BF16 does not,
so the **code predictor** (`embd`/`head`/`proj_in`/`out_embd` + 35 `blk`) owns
it. `Q8_0` backbone 1.85 GB + bf16 mmproj 0.67 GB = 2.52 GB.

## The card arm (measured-here, 2026-10-19, B60 PCI 8086:e211, dev binary sha256 0179adb5…, Q8_0 backbone ac7931ae… + bf16 mmproj b9503e95…)

The full clone path on `gate-fleurs` through `/v1/audio/speech`: eight German
sentences, seed 42, sampling t 0.8 / top_p 0.95 / top_k 40 / repetition 1.0,
`codec_eos` sampled, no cap (46–69 frames): **round-trip corpus WER 0.0455**
(3 errors / 66 ref words, floor 0.20) — **C4 PASS**; B3 real time FAIL
(0.258×, `qwen3tts-card-real-time`).

## What exists

- The arcint loop `src/exec/backend_llama.cpp` (`generate_speech`) and the
  routes; the unit's production model pair is now the requantized Q8_0 mmproj
  with `gen.code.head`/`gen.code.embd` kept at f16 (`mmproj-...-Q8_0-
  codeheadf16.gguf`, campaign `qwen3tts-card-real-time` lever (b), C4 + B3
  PASS); the official **bf16 mmproj** stays as fallback
  (`contrib/systemd/arcint-tts.service`).
- The mixed Q8_0 file with `a.gen.code` at BF16 (0.56 GB, second option):
  `llama-quantize --pure --allow-requantize --tensor-type "gen\.code=bf16"`.
- The gate `tools/tts_gate.py`; upstream llama.cpp's `tools/{tts,mtmd}` at the pin `3e4dfa8`;
  Qwen `Qwen/Qwen3-TTS-12Hz-1.7B-Base` config/generation_config/modeling code.
- The model set and staging: `docs/milestone-1.2.0.md` §2/§9.3;
  `docs/speech-service.md`.

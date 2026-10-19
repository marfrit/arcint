# qwen3tts-voice-similarity — C5: the clone's speaker embedding against the reference, calibrated

**Open 2026-10-19.** C5's floor is calibrated, not absolute. The named model,
its safetensors conversion in a torch container, the `--embed-cmd` hook and a
CC cross stand-in are **measured**; a two-clip same-speaker FLEURS pair proved
not obtainable, and the real same-person recordings remain the operator's final
step.

## Charter

`docs/milestone-1.2.0.md` §6 C5: the clone's speaker-embedding cosine against
the reference, the gate voice and (later) the enrolled ones, above a floor the
operator sets. The operator's floor (D8, 2026-10-19, operator-set): use the
**Seed-TTS-eval** speaker model (fine-tuned WavLM-large SV), named on the
record; the clone-vs-reference cosine must come near the cosine between two
real recordings of the same person, and clearly above the cross-speaker cosine
(one household speaker's clone against the other's reference and the reverse). Until
the real recordings exist, prove the method with CC stand-ins: two clips of one
FLEURS speaker against another speaker.

## Reference to follow

seed-tts-eval (`github.com/BytedanceSpeech/seed-tts-eval`, `code`): SIM is the
cosine of speaker embeddings from WavLM-large fine-tuned on speaker
verification — the UniSpeech `downstreams/speaker_verification` table's
"WavLM large, *Fix pre-train: No*" row, its best SV model (Vox1-O EER 0.431,
Vox1-E 0.538, Vox1-H 1.154). `code`: the checkpoint `wavlm_large_finetune.pth`
(1,301,926,579 bytes, sha256 `51f07e3b94d9…`) is a `torch.save` dict with two
top-level keys — `model` (711 tensors: `feature_extract.model.*`, the
WavLM-Large backbone in original-WavLM naming, the ECAPA-TDNN head
`feature_weight`/`layer1`–`layer4`/`conv`/`pooling`/`bn`/`linear`, and the
training-only AM-softmax `loss_calculator.projection.weight`) and an unset
`best_valid_eer` sentinel (`100.0`). Upstream runs it via `s3prl` +
`torch.hub`; the Drive URL needs a confirm-token dance and a vendored Azure URL
has an expired SAS token, so the **verbatim Hub mirror**
`bezzam/wavlm_large_finetune_seed_tts_eval` (license tag `cc-by-sa-3.0`)
re-hosts the unmodified weights (README, `code` for the provenance). The open
TTS leaderboard rebuilds the backbone as an HF `WavLMModel` and remaps the
checkpoint's keys onto it, reported **bit-exact** (0.0 max difference, all 25
hidden states and the final 256-d embedding) against UniSpeech's original
`WavLM.py` (README, `code`). Scale note (`code`): its cosines are **not**
interchangeable with `microsoft/wavlm-base-plus-sv`; on the UniSpeech VoxCeleb1
clips this model gives same-speaker **0.60–0.69** and different-speaker
**-0.17–0.18**, where base-plus-sv gives 0.89–0.96 / 0.60–0.84 — those two
ranges are the calibration anchors.

## Gate

`measured-here`: on the CC stand-ins, the same-speaker cosine near the
reference's same-speaker anchors and clearly above the cross-speaker cosine;
the clone-vs-reference cosine is reported with every C4 verdict through
`tools/tts_gate.py --embed-cmd` (the embedding computation stays outside the
gate as a one-number command). An absolute number only once the two household
speakers' real recordings exist.

## Current state (measured-here, 2026-10-19)

**The method runs.** The named model is converted to safetensors in a torch
**container** (an LXC) and embedded there: the mirror's `wavlm_large_finetune.pth`
(`bezzam/wavlm_large_finetune_seed_tts_eval`, CC-BY-SA-3.0) is fetched, its
sha256 verified (`51f07e3b…`), and saved as
`wavlm_large_finetune_seed_tts_eval.safetensors` (1,301,746,084 bytes); the
backbone is an HF `WavLMModel` with the checkpoint's original-WavLM names
remapped onto it, the ECAPA-TDNN head transcribed from UniSpeech, one clip per
forward pass, cosine — the `huggingface/open-tts-leaderboard`
`transformers/score_similarity.py` recipe (Apache-2.0), re-landed as
`tools/tts_voice_sim.py`.

| stand-in | cosine | evidence |
|---|---|---|
| same audio vs itself (sanity) | 1.000000 | `measured-here` |
| the gate-voice clone vs its reference (46 synthesized `gate-fleurs` sentences) | **mean 0.598038**, min 0.439814, max 0.751893 | `measured-here` |
| two FLEURS files of one sentence, two different speakers (cross, CC-BY-4.0) | **0.262419** | `measured-here` |
| published same-speaker VoxCeleb1 range (this model) | 0.60 – 0.69 | `paper`/`code`, mirror README |
| published cross-speaker VoxCeleb1 range (this model) | -0.17 – 0.18 | `paper`/`code`, mirror README |

The clone-vs-reference mean (0.598) sits at the published same-speaker anchors
and clearly above the measured CC cross pair (0.262): the method's calibration
holds. The cross pair is two files of the same FLEURS sentence, which by FLEURS'
design are two different speakers (each sentence is recorded by 2-3 speakers).
**A two-clip same-speaker FLEURS pair is not obtainable:** the local `de_de` dev
TSV has no speaker column (its 7 fields carried `id` == sentence, file name,
transcription, raw transcription, phonemes, samples, gender; 147 distinct `id`
== 147 distinct sentences over 363 rows), and the canonical `google/fleurs` on
the Hub lists no speaker field either; `measured-here` on both. Low-download CC
alternatives with speaker labels were surveyed and **not taken**: Common Voice
is gated (`common_voice_16_1` needs auth; `17_0` empty), VoxPopuli `de` is
ungated with `speaker_id` (license `cc0-1.0`/other) but its smallest split
parquet is **936 MB**, not low-download. The reference's published same/cross
anchors therefore stand as the same-side calibration; the real recordings
remain the final C5 (operator).

**The gate hook.** `tools/tts_gate.py --embed-cmd` runs one command that reads
the cases JSON on stdin and prints one cosine; the landed command is
`tools/tts_voice_sim.py stdin --ref <voice reference.wav> --model
<wavlm_large_finetune_seed_tts_eval.safetensors>`. The model file and the
container venv stay operator-local, outside the repository. Open: the real
same-person recordings (operator).

## Where it lives

- The gate and its one-number cosine hook: `tools/tts_gate.py --embed-cmd`.
- The C4 set and the enrolment envelope: `docs/milestone-1.2.0.md` §4/§6 D5,
  `docs/speech-service.md`.
- The embedding command: `tools/tts_voice_sim.py` (`pair`, `mean-vs-ref`,
  `stdin`); the model file, the container venv and the converted safetensors
  stay operator-local.
- The recipe source: `huggingface/open-tts-leaderboard`
  `transformers/score_similarity.py` (Apache-2.0); the head from
  `microsoft/UniSpeech`, which is **CC-BY-SA-3.0** by its root LICENSE
  (`code`, release review 2026-10-19; the leaderboard's "MIT" label is wrong),
  so `tools/tts_voice_sim.py` is CC-BY-SA-3.0 (`THIRD_PARTY.md`).
- The model: `bezzam/wavlm_large_finetune_seed_tts_eval` (mirror),
  `microsoft/UniSpeech` `downstreams/speaker_verification`; `code`.

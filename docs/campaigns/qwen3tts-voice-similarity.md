# qwen3tts-voice-similarity — C5: the clone's speaker embedding against the reference, calibrated

**Open 2026-10-19.** C5's floor is calibrated, not absolute; the method, the named model and its anchors are owed.

## Charter

`docs/milestone-1.2.0.md` §6 C5: the clone's speaker-embedding cosine against
the reference, the gate voice and (later) the enrolled ones, above a floor the
operator sets. The operator's floor (D8, 2026-10-19, operator-set): use the
**Seed-TTS-eval** speaker model (fine-tuned WavLM-large SV), named on the
record; the clone-vs-reference cosine must come near the cosine between two
real recordings of the same person, and clearly above the cross-speaker cosine
(the operator's own clone against his wife's reference and the reverse). Until
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
gate as a one-number command). An absolute number only once the operator and
his wife's real recordings exist.

## Current state

Not measured. The named model is `.pth`, not safetensors, and the mirror is
zero-download; per `docs/milestone-1.2.0.md` §2's rule and the operator's
step-3 condition ("safetensors and well known; if a low-download repo, run it
in a container"), it runs in a **container** (torch) on the host and the kept
local copy is converted to safetensors there. Stand-ins on the card host: the
enrolled `gate-fleurs` and `gate-de` (different FLEURS speakers), and the
`markus-test` / `frau-test` enrolment stand-ins. A same-speaker pair needs a
FLEURS speaker with two clips; the `de_de` dev TSV carries only a gender column
and no speaker id (`measured-here` on the TSV), so the reference's published
same/cross anchors stand in until a two-clip same-speaker source is found, or
the gate voice's own reference can be paired with a second recording. Open.

## Where it lives

- The gate and its one-number cosine hook: `tools/tts_gate.py --embed-cmd`.
- The C4 set and the enrolment envelope: `docs/milestone-1.2.0.md` §4/§6 D5,
  `docs/speech-service.md`.
- The model: `bezzam/wavlm_large_finetune_seed_tts_eval` (mirror),
  `microsoft/UniSpeech` `downstreams/speaker_verification`; `code`.

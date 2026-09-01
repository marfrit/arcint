# The 3.8 MTP head, verified against two independent sources (2026-09-01)

The reconstructed Qwen3.8-27B MTP head (`tools/export_mtp.py`) made three
semantic decisions by measurement alone, because no reference existed: the
zero-centred norms applied as (1+w), the sigmoid attention gate against a
config that says "swish", and the per-head q/gate interleave. Two public
artifacts now allow all three to be checked instead of trusted.

## Source 1: unsloth's GGUF carries the complete head

`unsloth/Qwen3.8-27B-GGUF` ships the MTP head as **block 64** (llama.cpp
`qwen35.nextn_predict_layers = 1`) — and its dynamic quantisation keeps the
head at Q6_K/Q8_0/F32 even inside the 3-bit files. The head tensors were
range-fetched from the Q8_0 variant by the offsets in the GGUF header
(~500 MiB instead of a 29 GB download), dequantised, and diffed against the
bf16 `mtp.*` tensors the reconstruction was built from:

| tensor class | result |
|---|---|
| all 8 weight matrices (q, k, v, o, gate/up/down, eh_proj) | cosine 0.9998–0.9999, maxdiff ≤ 0.0045 — Q8 rounding, identical layout, no permutation |
| all 7 norms (F32 in the GGUF) | **exactly `1 + raw`, maxdiff 0.0000** |

So the GPTQ side-channel the head was built from is faithful to the official
checkpoint, and the (1+w) norm convention is confirmed at the byte level:
llama.cpp's converter bakes the +1 (`conversion/qwen.py`,
`data_torch = data_torch + 1`) and runs plain RMS on the shifted weights.

## Source 2: llama.cpp's graph confirms the semantics

`src/models/qwen35.cpp` (`graph_mtp`):

- attention gate: `ggml_sigmoid`, applied post-attention, pre-o_proj —
  matches the measured sigmoid-vs-swish decision (66% vs 13%);
- q/gate: one projection viewed with stride 2×head_dim — the per-head
  interleave whose naive split cost 53 points;
- `ggml_concat(e_norm, h_norm)` — embedding first, hidden second, matching
  the exporter's fc input order;
- lm_head and final norm fall back to the base model's when the checkpoint
  carries none — same structure as `extract_lm_head`.

## The one divergence, measured to a tie

llama.cpp rotates rope half-split (rotate_half) where the exporter rotates
interleaved (even, odd) pairs — on byte-identical, unpermuted weights, so
the conventions are not mathematically equivalent. `--rope half` now exports
the alternative; A/B on the B60 (b7c1, 32768 ctx, greedy,
repetition_penalty 1.0, 400 tokens):

| rope | code accept | prose accept |
|---|---|---|
| interleaved (shipped) | 76.7% (174/227) | 77.4% (175/226) |
| half-split | 75.4% (172/228) | 80.6% (179/222) |

Opposite signs, single-prompt noise: the pairing does not decide acceptance
at this resolution, and the shipped head stays interleaved. If the question
ever matters, it needs the Prüfstand harness across seeds, not two prompts.

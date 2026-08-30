# Qwen3.8-27B MTP head for OpenVINO — reconstructed (draft model card)

**What this is.** The multi-token-prediction head of Qwen/Qwen3.8-27B as two
OpenVINO IR graphs, rebuilt from the checkpoint's own `mtp.*` tensors by
`tools/export_mtp.py` in the arcint repository:

| file | what | size |
|---|---|---|
| `openvino_mtp_layer.{xml,bin}` | the MTP transformer layer (f16) | 849 MB |
| `openvino_mtp_lm_head.{xml,bin}` | the LM head the draft is decoded with (f16) | 1.27 GB |

**Reconstructed, not official.** optimum-intel drops the MTP graph on export;
newer development builds export the layer as `openvino_mtp_model` but not the
lm_head. These two files are what arcint's speculative decoding path serves.

**Which IR it belongs to.** Any OpenVINO IR of Qwen3.8-27B with hidden 5120,
vocab 248320 and untied embeddings. Measured to pair with:

| body | draft acceptance (greedy, B60) |
|---|---|
| arcint's AWQ export (`qwen38-b7c1-ov`) | 93.2% |
| `OpenVINO/Qwen3.8-27B-int4-ov` (Intel's public int4) | 90.8% on the acceptance task (10/10), 96.3% code / 77.3% prose |

**Intel's own MTP layer works with this lm_head.** `OpenVINO/Qwen3.8-27B-int4-ov`
ships `openvino_mtp_model` (the layer, int4, no lm_head). Served through
arcint's `--mtp-layer exported` with the `openvino_mtp_lm_head` from this
card: 93.9% acceptance on code, 76.4% on prose, 37.7–38.1 t/s against 25.0
t/s without speculation on the B60. If you have Intel's IR, the lm_head is the
only file you are missing.

**How to use.** Place the four files beside `openvino_language_model.xml` and
serve with `arcint --mtp on`. Acceptance is printed on every decode line
(`draft accept 96.3% (157/163)`); a head that does not belong shows ~0%.

**Why acceptance is the oracle.** A drafted token is accepted only when it
equals what the sampler would have picked anyway, so a wrong head cannot
change the answer — it can only make speculation useless.

License follows the base model (Apache-2.0).


---

# Qwen3.6-35B-A3B MTP head for OpenVINO — reconstructed (draft model card)

The same reconstruction for the MoE model: `openvino_mtp_layer` (the MTP
layer with its 256-expert MLP, 1.69 GB f16) and `openvino_mtp_lm_head` (the
base IR's int8 lm_head, 509 MB). Built by `tools/export_mtp.py` from the
checkpoint's own `mtp.*` tensors; pairs with `OpenVINO/Qwen3.6-35B-A3B-int4-ov`.

| body | draft acceptance (greedy, B60) | decode, `--mtp on` vs `off` |
|---|---|---|
| Intel's public int4 IR | 93.9% code / 75.4% prose | **48–53 t/s vs 71.5 t/s** |

**Read the second column before using it.** The head is correct, and on an
Arc card today speculation with it is *slower* than plain decoding: the
OpenVINO GPU plugin runs a two-token forward of this mixture of experts on
its prefill path, at 1.8× the cost of a one-token step, and the head as
exported reads its full expert weights per draft. It is published as the
reference head — the weights and the graph nobody else has assembled for
OpenVINO — not as a speed-up. It becomes one when the plugin has a small-M
MoE decode path; the numbers above are the measurement to repeat then.

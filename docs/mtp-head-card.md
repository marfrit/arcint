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

**How to use.** Place the four files beside `openvino_language_model.xml` and
serve with `arcint --mtp on`. Acceptance is printed on every decode line
(`draft accept 96.3% (157/163)`); a head that does not belong shows ~0%.

**Why acceptance is the oracle.** A drafted token is accepted only when it
equals what the sampler would have picked anyway, so a wrong head cannot
change the answer — it can only make speculation useless.

License follows the base model (Apache-2.0).

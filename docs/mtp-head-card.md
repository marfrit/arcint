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

**Read the second column before using it.** The kernels are not the reason:
the plugin routes a two-token forward through its batched-GEMV decode path,
where it costs the device 1.15× a one-token step. Two other things cost. The
serving loop spends 36.5 ms of wall for 16.7 ms of device per accepted pair.
And the head itself is 6.7 ms of that device, three quarters of a full
64-layer body step, to draft a single token, because its 256-expert MLP is
read densely in f16. On device time alone the head is therefore a wash
today, 8.6 ms per token against 8.7 plain.

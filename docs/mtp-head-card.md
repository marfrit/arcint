# Qwen3.8-27B MTP head for OpenVINO — reconstructed (draft model card)

**What this is.** The multi-token-prediction head of Qwen/Qwen3.8-27B as two
OpenVINO IR graphs, rebuilt from the checkpoint's own `mtp.*` tensors by
`tools/export_mtp.py` in the arcint repository:

| file | what | size |
|---|---|---|
| `openvino_mtp_layer.{xml,bin}` | the MTP transformer layer (f16) | 849 MB |
| `openvino_mtp_lm_head.{xml,bin}` | the LM head the draft is decoded with (f16) | 1.27 GB |

**Reconstructed, not official.** optimum-intel does not export the MTP lm_head;
newer development builds export the layer as `openvino_mtp_model`. These two
files are what arcint's speculative decoding path serves. The reconstruction
matches unsloth's GGUF MTP block and llama.cpp's `graph_mtp` semantics
(`docs/mtp-head-verification.md`).

**Which IR it belongs to.** Any OpenVINO IR of Qwen3.8-27B with hidden 5120,
vocab 248320 and untied embeddings. Measured to pair with (`measured-here`):

| body | draft acceptance (greedy, B60) |
|---|---|
| arcint's AWQ export (`qwen38-b7c1-ov`) | 93.2% |
| `OpenVINO/Qwen3.8-27B-int4-ov` (Intel's public int4) | 90.8% on the acceptance task (10/10), 96.3% code / 77.3% prose |

**Intel's own MTP layer works with this lm_head.** `OpenVINO/Qwen3.8-27B-int4-ov`
ships `openvino_mtp_model` (the layer, int4, no lm_head). Served through
arcint's `--mtp-layer exported` with the `openvino_mtp_lm_head` from this
card: 93.9% acceptance on code, 76.4% on prose, 37.7–38.1 t/s against 25.0
t/s without speculation on the B60 (`measured-here`). If you have Intel's IR,
the lm_head is the only file you are missing.

**How to use.** Place the files beside `openvino_language_model.xml` and
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
| Intel's public int4 IR, stock OpenVINO | 93.9% code / 75.4% prose | 48–53 t/s vs 71.5 t/s |
| Intel's public int4 IR, int4 head, `marfrit-openvino` (patch 0003) | 84.0% code / 71.4% prose | 72.9 t/s vs ~62 t/s (code prompt) |

On a stock OpenVINO build a two-token verify forward rebuilds the MoE's
per-expert mask subbuffers on every inference (20,480 `create_subbuffer`
calls); patch 0003 of the `marfrit-openvino` series skips them below the
batched-GEMV threshold (verify forward 27.3 -> 18.1 ms, byte-identical output,
`measured-here`). The int4 head is NNCF INT4_ASYM, group 64, all layers, lm_head
int8 (layer 1.69 GB -> 455 MB). The exported head computes its 256-expert MLP
densely (the router's lowering does not match the plugin's MoE fusion pattern).
The 72.9 t/s row is a single prompt, not yet scored on the acceptance task.

License follows the base model (Apache-2.0).

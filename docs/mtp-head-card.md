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

**Read the second column before using it.** The head is correct, and on a
stock OpenVINO build speculation with it is *slower* than plain decoding on an
Arc card. The kernels are not the reason: a two-token forward takes the
plugin's batched-GEMV decode path and costs the device 1.15× a one-token step.
The cost has been found, and it is host-side churn. At `token_num > 1` the MoE
implementation's `prepare_internal_buffers` rebuilds its per-expert mask
subbuffers on **every** inference — 20,480 `create_subbuffer` calls per
two-token forward — for a per-expert prefill fallback that the batched-GEMV
path never reads. At one token the block is skipped, which is why plain
decoding never showed it. A plugin patch that skips the masks below the
batched-GEMV threshold takes the verify forward from 27.3 ms to 18.1 ms with
**byte-identical** output.

Separately, the head reads its 256-expert MLP densely in f16, 1.69 GB per
draft: the router *is* present in the exported graph, but its lowering does not
match the pattern the plugin's MoE fusion looks for, so every expert is
computed for every token and the unselected ones are weighted by zero.

With a patched plugin **and** an int4 head, the 35B's speculation does win —
72.9 t/s against ~62 plain on a code prompt, B60. That figure is single-prompt,
has not yet been through the acceptance harness, and needs a plugin you build
yourself; treat it as a direction, not a result. The second column is what the
published stack gives you today.

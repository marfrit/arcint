# The fit levers: the full-depth 35B all-resident on the 16 GiB card

Built (plugin patches 0054–0058, exporter `--dense-u8` in
`tools/q4e/dense_u8.py`): the full-depth packed Qwen3.6-35B-A3B
(`qwen3.6-35b-a3b-native-d40packed-u8`) loads and serves all-resident on the
A770 with `--offload-ratio 0 --moe-per-expert-dispatch --emb-device CPU
--dyn-quant off`, u8 KV: 13.11 GiB device-resident, 84,704 tokens of context
per lane, compile host peak 0.47 GB once the native decode chains fuse
(`measured-here`, A770). Dense Q6_K projections ride as u8 group-16 with the
shared expert and attention k/v kept f16 (depth-4 served A/B against f32
dense: KL 1.95e-4, argmax 983/1000, `measured-here`). Instruments:
`tools/bigalloc.c`, `tools/bigalloc_report.py`,
`tools/native_moe_match_probe.cpp`, `tools/native_moe_block_ab.cpp`. The
prefill rate on this artifact is in `docs/design-native-dpas-expert-kernel.md`
(952.1 t/s at 4096).

Owed: the plugin mechanism of the q/k/v horizontal-fusion defect under
compressed weights (all three compressed gives wrong logits); IQ2_S as down
and Q8_0 as gate/up under dispatch; a full-depth KLD of the u8 artifact; the
fit ledger's ≈0.4 GiB undercount at the edge.

Full history: `git show b0447b8:docs/design-fit-levers.md`.

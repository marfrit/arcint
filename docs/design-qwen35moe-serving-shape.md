# The `qwen3_5_moe` serving-shape emitter and the IQ2_S native expert format

Built: the serving-shape emitter (`tools/q4e/serving_shape.py`,
`tools/export_serving_artifact.py`) emits the `qwen3_5_moe` family (plain
pre-norm residual, 30 GDN + 10 attention layers, 256 experts top-8 plus one
shared expert, no hyper-connections, no n-gram table) from
`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, with the experts in the checkpoint's own
blocks: IQ2_S gate/up (plugin patch 0050; packed form `--native-packed`,
patches 0052–0055), IQ3_XXS/IQ4_XS down. Registry ids
`qwen3.6-35b-a3b-native-d4*` / `-d40*` (`src/core/model_registry.cpp`).
The all-resident native route (`--offload-ratio 0 --moe-per-expert-dispatch`)
exists through patch 0051 and arcint's `moe_offload_active` (`src/config.h`).
The current full-depth artifact and its fit are in `docs/design-fit-levers.md`;
its prefill rate (952.1 t/s at 4096, A770, `measured-here`) in
`docs/design-native-dpas-expert-kernel.md`; decode 28.1 t/s after 4096 tokens
with device-side routing (patch 0067, A770, `measured-here`, DESIGN §7.0.2cs).

Full history: `git show b0447b8:docs/design-qwen35moe-serving-shape.md`.

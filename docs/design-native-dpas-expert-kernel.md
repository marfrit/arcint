# The native per-expert kernels on the matrix unit

Built (plugin patch 0064, DESIGN §7.0.2co): on the A770 the IQ2_S-packed
gate/up of the per-expert route runs on the matrix unit (`dpas`, SIMD 8, tile
16 pairs, B operand exact in f16 with the super-block scale applied after each
16-deep chain), for every call size, with a K-split one-pair kernel for
decode; IQ3_XXS/IQ4_NL down stays on the scalar row kernel of patch 0061. The
full-depth packed u8 Qwen3.6-35B-A3B, all-resident, prefills at 952.1 t/s at
4096 tokens against 653.7 before, decode within run spread, Prüfstand 10/10
(`measured-here`, A770). Instruments: `tools/native_kernel_harness.py`,
`tools/cldump.c`, `tools/native_moe_block_ab.cpp`, `tools/logits_dump_diff.py`.
Owed: the B60 (Xe2) variant.

Full history: `git show b0447b8:docs/design-native-dpas-expert-kernel.md`.

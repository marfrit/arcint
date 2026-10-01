# GGUF opened in process on a template IR (built, 0.4.0–0.4.3)

`--gguf FILE --model DIR` serves a GGUF of an allowlisted architecture on the
served IR of the same architecture, which is the topology template: a pass
replaces each projection's decompression subgraph with the file's own tensor,
found by the HF module name the IR carries, after undoing the converter's
transforms (`A_log -> -exp`, the `+1` on norms, the V-head reorder of
`qwen35`'s 16 key / 48 value heads, the conv1d squeeze). The template's
tokenizer, chat template, embedding (unless `--gguf-embed file`), GDN state
tensors and MTP layer are served; its AWQ activation scales are set to one. A
file whose architecture, block count, geometry or vocabulary does not match is
refused by name (`src/core/gguf.*`, `src/core/gguf_dequant.*`,
`src/core/gguf_repack.*`).

Two forms reach the graph (`--gguf-mode repack|native|mixed`, default `mixed`:
Q4_K repacked, every other type native):

- **repack** (DESIGN §7.0.2ba): the K-quant rows in the plugin's own grouped
  compressed form, no zero point, f16 scale per group, so the runtime's
  compressed fully-connected path and its activation reservation apply. Q4_K
  and Q5_K carry their mins exactly as augmented columns under the
  super-block's dmin, with the activation widened by its group sums
  (`--gguf-mins exact|shared|nibble|split`). The deviation per weight is
  bounded and measured at load: under 1/64 of a quantisation step for Q4_K,
  1/32 for Q5_K and Q6_K, 1/16 for Q8_0 (`code` + `measured-here`); a tensor
  over its bound is refused. With f16 activations (`--dyn-quant off`, the
  default for a GGUF-opened model) the greedy output equals the native form's
  byte for byte.
- **native** (`--gguf-native`, plugin patches 0021 and 0028–0030): the GGUF
  bytes as a tagged u8 constant, decoded inside the fully-connected kernel;
  each quantised value is packed as the f16 bit pattern `0x6400 | q'` and the
  scale and offset are applied to the matrix-multiply sums.

The fit sees a GGUF-opened model as an artifact with different constants: its
weight term is the resident form's bytes, and the reservation, KV precision,
prefix cache, drafters and chunk belt are those of the IR path. `--gguf-check
once|always` keeps or re-runs the repack's deviation verdict between loads.

Dense Qwen3.8-27B Q4_K_M on the B60, u8 KV, one lane: warm prefill 1,001 t/s
at 856 tokens and 464 t/s at 71.7k, 10/10 on the Prüfstand (`measured-here`,
DESIGN §7.0.2bp). Owed: `--gguf` for MoE files and for the sub-4-bit types.

## GGUF block formats (reference)

Blocks run along the contraction dimension of a `[out, in]` weight, 256 values
per super-block (32 for Q8_0) (`code`, `ggml-common.h`). Q4_K: `d`, `dmin`
(f16), 12 bytes holding eight 6-bit scales and eight 6-bit mins, 128 bytes of
nibbles; value `d·sc·q − dmin·m` per 32-value sub-block (144 B). Q5_K adds 32
bytes of high bits (176 B). Q6_K: 128 bytes low nibbles, 64 bytes of 2-bit
highs, 16 signed 8-bit sub-scales, `d`; value `d·sc·(q − 32)` per 16 (210 B).
Q8_0: `d` and 32 int8 (34 B). The sub-4-bit set (IQ4_XS, IQ3_S, IQ3_XXS,
Q3_K) uses codebooks and sign tables, a different kind of decoder.
IQ4_NL is a codebook type at 4.5 bpw (18 B per 32: one f16 `d`, then 32
nibbles), decoded through a 16-entry non-uniform lookup, `value = d ·
kvalues[q]`, with levels `[−127, −104, −83, −65, −49, −35, −22, −10, 1, 13,
25, 38, 53, 69, 89, 113]` (spacings 11…24); the best affine fit over the 16
codes leaves 0.847 of a step, so no `Convert→Subtract(zp)→Multiply(scale)`
chain carries an IQ4_NL block exactly (`measured-here`, gated by
`tests/python/test_repack_route.py`).

| repacked block | weights | scale (f16) | zero point | group | the min |
|---|---|---|---|---|---|
| Q4_K | u4, the block's nibbles | d·sc | none | 32 | augmented columns |
| Q5_K | u8, the block's 5-bit values | d·sc | none | 32 | augmented columns |
| Q6_K | u8, the block's 6-bit values | d·sc, per 16 | 32, an integer | 16 | — |
| Q8_0 | i8, the block's bytes | the block's d | none | 32 | — |

Full history: `git show b0447b8:docs/design-gguf-native.md`.

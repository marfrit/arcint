# Design note: a `qwen3_5_moe` serving-shape emitter and the IQ2_S native expert format

Status: design + the IQ2_S decoder landed (2026-09-24). The emitter port and
the plugin format are OWED and scoped here. Written after the 2026-09-24
operator decision *"IQ3_XXS first with conversion and benchmark, then continue
the roadmap"* re-targeted to `Qwen3.6-35B-A3B` (see
`docs/campaigns/sub4bit-vram-kernel.md`, 2026-09-24 IQ3_XXS leg).

## 1. Why a new emitter is needed

The native serving-shape route (`tools/q4e/serving_shape.py` +
`tools/export_serving_artifact.py`) emits exactly ONE graph family today: the
Flash-Next `qwen4_exp` backbone. Its `build_serving_shape_ir` hardwires the
hyper-connection width (`hc = cfg.hc_count`, `:1267`) and the PLE
(`_ple_state`, `:1290`), and its layer loop is built on the hyper-connection
mixer (`_split_combine` / `_recombine`, `:1385`, `:1401`). The shard admission
keys on `qwen4exp.ple.eos_token_id` (`export_serving_artifact.py:78`), and
`piecewise_export.REAL_GEOMETRY` is 48 layers / 512 experts / `hc_count 4` /
`ple_layer_ids [2]`.

The requested checkpoint is `Qwen3.6-35B-A3B` (`unsloth/Qwen3.6-35B-A3B-GGUF`,
`Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`), `general.architecture = qwen35moe`:
40 layers, 256 experts, top-8, ONE shared expert, 262144 context, and **no
hyper-connection stage and no PLE** — those keys are absent from the shard's
54 metadata fields. So the emitter's layer for this family is a plain pre-norm
residual:

    hidden = hidden + attn_or_gdn(norm(hidden))     (30 GDN + 10 attention)
    hidden = hidden + moe_tiled(norm(hidden))        (router + 256 experts + shared)

## 2. The geometry (read from the shard and the served int4 export)

| field | value | source |
|---|---|---|
| `block_count` / `num_hidden_layers` | 40 | `code`/`measured-here`: GGUF, served config |
| `hidden_size` | 2048 | both |
| attention layers | `i % 4 == 3` (10 of 40) | `full_attention_interval 4` |
| attention | 16 heads, 2 kv heads, `head_dim 256`, rotary 64, theta 1e7 | GGUF |
| GDN | key dim 128 × 16 heads, value dim 128 × 32 heads, conv 4, state 128, dt rank 32 | GGUF `ssm.*`, served config |
| MoE | 256 experts, top-8, intermediate 512, shared 512 | GGUF/config |
| vocab | 248320 | config |

Expert formats in the fetched file (`measured-here`: gguf-py histogram
`Q6_K 252, F32 361, IQ3_XXS 37, IQ2_S 80, IQ4_XS 3`):
**`ffn_gate_exps` / `ffn_up_exps` = IQ2_S on all 40 layers**;
`ffn_down_exps` = IQ3_XXS on 37 layers, IQ4_XS on 3.

## 3. The IQ2_S native format

IQ2_S is ggml type 22, 82 B per 256 values (`code`: llama.cpp
`ggml-common.h`, `ggml-quants.c` `dequantize_row_iq2_s`, pinned clone
56b9eb28):

    f16 d | 32 B low 2-bit grid indices | 32 B sign masks (one per 8 values)
          | 8 B qh | 8 B 4-bit sub-block scales

    idx = qs[4*ib32 + l] | (((qh[ib32] >> (2*l)) & 3) << 8)   # 10-bit, l = 0..3
    db  = d * (0.5 + nibble) * 0.25       # low nibble l<2, high nibble l>=2
    y   = db * iq2s_grid[idx][j] * (sign_byte[l] & (1<<j) ? -1 : +1)

`iq2s_grid[1024]` is transcribed into `q4e.native_blocks` (the 1024-entry
table, 8 little-endian bytes per index). `iq2_s_split` returns one **u16**
grid index, one u8 sign byte and one f32 scale **per 8 values**, in the
IQ3_XXS convention (sign byte raw, unlike IQ3_XXS's 7-bit table index).
`iq2_s_decode` is the inverse.

**Exactness.** Measured 2026-09-24 against gguf-py's own `dequantize` on the
real shard, two experts, every row: `blk.0.ffn_gate_exps.weight` and
`blk.0.ffn_up_exps.weight`, **max|diff| 0.0, bit-exact**. Cells:
`tests/python/test_native_blocks.py::test_iq2_s_split_and_decode_on_a_hand_built_block`
(the byte layout, mutation-sensitive — moving the sign bytes to the
low-index region makes it fail) and
`::test_iq2_s_split_decodes_the_real_shard_exactly_as_gguf_py_does` (the
independent oracle).

## 4. What the plugin needs (OWED, the next patch)

Patch 0043's native formats are IQ4_NL=1 / IQ3_XXS=2 / Q8_0=3, all with a u8
weight slot. IQ2_S adds a 10-bit grid index and TWO 4-bit scales per 32
values, so it does not fit that shape as-is. The planned slot layout (to be
pinned by the patch and its red-first cells):

| slot | shape | contents |
|---|---|---|
| weight | u8 `[E, out, K/32, 8]` | bytes 0..3 = the four low index bytes; byte 4 = the qh byte (its bits 2*l,2*l+1 are l's high grid bits); bytes 5..7 zero |
| zero-point | u8 `[E, out, K/32, 4]` | the four raw sign bytes (same shape as IQ3_XXS) |
| scale | f16 `[E, out, K/32, 2]` | the two 4-bit sub-block scales (`d*(0.5+n)*0.25`) |

The emitter's opset chain gathers `iq2s_grid[1024]` with the assembled 10-bit
index, then applies the raw-bit sign and the two scales. The plugin patch adds
`kWeightFormatIq2S = 4`, a `NativeIq2SSBlock` pattern beside `NativeExpertBlock`,
the validation relaxation for the doubled scale, the tier row decoder (the
same tables as `src/core/gguf_dequant.cpp`) and the OpenCL decode — the shape
of patch 0043/0045 for a fourth format.

## 5. The emitter port (OWED)

A `build_qwen35moe_serving_shape_ir` that:
- emits plain pre-norm residual layers (no `_split_combine`/`_recombine`, no
  `_ple_state`, no n-gram ports);
- reuses `serving_shape.emit_stateful_attention` for the 10 attention layers
  (the qwen3.5 `q_proj` carries query and gate interleaved per head, and the
  gate is sigmoid — see `export_mtp.build_mtp_layer`, measured 66 % vs 13 %);
- reuses `q4e.gdn` for the 30 GDN layers (`gdn_key_head_map = "tiled"` for
  the GGUF, value/key head ratio 2);
- reuses `serve_moe_tiled` / `emit_moe_tiled` with the native filler, so the
  `MOECompressed` constants carry IQ2_S (gate/up) and IQ3_XXS/IQ4_XS (down);
- emits the separate embedding model and head as `export_serving_artifact`
  does.

**Facts to measure before coding** (each a small cell, not an assumption):
1. the GDN output gate activation (`sigmoid` vs `silu`) for `qwen35moe`;
2. the value-to-key head map (`tiled` vs `interleave`);
3. the norm convention (`(1 + w)` vs plain) on `attn_norm` / `post_attention_norm`
   for this converter;
4. whether the MoE tiled block for 256 experts top-8 fuses on the plugin the
   way the 512-expert block does.

The oracles: llama.cpp's `qwen35moe` graph, the served
`qwen36-35b-a3b-int4-ov` IR's own subgraph, and gguf-py's dequantize.

## 6. Gate

A full-depth `Qwen3.6-35B-A3B` artifact carrying its own IQ2_S/IQ3_XXS/IQ4_XS
expert blocks loads and serves through the plugin (patch series through
0049+), and on the A770 (PCI `8086:56a0`, GPU.1) reports decode t/s, TTFT and
digests at 1 / 4096 / 16384 against the int4 comparand (DESIGN §7.0.2v:
9.1 t/s at ratio 50, 8 GiB pool), with the fully-resident native route's
backlogged gap (patch 0043 assert) either cleared or recorded BLOCKED. No
result is claimed before it is measured.

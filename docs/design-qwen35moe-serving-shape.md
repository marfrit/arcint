# Design note: a `qwen3_5_moe` serving-shape emitter and the IQ2_S native expert format

Status: design + the IQ2_S decoder landed (2026-09-24); the emitter port, the
plugin IQ2_S format and a depth-4 native export landed the same day (§7). The
A770 window and the full-depth export remain OWED. Written after the
2026-09-24 operator decision *"IQ3_XXS first with conversion and benchmark,
then continue the roadmap"* re-targeted to `Qwen3.6-35B-A3B` (see
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

## 7. The leg, landed 2026-09-24 — the plugin format, the emitter, a depth-4 export

Evidence class per disposition: `code` = the cited source; `measured-here` =
run on this leg and cited with its raw output; `paper` unused here.

### 7.1 The four conventions, measured before any code (design note §5)

1. **GDN output gate = SILU** (gated RMSNorm on the `attn_gate` projection, not
   sigmoid). `code`: llama.cpp `src/models/qwen35moe.cpp` `build_norm_gated`
   computes `silu(gate)` then multiplies the RMSNormed core. `measured-here`:
   the served int4 IR's `linear_attn.norm` chain is
   `Multiply(core, rsqrt) -> Multiply(weight) -> Multiply(silu(z))`, the only
   `Sigmoid` in `linear_attn` sits on `in_proj_b`/beta, and the GGUF
   `blk.0.ssm_norm.weight` equals that IR Constant to `max|diff| 0.0`.
2. **Value/key head map = TILED.** `code`: llama.cpp `ggml_repeat_4d` (head
   `j` reads key head `j % 16`); `q4e.gdn._key_head_map`. `measured-here`: every
   value-head-indexed GDN tensor in the GGUF is the HF *interleave* order
   re-laid into llama's *tiled* order — sigma-map max|diff| vs identity:
   `attn_qkv` v 0.012 vs 0.395; `attn_gate` (z) 0.012 vs 0.297; `ssm_out`
   0.036 vs 0.413; `ssm_alpha` 0.0069 vs 0.171; `ssm_beta` 0.0043 vs 0.085.
   The served int4 IR is the HF interleave form (Broadcast `[16,2,…]` +
   GatherND); the GGUF-fed artifact must use tiled to match llama.cpp.
3. **Norms = PLAIN RMSNorm, no `(1 + w)`, pre-norm residual.** `code`:
   llama.cpp `build_norm` (`RMSNorm` then multiply); `measured-here`: GGUF
   `attn_norm`, `post_attention_norm`, `ssm_norm`, `attn_q_norm`,
   `attn_k_norm` all equal the served IR's Constants to `max|diff| 0.0`, and
   the IR chain is `Pow -> ReduceMean -> Add(eps) -> Sqrt -> Divide ->
   Multiply(x) -> Multiply(weight)` with no `+1`. `qattn._rmsnorm_hd` therefore
   takes `norm_plus_one=False` for this family.
4. **The tiled MoE lowering is E-agnostic.** `measured-here`, device-free: a
   256-expert top-8 tiled block compiles on the CPU plugin to **3
   GatherMatmul** primitives exactly as 512/top-10 does, and the constraint
   walker matches the one MoE root at 256/8 (real 2048/512 and tiny 256/128
   geometry). `code`: the matcher reads `E` off the weight's leading dim with
   no bound. The GPU compile at 256/IQ2_S stays OWED to the A770 window.

### 7.2 The plugin format (patch 0050)

`kWeightFormatIq2S = 4`, `is_native_format`, the `[E, ofm, K/32, 8]` weight
branch, `NativeIq2sWeightsBlock`, tier row decoder + `kIq2sGrid[8192]`, the
OpenCL `native_dot_iq2s`, and the IQ2_S scale-transpose skip. **IQ4_XS down
needs no new format**: `iq4_xs_split` lands on the IQ4_NL layout, so the three
IQ4_XS down tensors ride `kWeightFormatIq4Nl` (`measured-here`: `blk.34/38/39`
split -> decode vs gguf-py `max|diff| 0.0`). Built clean on the pinned tree
(`ninja openvino_intel_gpu_plugin`, rc 0). See the patch header and the
patches README.

### 7.3 The emitter and the depth-4 export

`build_qwen35moe_serving_shape_ir` (+ `qwen35moe_real_config`, the layer state
builder and the key table) emits the plain pre-norm residual layer, reuses
`q4e.gdn` (tiled), `emit_stateful_attention` (`norm_plus_one=False`) and
`emit_moe_tiled`, and is wired into `tools/export_serving_artifact.py` via
`--family qwen35moe` (auto-detected from `general.architecture`). The depth-4
export `qwen36-35b-a3b-d4n-ov` (operator-local): 4 layers = 3 GDN + 1
attention, 1476 nodes, 12 native expert bodies (2,415,919,104 B = 2.25 GiB),
LM `.bin` 4,284,499,713 B, embed `.bin` 1.89 GiB, peak host 10.88 GiB. It
loads (`ov.Core().read_model`, 1476 nodes) and every expert body is the
native format: **all 12 gate/up IQ2_S and all 4 down IQ3_XXS blocks read back
byte-exact (`iq2_s_split`/`iq3_xxs_split` codes and signs equal) and decode to
`max|diff| 0.0`** against the GGUF. Red-first cells in
`tests/python/test_qwen35moe_serving_shape.py`: the low-byte-only index mutant
is caught (real indices exceed 255), the 256-expert tiled cell asserts 3
GatherMatmuls, and the emitter cell compiles on CPU; the signed/scale cells of
`test_native_blocks.py`/`test_native_expert_gemv.py` are mutation-sensitive.

### 7.4 OWED

- **The A770 window (the gate, §6)**: the GPU compile of the native IQ2_S
  pattern and per-expert kernel, and the served rate/digests. No card was
  touched on this leg; the plugin was built and the pattern serialised only.
- **The full-depth 40-layer export** (the earlier full-depth runs are
  hour-scale; this leg exported depth 4).
- **The served binary's `qwen3_5_moe` load path**: the artifact's `config.json`
  carries the family geometry but `src/core/artifact.cpp`'s family admission
  and the served loop for a non-PLE family are unverified here — a depth-4
  `read_model` + CPU compile is what "loads" means on this leg.
- **The 256-expert/IQ2_S GPU fusion** (cell 4's card half).

## 8. The served-side admission, landed 2026-09-25 — no card window

Evidence class per disposition: `code`, `measured-here` (device-free), or a
build/test result.

### 8.1 The registry entry

`qwen3.6-35b-a3b-native-d4` admits the artifact directory
`qwen36-35b-a3b-d4n-ov` by the numbers read off its own `serving-shape.json`
and `.bin`, not guessed — `arcint --model <dir> --inspect-artifact` printed
arch `391bd21db6368d57`, template `55d4931433fe502b`, tokenizer
`87a7830d63fcf43b`, `4,284,499,713 B` in one segment (2026-09-25). The entry
states plainly that it is a measurement artifact: depth 4 of 40, not the
model's answers. `models/allowlist-raw.json` carries the same row, which
`tests/test_provenance.cpp` holds against the compiled registry.

### 8.2 `weights_bytes` becomes a contract

The allowlist pinned `weights_bytes` and `validate_artifact` never read it, so
a re-exported `.bin` passed admission on the strength of its xml hash alone.
`ArtifactInfo` now carries `weights_bytes` (set in `Artifact::to_info`) and
`validate_artifact` calls `check_u64`, which refuses a mismatch or an artifact
that reports nothing and warns only when the entry leaves it unpinned.

### 8.3 The no-PLE / no-n-gram path is first-class

- `ngram::check_declared_table(ngram_size, ple_embed_dim, plan)` returns `""`
  when the config declares no table and the IR declares no `ngram_table.K`
  port — the `qwen3_5_moe` case, where the binding is INERT and `--ngram-gguf`
  is not needed — and a named refusal when the config DOES declare a table the
  graph cannot carry (the silent-nullptr case: a table admitted and held that
  nothing reads).
- `bind_ngram_ports` calls it in the empty-plan branch (after the existing
  `--ngram-gguf` refusal) and returns inertly otherwise.
- `feed_ngram_ports` feeds the GDN `conv_mask` whenever the graph declares it,
  BEFORE the table-plan early return. The first form returned on
  `ngram_ports_.empty()` and left `conv_mask` unwritten on exactly this family.

### 8.4 Red-first, mutation-tested

`tests/test_registry.cpp` (19 cases) and `tests/test_ngram_ports.cpp` (13
cases) carry the cells. Mutation run 2026-09-25: making `check_u64` a no-op
fails `registry_the_native_qwen35moe_rung_is_admitted_without_a_ple`; making
`check_declared_table` always return `""` fails
`ngram_ports_a_declared_table_with_no_port_is_refused_by_name`. Both restored
and green. Whole device-free C++ suite: 609 cases, 0 failed, 2 skipped.

### 8.5 OWED

The GPU load of the native IQ2_S graph and the served arm (rate + digests
against the int4 comparand, the 15.1 GiB fit verdict, the V4/determinism
reading) — the A770 window. No card was touched (`pgrep -x arcint` = 0); the
served load could not be exercised device-free, so 8.3 is `code` + unit cells,
not a served reading.

## 9. The A770 window, landed 2026-09-25 — the native IQ2_S graph compiles and serves; V4 does not fire

Evidence class per disposition: `code`, `measured-here` (run on this leg,
raw rows in the operator-local packet and cited here), or
`previously-measured` (an earlier window's number, labelled). No number here
is an estimate.

The gate (§6) asks for a served reading on the A770 (PCI `8086:56a0`,
OpenVINO `GPU.1`) at 1 / 4096 / 16384 against the int4 comparand, and for the
fully-resident native route either cleared or recorded BLOCKED. This leg ran
it on the **depth-4** artifact; the full-depth 40-layer export stays OWED.

### 9.1 The runtime and the plugin fingerprint

- **Card**: A770 = `GPU.1` = PCI `8086:56a0` (confirmed by the plugin:
  `FULL_DEVICE_NAME` = `Intel(R) Arc(TM) A770 Graphics (dGPU)`,
  `GPU_DEVICE_TOTAL_MEM_SIZE` = 16,225,243,136 B = 15.11 GiB). The B60 =
  `GPU.0` = `8086:e211` was never opened. `code`+`measured-here`.
- **Plugin**: a **measurement** build of the pinned OpenVINO (`71640275`,
  2026.4.0) with patches 0003–**0050** applied, debug caps OFF, installed to
  its own prefix; GPU-plugin sha256
  `582c3230f8c0960ba8d91df75db20775a66de1449ab761d0d39566e35d19a1f1`, version
  string `2026.4.0-22849-71640275d29-marfrit-p19`. The pre-existing prefixes
  (`ov-0045`, `ov-0047`, `ov-0049`, `ov-dbg`, `ov-venice`) were not touched
  (`measured-here`: the `ov-0049` plugin still reads `2d83e2a6d2aa7894…`).
- **Binary**: the tip (`591f5bc`) built with OpenVINO on; sha256
  `30d55b67d13b07ce6c74dfa617871a92c1b6243dafdf1092d350af781932d6f1`.
- **Harness** (operator-local): the served-services runner (`bench-arm.py`,
  sha256 `03c76dc5…`) with its `unshare -rm` CPU-online shim
  (`352eaf78…`), the shared 23,680-id prompt
  (`31b351d6f95c5f08fc877e1750c3126610f16443ae568b930374deaad4c5e91e`), 32
  greedy tokens, `--paged-kv u8`, prefix cache 2048 MiB. Per the
  serving-shape convention the arms carry **`--no-logits-slice`**
  (`code`: `kld-d48n.sh:35`, the same flag the `qwen4_exp` serving-shape
  runs carry).

### 9.2 The compile (gate item 1) — PASS

`measured-here`: "compiling PAGED language model on GPU.1" then "language
model ready in 22.0 s (paged); device-resident 1.28 GiB" (arm n50). The
native IQ2_S MoE graph — the tiled expert Constants, the matcher, the
`MOECompressed` with `weight_format=4`, and the per-expert kernel path —
compiles on the A770.

The first attempt omitted `--no-logits-slice` and refused at the load's
logits-slice verification (`measured-here`, verbatim): *"logits slice did not
take: 128 row(s) for a 128-token forward (the slice keeps the last 1, so 1
expected), shape [1,128,248320] -- the token axis is not where the slice
assumed"*. That is the serving-shape layout (`[1, tokens, vocab]`, token axis
1) against a served-path slice axis of 0, and it is why `--no-logits-slice`
is the convention here — the same shape the `qwen4_exp` serving-shape export
has (§9.1). The compiled graph had already run the 128-token forward: the
native IQ2_S compute **executed on the card** before the check refused.

### 9.3 The served sweep (gate item 2) — the native IQ2_S depth-4 artifact

A770 `GPU.1`, u8 KV, chunk 2048, 8 GiB device pool, `--offload-ratio R
--moe-cpu-tier --moe-per-expert-dispatch`, `--no-logits-slice`, 1 lane,
`--n-ctx 32768`. Extension rate = `(prompt − cache-hit) / prefill_s`.

| arm | R | slots | `T_boot` s | depth | hit | ext tok | prefill s | ext t/s | TTFT s | decode t/s | digest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n50 | 50 | 128 | 380.377 | 1 | 0 | 1 | 0.50 | 2.0 | 0.806 | 4.2 | `3f6d0ab1f8d9` |
| n50 | | | | 4096 | 0 | 4096 | 143.00 | 28.6 | 143.006 | 14.2 | `8cccdbac48ed` |
| n50 | | | | 16384 | 2048 | 14336 | 497.13 | 28.8 | 497.132 | 14.0 | `e2a836c80c1f` |
| n75 | 75 | 64 | 351.375 | 1 | 0 | 1 | 0.34 | 2.9 | 0.566 | 8.3 | `3f6d0ab1f8d9` |
| n75 | | | | 4096 | 0 | 4096 | 214.41 | 19.1 | 214.407 | 13.4 | `8cccdbac48ed` |
| n75 | | | | 16384 | 2048 | 14336 | 744.45 | 19.3 | 744.455 | 12.4 | `e2a836c80c1f` |
| n50p | 50 | 128 | 299.222 | 1 | 0 | 1 | 0.33 | 3.0 | 0.587 | 8.4 | `3f6d0ab1f8d9` |
| n50p | | | | 4096 | 0 | 4096 | 141.80 | 28.9 | 141.805 | 14.3 | `8cccdbac48ed` |
| n25 | 25 | 192 | 251.168 | 1 | 0 | 1 | 0.33 | 3.0 | 0.555 | 8.6 | `3f6d0ab1f8d9` |
| n25 | | | | 4096 | 0 | 4096 | 80.27 | 51.0 | 80.270 | 15.1 | `8cccdbac48ed` |
| n99 | 99 | 2 | 395.360 | 1 | 0 | 1 | 0.37 | 2.7 | 0.627 | 6.1 | `3f6d0ab1f8d9` |
| n99 | | | | 4096 | 0 | 4096 | 281.87 | 14.5 | 281.871 | 8.6 | `8cccdbac48ed` |

All arms loaded (`ready`, exit 0). The sampler on the physical host ran
through all arms, **0 watchdog trips**, minimum `MemAvailable` recorded ≈
24.7 GiB (batch 1) / ≈ 50.6 GiB (batch 2) / ≈ 46.9 GiB (batch 3) — never near
the 4 GiB fence. The device pool placed variously (batch 1 fdinfo peak
`drm-resident-vram0` ≈ 10.8 GiB; batch 3 peak `vram0` ≈ 3.9 GiB against
`gtt` ≈ 10.4 GiB) — recorded, not explained; the fit arithmetic is the
authoritative claim (§9.5).

### 9.4 The int4 comparand (gate item 3) — same flags, different depth

`measured-here`: the existing full-depth int4 artifact (40 layers, 30 GDN +
10 attn, 262144 ctx) at the **same flags** (ratio 50, 8 GiB pool, tier,
dispatch, u8 KV): `T_boot` 568.341 s, ext prefill 16.0 t/s @4096 and 26.1 t/s
@16384, decode 5.4 t/s @4096 and 5.3 t/s @16384; digests `012397a89576`,
`f3e9eb2ffa08`, `2d0ff8b1f891`. Its served text is coherent.

`previously-measured`: §7.0.2v's 9.1 t/s at ratio 50/8 GiB is the **fused**
path, tier OFF; §7.0.2x measures 15.0/15.5 t/s at ratio 50/8 GiB, tier ON,
**no dispatch**. This window's comparand adds `--moe-per-expert-dispatch`
(the native route's own mechanism), under which the int4 40-layer arm reads
5.4 t/s — ~2.8× below the fused/tier reference. The comparator is therefore
**not** the §7.0.2v configuration, and the depth differs (native 4 layers vs
int4 40). The delta this window supports is a **per-layer** one: the native
IQ2_S per-layer decode cost is ≈ 3.8× the int4 affine per-expert per-layer
cost (native 1/(14.3·4) ≈ 17.5 ms/layer, int4 1/(5.4·40) ≈ 4.63 ms/layer).
A like-for-like depth-4 int4 artifact does not exist; exporting one is OWED.

### 9.5 The fit verdict (gate item 4) — the fully-resident native arm is BLOCKED

`measured-here`, the load's own reservation line (arm n50p, ratio 50):

    weights+graph 1.28 GiB + drafters 0.95 + expert slots 0.15 (probe-static)
    + activations 3.40 (all 1 lane, chunk 2048) + margin 0.25
    + 1 x (GDN rows 9.6 MiB + KV 1.1 KiB/token) of 15.11 GiB
    -> max ctx 8397168 per lane

The depth-4 native artifact fits with ≈9 GiB of headroom. The **fully-resident
native arm does not exist**, recorded BLOCKED with the compile-time assert:

    Check '!native || (_cpu_tier && _weight_provider->is_offloaded())' failed
    at .../moe_3gemm_swiglu_opt.cpp:1886: native expert formats (IQ4_NL /
    IQ3_XXS) need OFFLOAD_RATIO in (0, 100) and MOE_CPU_TIER=YES: every routed
    expert is computed on the CPU tier until the OpenCL decode exists (patch
    0043)

`code`: arcint sets the plugin's `OFFLOAD_RATIO` property only when the ratio
is `> 0`, and ratio 0 is exactly the fully-resident configuration — so the
native MoE cannot serve fully resident. The assert's message still names
IQ4_NL / IQ3_XXS (patch 0043's text predates IQ2_S); the assertion fires for
IQ2_S too. Lever is in the campaign status log.

### 9.6 The V4/determinism reading (gate item 5) — V4 does NOT fire here

`measured-here`: the served digest is **byte-identical across resident
shares**. At depths 1 and 4096, ratios 25 / 50 / 75 / 99 give
`3f6d0ab1f8d9` / `8cccdbac48ed`; at depth 16384 both 50 and 75 give
`e2a836c80c1f`. The per-expert route was demonstrably exercised
(`MOE_OTD_PERF_LOG=1` counters, `measured-here`):

| R | slots | gpu hit rate | per-expert GPU invocations | CPU-tier experts | decode t/s @4096 |
|---|---|---|---|---|---|
| 25 | 192 | 65.03 % | 407,480 | 1,629 | 15.1 |
| 50 | 128 | 43.15 % | 273,416 | 3,382 | 14.3 |
| 99 | 2 | 0.34 % | 1,590 | 6,818 | 8.6 |

The resident share spans 2 → 192 slots and the GPU/CPU expert counts differ by
two orders of magnitude, yet the answer does not move at any depth. This is a
**negative** against the standing §7.0.2cf expectation (native kernels not
bit-identical to the host tier). Two explanations remain open and this window
does not separate them: (a) patch 0050's IQ2_S kernel is bit-identical to the
CPU tier, or (b) the 4-layer artifact's greedy output degenerates to a
repeated-token attractor robust to the perturbation — `measured-here`, its
served text is repetitive at every depth (e.g. `…retienretienretien…`),
unlike the coherent int4 comparand. A logits-level A/B (the §7.0.2cf
one-layer method) is OWED; the served-digest reading is not evidence for (a).

### 9.7 262144 reachability (gate item 6) — reachable, from the run's own arithmetic

`measured-here`: the reservation's `max ctx 8397168 per lane` is the run's own
KV arithmetic (KV 1.1 KiB/token, GDN rows 9.6 MiB fixed), and an arm with
`--n-ctx 262144` loaded and served on the A770. 262144 is inside the cap; the
cap is ~32× the artifact's configured context. This is the depth-4 artifact's
arithmetic, not a full-depth estimate.

### 9.8 OWED

- The **full-depth 40-layer export** and its served reading (the mechanism is
  depth-independent; the number is not).
- A **logits-level V4 A/B** to separate (a) from (b) in §9.6.
- A **depth-4 int4 comparand** for a like-for-like format delta.
- The served `qwen3_5_moe` **answer quality**: the depth-4 artifact's greedy
  text is degenerate; whether that is truncation or an emitter weight-mapping
  defect is not resolved here. The device-free verification covered the expert
  bodies and the norm weights only.

## 10. The full-depth (40-layer) native export, landed 2026-09-25 — verified before any card

The A770 window (§9) left one decisive question open: was the depth-4
artifact's degenerate greedy text **truncation** or an **emitter/fill defect**?
A full-depth artifact answers it without a card. Evidence classes: `code`,
`measured-here`, `previously-measured`.

### 10.1 The export

`tools/export_serving_artifact.py --family qwen35moe --layers 40
--expert-format native` over the same checkpoint shard as depth 4, with the
served int4 artifact as the tokenizer passthrough. `measured-here` (its own
log): feed 6.8 s; **build 249.2 s** for **40 layers = 30 GDN + 10 attn,
14,499 nodes**; dense fill 612 tensors / 7.23 GiB; expert fill **120 bodies /
24,662,507,520 B**; save 236.7 s → language-model `.bin`
**23,429,144,641 B** (21.82 GiB) and embeddings `.bin` 2,034,237,448 B (1.89
GiB); `lm_xml_sha b94ecc6ab6b200ac`; chat template `55d4931433fe502b`;
tokenizer `87a7830d63fcf43b`. **peak_host_GiB 29.38** — inside the container's
44 GiB cgroup cap, so the full-depth build is a single process, not a
segmented one (`code`: `qwen35moe` refuses `--segment-layers`). Artifact about
25.9 GiB.

### 10.2 The all-body type census — 120/120 match the GGUF

`measured-here`, read off the saved IR's Constants (the emitter names them
`layer{i}/moe/experts_{kind}/...`), against gguf-py's own tensor types:

| role | IR bodies | GGUF types |
|---|---|---|
| gate | 40 × IQ2_S | 40 × IQ2_S |
| up | 40 × IQ2_S | 40 × IQ2_S |
| down | 37 × IQ3_XXS + 3 × IQ4_XS→IQ4_NL | 37 × IQ3_XXS + 3 × IQ4_XS |

The three IQ4_XS down layers fold onto the IQ4_NL layout
(`native_blocks.iq4_xs_split`), exactly as depth 4 did.

### 10.3 Sampled byte-exactness — the artifact, not the emitter

`measured-here`: the IR's packed Constant **bytes** (not a re-run of the fill)
compared against the emitter's packing recomputed from the GGUF — one gate and
one up body at layer 0 (IQ2_S), down at layers 0 and 19 (IQ3_XXS), down at
layers 34 and 39 (IQ4_XS→IQ4_NL). Every one: **grid indices, sign bytes and
block scales byte-equal**, `grid_ok/sign_ok/scale_ok = True`. Shapes read off
the artifact: IQ2_S `[256,512,64,8]` + `[256,512,64,4]` + f16 `[256,512,64,2]`;
IQ3_XXS `[256,2048,16,8]` + `[256,2048,16,4]` + f16 `[256,2048,16,1]`;
IQ4_NL codes packed `[134,217,728]` u8 + f16 `[256,2048,16,1]`.

### 10.4 Admission

`qwen3.6-35b-a3b-native-d40` (artifact `qwen36-35b-a3b-d40n-ov`) is admitted
with `n_layer 40`, 10 attention + 30 GDN, 256 experts, `arch_hash
b94ecc6ab6b200ac`, template `55d4931433fe502b`, tokenizer
`87a7830d63fcf43b`, **`weights_bytes 23,429,144,641`** — all read off the
artifact's own `serving-shape.json`. The depth-4 rung stays admitted beside
it. `measured-here`: `--inspect-artifact` prints
`admitted as qwen3.6-35b-a3b-native-d40`; `tests/test_registry.cpp` is 20
cases, 0 failed (the new entry has its own red-first cell), `test_provenance`
4/0, `test_ngram_ports` 13/0.

### 10.5 OWED

The **A770 reading** of the full-depth artifact — coherence, rate, digests,
fit, 262144, V4 — is the window that follows in this leg, recorded in §11.

## 11. The speed defect: the all-resident native route, 2026-09-25

Operator redirect, same day: *"before measuring more slow runs, spend the work
on making it faster. This is not a measurement project."* The native route's
only fast configuration -- every expert resident, only the routed experts
computed on the GPU per-expert kernel -- was **unreachable**, by a three-link
dead end. This section is the fix and the gate. Evidence classes: `code`,
`measured-here`, `previously-measured`.

### 11.1 The three links (each a bug, not a missing feature)

1. `code` — `src/config.cpp` refused `--moe-cpu-tier` when the ratio was 0.
2. `code` — `src/exec/backend_ov.cpp` set the plugin's `OFFLOAD_RATIO` (and
   `ov::weights_path`) only when the ratio was `> 0`, so an explicit `0` was
   swallowed. The plugin's own validator accepts 0 (`v <= 100`, patch 0005).
3. `code` — the plugin's `prepare_moe_otd_params` (`ops/moe.cpp`) computed
   `lru_expert_num = 0` when `otd_ratio == 0`, so `moe_3gemm_swiglu_opt.cpp`
   selected the **ResidentExpertWeightProvider** — no slot pool, **no native
   reader** — while patch 0043's assert demanded `_cpu_tier &&
   is_offloaded()`, a conjunction unsatisfiable at 0. Its stated rationale
   *"until the OpenCL decode exists"* was obsolete: patch 0045 added the
   decode, and patch 0050 carried it to IQ2_S.

### 11.2 The fix

- **Plugin patch `0051-native-fully-resident.patch`** (`code`): a native format
  at `otd_ratio == 0` enables OTD when the caller supplied `ov::weights_path`
  (an explicit 0 is thereby distinguishable from unset) and sizes the pool at
  `num_expert` — every expert resident, read through the offload provider's
  native reader. The assert drops the `_cpu_tier` requirement: it needs only
  `_weight_provider->is_offloaded()`, because the tier is unnecessary when
  there are no misses.
- **The arcint side** (`code`): a `offload_ratio_set` Config flag; the pure
  function `moe_offload_active(ratio, ratio_set, expert_format)` in `config.h`
  (ratio > 0, or an explicit 0 for a `native` artifact); `Artifact` now reads
  `serving-shape.json`'s `expert_fill.format`; the two guards relaxed
  (`--moe-cpu-tier` allowed at ratio 0 when dispatch is requested, and
  `--moe-per-expert-dispatch` allowed without the tier at ratio 0).
- **Red-first** (`measured-here`): `tests/test_config.cpp` gained
  `config_offload_active_covers_the_all_resident_native_case` and three parse
  cells. Mutation run 2026-09-25: reverting `moe_offload_active` to
  `offload_ratio > 0` fails exactly that cell; restored, `config` is **74
  cases, 0 failed** (up from 69). The OpenVINO build and the no-OpenVINO build
  both compile clean.

### 11.3 The gate — A770, depth-4 artifact (`measured-here`)

`--offload-ratio 0 --moe-cpu-tier --moe-per-expert-dispatch`, u8 KV, chunk
2048, `--no-logits-slice`, 1 lane, `--n-ctx 32768`, plugin `ov-0051`.

| quantity | all-resident (this leg) | tiered ratio 50 (`previously-measured`, same day) |
|---|---|---|
| load | ready in **4.4 s**, device-resident 2.71 GiB | ready in 22.0 s, device-resident 1.28 GiB |
| reservation max ctx | 7,066,560 / lane | 8,397,168 / lane |
| decode t/s @1 / @4096 | 49.9 / **56.8** | 4.2 / 14.2 |
| ext prefill t/s @4096 | **155.9** | 28.6 |
| digest @1 / @4096 | `3f6d0ab1f8d9` / `8cccdbac48ed` | `3f6d0ab1f8d9` / `8cccdbac48ed` |

The all-resident native route is **4.0× decode and 5.4× prefill** the tiered
arm at the same depth, with **byte-identical digests** — and the counters say
why: `cpu_tier_pairs=0`, `cpu_tier_experts=0`, `per_expert_dispatches=6847`,
`per_expert_gpu_invocations=544,832`, gpu hit rate 85.0 %. No expert touched
the CPU tier; the GPU per-expert kernel (patch 0050's IQ2_S path) computed
every routed expert.

Two caveats, stated: (a) the load's first fit pass failed on
**fragmentation, not arithmetic** ("allocation failed at a budget that said it
fits", pass 1/4), and pass 2/4 loaded by capping the prefix-cache spare at
219,805 pages — the arithmetic fits, the allocator did not on the first pass;
(b) the digests being identical again leaves the §9.6 question open (bit-exact
kernel vs degenerate attractor) for this depth-4 rung.

The dispatch-without-tier variant does **not** load: it reaches the plugin and
refuses at `moe_3gemm_swiglu_opt.cpp:1141` (`dispatch_cpu_tier: x/routing-weight
usm_host buffers not populated`) because the dispatch route needs the tier's
hoisted host buffers even when there are no misses. The reachable all-resident
configuration therefore carries `--moe-cpu-tier`.

### 11.4 The sizing census (operator item 3) — resolved, no mis-map

`measured-here`: the SAVED artifact's native body bytes, read off its own
Constants, against the GGUF's raw block bytes, over all 40 layers:

| role / GGUF type | artifact | GGUF | ratio |
|---|---|---|---|
| gate IQ2_S | 5.000 GiB | 3.203 GiB | **1.561** |
| up IQ2_S | 5.000 GiB | 3.203 GiB | **1.561** |
| down IQ3_XXS | 4.047 GiB | 3.541 GiB | 1.143 |
| down IQ4_XS→IQ4_NL | 0.422 GiB | 0.398 GiB | 1.059 |
| **experts, total** | **14.469 GiB** | **10.346 GiB** | **1.399** |

The inflation is the plugin's native layout, not a mis-map: grid indices ride
as four little-endian u16 (`8 B/32`) against the GGUF's packed qs+qh
(`5 B/32`), and the two sub-block scales as two f16 (`4 B/32`) against the
GGUF's 4-bit nibbles (`1 B/32`); gate/up is exactly 82 → 128 B per 256. The
sampled byte-exactness (§10.3) proves the same bytes, re-laid.

The **"24 GB / 600 MB per layer"** premise came from the manifest's
`expert_fill.filled_bytes` (24,662,507,520 B). That field sums the **f32 split
parts** (`q4e/native_blocks`' `scales` are f32), about 1.55× the artifact's
actual Constant bytes — a manifest-field overstatement, not artifact size. The
lm `.bin` (21.82 GiB) confirms the 14.47 GiB experts + ~7.35 GiB dense. A
packed native layout (store the GGUF's 82 B blocks and decode them) is the
size lever; it is not this leg's.

### 11.5 The full-depth artifact's coherence — truncation, confirmed

The interrupted full-depth sweep (`measured-here`; stopped the moment the
priority changed, so 1 and 4096 are complete and 16384 is not): the d40n
artifact **loaded and served** (compile 65.3 s, `weights+graph 3.68 GiB`,
reservation `max ctx 581600 per lane`). Its greedy text is **coherent** —
depth 1 `"\n# Tools\n\nYou have access to the following functions:\n\n1.
**execute_command**: Run a command in the terminal..."`, depth 4096
`"...system: You are a function calling AI model..."`. Rate: decode 0.5 t/s
@1 and 1.5 t/s @4096, ext prefill 2.6 t/s @4096. So the depth-4 rung's
degenerate text was **depth-4 truncation**, not an emitter/fill defect; the
emitter is right.

### 11.6 OWED

- The **full-depth all-resident arm** does not fit (14.47 GiB of experts alone
  against 15.11 GiB of VRAM), so the all-resident speed is a small-depth route
  until the packed native layout and/or a smaller expert representation.
- A **packed native layout** (the size lever) — the offload miss path copies
  the inflated body across PCIe.
- The **16384 point** of the full-depth sweep (interrupted by the redirect).
- A **logits-level V4 A/B** to separate bit-exact kernel from attractor
  robustness.

# Design note: opening GGUF files in process with native K-quant weights (0.4.0)

Status: draft for discussion, 2026-09-06. Written from two read-only recon
passes over the llama.cpp tree the operator keeps (a fork whose
converter, quant formats, `qwen35` graph and tokenizer are byte-identical
to upstream at its merge base) and over the pinned OpenVINO GPU plugin
with the patch series applied. Decisions taken by the operator before
this note: the allowlist keeps its families and the point is comparing
quantisations of the same model; the K-quant tensors are served natively,
not re-quantised; the engine opens the file in process; the served IR's
tokenizer and chat template govern for now; the order is the dense 27B at
Q4_K_M, then the pruned coder's MoE file, then the sub-4-bit set.

## 1. What a GGUF of this family is

Both target files are `qwen35` / `qwen35moe` with `block_count` one above
the model's layer count: the last block is the MTP ("nextn") layer, which
llama.cpp loads and runs as a one-layer draft graph (its constants file
still calls the tensors "preserved but unused"; the graph builder does
use them). Per-tensor types in the operator's Q4_K_M files:

| tensor (GDN block) | dense 27B | pruned coder (MoE) |
|---|---|---|
| `attn_qkv` (HF `in_proj_qkv`, fused q,k,v) | Q6_K | Q6_K |
| `attn_gate` (HF `in_proj_z`) | Q4_K | Q4_K |
| `ssm_alpha`, `ssm_beta` (HF `in_proj_a/b`) | F32 | Q4_K |
| `ssm_out` (HF `out_proj`) | Q5_K | Q4_K |
| `ssm_a` (= −exp(A_log)), `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, norms | F32 | F32 |
| `ffn_gate`, `ffn_up` / `ffn_*_exps`, `ffn_*_shexp` | Q4_K | Q4_K (experts as one tensor `[K, N, experts]`) |
| `ffn_down` / `ffn_down_exps` | Q6_K | Q6_K |
| `ffn_gate_inp` (router) | — | F32 |

Attention blocks (every fourth): `attn_q` Q4_K holding query and gate
fused (HF's `q_proj` is `2 × heads × 256` wide in this family, the IR's
too), `attn_k` Q4_K, `attn_v` Q6_K, `attn_output` Q4_K, q/k norms F32.
Non-block: `token_embd` Q4_K, `output` Q6_K, `output_norm` F32; the MTP
block adds `nextn.eh_proj` (Q8_0 in the dense file, Q4_K in the coder's)
and three F32 norms, with embedding and head tied to the model's.

So one Q4_K_M file needs **four decoders — Q4_K, Q5_K, Q6_K, Q8_0** —
before it opens, in three consumers: the fully-connected path, the
embedding gather, and (for the coder) the expert fusion.

**Block layouts** (`ggml-common.h`, unchanged in the fork): blocks run
along the contraction dimension of a `[out, in]` weight, 256 values per
super-block (32 for Q8_0). Q4_K: `d`, `dmin` (f16), 12 bytes holding eight
6-bit scales and eight 6-bit mins, 128 bytes of nibbles; value
`d·sc·q − dmin·m` per 32-value sub-block (144 B). Q5_K adds 32 bytes of
high bits (176 B). Q6_K: 128 bytes low nibbles, 64 bytes of 2-bit highs,
16 signed 8-bit sub-scales, `d`; value `d·sc·(q − 32)` per 16 (210 B).
Q8_0: `d` and 32 int8 (34 B). The sub-4-bit set (IQ4_XS, IQ3_S, IQ3_XXS,
Q3_K) uses codebooks and sign tables, a different kind of decoder.

**Converter transforms that the open must undo or reproduce** (the
converter's `qwen.py`, byte-identical to upstream): `A_log → −exp` into
`ssm_a`; `+1` on every norm weight except the GDN's own norm; the conv1d
weight squeezed to `[kernel, channels]`; and, because this family has 16
key heads against 48 value heads, a **V-head reorder** from HF's
grouped-by-key-head layout to ggml's tiled layout on the V rows of
`attn_qkv`, on `attn_gate`, `ssm_alpha/beta`, `ssm_a`, `dt_bias`, the V
channels of the conv1d, and the input columns of `ssm_out`. Row
permutations preserve K-quant blocks (a block is a run along the input
dimension of one output row); the column permutation of `ssm_out` does
not, so it is applied to the activation instead — a Gather on the GDN
output before that projection, exact and cheap.

**Tokenizer** (measured 2026-09-06 on the dense file): tokens identical
over the IR's 248,077 entries, merges identical, the same pre-tokenizer
regex (llama.cpp's `qwen35` type quotes it from `tokenizer.json`), eos
248046 in both; the file pads the vocabulary to 248,320 and its pad id
differs. The chat template differs (Unsloth merges leading system
messages, drops the no-user-query exception, maps `high` to `xhigh`
reasoning effort). Per the operator, the served IR's tokenizer and
template are used; the file's are read, compared, and the difference
logged at load.

## 2. What the plugin has and has not

The exported IRs carry every projection as `Const(u4 [N, K/64, 64]) →
Convert → Subtract(zp u4) → Multiply(scale f16) → Reshape → Convert →
MatMul`, group 64; the head is u8 per-channel; the embedding i8 per-row
with no zero point; the MoE IR's experts are `u4 [experts, N, K/64, 64]`
consumed by a batched MatMul that the plugin's matcher turns into its
expert fusion. The plugin's `ConvertFullyConnectedToFullyConnectedCompressed`
accepts u8/i8/u4/i4/u2/f8/f4 weights at the pattern level, but the
bf-tiled kernel implements only u4/i4/u8/i8/f16/f32; its group size is
flat (`K / scale columns`), its zero point is an integer, and its int4
unpack is two nibbles per byte. Weights are reordered at compile time by
a reorder node into a blocked layout (`os_is_yx_osv32_isv2` for int4).
The expert fusion matches exactly twelve u4 constants and nothing else.
Nothing in the patch series (0003–0020) touches the weight format: they
move placement, uploads, residency and attention. Route 1 of the
milestone note — reducing Q4_K to u4 with per-32 f16 scale and zero
point — would serve on today's kernels with the scales rounded to f16;
the operator chose the native route instead, which the rest of this note
designs.

## 3. Design

### 3.1 The open: template IR, GGUF tensors

arcint does not build the graph from scratch. The served IR of the same
architecture is the topology: `--model <file.gguf>` resolves, through the
allowlist, the template directory of its family (`qwen35` with this
geometry → the dense export; `qwen35moe` with 184 experts → the coder's
export) and reads the template with memory-mapped weights, which touches
nothing until compile. A new pass then walks the model: for every weight
constant whose decompression subgraph the exporter emitted, it finds the
GGUF tensor by the HF module name the IR already carries in the constant
name, checks shape and dtype class, and replaces the whole decompression
subgraph with one new constant. F32 tensors (norms, `ssm_a`, conv1d, dt
bias, router) are replaced by value after the converter's transforms are
undone. Only then is the model compiled. The template's own weights are
never read.

The MTP layer comes from the file's last block through the same pass
with the reconstructed MTP IR (`tools/export_mtp.py`'s graph) as its
template; `nextn.eh_proj` and the three norms map onto it, embedding and
head are tied. DFlash stays what it is (its own IR).

A refusal is by name: a file whose architecture, block count, geometry,
expert count or vocabulary does not match a template is refused at open
with the mismatch printed; a tensor missing or of an unexpected class is
a refusal, not a fallback.

### 3.2 The weight representation in the graph

OpenVINO has no user-defined element types. The K-quant tensor travels
through the graph as a **u8 constant holding the GGUF bytes verbatim**,
shape `[N, row_bytes]`, with runtime information on the constant naming
the block type and the logical `[N, K]`. A plugin transformation
(`ConvertMatMulToFullyConnectedKQuant`, new) matches `MatMul(x, Const u8
with that tag)` and emits a new internal op `FullyConnectedKQuant`; the
expert fusion's matcher gets a sibling that accepts the tagged constants
where it accepted twelve u4 ones. Because the model is built in process,
the tag never has to survive serialization. The u8 constant's byte count
is the GGUF's, so the fit's weight term is the file's size, and the
compile-time reorder is told to keep the layout (identity), since the
kernel reads the GGUF's own block bytes.

### 3.3 Kernels

Three consumers, in this order:

1. **Fully connected** (every projection and the head): a bf-tiled-style
   kernel with a K-quant unpack path in place of `UNPACK_INT4`: the inner
   loop walks 256-value super-blocks along K, unpacks the sub-block scale
   and min (Q4_K/Q5_K: the 6-bit scheme; Q6_K: the int8 sub-scales; Q8_0:
   one f16 per 32), and accumulates in f32 as the bf-tiled kernel does.
   Both the decode shape (few rows) and the prefill shape (a chunk of
   rows) come from the same kernel family, as today. Weight bytes are
   read in their GGUF layout; the tile is 256 wide along K by
   construction. Activation-side dynamic quantization stays off for this
   op in the first version.
2. **Embedding gather**: rows of the Q4_K embedding gathered by token id
   and dequantized to f16 in-kernel (a row is 20 super-blocks; the gather
   is block-preserving). The IR's i8 embedding is replaced by this.
3. **Expert fusion**: the `moe_3gemm_swiglu` kernels gain the same unpack
   path for expert weights stored as K-quant rows per expert; the host
   tier's expert kernel (patch 0011) stays u4-only and refuses the tagged
   format at load, so a GGUF-opened MoE serves resident or with the
   device slot pool only until that kernel learns the format (a later
   item, `kquant-host-storage`).

The sub-4-bit set (IQ4_XS, IQ3_S, IQ3_XXS, Q3_K) is the same three
consumers with codebook decoders, after the linear set is measured.

### 3.4 Exactness, in three layers

- **Host reference**: a C++ dequantizer for every block type in arcint
  (`src/core/gguf_dequant.*`), checked against llama.cpp's own
  `dequantize_row_*` output on real tensors from the file (a device-free
  unit test with a small GGUF fixture; a tool that compares whole
  tensors on the host).
- **Kernel against reference**: plugin unit tests that run
  `FullyConnectedKQuant` and the gather on random K-quant blocks against
  a MatMul on the host-dequantized f32 weights, with the tolerance the
  bf-tiled int4 tests already use for f16 accumulation.
- **Served against served**: the Prüfstand through the endpoint on the
  GGUF-opened dense model, greedy and at the artifact's sampling
  defaults; a same-file comparison against llama.cpp's Vulkan build on the
  host for greedy first tokens on a prompt set (report-only: different
  accumulation orders are not byte-identical); and the engine's own
  invariants — cold/warm prefix cache, one lane against two — byte-exact
  on the GGUF path exactly as on the IR path.

### 3.5 What the fit sees

Weights are the file's bytes: the dense Q4_K_M is 16.3 GB, above the IR's
12.8 (Q6_K on `attn_qkv`, `ffn_down` and the head is 6.6 bits per weight
against int4's 4.5). It fits the 24 GB card with room and does not fit the
16 GiB card; the sub-4-bit files are where that card gains (IQ3_M at
13.9 GB is tight beside any context). The reservation, KV precision,
prefix cache, drafters and the belt are unchanged: from the fit's point
of view a GGUF-opened model is an artifact with different constants.

## 4. Stages and gates

| stage | scope | gate |
|---|---|---|
| 0 | reader, allowlist entry by GGUF metadata, refusals, tensor map with the converter's transforms undone, host dequantizers | device-free: fixture GGUF opens, wrong files refused by name, dequantizers match llama.cpp on real tensors from the dense file |
| 1 | dense Qwen3.8-27B Q4_K_M: FC kernel for Q4_K/Q5_K/Q6_K/Q8_0, embedding gather, the template pass, MTP from the file | opens and serves on the 24 GB card; Prüfstand at its bit width recorded next to the IR's 10/10; prefill and decode rates reported against the IR's; cold/warm and lane byte-exactness green; the llama.cpp first-token agreement reported |
| 2 | the pruned coder's `qwen35moe` file: expert fusion with K-quant rows, shared experts, router | the coder's Prüfstand through the GGUF-opened model against the b5 IR's 10/10; resident on the 24 GB card; the offload path refused by name until the host kernel learns the format |
| 3 | IQ4_XS, IQ3_S, IQ3_XXS, Q3_K decoders in all three consumers | the UD-Q3_K_XL and IQ3_M dense files open and serve; the 16 GiB card's admitted context at each, on the record |

Each stage is a campaign-sized unit with its own red case first, one card
window at the end, review before commit, a DESIGN `§7.0.2x` record and a
CHANGELOG line. Plugin work lands as numbered patches after 0020 with the
series' header convention and unit tests.

## 5. Out of scope, named

Re-quantising K-quants to the plugin's u4 (the operator declined it);
reading the GGUF's tokenizer or template as the served ones; the host
tier's K-quant kernel (its own campaign); any architecture outside the
allowlist's families; llama.cpp byte-identity as a gate.

## 6. Open points for the discussion

1. **The MTP layer from the file** is in stage 1 above. It could be
   stage 1b: open with MTP off first, then add the draft layer once the
   dense path is measured. Which?
2. **The template requirement** means a GGUF opens only where its
   family's served IR directory exists. Acceptable for this repository's
   purpose (comparing quantisations of models it already serves); it is
   the reason this is not a general GGUF engine.
3. **Group-of-256 tiles** in the FC kernel against the bf-tiled kernel's
   current tiling: the recon did not read the tiling constants deeply
   enough to say whether a 256-wide K tile fits the kernel's register
   budget on Arc as-is; the first implementation step is that reading,
   and it may put the unpack in a separate per-super-block prepass.

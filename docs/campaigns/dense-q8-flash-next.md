# dense-q8-flash-next — Flash-Next's dense projections served in their checkpoint's Q8_0 form, not as f16 from f32

Charter: 0.5.x performance, the GPU half of a decode layer after
`host-expert-bank`.

## The defect, as measured

- **The GPU's decode time is dense weights.** With every host expert in RAM, a
  Flash-Next decode token on the B60 costs ~118 ms. Its GPU device time is
  33.3 ms, of which oneDNN `gemm_kernel` is 23.7 ms: 761 calls per token at
  31 µs (`measured-here`, CLIntercept, a 129- minus a 1-token process, d48s2,
  ratio 78 + tier + dispatch). The GPU work runs serially with the CPU tier in
  every layer, so it is on the critical path.
- **The artifact upcasts them.** Its non-expert constants are f32, 15.71 GiB
  (`measured-here`, the IR's constants of 1 MiB and more). The GPU plugin
  serves them as f16, ~7.9 GB a token.
- **The checkpoint is smaller.** It stores those projections as Q8_0
  (4.17 GiB) and Q6_K (0.49 GiB; the output head), with f32 norms and
  routers (0.29 GiB) (`measured-here`, gguf-py over shards 2–3).

## Known against hypothesised

**Known** (`code`):
- **Group layout.** The feed never re-orders inside a K group
  (`gguf_feed._materialise` concatenates whole rows or whole column blocks).
- **The Q8_0 scale.** Its largest |q| is 127 (`quantize_row_q8_0`), so
  max|w| / 127 is the checkpoint's f16 d exactly.
- **Naming.** The qwen4_exp and qwen35moe emitters share
  `emit_stateful_attention` and `emit_shared_expert`, which name attention
  k/v and the shared expert. So `--dense-u8`'s exclusions by name hold for
  this family too; the exporter's family refusal cited a reason the code no
  longer has.

**Hypothesised:**
- **Bytes.** i8 group-32 plus f16 scales halve the dense bytes the card reads
  (2 -> 1.0625 B a value), taking ~10 ms off `gemm_kernel` a token.
- **VRAM.** It frees ~3.5 GB of VRAM for expert slots.

## Gate

**The exposure the quality row must cover** (`code`). The k/v exclusion
exists because q, k and v all compressed were fused horizontally and served
garbage (DESIGN 7.0.2ci). The 36 GDN layers have the same shape: four
MatMuls off one input (`gdn.py`: in_proj qkv, z, a, b), of which qkv and z
are converted and a and b stay plain. That is two compressed siblings beside
two plain ones, the class measured clean at depth 4 on 2026-09-26 (k/v
compressed with q plain). So the risk is bounded by that evidence but not
measured on this artifact. The owed quality row therefore starts with the
depth-4 logits A/B, the instrument that found the k/v defect.

On the served configuration above, decode of a fresh process's first answer
is faster than d48s2's, and the text is coherent. The answer is expected to
change: the plugin's compressed FC rounds differently from the f16 path. A
quality row against the reference (KLD) is owed at the next measurement
window, as the operator deferred.

## Scope — in / out

In: `tools/q4e/dense_q8.py` (recovery, chain, plan through `dense_u8.plan`
with a format), `--dense-q8` in the exporter (Q8_0 first, then `--dense-u8`
over the rest), the family gate lifted, cells. Out: the embeddings model;
the QSA indexer (not emitted); plugin changes.

## Status

- 2026-09-28. `dense_q8.py` and six cells, each red on a named mutant: 127
  never tried, k/v not kept, `skip` ignored, a zero-point chain. The u8 cells
  are unchanged and pass. After the first full-depth export, an early exit:
  a tensor that 127 leaves more than 1 % unresolved is not Q8_0. Walking all
  127 candidates over the Q6_K head cost that export ~30 min.
- 2026-09-28. **Full-depth export `d48q8`** (`measured-here`, the export
  log; admitted in the registry and allowlist):
  - 328 Q8_0 projections converted, 6.17 GiB as f16 -> 3.28 GiB, max
    relative deviation 4.9e-4.
  - The Q6_K output head via `--dense-u8`: 1.18 -> 0.67 GiB.
  - 217 kept, exactly the expected set: 48 f32 routers, 144 shared-expert
    matrices, 24 attention k/v, and the head (taken by the u8 pass).
  - lm `.bin` 72.17 -> 60.70 GiB; host peak 51.1 GiB, which needs the whole
    52 GiB container.
- 2026-09-28. **B60 window** (`measured-here`; ratio 78 + tier + dispatch,
  12e9 device pool, census112 seed with d48q8's own layer keys, host bank
  46 GiB, fresh process, 257 greedy tokens):

  | | first answer | second prompt | device weights+graph | dense gemm a token |
  |---|---|---|---|---|
  | d48s2 (0072 final build) | 38.90 s (38.18–39.66 over three runs) | 33.77 s | 19.23 GiB | 23.7 ms (31 µs a call) |
  | d48q8 | 36.56 s | 33.78 s | 15.82 GiB | 17.2 ms (22.5 µs a call) |
  | d48q8, ratio 75, 15.4e9 pool, census128 | 35.83 s (7.2 t/s) | 32.11 s (8.0 t/s) | 18.66 GiB | — |

  Device time per decode token is 33.3 -> 28.2 ms. The saving is real but
  small on the wall: at 22.5 µs a call the 761 dense calls a token look
  launch-bound rather than bandwidth-bound. The freed VRAM carries 128
  instead of 112 resident experts a layer. The text differs from d48s2's and
  reads coherent on both prompts (a read, not a quality gate). The shared
  expert now arrives as an f16 Constant + Convert (this is the family's first
  `--dense-fp16` export) and still fuses: device weights fell by exactly the
  dense saving. **Quality row (KLD against the reference)
  owed**, as the gate says.

- 2026-09-28. **The quality row's first half is closed, device-free.** The
  depth-4 logits A/B the gate names -- the instrument that found the k/v
  defect -- now runs on CPU: `tests/python/test_dense_q8.py::test_the_q8_chain_
  leaves_cpu_logits_bit_identical_to_the_plain_graph` builds a small decoder
  whose dense projection is in the checkpoint's own Q8_0 form, runs the plain
  graph and the pass-applied graph on CPU, and asserts the logits are
  **bit-identical** (the pass recovers `w = d*q` exactly in f32). Red first:
  a mutant that scales the recovered f16 by 1.001 makes it fail at max|d|
  5.44e-4; restored it passes. This gates the recovery's numerics, not the
  plugin's fused compressed FC -- the k/v class's garbage is a plugin
  behaviour a CPU cell cannot see. The **served** depth-4 KL/argmax row
  therefore stays the card row, and it is **blocked** on the f32 reference
  re-capture (`docs/campaigns/qsa.md` T8), not on this artifact.

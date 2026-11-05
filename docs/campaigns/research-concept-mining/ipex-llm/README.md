# IPEX-LLM, mined for arcint (2026-11-05)

## Measured on our cards, after the mining (`measured-here`, 2026-11-05, windows 92-94)

The wheel `bigdl_core_cpp` 2.7.0b20260110, run in a test container on the card host (compute runtime 26.35, oneAPI runtime libraries 2025.0.2 from PyPI, Level Zero loader 1.34, xe). arcint's rows in the same container, the same hour: B60 2328 / 80.8, A770 1951 / 70.6, within 0.5 % of its rows at runtime 26.27. Llama 2 7B Q4_0, llama-bench, 5 repetitions. `paper` rows are a different, older build on other machines; these do not settle what those did.

| Card, flash attention | pp512 | tg128 | mean KL, whole chunks | mean KL, one token a ubatch | 8 fact questions |
|---|---|---|---|---|---|
| B60, off | 2825.67 | 80.54 | 0.0163 | 0.0162 | 8 right |
| B60, on | 2830.14 | 78.06 | 0.0163 | **6.72** | gibberish |
| A770, off | 2603.55 | 82.85 | 0.0162 | **inf** | 6 right |
| A770, on | 2618.29 | 80.96 | 0.0162 | **inf** | gibberish |
| arcint's series, B60, on (llama-bench) | 2482.36 | 80.58 | 0.00043 | 0.00043 | 8 right |
| arcint's series, A770, on (llama-bench) | 2074.21 | 69.71 | 0.00043 | 0.00043 | 8 right |

What this changes in the text below:

- **Only prefill is a real lead.** A whole-chunk KL of 0.016 nats is inside the answer-level bar, and prefill leads by 14 % on the B60 and 26 % on the A770. Decode is not one: on the B60 the valid arm equals arcint's, and every faster-looking A770 decode row in this table comes with broken logits. On this host, with this wheel, IPEX-LLM's one-token decode is right only on the B60 without flash attention.
- **Its KL is 38 times ours.** The KV cache is 8-bit in every arm (`K (i8)`, `V (i8)` in its log), and the primitive below takes f16 weights and an f16 source. Which of them makes the 0.016 is not separated.
- **The oneDNN hypothesis stands untested.** The primitive is what candidate 1 says (`measured-here`: `matmul jit:gemm:any`, `src:f16 wei:f16 dst:f32`, 442 calls for a 512-token prompt), so the mechanism is read right. But `GGML_SYCL_DISABLE_DNN=1` does nothing in this build: the 442 calls stay and the rate stays. "The only mechanism that explains" below is an inference. `IPEX_LLM_SYCL_USE_MKL=1` does change the route: 2767 on the B60, 2675 on the A770, which says a vendor f16 GEMM of either kind gets there and does not say what stock SYCL's fp16 build (1891) does differently.
- **It cannot serve our models.** This wheel, the newest published, refuses the files arcint serves: `unknown model architecture: 'qwen35moe'`.
- Under unitrace a prefill is `gemm_kernel` plus `dequantize_q4_0_kernel`, attention small; a decode is `mlp_forward_q4_0_kernel`, `qlinear_xpu_kernel_q4_0_2x16_qkv` and the matvec, which confirms that the fusions run, and on the B60's valid arm they do not beat arcint.

Priced lever: the f16-operand GEMM on prefill, only on the answer-level bar, against the KL it costs. Everything else in the ranking below stays `paper` until priced.

How from here (operator, 2026-11-05): by profiling. The kernels are not disassembled further, and the PyTorch path's kernel packages, which declare no licence, are not read at all. oneDNN's GEMM is open source and can be read upstream.

Why: IPEX-LLM's llama.cpp build reports the best Llama 2 7B Q4_0 rows for arcint's two cards
(`paper`: B60 pp512 **2861.84** / tg128 82.24 with flash attention on, 2861.53 / 82.92 off; A770
up to 2564.25 / 72.54, 2416.16 / 82.92 off) against arcint 1.5.1 (2482.36 / 80.58, 2074.21 /
69.71). Flash attention on/off moves prefill 0.01 % — the attention is not the lead.

Method, four passes, each written as it went:

0. **Binary inventory** — `binary.md`. The wheel `bigdl_core_cpp-2.7.0b20260110` carries the fast
   path as `libggml-sycl.so` (not stripped), `libllama.so`, `libggml-base.so` and `llama-bench`
   (build string commit `3d3c767`). Read with `nm -C`, `objdump -d`, `strings`, a SPIR-V reader
   (48 modules, 524 entry points, ESIMD kernel families), and the graph-side `ggml_bigdl_*` ops.
1. **Commit classification** — `records.md`, 4,113 commits oldest first, one record a commit.
   Counts at score 0/1/2/3: **3059 / 914 / 131 / 9**.
2. **Ranking** — `candidates.md`, the top 10 with evidence, numbers, cost and experiment; the
   next 10 and the lowers recorded.
3. **Research** — `research.md`, the top 10's mechanism at file:line in the Python tree and
   symbol/address in the binary; closed-binary gaps named with what a card measurement would tell
   apart.
4. **Audit** — `audit.md`, the kept candidates against arcint's code (workspace `d4b6169`, llama.cpp
   at the pin with the series 0001-0098): what we have, how the build differs, quality class,
   units, the pricing measurement, an implementation sketch with red-first tests.

Evidence classes: IPEX's numbers are `paper`; the wheel and the Python tree are `code`; nothing
here is `measured-here`.

## The top 5, two lines each

1. **Prefill GEMM: dequantize-then-vendor-GEMM.** IPEX dequantizes the whole quantized weight
   matrix to f16 into scratch and calls oneDNN (oneMKL with `IPEX_LLM_SYCL_USE_MKL=1`); arcint's
   0096 dequantizes each tile inside its own XMX GEMM. This is the only mechanism that explains
   +15 % (B60) / +24 % (A770) on a dense Q4_0 prompt with attention neutral.
2. **QKV projection fused with norm and RoPE, combined wqkv weight.** A graph-level op, BF16 out,
   small-row q4_0 kernel (plain SYCL); arcint has the norm+RoPE half (0074, opt-in) but not the projection or a
   merged weight. A launch/intermediate lever at decode/verify first; its prefill behaviour is
   unread from the binary.
3. **MLP fused gate/up/SiLU (and a single-weight upgate form).** IPEX fuses it for one row in the
   Python path (`mlp_fusion_check` one-row gate) and in the llama.cpp op; arcint fuses only a
   q4_K GEMV, not q4_0 and not the prefill GEMM.
4. **Weight repack at load, card-shaped matvec geometries.** `reorder_qw_q4_0` at load and
   `<32,16,2|8>`/`<64,32,8>` DMMV instances; arcint reads raw q4_0 blocks with legacy shapes.
   Exact class; a decode bandwidth lever.
5. **q4_0 small-batch GEMM for 2-64 rows.** IPEX's own `vec_q4_0_batch_kernel` with f32
   activations, gated at ≤48/≤64 rows (and >1 row on BMG); arcint's few-column XMX stops at 8.
   Rounding-only; the verify/MTP shape.

## What IPEX-LLM's prefill lead is most likely made of

Most likely the **vendor GEMM on materialized f16 weights**: their default prompt matmul is
`ggml_sycl_op_mul_mat_sycl` → dequantize-to-f16 in a pooled scratch → oneDNN matmul (the read
dispatch and the three `dnnl::primitive::execute` sites, `binary.md` §5), not their own quantized
batch kernels and not MMQ. The fused QKV/MLP kernels read as small-row ESIMD shapes (decode
class), and the attention is flash-XMX on both sides with fa on/off equal in their own table, so
neither explains +15 %. The remaining difference is tiling/scheduling quality of the large
f16 GEMM.

The measurement that confirms it: IPEX's `llama-bench` on the B60 and A770 with
`GGML_SYCL_DISABLE_DNN=0` and `=1` — if the lead collapses or falls to arcint's row with the
vendor GEMM off, the lead is the vendor GEMM; `ONEDNN_VERBOSE=1` names the primitive's dtype
(f16 or BF16) and blocking in the same run. The commands are in the handover below.

## Card commands (Claude runs them; nothing here was executed)

Same-host comparison, one window, both cards, Llama 2 7B Q4_0 and the served 27B (paths are the
model files Claude has):

```
cd ~/research/ipex-llm/x/bigdl/cpp/libs/llama_cpp
export LD_LIBRARY_PATH=$PWD
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=1
# Llama 2, flash attention on and off, interleaved pairs
./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 128 -r 5 -fa 1,0
# the isolating arms, same model, same window
GGML_SYCL_DISABLE_DNN=1 ./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 128 -r 5 -fa 1
IPEX_LLM_SYCL_USE_MKL=1  ./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 128 -r 5 -fa 1
GGML_SYCL_DISABLE_GRAPH=1 ./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 128 -r 5 -fa 1
GGML_SYCL_DISABLE_OPT=1 ./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 128 -r 5 -fa 1
# the mechanism's identity in the same window: primitives, dtype, shapes
ONEDNN_VERBOSE=1 ./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 0 -r 1 -fa 1
# the 27B agent: prefill at 512 / 4096 / 20000, decode 128
./llama-bench -m <qwen3.8-27b-q8_0.gguf> -p 512,4096,20000 -n 128 -r 3 -fa 1
# a kernel-level split of the 512-token prompt (tool availability unverified):
unitrace -t -d -- ./llama-bench -m <llama2-7b-q4_0.gguf> -p 512 -n 0 -r 1 -fa 1
```

On the arcint side of the same window: the served `llama-bench` at the pin with the series on the
same GGUF and flags, and `tools/llama_scoreboard.py` for prefill/decode; then one profile of the
q4_0 GEMM's achieved GFLOPs against the card's XMX peak (`GGML_OPENCL_PROFILING`).

What the commands separate: `GGML_SYCL_DISABLE_DNN` = vendor GEMM vs IPEX's own kernels (the
prefill hypothesis); `ONEDNN_VERBOSE` = dtype/blocking; `GGML_SYCL_DISABLE_GRAPH` = graph-capture
share; `GGML_SYCL_DISABLE_OPT` = the in-backend optimisation — **not verified** to gate the bigdl
graph fusions, so read its delta with that caution.

## What could not be verified

- The oneDNN primitive's dtype and blocking (closed binary); the fused-op shape gate at prefill.
- `GGML_SYCL_DISABLE_OPT`'s relation to the `ggml_bigdl_*` graph fusions (no read path ties them).
- Whether `GGML_SYCL_DISABLE_DNN=1` routes to `batch_forward_*` in the llama.cpp binary (a
  negative not fully tried).
- The base commit `3d3c767`'s relation to our pin `3e4dfa8` (no network; the pin's source has ops
  the binary lacks).
- No number is stated by the commits for the mechanisms (only two source commits carry a
  measurable figure); the numbers are the benchmark report's.

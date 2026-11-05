# IPEX-LLM, mined for arcint (2026-11-05)

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

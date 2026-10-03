# llama-engine-kquant-kernels — Intel K-quant kernels for the libllama engine

**Closed 2026-10-03.** `contrib/llama.cpp/patches/0001`: on Intel Arc,
K-quant matrix-vector products on integer dot products (1 to 4 columns, a
two-term int8 activation), an fp16 DPAS GEMM on the XMX units for `MUL_MAT`
and `MUL_MAT_ID`, K-quant `MUL_MAT_ID` on the device, and the placement fix
(no K-quant tensor on the CPU through the stock Adreno `supports_op` rule).
The record, with every number, is `contrib/llama.cpp/README.md` and
`docs/llama-engine.md`.

Reference followed (`code`): llama.cpp's CUDA MMVQ (`vec_dot_q4_K_q8_1` and
siblings; the q8_1 activation, multi-column matvec) and its dequantize-then-
GEMM path; the DPAS operand layouts from the emulation in
bashbaug/SimpleOpenCLSamples `20_matrixexperiments-{i8,bf16}`.

Gate (`CLAUDE.md` answer-level bar, baseline arm ggml's float kernels on the
same card), result (`measured-here`, 2026-10-03):
- KL against the same GGUF on the CPU backend, prefill and decode, both
  models: within 0.0001 nats of the baseline arm or better; top-1 agreement
  within 0.02 points or better.
- Acceptance task: 10/10 at temperature 0 on both served models; sampled
  runs at temperature 0.7 in `docs/llama-engine.md`.
- Speed: Qwen3.8-27B on the B60 prefill 71 -> 418 t/s, decode 10.3 -> 18.0;
  the Qwen3.6 coder on the A770 prefill 111 -> 605, decode 7.7 -> 37.7.

Dropped on the way (`measured-here`): an int8-activation DPAS GEMM (CUDA
MMQ's shape; 20-30 TOPS) and a single-term q8_1 matvec moved the dense
model's KL by 0.0026-0.0028 nats and top-1 by 0.9 points; the fp16 GEMM and
the two-term activation replaced them at equal or better speed.

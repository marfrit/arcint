# Kernel selector reads (2026-08-29/30)

Why served nodes landed on reference kernels, read from the pinned runtime's
selectors (`code`, build `2026.4.0-22849-71640275d29`):

- the shared-expert gate's `M x 2048 : 2048 x 1` int8-by-u4 GEMM has no oneDNN
  catalog entry at M >= 128 (`ONEDNN_VERBOSE=dispatch`); `--gate-pad 16`
  widens the gate to a catalogued N (DESIGN §7.0.2g);
- `paged_causal_conv1d_ref` is the only registered implementation of its op;
- `generic_eltwise_ref` takes the `Add` nodes whose broadcast operand
  disqualifies `EltwiseKernel_vload8`;
- no permute node exists on the paged path.

Full history: `git show b0447b8:docs/kernel-selectors.md`.

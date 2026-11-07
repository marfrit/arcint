# The folded series

Patches 0001-0096 of the unfolded series were folded into the 12 patches
named `0000-NN-*.patch` (2026-11, operator decision: the patches need not stay
PR-shaped, since people mine the repository for concepts rather than read PRs).

The unfolded series is in git at commit `498f9d95` (`contrib/llama.cpp/patches/` there,
97 files up to 0096 plus 0097-0104). `0097` and everything after it are
unchanged and apply on top of the folded series as before.

**Identity.** pin `3e4dfa8` + the folded patches (applied with `git apply --whitespace=nowarn`
in sorted glob order, as the recipes do) is byte-identical to pin + old 0001-0096: same files, same
bytes, same modes, same new and deleted files. A folded patch is not a stand-alone change: it is the
diff between two intermediate trees of that fold, and the intermediate trees need not build.

**Numbering.** Every number 0001-0096 is cited as a label in the README, the campaign documents and
commit messages (about 1,800 places), and those citations were not rewritten. So the folded files do
not reuse those numbers: they are `0000-01` to `0000-12`, which sort before `0097` in the C and the
en_US collations alike. Numbers below 0097 in prose are history labels; this file resolves them.
The old numbers 0031, 0055, 0061, 0073, 0076, 0078 and 0085 were dropped or retired before the fold.

## Folded patches

| Folded patch | Old patches |
|---|---|
| `0000-01-kquant-integer-dpas.patch` | 0001, 0002, 0004, 0005, 0006, 0009, 0011, 0012, 0013, 0014 |
| `0000-02-lowbit-iq-types.patch` | 0021, 0022, 0023, 0024, 0027, 0030, 0034, 0035, 0036, 0053, 0054, 0058, 0096 |
| `0000-03-expert-cache-and-bank.patch` | 0032, 0042, 0043, 0044, 0046, 0056, 0063, 0075 |
| `0000-04-attention-at-depth.patch` | 0007, 0008, 0015, 0016, 0017, 0018, 0019, 0020, 0028, 0037, 0038, 0039, 0040, 0041, 0045, 0048, 0088, 0089 |
| `0000-05-image-speech-and-transform-ops.patch` | 0059, 0060, 0062, 0095 |
| `0000-06-graph-launches-and-transfers.patch` | 0025, 0047, 0049, 0065, 0072, 0079 |
| `0000-07-gdn-and-fusions.patch` | 0003, 0050, 0051, 0057, 0066, 0074, 0081, 0082, 0086, 0087, 0090, 0091 |
| `0000-08-kv-windows-and-slot-loan.patch` | 0026, 0029, 0052, 0068 |
| `0000-09-moe-routing-gemm-tiles.patch` | 0033, 0064, 0071 |
| `0000-10-mtp-and-speculation.patch` | 0010, 0067, 0080, 0083, 0084, 0094 |
| `0000-11-expert-cache-diagnostics.patch` | 0069, 0070, 0077 |
| `0000-12-tests.patch` | 0092, 0093 |

## Old number to folded patch

| Old patch | Folded into |
|---|---|
| `0001-opencl-intel-kquant-integer-dot-and-xmx.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0002-opencl-intel-iq3xxs-iq4xs-iq4nl.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0003-opencl-intel-gated-delta-net.patch` | `0000-07-gdn-and-fusions.patch` |
| `0004-opencl-intel-decode-small-ops.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0005-opencl-intel-f32-few-row-matmul.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0006-opencl-intel-moe-tokens-as-dpas-a.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0007-opencl-intel-decode-attention-split.patch` | `0000-04-attention-at-depth.patch` |
| `0008-opencl-intel-prompt-attention-dpas.patch` | `0000-04-attention-at-depth.patch` |
| `0009-opencl-intel-few-column-kquant-xmx.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0010-mtp-masked-nextn-rows-of-the-outputs.patch` | `0000-10-mtp-and-speculation.patch` |
| `0011-opencl-intel-kquant-gemm-2d-block-loads.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0012-opencl-intel-kquant-gemm-2d-order.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0013-opencl-intel-kquant-gemm-packed-decode-int8.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0014-opencl-intel-searched-tiles.patch` | `0000-01-kquant-integer-dpas.patch` |
| `0015-opencl-intel-quantized-kv-attention.patch` | `0000-04-attention-at-depth.patch` |
| `0016-opencl-intel-gqa-decode-attention.patch` | `0000-04-attention-at-depth.patch` |
| `0017-opencl-intel-head-size-128-xmx-attention.patch` | `0000-04-attention-at-depth.patch` |
| `0018-opencl-intel-gqa-decode-searched.patch` | `0000-04-attention-at-depth.patch` |
| `0019-opencl-intel-attention-agent-searched.patch` | `0000-04-attention-at-depth.patch` |
| `0020-opencl-intel-a770-attention-gemm-agent-searched.patch` | `0000-04-attention-at-depth.patch` |
| `0021-expert-cache-slots-and-usm-bank.patch` | `0000-02-lowbit-iq-types.patch` |
| `0022-opencl-intel-low-bit-iq-and-f16-small-n.patch` | `0000-02-lowbit-iq-types.patch` |
| `0023-opencl-one-context-per-platform-multi-device.patch` | `0000-02-lowbit-iq-types.patch` |
| `0024-expert-cache-per-device.patch` | `0000-02-lowbit-iq-types.patch` |
| `0025-opencl-strided-rows-f32-copy.patch` | `0000-06-graph-launches-and-transfers.patch` |
| `0026-kv-cache-cell-windows.patch` | `0000-08-kv-windows-and-slot-loan.patch` |
| `0027-opencl-kq-cache-stack-temporary.patch` | `0000-02-lowbit-iq-types.patch` |
| `0028-opencl-fa-row-blocks.patch` | `0000-04-attention-at-depth.patch` |
| `0029-kv-windows-per-sequence-attention.patch` | `0000-08-kv-windows-and-slot-loan.patch` |
| `0030-opencl-iq2-xs.patch` | `0000-02-lowbit-iq-types.patch` |
| `0032-opencl-bank-stream.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0033-opencl-b60-mm-id-tile.patch` | `0000-09-moe-routing-gemm-tiles.patch` |
| `0034-opencl-iq-swar.patch` | `0000-02-lowbit-iq-types.patch` |
| `0035-opencl-iq4-local-codebook.patch` | `0000-02-lowbit-iq-types.patch` |
| `0036-opencl-get-rows-q8-0.patch` | `0000-02-lowbit-iq-types.patch` |
| `0037-opencl-fa-dpas-f16kv.patch` | `0000-04-attention-at-depth.patch` |
| `0038-opencl-fa-dpas2.patch` | `0000-04-attention-at-depth.patch` |
| `0039-opencl-fa-sg8-q-global.patch` | `0000-04-attention-at-depth.patch` |
| `0040-opencl-fa-sg8-load8-words.patch` | `0000-04-attention-at-depth.patch` |
| `0041-opencl-fa-dpas2-q-pairs.patch` | `0000-04-attention-at-depth.patch` |
| `0042-expert-cache-nb-admission.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0043-opencl-cache-mm-id-min-tok.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0044-opencl-bank-stream-min-tok.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0045-opencl-fa-dpas2-split-kv.patch` | `0000-04-attention-at-depth.patch` |
| `0046-expert-cache-tables-one-write.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0047-input-copies.patch` | `0000-06-graph-launches-and-transfers.patch` |
| `0048-opencl-fa-sg8-v-pairs.patch` | `0000-04-attention-at-depth.patch` |
| `0049-decode-launches.patch` | `0000-06-graph-launches-and-transfers.patch` |
| `0050-gdn-snapshots-in-cache.patch` | `0000-07-gdn-and-fusions.patch` |
| `0051-gdn-gating.patch` | `0000-07-gdn-and-fusions.patch` |
| `0052-reserve-cells.patch` | `0000-08-kv-windows-and-slot-loan.patch` |
| `0053-opencl-lowbit-decode-spread.patch` | `0000-02-lowbit-iq-types.patch` |
| `0054-opencl-lowbit-gemm-decode.patch` | `0000-02-lowbit-iq-types.patch` |
| `0056-expert-cache-cpu-misses.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0057-opencl-intel-rope-apply-fused.patch` | `0000-07-gdn-and-fusions.patch` |
| `0058-opencl-intel-q8_0-xmx-gemm.patch` | `0000-02-lowbit-iq-types.patch` |
| `0059-opencl-intel-conv-3d.patch` | `0000-05-image-speech-and-transform-ops.patch` |
| `0060-opencl-intel-leaky-relu.patch` | `0000-05-image-speech-and-transform-ops.patch` |
| `0062-mtmd-qwen3tts-gen-valid-frames.patch` | `0000-05-image-speech-and-transform-ops.patch` |
| `0063-opencl-intel-cpu-miss-rows.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0064-opencl-intel-topk-moe-subgroup.patch` | `0000-09-moe-routing-gemm-tiles.patch` |
| `0065-opencl-intel-cpy-run.patch` | `0000-06-graph-launches-and-transfers.patch` |
| `0066-opencl-intel-gdn-tg-verify.patch` | `0000-07-gdn-and-fusions.patch` |
| `0067-llama-embeddings-nextn-skip.patch` | `0000-10-mtp-and-speculation.patch` |
| `0068-llama-prefill-slot-loan.patch` | `0000-08-kv-windows-and-slot-loan.patch` |
| `0069-expert-route-trace.patch` | `0000-11-expert-cache-diagnostics.patch` |
| `0070-expert-cache-save.patch` | `0000-11-expert-cache-diagnostics.patch` |
| `0071-opencl-intel-moe-int8-dpas.patch` | `0000-09-moe-routing-gemm-tiles.patch` |
| `0072-opencl-uniform-work-groups.patch` | `0000-06-graph-launches-and-transfers.patch` |
| `0074-opencl-intel-qk-norm-rope.patch` | `0000-07-gdn-and-fusions.patch` |
| `0075-opencl-intel-cpu-miss-prefetch.patch` | `0000-03-expert-cache-and-bank.patch` |
| `0077-opencl-alias-pages.patch` | `0000-11-expert-cache-diagnostics.patch` |
| `0079-opencl-dg2-leo-image-limit.patch` | `0000-06-graph-launches-and-transfers.patch` |
| `0080-llama-mtp-kv-only.patch` | `0000-10-mtp-and-speculation.patch` |
| `0081-llama-gdn-state-rows.patch` | `0000-07-gdn-and-fusions.patch` |
| `0082-llama-gdn-qk-l2-norm.patch` | `0000-07-gdn-and-fusions.patch` |
| `0083-round-trip-trim.patch` | `0000-10-mtp-and-speculation.patch` |
| `0084-llama-fast-topk.patch` | `0000-10-mtp-and-speculation.patch` |
| `0086-ggml-gdn-replay-op.patch` | `0000-07-gdn-and-fusions.patch` |
| `0087-gdn-replay-memory-opencl.patch` | `0000-07-gdn-and-fusions.patch` |
| `0088-opencl-gqa-int8-qk.patch` | `0000-04-attention-at-depth.patch` |
| `0089-opencl-gqa-staged-1-3-rows.patch` | `0000-04-attention-at-depth.patch` |
| `0090-opencl-gdn-rows-write-layout.patch` | `0000-07-gdn-and-fusions.patch` |
| `0091-replay-hybrid-input.patch` | `0000-07-gdn-and-fusions.patch` |
| `0092-tbo-replay-chain-pool.patch` | `0000-12-tests.patch` |
| `0093-tbo-gdn-rows-ids-set.patch` | `0000-12-tests.patch` |
| `0094-defaults-kv-only-sync-trim.patch` | `0000-10-mtp-and-speculation.patch` |
| `0095-opencl-fwht-hadamard.patch` | `0000-05-image-speech-and-transform-ops.patch` |
| `0096-opencl-intel-dense-q4_0.patch` | `0000-02-lowbit-iq-types.patch` |

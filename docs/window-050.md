# window-050 — the 0.5.0 window: measured constants and coherence rows

The 0.5.0 window (2026-09-12/13) for Qwen3.8 Flash-Next, closed by the 0.5.0
mechanism release (2026-09-13); the Paris row moved to 0.5.1
(`docs/window-051.md`, row B3). Below are its measured rows only.

Markers: `RUN@<sha>` = executed on the tree `<sha>`; `RUN@wt+<sha>` = executed
on a working tree at `<sha>` with uncommitted deltas; `RUN@unrecorded` = tree
never recorded (none remain); `DRY` = exercised in a no-op form; `UNTESTED` =
never executed. `tests/python/test_window_manifest.py` checks that every marker
below names a commit. Evidence class of every row: `measured-here` unless the
row says otherwise.

---

## Measured constants

| term | value | provenance |
|---|---|---|
| card under test | A770, 15.11 GiB reported (16,225,243,136 B) | `RUN@e78812d` GPU.1 enumeration |
| alternate card | B60, 22.71 GiB reported (24,385,683,456 B) | `RUN@e78812d` GPU.0 enumeration |
| CARD-tier weights at f32 | 18.492 GiB | `RUN@wt+2e99661` size ledger |
| → fits A770 reserve | no, −3.492 GiB | `RUN@wt+2e99661` size ledger |
| → fits B60 | yes, +4.218 GiB | `RUN@wt+2e99661` size ledger |
| offload tier | 450.000 GiB f32 (48 × 9.375) | `RUN@wt+2e99661` size ledger, file-sourced |
| host-mmap tier | 190.736 GiB f32 n-gram table | `RUN@wt+2e99661` size ledger, file-sourced |
| f32 : quantized ratio | 7.72× | `RUN@wt+2e99661` size ledger vs WP6b 85.38 GiB |
| GPU inference precision | f32, pinned explicitly | `RUN@829a213` |
| GPU.1 max single allocation | 4,294,959,104 B (4.00 GiB) | `RUN@be57428` |
| GPU.0 max single allocation | 24,385,683,456 B (whole VRAM) | `RUN@be57428` |
| serving-shape boot, 1 layer | compiles + infers on both cards | `RUN@be57428` |
| → GPU.1 compile / infer | 14.02 s / 0.400 s, 7.63 GiB host | `RUN@be57428` |
| → GPU.0 compile / infer | 10.48 s / 0.629 s, 7.57 GiB host | `RUN@be57428` |
| → CPU host cost, same graph | 28.50 GiB (3.7× the GPU path) | `RUN@be57428` |
| QSA→dense price, T ≤ 2051 | 0.0, exact (boundary derived, not the budget 2048) | `RUN@692c0a6` attention piece |
| QSA→dense price, T = 2052 | 2.307817e-06 over 1/2052 rows | `RUN@692c0a6` attention piece |
| QSA→dense price, T = 2080 | 2.385560e-02 over 29/2080 rows = T−2051 | `RUN@692c0a6` attention piece |
| attention piece floor vs pin | 1.855e-07 (T=64), 1.535e-07 (T=96) | `RUN@e78812d` attention piece |
| serving-shape IR, 48 layers | 84,372 nodes, 36 GDN + 12 dense-causal | `RUN@198b736` `--serving-shape` |
| → declared constants | 183.07 GiB | `RUN@198b736` |
| → build cost | 6.43 s, 4.6 GiB RSS | `RUN@198b736` |
| expert body declared type | u4, rank-4 [E, out, groups, 128] | `RUN@198b736` contract test |
| per-expert int4 slice | 2,457,600 B (gate+up+down) | `flash_next_offload.h:45` (`code`) |
| → as the IR walk reads it | 4,915,200 B = exactly 2× (u4 ceiled to 1 B) | `RUN@198b736` |
| MoE router on GPU | scatter shape OK on both cards | `RUN@be57428` |
| GDN on GPU, batched emission | first bad row 65 at every T ≥ 66, both cards; fixed in the emitter | `RUN@be57428`, fix `RUN@61bd61a` |
| GPU acceptance doctrine | \|ov−r64\| ≤ 20 × \|r32−r64\| | `RUN@be57428` |
| 86k-node GPU compile | 204.90 s on B60, 86,143 nodes | `RUN@be57428` |
| → compile cost scaling | 0.55 ms/node to 29k, 2.38 ms/node at 86k | `RUN@be57428` |
| shipped expert bodies, census | IQ3_XXS 94, IQ4_NL 43, IQ4_XS 2, Q8_0 5 = 144 over 48 layers | `RUN@bd5f53c` `test_repack_route.py` |
| → shipped expert bytes on disk | 51.99 GiB (55,823,564,800 B) | same cell |
| → the same experts as emitted u4 | 56.25 GiB (60,397,977,600 B = 512 × 48 × 2,457,600) | `flash_next_offload.h:45` (`code`) |

## Coherence rows (the served path at depth 4)

| probe | served model | result | tokens |
|---|---|---|---|
| "The capital of France is" (ids `760,6511,314,9338,369`) | serving-shape IR at depth 4, zero weights, `RUN@806b76f`, A770 then B60 | structure witness: logits `(1, 5, 248320)`, finite, absmax 0, greedy id `0` on both cards | 1 |
| same | serving-shape IR at depth 4, real weights (Q3_K_XL shards; 95 dense tensors, 12 expert bodies, the IQ4_NL table as 7 ports), `RUN@2413cab`, A770 then B60 | logits finite, absmax 9.1495 (A770) / 9.1205 (B60); greedy id `5613` (`ramework`) on both cards | 1 |
| same | the served binary: `arcint --model qwen38-flash-next-d4-ov --ngram-gguf <shard 2>`, `RUN@wt+6743ffb`, A770 and B60 | greedy, 8 tokens, byte-identical across two A770 runs: `rameworkenessooter5ussionxigy引`; B60 shares the first four tokens; warm decode 38.2 / 37.9 t/s (A770), 47.1 t/s (B60) | 8 |

Depth 4 of 48 is a mechanism witness. The full-depth answer is row B3 of
`docs/window-051.md`.

Full history: `git show b0447b8:docs/window-050.md`.

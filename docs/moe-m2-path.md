# The MoE two-token verify forward (2026-08-30)

On the 35B MoE a two-token verify forward rebuilt the per-expert mask
subbuffers on every inference (20,480 `create_subbuffer` calls per forward).
Plugin patch 0003 (shipped since `+p1`) skips them below the batched-GEMV
threshold: verify forward 27.3 -> 18.1 ms, MoE host execute 8.91 -> 0.74 ms,
output byte-identical (B60, u8 KV, `measured-here`). With an int4 MTP head the
35B decodes 72.9 t/s with `--mtp on` against ~62 plain on a code prompt
(single prompt, `measured-here`).

Full history: `git show b0447b8:docs/moe-m2-path.md`.

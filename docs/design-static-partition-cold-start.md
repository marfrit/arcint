# Design note: the static partition's cold start

A cold process's slow first requests under the static partition are removed by
two pieces: `--fit-ledger-dir PATH` persists the admission fit terms (slot
pool, activation fit, chunk cap) so a restart skips the load-time probes, and
on a ledger hit a 128-token pre-warm forward on lane 0 fills the pinned expert
slots at load (`backend_ov.cpp`, end of `load_paged()`), before the first real
request. Measured on the A770 (35B int4, ratio 50, 8 GiB pool, u8 KV, one
lane, n_ctx 65,536, 64 greedy tokens, `measured-here`): the ledger-hit
process's first request decodes at 15.2 t/s against 18.4 warm (ratio 1.21,
gate at least 0.5), pre-warm 8.4 s, output identical across the two processes.

Full history: `git show b0447b8:docs/design-static-partition-cold-start.md`.

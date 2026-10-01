# Served-services benchmark (2026-09-24)

Ten served sessions of the two deployed services, the packaged 0.5.0 binary
(`+p17`) against the qfndev tip (plugin 0003–0049), one fresh process per arm,
32 greedy tokens after a 1 / 4,096 / 16,384-token prompt (`measured-here`):

| service | card | extension prefill t/s @ 4,096 / 16,384 | decode t/s @ 4,096 / 16,384 |
|---|---|---|---|
| coder `qwen3.6-27b-a3b-coder`, u8 KV, 98,304 ctx, 2 GiB prefix cache | A770 | 1,379 / 1,209 | 43.9 / 42.4 |
| dense agent `qwen3.8-27b`, `i8:u8` KV, MTP on, 122,880 ctx | B60 | 1,141 / 852 | 24.6 / 20.2 |

Configuration deltas from the same window: the coder at `--offload-ratio 75
--moe-cpu-tier` reads 13.4 / 15.7 t/s prefill and 7.2–7.3 t/s decode with a
~545 s boot; the dense agent with `--mtp off` reads 1,253 / 953 t/s prefill
and 21.4 / 21.3 t/s decode; `--paged-kv u8` in place of `i8:u8` is
rate-neutral.

Full history: `git show b0447b8:docs/benchmark-served-services.md`.

# M8 — asymmetric paged KV (built)

`--paged-kv KEY[:VALUE]` serves u8 keys with i4 values: plugin patches 0008
(`VALUE_CACHE_PRECISION`), 0009 (per-side kernel plan) and 0010 (per-side
decode kernels), with the u8:i4 mixed prefill stage on micro-SDPA from patch
0020 (`+p6`, at parity with u8). u8:i4 costs 8.8 KiB/token against u8's 11.3,
scores 10/10 on the coder, and auto-fits 171,392 tokens on the 16 GiB card
(`measured-here`, DESIGN §7.0.2at). Owed: cold/warm prefix-cache byte-identity
at u8:i4.

Full history: `git show b0447b8:docs/design-m8-asymmetric-kv.md`.

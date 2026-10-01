# u8i4-prefill-price — `--paged-kv u8:i4` prefills at u8's rate

**Closed 2026-09-05.** Patch 0020 lets micro-SDPA serve the u8-key / i4-value
mixed stage, reading the packed values in place (values stay four-bit in
VRAM). 16 GiB card, coder, chunk 128: u8:i4 459 vs u8 457 t/s at 37,707
tokens, 401 vs 398 at 71,727; Prüfstand 10/10 through the u8:i4 server
(`measured-here`). Packaged from `+p6`; DESIGN §7.0.2as. Prior art:
`research-kv-quantisation.md`.

Full history: `git show b0447b8:docs/campaigns/u8i4-prefill-price.md`.

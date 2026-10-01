# DFlash2 drafter (built)

The public block-diffusion draft head `incoai/Qwen3.8-27B-DFlash2` pairs with
arcint's int4 Qwen3.8-27B (offline acceptance length 3.39 code / 3.76 prose
against a shuffled-feature null of 1.23 / 1.10, `measured-here`). It is
exported by `tools/export_dflash.py` (`--compress int4` for the served head)
and served with `--dflash DIR` (`--dflash-device` parks it on the other card).
The head's residual stream is rescaled so it fits f16 exactly, and plugin patch
0014 plus `src/core/dflash_window.h` keep the 2,048-row state window drafting
past its edge. Served on the B60 (int4, u8 KV, greedy, 32k context): 44.8 t/s
at 3.13 tokens per verify cycle against 24.0 t/s plain and 33.0 with the MTP
head (`measured-here`). One arcint process per card: two processes on one card
wedge under load (`measured-here`).

Full history: `git show b0447b8:docs/dflash-pairing-probe.md`.

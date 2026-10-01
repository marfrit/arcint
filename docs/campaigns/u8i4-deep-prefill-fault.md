# u8i4-deep-prefill-fault — deep u8:i4 prefill without the out-of-resources fault

**Closed 2026-09-05 for the served pairing.** On the micro-SDPA path (patch
0020, `+p6`) a 118,454-token u8:i4 prefill runs on the 16 GiB card at every
chunk from 128 to 2,048 with free VRAM flat, and the depth ladder is green on
both cards at both precisions (`measured-here`, DESIGN §7.0.2at–§7.0.2av).
`src/exec/fit.h` charges no scratch term and caps the chunk at the measured
2,048 when `packed_values_mixed_stage_on_micro` holds. The pairings 0020 does
not admit (i4:i4, four-bit keys) keep the 128-token belt
(`prefill_chunk_cap_for_packed_values_ex`) and the scratch charge; the fault's
owner on that generic path is not attributed.

Full history: `git show b0447b8:docs/campaigns/u8i4-deep-prefill-fault.md`.

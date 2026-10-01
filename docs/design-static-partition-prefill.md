# Design note: static-partition-prefill — hybrid grouped-GEMM/host prefill

Under the static partition a prefill layer splits its routed batch by
residency: resident experts run through the device grouped GEMM, the rest
through the host CPU tier, in the same layer (plugin patch 0037; patch 0042
sizes the gather by the filled count, shipped in `marfrit-openvino` `+p18`).
`grouped_fallbacks` went from 400 to 0; on the 16 GiB card (35B int4, ratio
50, 8 GiB pool) tier-ON prefill reads 27.9 t/s against 87.2 tier OFF, decode
18.2 t/s; on the B60 the 35B serves at ratio 99 with the tier at 23.3 t/s
decode (`measured-here`). Prefill with host-resident experts is carried on by
`docs/campaigns/prefill-expert-streaming.md` (stream the missing experts to the
card and compute every expert there, as Strata and FreeToken do).

Full history: `git show b0447b8:docs/design-static-partition-prefill.md`.

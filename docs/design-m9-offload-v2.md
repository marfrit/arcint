# M9 — expert offload v2 as a plugin patch series

The OTD expert offload's slot pool lives in device memory with batched
asynchronous uploads: plugin patches 0004 (OTD perf counters: evictions,
staging bytes, device and host slot buffers), 0005 (device-resident slot pool
with a per-buffer fallback to host memory), 0006 (async batched slot uploads)
and 0007 (the redundant per-MoE-layer stream finish dropped), sized by the fit
pass's `expert_slot_bytes` and verified by its plateau probe. Measured on the
16 GiB card with the 35B (`measured-here`): 0.4 t/s unpatched at ratio 25 to
9.1 t/s at ratio 50 with an 8 GiB pool (16-token probe), 10.4 t/s on a
64-token probe.

Full history: `git show b0447b8:docs/design-m9-offload-v2.md`.

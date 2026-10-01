# static-partition-prefill — tier-ON prefill splits a layer's batch by residency

**Closed 2026-09-17.** Patch 0037 runs a layer's resident experts through the
device grouped kernels and the rest through the host tier in the same layer
(`grouped_fallbacks` 400 → 0); patch 0042 sizes the grouped gather by the
filled pair count. Shipped in `+p18` (`measured-here`, DESIGN §7.0.2bx). The
prefill-rate lever for the host-tier experts is `prefill-expert-streaming`.

Full history: `git show b0447b8:docs/campaigns/static-partition-prefill.md`.

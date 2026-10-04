# hybrid-expert-fetch — merged into expert-hot-set-lru

**Merged 2026-10-01.** The split of each layer's missed experts between the
link and the CPU tier is part of the adaptive expert cache in Strata, the
reference written for this model, so one campaign owns both, with the bank
pinning (`decision`, operator's architect, 2026-10-01). The reference, the
share formula, the gate and the state are in
[expert-hot-set-lru](expert-hot-set-lru.md); DESIGN §8.6.

Full history: `git show b0447b8:docs/campaigns/hybrid-expert-fetch.md`.

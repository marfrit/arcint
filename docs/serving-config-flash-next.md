# Flash-Next streaming serving config (WP7 dry-run)

`arcint --flash-next-offload-plan HIT` prints a device-free residency plan for
Flash-Next on the 16 GiB card at a measured per-layer cache hit rate `HIT`
(0..1) and exits ADMIT/REFUSE (`src/exec/flash_next_offload.h`; replay tool
`tools/expert_lru_replay.py`). The plan's projections are superseded by served
measurements: the model serves on the B60 at 6.6 t/s decode, ~61–65 t/s
prefill at 20–27k tokens (`measured-here`, d48q8, CPU tier), with the n-gram
table disk-staged per forward (`docs/campaigns/ple-disk-backend.md`). The
expert cache is `docs/design-expert-hot-set-lru.md`; the MTP head is in
`docs/design-qwen-flash-next.md`.

Full history: `git show b0447b8:docs/serving-config-flash-next.md`.

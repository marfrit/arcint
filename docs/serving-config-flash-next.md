# Flash-Next streaming serving config (WP7 dry-run)

`arcint --flash-next-offload-plan HIT` prints a device-free residency plan for
Flash-Next on the 16 GiB card at a measured per-layer cache hit rate `HIT`
(0..1) and exits ADMIT/REFUSE (`src/exec/flash_next_offload.h`; replay tool
`tools/expert_lru_replay.py`). The plan's projections are superseded by served
measurements. With plugin patch 0076 (runtime `+p26`) the model serves on the
B60 at 10.3 t/s decode on a 20,085-token needle and 65.5 t/s prefill
(`measured-here`, d48q8; 7.5 / 64.0 t/s without the adaptive cache), with
the n-gram table disk-staged per forward (`docs/campaigns/ple-disk-backend.md`).

The served configuration (B60, 24 GB card, a host with ~52 GiB):

    ARCINT_MOE_DEVICE_POOL_BYTES=15400000000   # slots in VRAM, not host memory
    MOE_CPU_TIER_SEED=<census128 seed>         # start placement
    MOE_CPU_BANK_BYTES=32212254720             # 30 GiB host bank
    MOE_CPU_BANK_SEED=<bank rank file>
    MOE_CPU_TIER_ADAPTIVE=1                    # the cache follows decode
    MOE_CPU_BANK_FIXED=1                       # misses read the page cache
    MOE_CPU_TIER_LOOKAHEAD=<routers file>      # tools/export_router_lookahead.py
    arcint --offload-ratio 75 --moe-cpu-tier --moe-per-expert-dispatch \
           --paged-kv u8 --prefill-chunk 2048 --ngram-gguf <table> ...

Without `ARCINT_MOE_DEVICE_POOL_BYTES` the slot pool sits in host memory and
every GPU "hit" crosses the link (adaptive then loses: 9.4 to 7.4 t/s). The
expert cache is `docs/campaigns/expert-hot-set-lru.md`; the MTP head is in
`docs/design-qwen-flash-next.md`.

Full history: `git show b0447b8:docs/serving-config-flash-next.md`.

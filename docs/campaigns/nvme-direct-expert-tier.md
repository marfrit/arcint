# nvme-direct-expert-tier — LISBON's byte path: NVMe DMA straight into VRAM at load

**Closed 2026-09-24 (LISBON-001 gate met).** Patches 0048/0049 fill the
pinned expert slots at load from an expert store on an ext4 partition through
arcwell (github.com/marfrit/arcwell): a caller-created xe VRAM BO exported as
a dma-buf, `AW_IOC_SUBMIT_BATCH`/`AW_IOC_BATCH_WAIT`, imported into OpenCL
(`moe/pinned_nvme_transport.hpp`, `moe/aw_uapi.h`; store writer
`tools/q4e/expert_store.py`, `tools/test_expert_store.py`). B60, depth 4,
ratio 86: cold TTFT 92.49 s against host-fed 99.68 s, boot RSS 3.70 GiB,
decode 4.1 vs 3.4 t/s, `via_host_bounce` 0 (`measured-here`,
`docs/window-053.md` rows 1–3). Owed: the arcwell arm's restart determinism
(arcwell is B60-only) and a full-depth store keyed to a served artifact.
Expert misses at serving time are served by the host bank
(`host-expert-bank`). Design note `docs/design-nvme-direct-expert-tier.md`.

Full history: `git show b0447b8:docs/campaigns/nvme-direct-expert-tier.md`.

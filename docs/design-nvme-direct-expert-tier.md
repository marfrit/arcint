# design-nvme-direct-expert-tier — LISBON's load-time pinned fill

The static partition's resident experts can be filled at load straight from an
NVMe expert store into xe VRAM buffer objects by arcwell
(`github.com/marfrit/arcwell`), with no host bounce: the store is written by
`tools/q4e/expert_store.py` (one plain extent per expert, the three u4 weight
matrices in device order; scales and zero points stay on the host path), the
schedule is plugin patch 0048 (`MOE_OTD_PINNED_NVME_FILL=1`, device-free twin
`src/exec/pinned_nvme_fill.h`) and the transport plus OpenCL import of the
dma-buf slots is patch 0049 (`tools/arcwell_bo_dma_proof.c` and
`tools/arcwell_cl_slot_proof.c` are the standalone proofs). Gate met at depth 4
on the B60 (ratio 86, `measured-here`): cold TTFT 92.5 s against 99.7 s
host-fed, 697,958,400 B moved with `via_host_bounce` 0, four batches in
flight, child peak RSS 3.70 GiB, decode 4.1 against 3.4 t/s; two cold boots of
the host-fed arm byte-identical on the A770. Owed: the arcwell arm's
restart reading (arcwell runs on the B60 only) and the full-depth store.

Full history: `git show b0447b8:docs/design-nvme-direct-expert-tier.md`.

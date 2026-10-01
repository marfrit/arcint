# ple-disk-backend — the n-gram table staged per forward from disk, as FreeToken does

**Closed 2026-09-28.** An artifact exported with `--ngram-staging-rows N`
(`tools/export_serving_artifact.py`) declares one bounded `ngram_table.0`
port, and `bind_ngram_ports` fills it per forward with `pread` of the named
rows (`src/exec/ngram_staging.h`, cells `tests/test_ngram_staging.cpp`;
`ngram_staging_bytes` in `src/exec/fit.h`), the shape of FreeToken's default
disk backend (`code`: `~/src/FreeToken-ref/python/freetoken/models/qwen4_exp/ple_disk.py`,
`python/freetoken/engine/config.py:32`). The 26.82 GiB USM-host pin becomes a
2.884 MiB staging buffer, the 37.5 s load copy is gone, and the full-depth
staged artifact (`d48s`) decodes 2.2–2.8x faster than the pinned one (`d48n`)
on the A770 (`measured-here`,
DESIGN §7.0.2cg, §7.0.2cz). Owed: a layer key for the static partition that
does not move with the export's `.bin` layout (today the layer's `weight_0`
offset, patches 0013/0018).

Full history: `git show b0447b8:docs/campaigns/ple-disk-backend.md`.

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

**2026-10-02: Strata's reader.** The staged rows are read the way Strata
reads its PLE table (`code`: `~/src/Strata-ref/src/ngram/ple_reader.cpp`,
`src/platform/direct_file.cpp`): `O_DIRECT` page reads on 16 I/O threads,
rows that share a page read once, a 1M-row set-associative row cache, the
SSD keep-alive, and the rows issued at the top of the forward and collected
just before its infer (`src/exec/ngram_reader.{h,cpp}`;
`ARCINT_NGRAM_READER=0` keeps the synchronous `pread`). Cells: the reader's
bytes are the `pread`'s (edges, duplicate, the file's last row), a repeated
forward comes from the row cache, rows on one page are one read; a mutant
that shifts the row inside its page fails. Served (`measured-here`, B60,
d48q8, Flash-Next served configuration, one process per arm): 58–62 % of
rows from the row cache, collect waits 0.36–0.41 ms a forward; 500-token
decode 14.5–15.0 t/s against 14.7 without it, 2,076-token prefill 56.4–59.7
against 57.7 t/s, the needle right in every arm — within the run-to-run
spread. The table's random pages no longer go through the page cache the
CPU tier's file-tier experts live in.

Full history: `git show b0447b8:docs/campaigns/ple-disk-backend.md`.

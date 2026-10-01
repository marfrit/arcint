# design — the n-gram table as a per-forward disk staging buffer

Flash-Next's n-gram (PLE) table is read from disk per forward instead of being
pinned whole: the host computes each token's hashed row ids, `pread`s only
those rows into a bounded USM staging buffer and binds it to the
`ngram_table.K` ports, as FreeToken's default `DiskRowTable` does
(`docs/research-freetoken.md` §8). Code: `src/exec/ngram_staging.h`,
`backend_ov.cpp::bind_ngram_ports`, `tests/test_ngram_staging.cpp`; exported
with `tools/export_serving_artifact.py --ngram-staging-rows N`. Depth-4 gate
passed (`measured-here`): staged and pinned byte-identical, the 26.82 GiB of
USM host down to 2.884 MiB of staging, the 37.5 s load copy gone. At full
depth the staged `d48s` decodes 2.2–2.8x faster than `d48n` on the dispatch
route. Owed: the seeded dispatch-on d1/d512 digests and KLD window, a
plugin layer key independent of the export's `.bin` offsets, the tier-only twins'
KLD.

Full history: `git show b0447b8:docs/design-ple-disk-backend.md`.

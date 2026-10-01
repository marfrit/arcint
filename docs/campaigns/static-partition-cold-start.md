# static-partition-cold-start — a tier-ON process is warm from its first request

**Closed 2026-09-16.** `--fit-ledger-dir PATH` persists the plateau-probe and
activation-fit results per (artifact, device, flags, runtime) (`FitLedgerKey`,
`src/exec/fit.h`), so a matching start skips the load-time probe forwards; on
a ledger hit `load_paged()` runs one 128-token pre-warm forward that fills the
pinned expert slots. A770 reference cell: first / warm request decode 15.2 /
18.4 t/s (ratio 1.21, gate ≤ 2), outputs identical with and without the
ledger (`measured-here`). The tier-reference cell emits the cold metrics
(`decode-cold-1st-on/off`, `decode-cold-warm-ratio-on`). Design note
`docs/design-static-partition-cold-start.md`; DESIGN §7.0.2aq. Prior art:
`research-cold-start.md`.

Full history: `git show b0447b8:docs/campaigns/static-partition-cold-start.md`.

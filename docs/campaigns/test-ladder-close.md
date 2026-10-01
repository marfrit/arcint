# test-ladder-close — the acceptance references filled from the runners' own windows

**Closed 2026-09-05.** `tests/acceptance/run.py --all` runs the filled
enumeration (`tests/acceptance/cells.json`): `decode-warm-2nd-on` gates at
14.8 t/s, `decode-ratio-on-off` at 1.17, the large-card warm decode at 60 t/s;
nineteen references are report-only (`gate_at: null`). The first real run and
the fill windows are DESIGN §7.0.2aj, §7.0.2ak and §7.0.2al (`measured-here`).

Full history: `git show b0447b8:docs/campaigns/test-ladder-close.md`.

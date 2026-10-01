# The Prüfstand acceptance cell (built)

The acceptance target's external `pruefstand` cell reaches the operator's
harness wrapper through the run manifest's `pruefstand` key
(`-DARCINT_ACCEPTANCE_PRUEFSTAND`, written into `run_manifest.json`, never into
a tracked file). The wrapper prints `ACCEPTANCE-METRIC score <n> points`, and
`tests/acceptance/run.py` gates it at 10 like any other metric; an empty key is
a named skip. Campaign `pruefstand-cell-remote`, closed 2026-09-05.

Full history: `git show b0447b8:docs/design-pruefstand-cell.md`.

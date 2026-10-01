# pruefstand-cell-remote — the Prüfstand acceptance cell runs from the card window

**Closed 2026-09-05.** The run manifest carries a `pruefstand` key from the
CMake cache variable `ARCINT_ACCEPTANCE_PRUEFSTAND` (empty by default, nothing
host-shaped in a tracked file); the wrapper prints `ACCEPTANCE-METRIC score
<n> points` and the cell gates it at 10. First real run: 10/10 against the
deployed package (`measured-here`). Design note
`docs/design-pruefstand-cell.md`; DESIGN §7.0.2ao.

Full history: `git show b0447b8:docs/campaigns/pruefstand-cell-remote.md`.

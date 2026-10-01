# turnstile-wall-time — the turnstile test orders its threads by synchronisation

**Closed 2026-09-05.** `tests/test_turnstile.cpp` waits on
`Turnstile::issued()` (test-only, beside `served()`) instead of wall-clock
sleeps; `tests/roundtrip.sh` picks every server port with the kernel probe
(`free_port`) and polls instead of sleeping. 120 of 120 green at the failing
run density under build load (`measured-here`). DESIGN §7.0.2am, §7.0.2an.

Full history: `git show b0447b8:docs/campaigns/turnstile-wall-time.md`.

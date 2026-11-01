#!/usr/bin/env bash
# ops-2 item 6: in an ARCINT_TLS build a bad --tls-cert/--tls-key must be
# refused, and the refusal must come BEFORE the model load (the stub logs a
# "stub | n_ctx" line when it loads). A build without ARCINT_TLS refuses the
# flags at parse; this script then exits 77 (SKIP), so the default unit gate
# reads it as skipped and the ARCINT_TLS=ON gate arm runs the real assertion.
set -uo pipefail

BIN="${1:-}"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "usage: tls.sh /path/to/arcint" >&2
  exit 2
fi

out=$("$BIN" --stub --port 1 --tls-cert /nonexistent.pem --tls-key /nonexistent.key 2>&1)
rc=$?

if grep -q "ARCINT_TLS" <<<"$out"; then
  echo "tls: this build has no ARCINT_TLS (parse refuses the flags); the ON-arm check is skipped"
  exit 77
fi

fail=0
if [[ "$rc" == "0" ]]; then
  echo "  FAIL the bad-cert boot exited 0"
  fail=1
fi
if ! grep -q -- '--tls-cert' <<<"$out"; then
  echo "  FAIL the refusal does not name --tls-cert/--tls-key"
  fail=1
fi
if grep -q -E 'stub \| n_ctx|stub backend' <<<"$out"; then
  echo "  FAIL the model load ran before the TLS refusal"
  fail=1
fi

if [[ "$fail" == "0" ]]; then
  echo "tls: the bad certificate is refused before the model load"
fi
exit "$fail"

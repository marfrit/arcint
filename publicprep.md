# publicprep — checklist for taking arcint public on GitHub

Trigger state reached: speculative paged mode measured on the A770 —
MTP on 26.6 t/s @512 / 22.4 @4096 vs 18.1/17.7 paged-plain vs 17.3 stateful
baseline = 1.54× end to end, speculation alone 1.47×, depth collapse reduced
to a slope. That is the closing number of the launch story.

Publish under **marfrit** (consistent with the already-public B5 artifact on
HF that the allowlist references). Target: GitHub, public.

## 1. Fleet scrub (blocking)

- [ ] **CLAUDE.md must not ship as-is** — it contains home-infrastructure
      internals (power-plug AINs, netconsole listener addresses, nightly
      shutdown schedule, container paths). Replace with a public
      `DEVELOPMENT.md`: hardware needed (A770/B60, xe KMD, driver version),
      OV build dependency, how to run the suites. The session-rules content
      stays private.
- [ ] Sweep every file for fleet hostnames/paths and decide keep-or-trim
      deliberately: `grep -rniE 'dirac|bosch|boltzmann|hertz|pica|noether|fritz.box|192\.168\.' --include='*' .`
      Opinion docs mentioning "pica" or "dirac" are colour, not leaks — but
      the decision should be made once, on the list, not by default.
- [ ] `models/allowlist-raw.json` provenance comment mentions internal paths —
      trim to artifact facts.
- [ ] **Commit-trailer decision**: history carries Claude session URLs and
      co-author trailers. Either (a) squash to a curated initial commit
      (clean, loses the archaeology) or (b) keep history, strip/accept
      trailers via filter-repo (the honest-lab narrative argues for keeping
      the retraction commits — §7.0.1's value is its git history). Decide
      once; pushed history is public forever.
- [ ] Secrets sweep before the first push, from a clean clone:
      `grep -rniE 'token|passw|secret|Bearer|ssh-|BEGIN (RSA|OPENSSH)' .`
      plus a look at every file > nothing-should-be-binary.

## 2. Licensing (blocking)

- [ ] Choose Apache-2.0 (NInfer-compatible neighbourhood) or MIT; add LICENSE.
- [ ] third_party notices: cpp-httplib (MIT), nlohmann/json (MIT),
      minja (MIT) — keep their headers, add a THIRD_PARTY.md with versions.
- [ ] `tools/export_mtp.py`: one README sentence — the tool is repo-licensed,
      the IRs it emits derive from Qwen weights and inherit the Qwen
      Community License; arcint ships no weights.

## 3. Expectation management (README additions)

- [ ] NInfer-style scope statement is already there; add the operational
      caveats a stranger needs: OpenVINO dev-build pin (exact version/commit),
      GPU driver version measured against, Xe KMD only, **LAN use only — no
      auth, permissive CORS** (same warning class llama-server prints).
- [ ] Contribution policy, one paragraph: issues with measurements welcome;
      "support my GPU/model" requests out of scope by design; PRs must come
      with the relevant gate green and, for performance claims, a profile.
      (The additionalscope.md preamble, generalised: defects require a
      root-cause measurement.)
- [ ] State the measurement rules once: every throughput number in the tree
      names card, context depth, and config; correctness and throughput are
      separate columns, never one number.

## 4. Launch assets

- [ ] The report is the announcement: one page, NInfer-style — the matrix
      (stateful vs paged vs paged+MTP, both cards, depth 512/4096), the
      byte-equality gates that ran, the two upstream findings
      (#37607 device/path dependence, chunk non-exactness reproducer), and
      the retraction section referenced as a feature.
- [ ] Keep the outside reviews in-tree as `docs/reviews/`
      (secondopinion, thirdopinion, additionalscope + this file's final
      form) — a repo that publishes its own adversarial reviews is the
      differentiator; sanitise fleet colour per §1.
- [ ] Upstream link-backs on day one: the chunk-boundary reproducer to a new
      openvino.genai issue (cousin of #4367), the #37607 device/path table as
      a comment — arriving with evidence is the introduction.

## 5. Pre-flight (mechanical)

- [ ] Fresh-clone build on a machine that is not the dev box: cmake, ctest,
      roundtrip.sh, equivalence run documented with expected skips when no
      card is present.
- [ ] `-DARCINT_WERROR=ON` clean; ASan/UBSan run recorded (x86_64 note kept).
- [ ] Doc drift zeroed: README status table, llm.txt, DESIGN §3.5/§7 rows
      agree with each other and with the registry (the M4 row and the
      "provisional 7/10" entry were the known liars).
- [ ] Tag the launch commit; the version string in /props builds from it.

## 6. Explicit non-goals of going public

No roadmap promises, no Discord, no benchmarks-vs-everyone marketing table.
The repo argues by its gates. If attention arrives, the contribution policy
carries the load; if it does not, the repo still serves its two cards.

# Upstream sibling report: B60 GDN run-to-run nondeterminism (openvino#38099)

The B60 (`bmg-g21`, Xe2) run-to-run floor of the chunked GatedDeltaNet path was
reported upstream as a sibling of openvinotoolkit/openvino#38099
(`issuecomment-5751935449`), with the A770 (ACM-G10, Xe-HPG) as the
bit-identical control; the die-label correction was posted 2026-09-26
(`issuecomment-5843016598`). Nothing is owed on the thread. The defect itself
is tracked in `docs/design-served-prefill-determinism.md`.

Full history: `git show b0447b8:docs/upstream-38099-comment.md`.

# Parked patches — NOT part of the applied series

These are complete, reviewed work that is **not applied to the production
plugin** and must not be added to the numbered `patches/NNNN` series until
the blocker below is resolved. The production plugin (`marfrit-openvino
+p1`) does not carry them; nothing here has ever run outside test windows.

## 0008 / 0009 — asymmetric paged KV (u8 keys / i4 values)

The arcint-side contract for this feature shipped (see the M8 commit and
`docs/design-m8-asymmetric-kv.md`); these two patches are its plugin half.

- **0008** registers `VALUE_CACHE_PRECISION` and stops the pipeline
  collapsing key/value precision to one value. It also carries a genuine
  hardening fix (a positional model-cache-blob schema guard).
- **0009** declines micro-SDPA on a packing-class mismatch and documents
  why the decode fast path cannot serve u8:i4 without a kernel-source
  rewrite (one `is_i4_u4` boolean gates both operands).

**Blocker (DESIGN §7.0.2w):** building the plugin with 0008 present makes
plain `--paged-kv u8` serve deterministic garbage. A four-way bisect cleared
0004–0007, then a build of 0008 carrying only an unused env-gated debug
method served correctly — a layout-sensitive miscompile (latent UB the patch
perturbs) or a scratch-build staleness artifact. Resolving it starts with one
cold-from-scratch A/B (0001–0008 vs 0001–0007), not by editing source.

Because the reachable behaviour is asymmetric *refusal* (the kernels block
serving), this was parked rather than chased further, per the session's
measurement-cadence rule.

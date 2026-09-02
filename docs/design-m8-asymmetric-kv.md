# M8 — asymmetric paged KV (u8 keys, i4 values): design

## 0. What the i4 tax actually was (§7.0.3, read precisely)

- The u4 tax on the record is a **speed** tax on the **dequant path inside the PA kernel**, and it is **depth-dependent**: u4 is *faster* than f16 at 512/4096 (65.2 / 65.4 t/s) and 6% slower at 32768 (52.0 vs 55.4). Mechanism measured, not narrated: at 32k the profiled step puts `PagedAttentionExtension` at 58.6 ms (f16) vs **95.4 ms (u4), +63%**, every other kernel line identical to the tenth of a millisecond. Dequant work scales with cache length; the 4× bandwidth saving does not pay for it.
- Quality was **not** the tax. §7.0.3 has only greedy divergence at token 4–37 under u8/u4 vs f16 — a §3.2 near-tie numerics change, explicitly *not* evidence of degradation. So the milestone row's premise ("keys hold the rope structure and are the sensitive half") is an **unmeasured hypothesis**; the KLD probe tests it, and this design must not assume it.
- **The arithmetic that makes M8 worth doing:** the tax is per-dequantised-element over both operands. Taking i4 to one side halves the dequant surface → predicted ~3% at 32k against the ≤5% gate. That prediction is the first thing to try to falsify.

## 1. Property surface

**Choice: a second property, `VALUE_CACHE_PRECISION`** — config key string `"VALUE_CACHE_PRECISION"`; public decl if wanted `ov::intel_gpu::hint::value_cache_precision`, type `ov::element::Type`, default `ov::element::undefined` meaning "mirror KV_CACHE_PRECISION".

Rejected: overloading `KV_CACHE_PRECISION`. It is declared `ov::element::Type`, not string — encoding `"u8:i4"` means changing a public property's type, breaking every other consumer and the plugin's own validate list. A second property of the same type is additive, defaults to today's behaviour exactly, and keeps the diff at three lines in the one place that collapses the two.

`patches/0004-paged-kv-asymmetric-precision.patch` (PR-shaped; per §1.1 the upstream PR is filed first, the patch carried meanwhile):

- `execution_config.cpp` (~361-367): register `VALUE_CACHE_PRECISION`, default `undefined`; extend the existing allowed-set validation (f16/u8/i8/u4/i4) with `undefined`. One hunk.
- `transformations_pipeline.cpp:854-855`: `keyCachePrecision = kv_cache_precision` unchanged; `valueCachePrecision = value_cache_precision == undefined ? kv_cache_precision : value_cache_precision`. One hunk.
- Nothing else: `convert_pagedattn_inputs.hpp:28-29` already carries both fields and `convert_pagedattn_inputs.cpp:83-85` already sets both port element types from them. Smallest-sufficient-divergence jackpot — the struct was written for asymmetry; only the pipeline collapses it.

## 2. Kernel plan

`patches/0005-paged-attention-per-side-kv-dtype.patch`, kept **separate** because 0004 alone is testable and its failure is the red case (§4).

- **Decode — `paged_attention_opt.cpp:245-280`.** Already emits *two* JIT constants (`PACKED_K_HEAD_SIZE`, `PACKED_V_HEAD_SIZE`) but derives both from one `kv_cache_dt`. Hunk: read `k_dt`/`v_dt` from the respective `key_cache` / `value_cache` input layouts (correct post-0004) and give each side its own constant and compression flag. This is where the win is — the PA node is the line the u4 tax showed up on.
- **Prefill — `sdpa_gen_micro.cpp:1576-1585`.** `is_int4_kv_cache` steers `problem.Ta_ext` — the **A operand, keys only**. Nothing there parameterises V's external type. Named risk: with K u8 / V i4 the generator either asserts (benign, the selector falls back) or reads V as u8 (**garbage, silent**). First cut refuses the ambiguity: a selector guard declining micro-SDPA when `k_dt != v_dt`, so prefill runs the opt kernel — and **price that fallback**; §7.0.3's prefill half is the warning that a precision change costs up to 22% of prefill at 115k, growing with depth.
- **Open question the profile must answer, not assume:** whether the OCL source behind the opt path unpacks both operands through one shared macro. If it does, the divergence becomes a kernel-source rewrite → fallback (d).
- Both patch headers carry the ground-rule-2 fusion-impact profile in the header comment block, per the `patches/0002`/`0003` convention, including the negative result.

## 3. arcint flag surface

`--paged-kv <key>[:<value>]`, each side ∈ {f16, u8, i8, u4, i4}; no colon = both sides, so `u8` and `f16` mean exactly what they mean today.

- `config.h:139` keeps `paged_kv` a string; the pair is parsed, not stored twice.
- `config.cpp:268-270` parse, `443-444` validate: split on `:`, each side against the allowed set, error text naming the grammar.
- `backend_ov.cpp:2136-2141`: **today's `ARCINT_PAGED_KV` handling is a trap** — `std::string(env) == "f16" ? f16 : u8` maps every unrecognised value silently to u8, so `ARCINT_PAGED_KV=i4` already lies. The env override must use the same parser and refuse what it does not understand.
- `:2150`: `KV_CACHE_PRECISION` = key type; set `VALUE_CACHE_PRECISION` only when the sides differ, so an unpatched plugin still serves the symmetric cases.

**Pickup by `kv_bytes_token_` / M7 — three checks, not an assumption:**

1. `:2269-2292` computes `block_elems * port.get_element_type().size()`. `Type::size()` is `ceil(bits/8)` → **1 byte for i4**, a 2× over-count. Must become bitwidth arithmetic (`block_elems * bitwidth() / 8`); f16/u8 unaffected.
2. The 16-token page tripwire at `:2305-2317` checks `value_cache` shapes for a bare 16. An i4 V layout may pad differently and make it fire — a *good* loud failure; extend the condition for the i4 layout, never delete the tripwire.
3. M7 then follows automatically (`fterms.kv_bytes_token`, `shrink_n_ctx`, `status_.reservation.kv_bytes_per_token` all read `kv_bytes_token_`). The check is not the printed number: **re-run M7's own exit test** — load at the new printed max context, complete a generation at full depth, ≤2% promised-vs-actual.

**A row number that does not survive arithmetic.** u8 is 11.3 KiB/token and its halves are unequal (K pads the token dim 16→20, V pads the head dim). Halving V's contribution lands near −18% to −25%, not the row's **−36%**. `kv_bytes_token_` at load reports the truth for free, before any kernel work. If it lands short, correct the exit criterion with the layout arithmetic — do not restate the target.

## 4. Gates, and the red case first

Protocol: **the one §7.0.3 ran to adopt the u8 default on the C++ endpoint (2026-08-29)** — Prüfstand 10/10 on the coder at base depth *and* at the ~30k depth probe; decode A/B greedy, 100 tokens, one process per cell, warm prefix so the number is decode and not a mixture; the 512/4096/32768 sweep **plus the 53.5k point**, because §7.0.3's lesson is that a precision verdict from one depth is a property of a point. Card, depth, KV precision and lane count named per cell. B60, coder b5, one lane.

- Prüfstand 10/10 at u8-K/i4-V; decode within 5% of u8/u8 at 32k; KV bytes/token delta measured (with the caveat above).
- **Byte-exactness applies, but not against u8/u8** — M8 does not claim invisibility (u8 vs f16 already diverges at token 4–37). What must be byte-exact is the engine's own invariants *at* u8:i4: cold vs warm prefix cache, lane against lane.
- **KLD probe**: teacher-forced logits, fixed corpus, fixed depth, u8/u8 reference vs u8/i4; mean KL, p99, top-1 agreement. `tools/paged_driver.py` already reads `logits` (:101) — add a dump and a compare. **Null control: u8/u8 against u8/u8 in two processes must be ~0**; if it is not, the probe measures noise and no verdict may be read off it.
- **Red first, three layers:** (a) before the arcint change, `--paged-kv u8:i4` is refused by validate (`must be u8 or f16`); (b) arcint patched, plugin unpatched — the property is unknown or ignored, so add a **port audit** at load: log the actual `key_cache`/`value_cache` element types from the existing `kv_pool_types_` and **refuse to serve** when the requested asymmetry is not reflected in the ports (silently-ignored-and-still-u8 is the quiet-downgrade class this engine exists to refuse); (c) `kv_bytes_token_` at u8:i4 must print *below* u8:u8 — equal means check 1's `Type::size()` rounding.

## 5. Fusion-impact profile (ground rule 2 — the graph judges the patch)

Deoptimisation candidates: `kv_cache_compression.cpp`'s transformation may split a shared Convert/DynamicQuantize into per-side nodes; PA kernel selection may move (the micro-SDPA guard, §2); the padded V head dim changes page byte size, hence pool page count, hence the prefix-cache reserve (§7.0.3: everything that costs bytes comes out of the reserve, because the live side is a count and cannot absorb it).

Method: `ARCINT_PROFILE=1` on the **served** path, u8:u8 vs u8:i4, same card, same depth, one decode step and one prefill chunk. Compare (i) the launched-kernel class histogram in the `docs/kernel-selectors.md` format, (ii) the `PagedAttentionExtension` line, (iii) **every other line to the tenth of a millisecond** — the isolation that made the u4 tax a measurement instead of a story. Profile a **mid-sequence chunk, never chunk 0**: a past-0 chunk overstates every node share and has already produced two retracted headlines. Profiled runs are for shares, never for rates.

## 6. Fallback ladder

- **(a0) First, and it costs no code:** re-measure i4/i4 through today's single property at the same depths. It prices the tax M8 is trying to halve and gives the two endpoints the asymmetric point must sit between.
- **(a)** u8-K/i4-V via per-side dt in `paged_attention_opt.cpp` — the target; C++-only if the OCL source already unpacks per side.
- **(b)** If micro-SDPA cannot express the asymmetry: keep it disabled for `k_dt != v_dt`, ship on the decode gate, and **state the prefill cost** at the measured depths rather than omitting it.
- **(c)** If the decode gate fails, close with i4/i4 as the recorded reference point (memory ceiling, speed floor), the asymmetric point between them.
- **(d)** Defer if the OCL kernel hard-codes one KV quant type across both operands — a kernel-source rewrite is over the smallest-sufficient-divergence bar for a memory lever. Close with the blocking symbol and file named, and offer 0004 upstream regardless: it is correct and three lines wide either way.

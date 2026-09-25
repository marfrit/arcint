# window-051 — 0.5.1 BERLIN-AS-LISBON acceptance (the first 0.5.1 commit; every measured row EMPTY)

Recorded 2026-09-13, before any segmentation code exists in this repository
and before the 12-layer artifact exists. This file is the acceptance commit
of 0.5.1 in the form the roadmap's law demands: the criteria first, the
measured rows empty, the predictions dated, every clause falsifiable. The
commit that fills a row is a measurement commit and pastes the raw output.

The markers are `docs/window-050.md`'s (`RUN@<sha>`, `RUN@wt+<sha>`,
`RUN@unrecorded`, `DRY`, `UNTESTED`); a row with no marker is EMPTY, and
EMPTY is the honest state of every measured column in this commit.

## 0. What 0.5.1 is, in one sentence

All 48 layers of the Qwen3.8 Flash-Next serving-shape IR served on one card
by a **segmented forward** — K compiled sub-models, the hidden state carried
between them as ports, the expert bodies of one segment at a time streamed
from the GGUF mmap into a single host buffer set — so that the model's own
token for "The capital of France is" exists at full depth, on this host, with
its residency bounded by REAL host headroom. Paris ships inside it because the
answer IS the depth proof (operator CR #2, 2026-09-13).

> [AMENDED 2026-09-13, REVIEW MEDIUM-1: the refill source is the artifact's `expert_bodies.u8` blob, NOT the GGUF - the bodies are the filler's re-quantisation (contradiction recorded at d93508c). The GGUF-mmap wording stands as the pre-contradiction plan of record, marked not erased.]


Route (a) of the HANDOFF's change request, priced there; route (b), the
12-layer rung, is its first number and prices the depth ladder before the
segmentation lands.

## 1. The device facts this design is built WITH (measured; not re-litigated here)

| fact | value | where measured |
|---|---|---|
| A770 per-object cap | 4,294,959,104 B (4.00 GiB) | window-050 §4.4 |
| B60 per-object cap | its whole 24,385,683,456 B | window-050 §4.4 |
| prefill block cap under tiled MoE | 1,638 tokens (`[512, M, 2560]` tile object) | window-050 §4.7 |
| the plugin stages EVERY constant in USM-host memory at compile | measured on the A770: `drm-resident-gtt` to 11.8 GiB, `vram0` flat at 7.6 MiB, `CL_OUT_OF_HOST_MEMORY` at one expert body with the host at 2.6 GiB free | window-050 §4.10 F2 |
| cgroup / systemd memory fences do not charge driver / USM-host memory | the `MemoryMax=44G` fence never engaged while the physical host ran out | window-050 §4.10 F2; fleet mneme 372 |
| host RAM | 48 GiB, ~46 usable; the n-gram table takes 26.82 GiB of it as pinned USM host while bound | window-050 §4.8 R3 |
| the n-gram table | seven `ngram_table.K` USM-host PORTS (26.8 GiB, ~1.9 s bind), the IQ4_NL bytes decoded in-graph — ports, NOT constants | window-050 §4.8 |
| state constructs | slice with negative indices (the ShapeOf-swallow workaround, pinned by a cell) | window-050 §4.7 |
| GDN emission | perchunk (default) | window-050 §4.2 |
| QSA boundary | 2051; price 0.0 below, 2.385560e-02 over 29/2080 rows above | window-050 §8 |
| paged ports | 13, all from `SDPAToPagedAttention` (a pattern over emitted constructs; nothing to declare by hand); the pass asserts at least one `v13::ScaledDotProductAttention`, so a segment must hold at least one full-attention layer (index ≡ 3 mod 4) | window-050 §4.6, CHANGELOG 0.5.0 |
| served binary needs | `--ngram-gguf`; `position_ids` at port rank 1; `qwen_sparse_attention` counted as attention | CHANGELOG 0.5.0 |
| depth-4 served, both cards | warm decode 38 / 47 t/s (A770 / B60); first token 5613 `ramework` — MECHANISM, never an answer | window-050 §4.9 |
| full-depth constants | 1,030 dense f32 tensors 17,169,341,952 B (15.99 GiB); 144 u4 expert bodies 64,644,710,400 B (60.2 GiB), 448,921,600 B each; `.bin` 81,948,724,009 B; fill 82 min at 32.8 GiB peak host | window-050 §4.10 |
| the KLD reference | llama.cpp master `56b9eb28`, capture `af7993b7…`, n_ctx 2735, 684 rows below / 683 at-or-above 2051 per window, min clamped at max−16, uint16 reconstruction, bit-reproducible across runs | window-050 §4.11 |
| the served logits' own floor at depth 4 | KL(A‖B) 2.1e-4 nats mean (one window bit-identical, one with 41 of 1,367 argmaxes moved), a per-window event; mechanism: window-050 §4.11 | window-050 §4.11 |
| the bar | 0.0599 nats, **PROVISIONAL** = 1.5 × another model's R0 (window-050 §7); re-derived in row (d) below | window-050 §7 |

## 2. The route, as this commit fixes it (design, not measurement)

- **K sub-models** over the 48 layers, each a serving-shape IR of its own
  layer range with at least one full-attention layer inside it; the
  candidate cut is 4 × 12 (3 SDPA each); 6 × 8 and 12 × 4 are the fallbacks
  the arithmetic in row (a) prices, and the measured 12-layer staging (row
  (b), WP3) decides between them.
- **The hidden state** `[1, T, 10240]` (the hc-width residual stream) is an
  output port of segment k and an input port of segment k+1; segment 0 takes
  `inputs_embeds`, the last segment carries the final mixer and the head.
  Rope positions are recomputed per segment from `position_ids` (the tables
  are shared constants of 134 MB); the GDN / conv state tables and the KV
  pools are per segment, bound once per lane, exactly as the single model's
  are today.
- **Expert bodies as ports**: u8-packed USM-host ports (a u4 port is
  converted and copied to the device by the plugin; u8 shares). The MoE
  fusion matches a Constant, not a Parameter, so every expert computes for
  every token inside a segment: ~5 GFLOP / token / layer, "slow" is the
  accepted regime of 0.5.1 (Venice buys speed on top of a model that answers).

> [CORRECTED 2026-09-15 — this bullet IS the deviation, now measured and sourced.
> (a) MEASURED (window-051 B.3): the port route costs **7.06x the device residency
> per layer** and **623x the warm forward** against the same graph with the bodies
> as Constants; "slow" is 9.15 s for ONE token at 4 layers, ~110 s/token at 48.
> There is no model underneath for Venice to buy speed on top of.
> (b) SOURCED (`docs/research-freetoken-code.md`): the reference implementation
> never computes an unrouted expert — `ensure_experts(layer_id, expert_ids)` takes
> the router's own ids (`moe/offload_cache.py:843`) — and never materialises a
> dequantised weight (`:43,45,48`: dequant in the GEMM K-loop, inside the ggml
> kernels, or by Triton inline-dequant). Flash-Next activates **10 of 512** experts
> per token (`num_experts_per_tok: 10`), so "every expert computes for every token"
> is **51.2x** the work the architecture requires, by construction.
> The sentence stands as the decision that was taken; it was never the mandate.]

- **The runtime** drives the K compiled models per forward in order and
  refills ONE expert buffer set (the 15 GiB class at 12 layers) per segment
  from the GGUF mmap before that segment runs — so at most one segment's
  bodies are host-resident at any time, and the compile of a segment stages
  only that segment's dense constants.
- **Nothing is pinned by a cgroup**: the staging budget is REAL host
  headroom (`free`'s available minus the table minus the buffer set minus
  the process), measured before every compile, refused with the numbers.

> [AMENDED 2026-09-13, REVIEW MEDIUM-1: the refill source is the artifact's `expert_bodies.u8` blob, NOT the GGUF - the bodies are the filler's re-quantisation (contradiction recorded at d93508c). The GGUF-mmap wording stands as the pre-contradiction plan of record, marked not erased.]


## 3. Acceptance rows — EMPTY at this commit

### (a) Per-segment staging arithmetic against REAL host headroom

Per-layer terms from the measured full-depth artifact: dense f32 per layer
= (17,169,341,952 − 2,542,796,800 head) / 48 ≈ 304.7 MB (the PLE and final
mixer ride in the first and last segment; the head, 2.54 GB f32, in the last);
expert bodies per layer = 3 × 448,921,600 B = 1,346,764,800 B.

| cut | layers / segment | dense f32 staged per compile (predicted) | expert buffer set, host (predicted) | table, host | predicted host peak while compiling one segment (table + buffer + staging + ~3 GiB process) | measured peak | fits ~46 GiB? (predicted → measured) |
|---|---|---|---|---|---|---|---|
| 4 × 12 | 12 | 3.66 GB (+2.54 GB head, last segment) | 16.16 GB (15.05 GiB) | 26.82 GiB | ≈ 26.8 + 15.1 + 3.4 (+2.4) + 3 = **48.3–50.7 GiB** — predicted **NOT to fit** unless the table leaves pinned host or the buffer set is halved | EMPTY | predicted no → EMPTY |
| 6 × 8 | 8 | 2.44 GB (+2.54 GB head) | 10.77 GB (10.03 GiB) | 26.82 GiB | ≈ 26.8 + 10.0 + 2.3 (+2.4) + 3 = **42.1–44.5 GiB** — predicted to fit with 1.5–4 GiB to spare | EMPTY | predicted yes → EMPTY |
| 12 × 4 | 4 | 1.22 GB (+2.54 GB head) | 5.39 GB (5.02 GiB) | 26.82 GiB | ≈ 26.8 + 5.0 + 1.1 (+2.4) + 3 = **35.9–38.3 GiB** | EMPTY | predicted yes → EMPTY |

The n-gram table stays PORTS (26.8 GiB pinned, shared by every segment that
carries the PLE layer — only segment 0 does); it is never re-emitted as a
constant, or segment 0 dies the way the 48-layer compile died. The cut is
DECIDED by the measured 12-layer staging in (b): if 4 × 12 measures over
the headroom, 6 × 8 is the cut and this table's row says so with the
number.

> [AMENDED 2026-09-14, B.1 — the artifact side of row (a) is now readable off disk, and it is still arithmetic, not a peak.
>
> `arcint --model <dir> --inspect-artifact` (0.5.1 B.1, commit 84b63b8) loads an artifact the same way the served path does and prints the contract; run against the real segmented export `/models/ov/qwen38-flash-next-seg12-ov` (`--layers 48 --segment-layers 12`, tree c00500f, filled 2026-09-13 22:09–23:39Z), it read:
>
> | quantity (artifact arithmetic, read off disk) | value |
> |---|---|
> | segments | 4 × 12, `attn 3 gdn 9` each, hidden boundary port 4 × 2560 = **10240** wide |
> | segment `.bin` bytes | 4,944,946,409 / 4,813,218,673 / 4,813,218,673 / 7,382,270,833 = **21,953,654,588** |
> | chain `arch_hash` | **32d3060ca30238d1** (the four segment xml digests in segment order; no file has this digest) |
> | expert bodies | `expert_bodies.u8` **60,397,977,600 B (56.25 GiB)**, 144 bodies, all 419,430,400 B |
> | one buffer set, every segment | 36 slots × 419,430,400 = **15,099,494,400 B (14.06 GiB)** |
> | per forward, the chain reads | the **whole blob: 56.25 GiB** (K refills of 14.06 GiB each) |
>
> Against the 48 GiB host: buffer set 14.06 GiB **plus** the table's 26.82 GiB pinned = **40.9 GiB before any dense staging or the process** — which is row (a)'s own 4 × 12 arithmetic coming out the same way from the other direction, and it is why C3 ("4 × 12 does NOT fit") is still the standing prediction rather than a worry.
>
> Every `measured` column in row (a) **stays EMPTY**: no compile of a segment has run beside the table, and no seconds appear in the instrument's output on purpose. C8's ~35 s/forward cold stays a clause until a card leg reads it.]

> [AMENDED 2026-09-15, B.3 — row (a) adds two terms that live in DIFFERENT POOLS, and it assumes an ORDER the export does not force. Both are now read off the artifact device-free; every measured column below stays EMPTY until a card leg fills it.
>
> **(i) The table is segment 0's alone.** The four segment xml files read with no device (`tools/boot_serving_shape.py --stage pass --artifact <segN>`, 2026-09-15, tree e8cb312):
>
> | segment | `ngram_table.*` ports | expert ports | `inputs_embeds` |
> |---|---|---|---|
> | segment0 | **7** (6 x 47,718,400 + 1 x 33,691,136 rows x 90 B = 28,800,138,240 B = **26.82 GiB**) | 36 x 419,430,400 B | `[1, -1, 2560]` |
> | segment1 | **0** | 36 x 419,430,400 B | `[1, -1, 10240]` |
> | segment2 | **0** | 36 x 419,430,400 B | `[1, -1, 10240]` |
> | segment3 | **0** | 36 x 419,430,400 B | `[1, -1, 10240]` |
>
> The PLE rides in segment 0, so only segment 0 declares the table. The table is bound to a REQUEST, and a request exists only after its own model is compiled. In the serving order — compile every segment, create the requests, then bind — **no compile ever runs beside a bound table**. Row (a)'s peak column adds the table to a segment's staging; that co-residency is a property of the ORDER, not of the artifact, and the order is free.
>
> **(ii) "the 48 GiB host" is two budgets.** The table is USM host, and window-050 §4.10 F2 measured that driver / USM-host memory is NOT charged to the container's cgroup — the `MemoryMax=44G` fence never engaged while the PHYSICAL host ran out (fleet mneme 372). The heap buffer set is ordinary process pages and IS charged. Read on the dev host 2026-09-15: the container's limit is `memory: 49152` MB (**48 GiB**, the number §1 records); the PHYSICAL host's `MemTotal` is **65,765,352 kB = 62.7 GiB**. Row (a)'s `26.8 + 15.1 + 3.4 + 3 = 48.3–50.7 GiB` compares a sum whose largest term is charged to the 62.7 GiB pool against the 48 GiB one. The two columns are not addable and this file will not add them again.
>
> **The predictions B.3 measures (stated before the card leg, falsifiable):**
>
> | # | prediction | dies if |
> |---|---|---|
> | P1 | in `compile-then-table`, no compile's peak carries table bytes; a segment's staging is dense-only (its `.bin` is 4.81–7.38 GB, the experts being ports) | a compile's peak host GTT exceeds its own dense bytes by the table's order of magnitude |
> | P2 | the table's 26.82 GiB does NOT appear in the process's RSS or the cgroup's `MemAvailable`; it appears on the physical host | the container's RSS climbs by ~26.8 GiB when the table is allocated |
> | P3 | `--buffer-set heap`: the 14.06 GiB set DOES appear in RSS (cgroup), so cgroup peak ≈ set + process ≈ 17 GiB of 48 | RSS stays flat across the buffer-set allocation |
> | P4 | **4 x 12 FITS** — physical ≈ table 26.82 + set 14.06 + staging + process ≈ 44–48 GiB of 62.7, with the four segments' 20.45 GiB of dense on the card, not the host | any leg refuses with an allocation error, or the physical host's MemAvailable crosses the 4 GiB watchdog |
> | P5 | `--order table-then-compile` (row (a)'s own premise) is the ONLY order that can die, and if it dies it dies on the PHYSICAL host | it completes with room to spare, which would retire the premise rather than the order |
>
> **C3 is therefore predicted FALSE as written** ("4 x 12 does NOT fit"): not because the arithmetic is wrong but because its budget is the container's and its peak assumes an order the geometry does not force. C3 stands unerased until the card leg reads P1–P5; if P4 holds, the cut stays 4 x 12 and no re-export is owed.
>
> | leg (`tools/probe_segment_residency.py`) | order | buffer set | measured cgroup peak | measured physical peak | outcome |
> |---|---|---|---|---|---|
> | L1 | compile-then-table | heap | EMPTY | EMPTY | EMPTY |
> | L2 | compile-then-table | usm | EMPTY | EMPTY | EMPTY |
> | L3 | table-then-compile | heap | EMPTY | EMPTY | EMPTY |
> | L4 | compile-then-table, `--table skip` (control) | heap | EMPTY | EMPTY | EMPTY |
>
> No seconds and no peaks appear above on purpose: this is the prediction commit.]

> [MEASURED 2026-09-15 07:17–07:26Z, B.3, **GPU.0 = Intel Arc Pro B60 22.71 GiB**, tree `1bd2121` staged at `~/wp/wt-1bd2121` and byte-verified both ends (395 files, per-file sha256 list identical, chain `7983bdd9ebd1a79661474aa9ed68464710bba341cbe3ffad90c00ada48b5438b`), OpenVINO 2026.4.0-22849 from `~/openarc-venv`, both units inactive throughout, one process per leg, `pgrep`+SIGKILL after each.
>
> **THE CUT QUESTION WAS THE WRONG QUESTION.** No leg ever reached the table, the buffer set or a pool comparison, because **no 12-layer segment of this artifact compiles at all**:
>
> | leg | segment | table ports | outcome |
> |---|---|---|---|
> | L0 | segment0 | 7 | SIGKILL at 49 s, physical MemAvailable 43.3 → 3.5 GiB |
> | R1 | segment1 | **0** | SIGKILL at 58 s |
> | R2 | segment3 | **0** | SIGKILL at 64 s |
> | R3 | segment0 | 7 | SIGKILL at 53 s |
>
> The probe allocated NOTHING of its own in any of them (`--table skip --buffer-set none`). R1 and R2 carry no table port at all and died the same way, so **the table is not the cause** and the first hypothesis (a declared-but-unbound port costing its bytes) is not what this is either.
>
> **THE VARIABLE, ISOLATED** (`tools/boot_serving_shape.py --stage compile`, GPU.0, same graph, `--expert-ports` the only difference; `device_resident` is the plugin's own `GPU_MEMORY_STATISTICS` `usm_device + cl_mem`, the column A.1 used):
>
> | cell | layers | expert bodies as | declared | compile | peak host RSS | **device_resident** |
> |---|---|---|---|---|---|---|
> | c1 | 4 | constants | 13.51 GiB | 9.38 s | 9.75 GiB | **6.72 GiB** |
> | c2 | 4 | **PORTS** | 4.13 GiB | 8.78 s | 5.28 GiB | **39.38 GiB** |
> | c3 | 8 | constants | 24.37 GiB | 10.99 s | 16.08 GiB | **12.11 GiB** |
> | c4 | 8 | **PORTS** | 5.62 GiB | 16.96 s | 6.81 GiB | **77.44 GiB** |
>
> All four rc=0. The port route makes the graph SMALLER to declare (13.51 → 4.13 GiB) and its device residency LARGER by 5.9×.
>
> | route | slope | intercept | 12 layers | 48 layers |
> |---|---|---|---|---|
> | experts as constants | **1.347 GiB/layer** | 1.33 GiB | 17.50 GiB | **66.0 GiB** |
> | experts as PORTS | **9.515 GiB/layer** | 1.32 GiB | **115.50 GiB** | 458.0 GiB |
>
> **Ratio of slopes 7.06×.** Expert bytes per layer are 3 × 419,430,400 = 1.172 GiB, and the extra device cost per layer is 9.515 − 1.347 = 8.167 GiB = **6.97× the port bytes**. WHY it is ~7× is NOT measured here — an in-graph u4→f16 unpack would be 4×, the u8 source another 1×, and the rest is unaccounted. Named as unmeasured, not explained.
>
> **THE INSTRUMENT IS VALIDATED BY A.1.** c1 and c3 reproduce row (b′)'s c01 and c03 **exactly** — 6.72 and 12.11 GiB, same property, same host, a different session and a different leg script. And the constants slope gives 48 × 1.347 + 1.33 = **66.0 GiB**, which is row (b′)'s independently derived "the 48-layer set needs ~66–69 GiB" arriving from the other direction. The ports numbers are trustworthy for the same reason the constants numbers are.
>
> **WHAT THIS KILLS.** The largest segment that fits the B60's 22.71 GiB without spilling is **2.2 layers** on the port route (15.9 on the constants route). Every cut row (a) prices — 4 × 12, 6 × 8, 12 × 4 — is dead, and so is the premise that segmentation rescues the 48-layer model: **the port route is 7× worse per layer than the constants route it was introduced to escape.** A 12-layer segment needs 115.5 GiB of device residency; the three that died were each asking for that.
>
> **C3 ("4 × 12 does NOT fit") is CONFIRMED — and not for its own reason.** Its arithmetic is host staging against a container budget; the cause is device residency of the port route. Both the row and this amendment stand.
>
> **MY OWN PREDICTIONS, SCORED HONESTLY:**
>
> | # | verdict |
> |---|---|
> | P1 "a segment's staging is dense-only" | **DEAD**. ~50 GiB of shmem for a 4.94 GB `.bin`. |
> | P2 the table is absent from RSS / cgroup | **UNREACHED** — no leg reached the table. |
> | P3 heap buffer set shows in RSS | **UNREACHED** — no leg reached the buffer set. |
> | P4 "**4 × 12 FITS**" | **DEAD**, and it was the headline. |
> | P5 table-then-compile is the only order that can die | **DEAD**: every order dies, before the table is ever touched. |
>
> Two of five unreached and three dead. The two-pools reading of row (a) (container 48 GiB vs physical 62.7 GiB) is still correct as far as it goes, and it is now also **moot**: the binding constraint is the card, not either host pool.
>
> **A HOST FACT THAT RE-PRICES EVERY STAGING NUMBER IN THIS CAMPAIGN.** `data` is ZFS with **`arc_c_max = 40 GiB`** on a 62.7 GiB host, and **ARC is not counted in `MemAvailable`**. L0 began with MemAvailable 43.3 GiB against a warm ARC and spent its life racing ARC eviction; after the kill the host settled at 59.7 GiB available with ARC at 2.72 GiB. Rounds 2 and 3 ran against a cold ARC and sample ARC as its own column. A.1's staging numbers were taken without that column.
>
> **NOT MEASURED, and owed before the next cut is chosen:** whether BINDING the expert ports as USM-host tensors before the first inference changes the compile's device residency at all (the probe allocates its buffer set after the compile, and the compile is where the cost appears — so this instrument cannot answer it); the A770 leg (GPU.1 was untouched, reserved for the city); and the ~7× mechanism.]

> [MEASURED 2026-09-15 07:27–07:29Z, B.3 round 4 — **the question round 3 left owed is answered, and the answer is the bad one.** GPU.0 (B60), same tree, `boot_serving_shape.py --stage forward`, 4 layers, 8 tokens, zeros; `--expert-ports` the only difference between the pair.
>
> The design's premise (§2 here, and the n-gram table's precedent in window-050 §4.8) is that a **USM-host u8 port is SHARED with the graph, not copied to the device**. That premise is CONFIRMED on the host side and IRRELEVANT on the device side, in the same leg:
>
> | step | `usm_device` | `usm_host` |
> |---|---|---|
> | after compile (nothing bound) | **39.38 GiB** | 0.01 GiB |
> | after the n-gram table bound | 39.38 GiB | 26.83 GiB (+26.82, exactly the table) |
> | after the 12 expert ports bound | **39.38 GiB** | 31.52 GiB (+4.69, exactly the port bytes) |
> | after a real forward | **39.42 GiB** | 31.52 GiB |
>
> The plugin does share both the table and the expert bodies from host memory without copying them to the device — `usm_host` grows by exactly the declared bytes and `usm_device` does not move. **But `usm_device` was already 39.38 GiB when `compile_model` returned, before any tensor existed to bind.** Binding moves it by 0.04 GiB. The port route's device cost is committed at compile and no binding discipline can reduce it. The control at the same geometry sits at **6.72 GiB** through the identical sequence.
>
> **AND THE FORWARD IS 623× SLOWER**, warm (`--repeat 2`, so cold and warm are both read and the jit is separated):
>
> | 4 layers, 8 tokens, GPU.0 | experts as PORTS | experts as constants | ratio |
> |---|---|---|---|
> | forward #1 (cold, pays the jit) | 15.203 s | 0.097 s | 157× |
> | **forward #2 (warm)** | **11.211 s** | **0.018 s** | **623×** |
> | decode probe, 1 token after 8 | 9.152 s | 0.045 s | 203× |
>
> Risk R1 (the unpack materialisation) has its first number, and it is not a tax, it is the whole cost: **9.15 s for a ONE-TOKEN forward at 4 layers**. At 48 layers on the same slope that is ~110 s per token, before any of the 56.25 GiB blob is refilled.
>
> Both routes return digest `b31205cb6685` — **bit-identical, and that is not a determinism claim**: the bodies are zeros, so the output is all zeros (`absmax=0.0`) and bit-identity is trivial. What it does say is that the two routes are the same graph; the port route is not wrong, it is ruinous.
>
> **VERDICT ON THE PORT ROUTE:** dead on two independent axes measured in the same hour — 7.06× the device residency per layer (round 3) and 623× the warm forward (round 4). Segmentation was introduced to escape the constants route's 66 GiB staging; its own mechanism costs more on both. **0.5.1 needs a different mechanism for host-resident experts, not a different cut.** No re-export of `qwen38-flash-next-seg12-ov` at 6 × 8 or 12 × 4 is worth 82 minutes: the per-layer numbers above are what any cut multiplies.
>
> What is NOT measured: the ~7× and the 623× are both unexplained mechanisms (an in-graph u4→f16 unpack accounts for 4× of the first and none of the second by itself); whether a u8 port that skips the in-graph unpack (bodies pre-unpacked to f16 on the host, 2× the bytes) changes either number; and the A770 — **GPU.1 was never touched in this session**.]

The admission side, for the record (de48de5): the artifact is allowlisted as
`qwen3.8-flash-next-seg12`, pinned to that chain hash, and `serve_refusal_for`
(3e69176) refuses to SERVE it — the single-graph path would open segment 0
(layers 0..11) and answer under an entry that says 48. "Admitted" is a pin, not
a capability, until §2's runtime exists.

### (b) Depth ladder prices

| depth | fill (predicted → measured) | compile, served binary (predicted → measured) | device-resident after compile | warm decode t/s | per-forward NVMe reads (predicted → measured) | card |
|---|---|---|---|---|---|---|
| 4 | measured 556.8 s (window-050 §4.9) | measured: server up in 1 min 58 s on a quiet host | 6.79 GiB | 38.2 / 47.1 | 0 (all resident) | A770 / B60 |
| 12 | **~25 min predicted** → **measured 1,450.9 s (24.2 min)** build + 197.9 s save + 97.8 s hash, `RUN@d93508c` 2026-09-13 17:36–18:06Z; 265 dense tensors 5.88 GiB f32, 36 bodies 16,161,177,600 B, `.bin` 22,613,492,905 B (21.06 GiB), peak host 30.39 GiB; xml sha `7738fa87cddca8e2`, allowlisted `qwen3.8-flash-next-d12` — **C1 holds** | predicted minutes, ~19 GB staged in host → **measured: compiled on the B60 in 89.7–133.9 s** [AMENDED 2026-09-13, REVIEW LOW: the headline is the RANGE of the recorded compiles of this artifact — 89.7 s the first served process (`RUN@b4f593e`, 18:08Z), 128.0 s the second, 131.8 s the python driver, 133.0 s the reviewer's R3 served process, 133.9 s the reviewer's driver leg; the low end is the first process of the day and the generation rule (the compute runtime's kernel cache) is UNTESTED as its cause; the earlier headline "89.7 s" stands as written, marked] — **C2 holds**. The host-side staging did NOT reach the arithmetic's 19 GB: `fdinfo` `drm-resident-gtt` peaked at **4.22 GB** in 15-s samples during the compile while `drm-resident-vram0` went 2 MB → 19.9 GB → 21.8 GB within two minutes; the 48-layer compile (§4.10) had shown gtt to 11.8 GiB and vram0 flat at 7.6 MiB for six minutes. The two compiles behave differently, not just by size; row (a)'s "every constant staged in host at once" is therefore NOT what this rung measured, and the 48-layer refusal's mechanism is re-opened by this number (a 24-layer rung would say whether it is a per-graph reservation the plugin makes when the total exceeds the card) | **17.73 GiB** (plugin, both processes; predicted ≈ 17) + table 26.82 GiB USM host (bind 37.5 s) | **18.3–18.5 t/s** greedy decode (B60, chat 32 tokens 18.5, raw 64 tokens 18.3; first request 13.8 with the jit) against 47.1 at depth 4; prefill 2,735 tokens at chunk 512 **84.4 s (32.4 t/s)**, later windows 97–109 s under the 2.7 GB/window logits dump | 0 (resident) | B60 |
| 48, segmented (4 × 12 or the cut (a) decides) | measured 82 min (the artifact exists) | per segment: EMPTY | per segment: dense f16 ≈ 1.8 GiB + one segment's expert ports (host) + KV + state | EMPTY | route (a) estimate **60 GB / forward at 1.8 GB/s ≈ 35 s** (cold page cache) → EMPTY; warm-cache case: EMPTY | A770 or B60 |

**The 12-layer rung's France line — mechanism, never an answer (row (f)):**
`"The capital of France is"` → **` impkatanRGRIESIRES fiNDamespace`** (8
greedy tokens, byte-identical on the warm repeat; 64 tokens continue
`…lerotimo渐进triceELAClassLoaderHit…复制复制复制…`); chat form
`reasoning_content` likewise. `RUN@b4f593e`, B60, 18:10Z. 36 layers are
missing.

**The 12-layer rung's KLD (REPORT ONLY, red as at depth 4):** mean KL
below / at-or-above 2051: window 0 **11.93 / 11.63**, window 1 **11.89 /
11.83** (leg 1); 12.01 / 11.60 and 11.99 / 11.89 (leg 2); argmax agreement
0.0007 / 0.0000. **The floor at this rung is NOT the depth-4 floor**: leg 1's
two replays differed by KL(A‖B) **0.743 / 0.822** mean, argmax agreement
**0.118 / 0.111**, max |logit diff| 18.0; leg 2 (`--warmup 2 --repeat 2`,
four passes per window) every one of 6 window pairs moved, the counted
pair by 0.687 / 0.787, agreement 0.123 / 0.103, and the greedy tokens
walked (window 0: `int`, `页`, ` inde`, `atom`; window 1: `loat`, `loat`,
`arul`, `cito`). The prediction written before leg 2 (the counted pair
bit-identical after two warm-ups, window-050 §4.11.1's rule at depth 12)
**died**. The python driver on the same artifact, single 1,024-token
forwards on the same request, state zeroed (`RUN@b4f593e`, B60, 18:37Z):
forward #2 differs from #1 from row 32 (750 of 1,024 argmaxes moved, max
|diff| 13.9), #3 from #2 (569 moved), #4..#8 each from its predecessor
(673–804 moved; first differing row 14, 25 or 32 every time) — **per-forward,
not a one-time step**, rows 0–13 identical every time. On the A770 the same
driver at depth 4 was bit-identical over 12 forwards (window-050 §4.11.1).

**Localised, then attributed to the CARD (the cut bisect, `RUN@b4f593e`,
B60, 18:48–19:12Z, 3–4 forwards per cut):**

| cut (artifact, card) | forward-to-forward | first differing row | max \|diff\| (max \|#1\|) | argmaxes moved / 1,024 |
|---|---|---|---|---|
| `layer0/out` (d12, B60) | every forward differs | 32 | 2.44e-4 (2.54) | 0 — 1–3 rows touched |
| `ple/out` (d12, B60) | #2 = #3 ≠ #1 (one step) | 777 | 5.86e-3 (2.54) | 1 |
| `layer1/out` (d12, B60) | every forward differs | 14 / 264 | 8.1e-3 (3.16) | 0 |
| `layer2/out` (d12, B60) | every forward differs | 14 / 137 | 0.26–0.44 (32.6) | 3–11 |
| `layer3/out` (d12, B60) | every forward differs | 14 | 0.25–0.30 (32.2) | 6–15 |
| `layer7/out` (d12, B60) | every forward differs | 32 | 15.6–26.1 (85.1) | 85–136 |
| logits (d12, B60) | every forward differs | 14 / 25 / 32 | 12.7–16.9 (21.1) | 569–804 |
| **`layer2/out` (d4, B60)** | **every forward differs** | 14 | 0.11–0.36 (32.4) | 3–12 |
| **logits (d4, B60)** | **every forward differs** | 14 / 51 / 148 | 1.60–1.69 (17.1) | 33–54 |
| `layer2/out` (d4, **A770**, window-050 §4.11.1) | **bit-identical ×8** | — | 0 | 0 |
| logits (d4, **A770**, warm) | **bit-identical ×12** | — | 0 | 0 |

The expert bodies, zero-points, scales and rope tables of layers 0–2 are
byte-identical between the two artifacts (29 named constants, same
digests, checked device-free the same hour). The variable that separates
"bit-identical" from "every forward differs" is the card: **the B60 (GPU.0,
Xe2) runs this graph's first layer with a per-forward nondeterminism of one
f16 ulp in a few rows at or after row 32** (`layer0/out`: 1–3 of 1,024 rows,
2.44e-4 against 2.54), which twelve layers amplify to an 11 % argmax
agreement at the logits and four layers to 33–54 moved argmaxes; the A770
(GPU.1, Alchemist) runs the same bytes bit-identically. Which kernel inside
layer 0's block (short-conv, GDN core, hyper-connection, MoE) is not
localised — the cut names the block, and the kernel-level cut needs names
the emitter does not set yet. The reviewer's B60 leg of the ba2d5de review
(two identical replays differing from row 14 / 261) was this, not the A770's
one-time kernel step.

**Consequences.** (1) A KL floor is per card: on the A770 it is the settled
kernel set's zero with the one-time step named; on the B60 it is a
per-forward floor whose size grows with depth (0.74 nats mean at 12 layers)
and against which no bar is readable at 48 layers without either the
kernel named and fixed or the A770 as the measurement card. (2) The
12-layer rung's France token above is ONE draw of that floor on the B60,
byte-identical only within its own process's warm repeat. (3) Row (c)'s
cold-boot determinism is predicted to FAIL on the B60 as measured and to
hold on the A770; the row stays EMPTY until the segmented service exists,
and it names the card when it is filled. (4) The kernel-level localisation
is an open row: `--cut` at the emitter's next finer names inside layer 0,
B60, cold and warm.

> [AMENDED 2026-09-13, REVIEW HIGH-1: "settled" here and in the A770 floor row means the OBSERVED floor pair of THIS run, never a cache-state claim from any earlier run - see window-050 §4.11.1 review amendment.]


### (b′) THE STAGING-VS-DEPTH CURVE — A.1, measured 2026-09-13 21:13–21:41Z, B60 (GPU.0), `RUN@87d0ae5` (tools byte-identical to b689d6d), fresh process per cell

The question the monster session left open: depth 12 staged 4.2 GB host-side
in 15-s samples while depth 48 died at 37 GiB of shmem — 4× the layers, 9×
the staging. The instrument here: the python driver's `--stage compile` on
the sparse-arena (zeros) graph at --layers 4/8/12/16/20/24, the real d4 and
d12 artifacts as controls, and a sampler on the PHYSICAL host (root, every
2 s): the driver process's `fdinfo` (`drm-resident-gtt` / `-vram0` summed
over its DRM clients), its VmRSS/VmHWM, and the host's own MemAvailable /
Shmem / MemFree — never a cgroup counter (mneme 372).

| cell | layers | bytes | compile | device_resident (plugin GPU_MEMORY_STATISTICS) | peak gtt (host) during compile | vram0 after transfer | host Shmem peak | host MemFree min | outcome |
|---|---|---|---|---|---|---|---|---|---|
| c01 | 4 (zeros) | declared 13.51 GiB | 8.8 s (pass 2: 9.1 s) | 6.72 GiB | 8.4 GiB (pass 2) | 2.1 GiB sampled (transfer missed at 2 s) | 0.7 GiB | 38.9 GiB | OK |
| c02 | 4 (real d4) | .bin 9.43 GB | 42.9 s (pass 2: 57.0 s) | 6.79 GiB | 8.1 GiB (pass 2) | 2.2 GiB sampled | 1.1 GiB | 31.3 GiB | OK |
| c03 | 8 (zeros) | declared 24.37 GiB | 21.3 s (pass 2: 14.1 s) | 12.11 GiB | 7.8 GiB (pass 2, 9 samples — the plateau missed) | 10.1 GiB | 0.4 GiB | 18.7 GiB | OK |
| c04 | 12 (zeros) | declared 35.23 GiB | 40.9 s (pass 2: 21.3 s) | 17.51 GiB | **20.7 GiB** (pass 2) | 18.0 GiB | 0.7 GiB | 13.5 GiB | OK |
| c05 | 12 (real d12) | .bin 22.61 GB | 202.1 s (a concurrent CPU job of mine on the host, pass 2 alone: 180.6 s) | 17.73 GiB | **13.2 GiB** (pass 2; pass 1 caught 8.6 → 3.5 mid-transfer) | 17.8 GiB | 9.0–10.8 GiB | 11.4–15.6 GiB | OK |
| c06 | 16 (zeros) | ≈ 46 GiB declared | 57.2 s | **22.90 GiB** (over the 22.71 GiB the card reports) | **18.3 GiB** | **23.5 GiB** (the physical 24 GiB) | 10.97 GiB | 4.1 GiB | OK — fits the physical card, not the reported one |
| c07 | 20 (zeros) | ≈ 57 GiB declared | 55.1 s | **28.30 GiB** | **28.6 GiB** (= the whole compiled set, one sample before the transfer) | 24.4 GiB + **13.3 GiB left in host gtt** | 4.8 GiB | 0.26 GiB | OK — **SPILLED**: the compile completes with the card full and the rest host-resident |
| c08 | 24 (zeros) | ≈ 68 GiB declared | 62.0 s | **33.69 GiB** | **32.8 GiB** (plateau 12 s at 32.8 with vram0 5.7) | 24.4 GiB + **10.2 GiB in host gtt** | 9.9 GiB | 1.1 GiB | OK — spilled |

**What the sampler saw, every cell alike (three phases).** (1) The build and
the pass read the constants: the process RSS climbs to the arena's pages
(c06 24.5 GiB, c08 26.2 GiB — file-backed, reclaimable; the host's MemFree
falls to 0.26–4 GiB while MemAvailable stays 15–27 GiB); no DRM client holds
anything yet (gtt ≈ 0, vram0 16 MiB). (2) The compile accumulates the
converted constants in **host GTT** (`drm-resident-gtt`), ~1.4 GiB per layer,
up to a plateau that equals the compiled size (c07 28.6 GiB for 28.3 compiled;
c08 32.8 GiB with 5.7 GiB already on the card); the host's Shmem column
follows only part of it (peaks 4.8–11 GiB), so USM-host staging here is
mostly userptr in the process, and the physical host's shmem at the 48-layer
death (37.3 GiB) was a lower bound on the staging, not the whole of it.
(3) A transfer of ≤ 10 s moves the GTT set into VRAM — **up to the physical
24 GiB, past the 22.71 GiB the OpenCL device reports** (c06 23.5 GiB resident
and OK) — and what does not fit **stays in host GTT and the compile
COMPLETES** (c07: 13.3 GiB host-side, c08: 10.2 GiB; the plugin's statistics
count it as `usm_device`, hence 28.30 and 33.69 "device-resident" on a 24 GiB
card). The "UNTESTED alternative" of window-050 §4.10 F2 — that the driver
spills device allocations to system memory — is therefore MEASURED TRUE on
the B60 at 20 and 24 layers; its forward cost is not measured here (`--stage
compile`), and a spilled graph binds its n-gram table on top of the spill.

**The 9× was a sampling artefact, and the curve is linear.** The 12-layer
"4.22 GB peak" of row (b) came from 15-s samples that missed a transfer
which takes seconds; at 2 s the same rung's gtt peak is **13.2 GiB (real d12) / 20.7 GiB (zeros)** — the artifact and the graph agree to within the f32→f16 conversion's transient. Per
layer the staged GTT is 1.3–1.5 GiB (c04→c06→c07→c08: 20.7 / 18.3 / 28.6 / 32.8 GiB; the
c06 reading is a sample inside the transfer, c04 one above the compiled 17.5 — the conversion's transient), which is the compiled size —
f16 dense + u4 expert bodies — of the layers; extrapolated to 48 layers the
plateau is **~66–69 GiB of host GTT before the transfer starts**, against a
host of 48 GiB (62.7 physical, 48 for the container): that is the death
window-050 §4.10 F2 recorded (gtt 11.8 GiB and shmem 37.3 GiB at a page
allocation failure), now with its mechanism — the compile needs the WHOLE
constant set host-resident once, whatever the card.

**VERDICT (A.1).** Single-graph 48 layers does NOT reopen: the compile's
host-side staging is linear in depth at ≈ the compiled bytes and crosses the
host at ~30 layers (the 24-layer cell already drove MemFree to 1.1 GiB
with its 26 GiB of arena pages next to 32.8 GiB of GTT); the n-gram table's
26.8 GiB is bound after that and the spill would have to live beside it.
Segmentation is confirmed as the only route to 48 on this host. Two things
re-price WP-B: (i) a SEGMENT may be compiled larger than the card — the
plugin spills and completes — so the cut is priced by host GTT at compile
(segment compiled bytes ≤ host headroom − table − buffer set), not by VRAM;
(ii) the per-layer GTT term measured here (1.3–1.5 GiB) is the
SINGLE-graph term — u4 bodies plus f16 dense; a segment of the WP4.1 emitter
carries its bodies as u8 PORTS, so its compile stages the dense f16 term
only (row (a)'s 3.66 GB at 12 layers, unmeasured here — B.3 measures it
beside the table and the buffer set); the host term that decides the cut is
therefore table + buffer set + dense staging, as row (a) prices it, and
nothing in this curve moves C3/C4 — they stay for B.3.

**Shared-process variant** (`~/wp/wpA1-shared.py`, 4 → 8 → 12 layers in one
process, compiled model dropped between depths, 21:40–21:41Z): compiled in
7.8 / 19.1 / 21.1 s, resident 6.72 / 12.11 / 17.51 GiB as in the fresh
processes; gtt 1.29 GiB flat throughout (the plugin's own working set),
vram0 after each drop 0.95 → 1.68 → 1.94 GiB — a residual of ~0.5 GiB per
compile-and-drop, no staging accumulation. "shmem accumulates across
compiles in one host session" is NOT the mechanism.

**Not measured here**: a forward on a spilled graph (its cost over PCIe),
the A770 (whose reported 15.11 GiB vs physical 16 GiB may hold the same
~1 GiB of headroom c06 found on the B60), and the segmented compile's
dense-only staging term (B.3). Contaminations on record: the host sampler
sampled the `timeout` wrapper for pass 1's c01–c05 (host columns valid,
fdinfo re-measured in pass 2), and a CPU tiny-geometry driver job of mine
ran on the host during c05/c06 (ZFS contention; killed 21:24Z).

### (c) Cold-boot determinism

Two cold boots of the segmented 48-layer service (process restarted, page
cache dropped between them), the same greedy requests, byte-identical
answers in digest form:

| probe | boot 1 digest | boot 2 digest | identical? |
|---|---|---|---|
| "The capital of France is", 8 greedy tokens | EMPTY | EMPTY | EMPTY |
| KLD capture window 0, greedy token | EMPTY | EMPTY | EMPTY |

A digest that differs between boots is a finding, and its mechanism is
measured with `tools/boot_serving_shape.py --repeat` (window-050 §4.11's
instrument), not narrated.

### (d) The KLD bar, re-derived from THIS model's own reference round-trip

**The pair, DECIDED here** (BF16 of this model fits nowhere this repository
can run): **P_ref** = llama.cpp master `56b9eb28`, CPU, f32 accumulation,
over the SAME UD-Q3_K_XL shards — the pinned capture `af7993b7…` (n_ctx
2735, two windows); **P_cand** = arcint's segmented 48-layer served path
over the same shards (f16 kernels, the IQ4_NL table decoded in-graph, the
experts dense per token). The two forwards read the same quantised bytes, so
the ideal KL is 0 and what the instrument measures is arcint's
implementation divergence.

**The reference's own rounding error** (the doctrine's base): the capture
stores each row as an f32 scale, an f32 min log-prob and n_vocab uint16
steps (`log_softmax(int, const float*, uint16_t*, int)`, perplexity.cpp),
the min clamped at max−16 — so a row's reconstruction differs from the
writer's own f32 log-probs by at most half a step (≤ 8/65535 nats per logit)
plus the clamp's tail. **F_ref** is that error measured at 248,320-wide on
real rows: served f32 rows round-tripped through the writer's transcription
(`tests/python/test_kld_served.py::llama_row`) and compared with
`kl_ref_vs_served` (the 257-wide cell measured 2.9e-6 nats). llama.cpp's own
run-to-run floor is measured **0** (bit-reproducible, window-050 §4.11) [AMENDED 2026-09-13, REVIEW HIGH-1: the cited §4.11 warm/cold generalisation was refuted on re-run - A770 steps once at an unpredictable forward; settle is defined by the observed FLOOR pair.].

**The bar, stated before any number**: `bar_0.5.1 = 100 × F_ref` below row
2051, and `bar_0.5.1 + the measured QSA price` (2.385560e-02 over the rows
above the boundary, window-050 §8) at or above it — a stated multiple of the
reference's own rounding error, never an imported multiplier. The multiple
is 100 because two decades over the instrument's resolution is where "the
same distribution to the instrument's precision" stops being a defensible
sentence; it is a choice, and it is written down before F_ref exists. The
served path's own floor **F_served** (KL(A‖B) over replays, §4.11's
instrument at depth 48) is printed beside every reading; a bar below
F_served is unreadable and the row says UNREADABLE, not PASS. The inherited
0.0599 is printed beside it for continuity and decides nothing.

| quantity | predicted | measured |
|---|---|---|
| F_ref (uint16 reconstruction error at 248,320-wide, mean over the capture's rows) | between 2.9e-6 (the 257-wide cell) and 8/65535 = 1.2e-4 nats | **3.0905e-05 nats** mean over 816 served rows (max 1.0533e-02, min 2.5e-10), `RUN@c00500f` A770 d4 f16 T=816, 2026-09-13 22:03Z — C6 holds on the mean; the per-row max is the clamp-tail term (below) |
| F_served at depth 48 (KL(A‖B) mean, ≥ 2 replays, both windows) | of the order of depth 4's 2.1e-4, per-window event | **MEASURED 2026-09-19** [measured-here] (KV u8) (`RUN@b6dbca5`, B60 GPU.0, d48n, ratio 99 + tier, KV u8, chunk 512, 14:04–18:44Z, dump `d48n-fserv.bin`, 4 replays of both windows): KL(A‖B) **mean 0.1361 (w0) / 0.1512 (w1)**, max 4.425 / 9.648, argmax agreement 0.850 / 0.902, max \|logit diff\| 13.22 / 17.81; 2 of 2 window pairs moved. **The decided bar (3.0905e-03) sits ~44x BELOW this floor, so this row is UNREADABLE, not PASS.** At depth 4 on the rows above: **bit-identical ×3** (KL(A‖B) 1.0e-18, max \|diff\| 0), the observed floor pair of that process. |
| bar_0.5.1 below 2051 = 100 × F_ref | 3e-4 .. 1.2e-2 nats | **3.0905e-03 nats** (PROVISIONAL: derived from served f16 rows, see the note) |
| bar_0.5.1 at/above 2051 = bar + QSA price | + 2.385560e-02 | **2.6946e-02 nats** |
| mean KL(P_ref‖P_served) below 2051 | not predicted (the first full-depth number) | EMPTY |
| mean KL(P_ref‖P_served) at/above 2051 | not predicted | EMPTY |
| verdict (REPORT ONLY at this commit; the gate is the tag's) | — | EMPTY |

**Measured — A.2, `RUN@c00500f` (tree byte-verified 1c2a9b5f5b73fdae), A770 (GPU.1), depth-4 artifact with real weights, capture window 0's first 816 ids, 2026-09-13 21:50–22:09Z (logs `~/wp/logs/wpA2-fref/`, rows `/models/ov/_kld/wpA2/`, the instrument `tools/kld_bar.py`).** The rows the derivation reads are the SERVED PATH's own f16 rows, not f32 ones: the f32 hint compiled (56.9 s, 8.82 GiB resident) and the forward was refused, verbatim — at T=816 `primitive_onednn_base.h:559: could not execute a primitive` (10.3 s in, rc=134), at T=512 `paged_attention.cpp:72: Check 'valid_block_size' failed ... Incorrect block size for Paged Attention operation for key cache quant mode BY_CHANNEL` (104 s in); the f32-hint + `--paged-kv f16` variant refused at COMPILE after 62.3 s — `program_builder.cpp:168: ProgramBuilder build failed ... ocl_kernel_builder.hpp:73: clBuildProgram, error code: -11 CL_BUILD_PROGRAM_FAILURE` (22:08Z). Three f32 attempts, three refusals: the served graph at depth 4 has no f32 forward on the A770 under this plugin, and the f32 row of the round-trip pair is NOT AVAILABLE from the card. The f16 leg: compile 47.6 s, 6.79 GiB, three forwards of the same request bit-identical (digest `0fddacdb5d44`), a 1-token decode probe after the block accepted. F_ref is the writer transcription's error on real rows of the served path's width and distribution; whether f32 rows would move it is UNMEASURED on this card, so the bar keeps its PROVISIONAL tag with a different reason than before: the rows are f16-served, dated, attributed.

> [AMENDED 2026-09-14, A.2: **C6 did not price the clamp-tail term.** The writer clamps the row's minimum at max−16, so every logit further down reconstructs to exactly max−16, and at 248,320-wide those entries carry e^-16 each — a tail the half-step bound never counts. `tests/python/test_kld_bar.py` found it red-first (40-nat synthetic rows: F_ref a decade over the step bound) and pins it as its own cell; on the served rows the MEAN stays inside C6's band (3.09e-5) while the per-row maximum (1.05e-2) does not. C6 stands as written for the mean; its bound is not a per-row bound.]

> [AMENDED 2026-09-14, A.2: the inherited **0.0599** (window-050 §7, another model's number) is SUPERSEDED BY LINEAGE by bar_0.5.1 = 3.0905e-03 below / 2.6946e-02 above 2051 wherever this document reads a bar; it stays printed beside every verdict for continuity and decides nothing — mark, not erasure. The new bar is itself PROVISIONAL on the f16-rows caveat above.]

> [AMENDED 2026-09-19, REVIEW: **bar_0.5.1 is an INSTRUMENT-RESOLUTION BOUND, not an acceptance bar**, and the record now says so. Three measured facts decide it. (1) It is unmeetable by the reference implementation: llama.cpp's own error against the same f32 reference is **mean 0.3387 / median 0.0649** on window 0 — ~110x the 3.0905e-03 bound. (2) Its own tail defeats its own mean: `F_ref`'s **per-row max is 1.0533e-02**, ~3.4x `bar_below`, so one row can carry more reconstruction error than the bound; it is a MEAN bound only, which C6's own amendment already conceded. (3) The READABLE acceptance candidate is the floor between implementations — llama.cpp vs the same reference, median **0.0649 (w0) / 0.0283 (w1)**, mean 0.3387 / 0.4415 — and above both of these sits the served path's own **`F_served` at depth 48**, which is still EMPTY and is the actual gate-blocking gap, not the 100x. The tooling was wired to match: `tools/kld_bar.py` marks the bound `bound_is_acceptance: false` and flags the clamp tail per run; `tools/kld_served.py` prints the decided bound, the between-implementations floor and the superseded literal together, so no reader can see only the 0.0599. Cells: `tests/python/test_kld_bar.py`, `tests/python/test_kld_served.py` (15 passed, 2026-09-19). No measured column here changes.]

> [MEASURED 2026-09-19 [measured-here] (KV u8), the one column that did change: **clause (d)'s `F_served` is filled.** The depth-48 served path's own run-to-run floor is **KL(A‖B) mean 0.1361 (w0) / 0.1512 (w1)**, max 4.425 / 9.648, argmax agreement 0.850 / 0.902, max |logit diff| 13.22 / 17.81 — from four replays of both 2,735-token windows (`RUN@b6dbca5`, B60 GPU.0, d48n, ratio 99 + tier, KV u8, chunk 512, 14:04–18:44Z, dump `d48n-fserv.bin`). That floor is **~44x ABOVE the decided bound (3.0905e-03)**, so clause (d)'s own rule applies and **the row reads UNREADABLE, not PASS**. The indeterminism is the chunk-boundary non-exactness the loader already warns about (DESIGN.md 3.2): 2,735 tokens over 6 chunks at chunk 512, 2 of 2 window pairs moved. Consequence for BERLIN: the artifact's own KL-vs-reference means (0.51 below / 0.32 above on window 0) are read to no better than ~0.14 nats, i.e. dominated by this floor, not by model fidelity — so closing the gate needs **determinism first** (a single-chunk prefill, or a bit-exact chunk-carry), not a tighter bar. Upstream: openvinotoolkit/openvino **#38099** (OPEN, acknowledged by Intel 2026-09-16, self-contained reproducer supplied the same day, no fix) is the same GPU chunked-GatedDeltaNet unroll family — deterministic wrong values for chunk ≥ 2 on both cards; and **#37607** (CLOSED 2026-09-18 as COMPLETED with **no upstream fix**, only the operator's workaround `CACHE_MODE=OPTIMIZE_SPEED`) governs the cached warm start. The defect now owns its campaign: `docs/campaigns/served-prefill-determinism.md`.]

> [CORRECTED 2026-09-20, REVIEW: **this amendment's chunk-boundary attribution
> is refuted.** An unchunked arm on the same served config (`--prefill-chunk 0`,
> B60 GPU.0, ratio 99 + tier, dump `d48n-d4-unchunked.bin`, all three forwards
> complete 13:37:30Z) gives a floor **higher** than the chunked one: warmup vs
> replay 0 mean **0.095497** (143/1367 moved), **replay 0 vs replay 1 mean
> 0.202143** (**284/1367 moved**, argmax **0.7922**) — against chunk-512's
> 0.072601 / 0.073234. Removing the chunk boundaries does not remove the
> divergence, so the depth-48 served floor is the **B60/Xe2 per-forward step
> this document's cut table already isolates** (layer 0, one f16 ulp at or
> after row 32; the A770 bit-identical), **not** §3.2's chunk non-exactness.
> The fix named above ("a single-chunk prefill, or a bit-exact chunk-carry")
> is superseded: determinism comes from a **different measurement card
> (A770)** or from **naming and pinning the kernel**. Campaign,
> design note and handoff updated the same day.]

> [MEASURED 2026-09-20 [measured-here] (KV u8), **the row turns READABLE — on a named card**: on the
> **A770 the depth-48 served path is bit-identical.** `RUN@b6dbca5`, A770
> (GPU.1), d48n, ratio 99 + tier, KV u8, chunk 512, served path, `WARMUP=0 REPEAT=2`,
> dump `d48n-a770-d48.bin`, two forwards r0<->r1 read with `floor_pair` over
> the tool's own 1367-row subset: mean **-0.000000**, max 0.000, **0/1367 rows
> moved**, argmax **1.0000**, max |diff| **0.000**, `bit-identical True`. So
> `F_served = 0`, the decided bar (3.0905e-03) sits ABOVE that floor, and
> clause (d)'s `F_served` requirement — "a bar below F_served is unreadable" —
> is **satisfied on the A770 as the measurement card**: the artifact's own
> KL-vs-reference means can now be read. The B60's floor (0.1361 / 0.1512 here,
> and every arm since: D2 0.072601 / 0.073234, force-the-tier 0.081553, D4
> unchunked 0.095497 / 0.202143) is therefore a **per-CARD defect**, not a
> property of the served path. **CORRECTED 2026-09-20 (same day, after the ISA
> test): the width is NOT the mechanism and the pin is DEAD.** `xe2`
> *requires* subgroup size 16 — a kernel with `intel_reqd_sub_group_size(8)`
> [DATED CORRECTION 2026-09-25: the A770 named in this row is **ACM-G10**
> (DG2-512); `acm-g12` is a different DG2 die (DG2-256, shipped in Arc Pro
> A60 / A570M / A530M). Both are Xe-HPG, so the subgroup-width argument below
> is unaffected — only the die label changes.]
>
> fails to compile on every Xe2 target (`bmg-g21`, `bmg-g31`, `lnl-m`,
> `ptl-h`) with *"Kernel compiled with required subgroup size 8, which is
> unsupported on this platform"* (it compiles for the A770, `ACM-G10`) — so
> `get_subgroup_size`->16 is **forced by the platform**, not a choice we can
> flip. And the disassembled reduction is a **fixed deterministic tree at both
> widths** (`add(8)+add(4)+add(1)+add(1)` at 16 on `bmg-g21`;
> `add(4)+add(1)+add(1)` at 8 on `ACM-G10` (A770); no SLM, no barrier). So the width
> explains the **card-to-card VALUE difference**, not the **run-to-run
> variance**; it is a correlate of the card, not the mechanism. **The
> mechanism is OPEN**, and after this amendment the peer session **refuted the
> ordering-race shape** (a `clFinish` after each of 233 enqueues leaves the
> repeats differing) and the JIT shape (`ocloc` twice -> byte-identical
> binaries), so what remains is a **within-kernel nondeterminism in the GDN
> arithmetic at execution level**, witnessed by the state digest being
> stochastic while the co-resident conv state is stable. What remains pinned is only the location: `layer0/mixer_out`
> with input-bit-identity across 8 repeats (nine ports bit-identical, both
> state tables exactly the all-zero hash) while the output differs every
> forward, and the A770 bit-identical.
>
> **Caveats, stated rather than smoothed.** [DISCHARGED 2026-09-21 — see the
> MEASURED 2026-09-21 amendment below.] (1) This is **x2**, not the
> x8/x12 of the **depth-4** evidence (both A770 rows in this document's cut
> table are depth 4; no A770 depth-12 leg exists), and §4.11's amendment *"the A770 steps once
> at an unpredictable forward; settle is defined by the observed FLOOR pair"*
> is **not refuted by two forwards** — a repeat-8 A770 arm is the follow-on
> and should be run before clause (d) is quoted as settled. (2) The A770 is
> the **measurement** card, not the product's target card: the B60 remains the
> deployment card. [CORRECTED 2026-09-20: the width pin is **DEAD** (`xe2`
> *requires* subgroup size 16), so the B60's row cannot be made readable that
> way; it stays a per-card caveat until the within-kernel mechanism is found or
> upstream fixes it. The honest sentence is *"the gate is readable on the A770
> now; the B60 is a per-card caveat."*]

> [MEASURED 2026-09-21 [measured-here], REVIEW: **the ×2 caveat is DISCHARGED — the repeat-8
> A770 arm landed and every pair is bit-identical.** `RUN@b6dbca5`, A770
> (GPU.1), d48n, ratio 99 + tier, KV u8, chunk 512, served path, `WARMUP=0
> REPEAT=8`, dump `d48n-a770-rep8.bin`, eight forwards at 3,563-3,593 s each
> (replay done 2026-09-21T05:41:50Z): all seven consecutive pairs
> `win0↔win1 … win6↔win7` read KL(A‖B) mean **-0.000000**, **0/1367 moved**,
> argmax **1.0000**, max |diff| **0.000**, `bit-identical True`. So
> `F_served(A770, d48) = 0` at the **same repeat count as the depth-4 ×8/×12
> evidence**, and clause (d) is settled **on the A770 as the measurement
> card**. No A770 depth-12 leg exists on the record; the ×8 here is depth 48.]

### (e) THE PARIS LINE — EMPTY, with its falsifiable clause

| probe | served model | answer (raw, pasted) | token id | date |
|---|---|---|---|---|
| "The capital of France is" (ids `760,6511,314,9338,369`), greedy | the segmented 48-layer served path, both cards | EMPTY | EMPTY | EMPTY |
| "What is the capital of France? Answer in one word." (chat form), greedy | same | EMPTY | EMPTY | EMPTY |

**The clause**: the 48-layer served answer to "The capital of France is" is
the token whose surface form is `Paris` (leading space or not; its id is
recorded when measured, from the GGUF's own vocabulary). The clause dies if
the greedy token is anything else.

**The named-refusal fallback, written now**: if the token is not `Paris`,
the measurement commit does NOT explain; it localises with the instrument
that exists — `tools/boot_serving_shape.py --cut layerN/out --repeat` over
the segmented graphs — and names WHICH segment / layer first departs from the
llama.cpp reference's hidden state for the same prompt (the reference's
per-layer activations captured once with `llama-eval-callback` or the
equivalent at `56b9eb28`), or which knowledge refuses: a wrong token with
argmax agreement near 1 against the reference elsewhere is a head/mixer
defect; a wrong token with the per-layer divergence starting at the first
segment boundary is the boundary's; one starting inside a segment is that
segment's kernel. The refusal row names the layer and the measurement that
named it.

### (f) Explicit NOT claims

- A 12-layer rung answer (row (b)) is mechanism, never an answer: 36 layers
  are missing, whatever token it returns.
- A depth-4 token is mechanism, never an answer (window-050 §4.9).
- Nothing in this file compares to FreeToken; the comparison row is
  GENEVA's, by our own runs.
- "Fits" in row (a) is a measured host peak under a real compile, never a
  cgroup reading.
- No KL mean is quotable without F_served beside it.

## 4. Falsifiable clauses of this commit, listed

| # | clause | dies if |
|---|---|---|
| C1 | the 12-layer fill takes ~25 min | it takes under 15 or over 40 |
| C2 | the 12-layer single model compiles on the B60 through the served binary | the host refuses its staging (~19 GB next to the table) — then the new edge is the number WP4 designs with |
| C3 | 4 × 12 does NOT fit the host while compiling one segment | its measured peak sits under ~46 GiB |
| C4 | 6 × 8 fits | its measured peak sits over ~46 GiB — then 12 × 4, and the row says so |
| C5 | a cold-boot pair is byte-identical | any digest differs |
| C6 | F_ref lies in [2.9e-6, 1.2e-4] | it does not — the writer transcription is then re-checked against perplexity.cpp before anything else |
| C7 | Paris: the 48-layer greedy token is `Paris` | it is not — the named-refusal fallback runs |
| C8 | one full forward at 48 layers costs ~35 s of NVMe reads cold | measured under 15 s (page cache) or over 90 s (a mechanism other than bandwidth) |

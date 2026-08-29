# Prefill baseline, B60, 2026-08-29

Everything below is measured on GPU.0 of the dev host (Arc Pro B60, 22.71 GiB), artifact
`/models/ov/qwen36-coder-b5-ov` (Qwen3.6-27B-A3B-Coder q4, 30 GDN + 10 attn),
u8 paged KV, one binary (`arcint 0.2.1`, commit `c7f84e6`), one lane,
`--n-ctx 262144` unless stated. Prompts come from `tools/bench_slots.py`
(`single`, `--max-tokens 8`), so the number is prefill and nothing else. Each
cell is the second run at that configuration; the first is discarded as
shape warm-up.

Context: every performance bar this project has ever set is a **decode** number.
Prefill is named in DESIGN §5 as "a first-class regression metric per card" and
has been exactly that — tracked, never optimised, never profiled.

## 1. The chunk knob, fixed depth (28906 tokens)

| requested chunk | served chunk | activation (measured) | prefill |
|---|---|---|---|
| 512 | 512 | 1.11 GiB | 1339.1 t/s |
| 1024 | 1024 | 1.73 GiB | 1638.6 t/s |
| 2048 | 2048 | 2.99 GiB | **1878.1 t/s** |
| 4096 | **2048** | 2.99 GiB | 1879.6 t/s |
| 8192 | **2048** | 2.99 GiB | 1875.8 t/s |

Decode is flat at 67.6 t/s across all of these (measured separately): the chunk
is a prefill knob and nothing else.

**4096 and 8192 are not served.** The chunk climb in `load_paged` stops at 2048,
and the reason is arithmetic rather than hardware. The admission fit for this
configuration is `0.480 GiB fixed + 1283.9 KiB per chunk token`, so a 4096 chunk
predicts 5.49 GiB. The fixed side is weights 12.83 + margin 0.25 + GDN rows 0.093
+ KV at 262144 (2.83) = 16.00 GiB. With the 25% headroom factor the step is
priced at 16.00 + 6.87 = 22.87 GiB against 22.71 available and is refused; without
it, 21.49 would have fit. The headroom exists because the fit was measured to land
~11% low, and it costs the largest chunk by 0.16 GiB.

## 2. The depth curve, at the served chunk 2048

| prompt tokens | seconds | prefill | exponent vs previous row |
|---|---|---|---|
| 982 | 2.10 | 467.6 t/s | — |
| 3868 | 3.10 | 1246.8 t/s | — |
| 15412 | 7.89 | **1954.5 t/s** | — |
| 30830 | 17.00 | 1813.8 t/s | 1.11 |
| 61640 | 48.33 | 1275.4 t/s | 1.51 |
| 123260 | 152.54 | 808.0 t/s | 1.66 |
| 240754 | 508.15 | 473.8 t/s | 1.79 |

Two shapes in one curve. Below ~15k the rate *climbs* — fixed per-request cost
(tokenize, embeddings, the first chunk's kernel selection) dominates, and a 982
token prompt gets 467.6 t/s because 2.1 s is mostly not prefill. Above ~15k the
rate falls and the **local exponent rises monotonically, 1.11 → 1.79**: the
quadratic attention term taking over. Overall 15412 → 240754 is L^1.52.

At the artifact's full context this is **8.5 minutes to first token**. That is the
pain this project was started over, still present on the served path.

## 3. The served path is slower at prefill — and it is two causes, not one

**Corrected 2026-08-29, same session.** The first version of this section compared
the paged path at its u8 KV default against the stateful path at its fp16
default and reported a single 22–30% figure. That conflated the graph with the
KV precision. Re-run at matched token counts and with the paged path forced to
f16 (`ARCINT_PAGED_KV=f16`), the two separate cleanly:

| prompt tokens | paged u8 (served) | paged f16 | stateful fp16 |
|---|---|---|---|
| 14450 | 2245.6 t/s | 2178.0 t/s | **2519.1 t/s** |
| 57792 | 1337.3 t/s | 1601.5 t/s | **1818.1 t/s** |
| 115564 | 850.5 t/s | 1092.0 t/s | **1160.8 t/s** |

**The paged graph costs 6–14%, and the penalty shrinks with depth**
(−13.5% / −11.9% / −5.9% at matched KV). That is the honest price of the port,
and it is smallest exactly where the port's decode win is largest.

**u8 KV costs prefill 0–22%, and the penalty grows with depth**
(+3.1% / −16.5% / −22.1% against f16 on the same graph). This is new. §7.0.3
chose u8 as the default on decode evidence — "never slower, +2.5% at 32k, halves
KV memory" — and on the quality harness. Nobody measured prefill, and at depth it
is the larger of the two costs.

The served configuration against the stateful default is therefore −10.9% /
−26.4% / −26.7%, which is where the original 22–30% came from; it is a real
number for the shipped config, but it is not one fact about the paged port.

**The trade u8 actually represents.** It halves KV — 2.83 GiB against 5.00 GiB at
262144 — and that halving is what makes two lanes at agent depth fit on the B60
at all (§7.2). So this is not "u8 was a mistake": it is a lever with a price tag
nobody had read. A one-lane deep-context endpoint and a two-lane agent endpoint
plausibly want different answers, and `ARCINT_PAGED_KV` already exists as the
switch.

## 3b. The original paged-vs-stateful table (superseded, kept for the record)

Same binary, same card, same chunk 2048, same `--n-ctx`, same session —
`--no-paged` selects the stateful reference executor:

| depth | paged (served) | stateful (`--no-paged`) | delta |
|---|---|---|---|
| ~15k | 1954.5 t/s (15412 tok) | **2519.1 t/s** (14450 tok) | **−22%** |
| ~60k | 1275.4 t/s (61640 tok) | **1818.1 t/s** (57792 tok) | **−30%** |
| ~120k | 808.0 t/s (123260 tok) | **1160.8 t/s** (115564 tok) | **−30%** |

The paged port was justified on decode and delivered it: 68 t/s flat to 30k
against a stateful path that collapsed to 3.6 t/s at 2.3k and 0.1 t/s at 18k
(DESIGN §5). Nobody measured what it did to prefill, because prefill was not a
bar. It cost 22–30%.

This also reconciles an inconsistency in the record: DESIGN §5 reports 257,167
tokens at 528 t/s, measured on the **stateful** path. The served path does
473.8 t/s at 240,754 — the document's deepest prefill number describes a graph
the fleet no longer runs.

## 4. Per-kernel profile of the served path

New in 0.2.1: `enable_profiling` was wired on the stateful path only, so every
kernel table in DESIGN describes a graph the fleet does not run. `ARCINT_PROFILE=<n>`
now profiles the paged path too, dumping one prefill chunk of `n` tokens and one
decode step at that depth. PERF_COUNT inflates wall clock (§7.0), so **read the
shares, not the totals**.

**Prefill chunk, 2048 tokens** (node total 206.26 ms, 100.7 us/token):

| share | nodes | op | kernel |
|---|---|---|---|
| **33.1%** | **40** | FullyConnectedCompressed | **`ocl:ref:any__i8`** |
| 18.9% | 331 | FullyConnectedCompressed | `jit:gemm:any__i8` |
| 15.4% | 30 | PagedGatedDeltaNet | `ocl::paged_gated_delta_net::opt___f16` |
| 8.2% | 40 | MOECompressed | `ocl::moe::moe_3gemm_swiglu_opt___f16` |
| **6.3%** | **30** | PagedCausalConv1D | **`ocl::paged_causal_conv1d::ref___f16`** |
| 4.7% | 104 | Add | `generic_eltwise_ref__f16` |
| 3.1% | 60 | Swish | `activation_ref__f16` |
| 2.1% | 131 | RMS | `rms_gpu_bfyx_opt__f16` |
| 2.1% | 161 | DynamicQuantize | `dynamic_quantize_gpu_opt__f16` |
| 1.6% | 10 | PagedAttentionExtension | `ocl::paged_attention::opt__f16` |
| 1.2% | 41 | StridedSlice | `strided_slice_ref__f16` |
| 0.8% | 40 | Multiply | `generic_eltwise_ref__f16` |
| 0.7% | 10 | Crop | `generic_eltwise_ref__f16` |
| 0.6% | 10 | VariadicSplit | `generic_eltwise_ref__f16` |
| 0.6% | 40 | MoERouterFused | `ocl::moe::moe_router_fused_opt___f16` |
| 0.4% | 20 | Concat | `concatenation_gpu_simple_ref__f16` |

**Decode step, 1 token at the same depth** (node total 178.16 ms — inflated,
shares only):

| share | nodes | op | kernel |
|---|---|---|---|
| 51.9% | **371** | FullyConnectedCompressed | `jit:gemm:any__i8` |
| 15.3% | 30 | PagedGatedDeltaNet | `ocl::paged_gated_delta_net::opt___f16` |
| 8.1% | 40 | MOECompressed | `ocl::moe::moe_3gemm_swiglu_opt___f16` |
| 6.3% | 30 | PagedCausalConv1D | `ocl::paged_causal_conv1d::ref___f16` |

### What the two tables say when read against each other

1. **Forty FullyConnectedCompressed nodes fall back to a reference OpenCL kernel
   during prefill and consume a third of the chunk.** In decode, all 371 FC nodes
   are on `jit:gemm:any__i8` and there is no `ocl:ref` row at all. The count is
   exactly the layer count, so it is one FC per layer selecting a different
   implementation at a 2048-token shape than at a 1-token shape. Those 40 nodes
   cost 68.3 ms where the other 331 together cost 38.9 ms.
2. **`PagedCausalConv1D` is a reference kernel in both phases**, 6.3%, 30 nodes —
   one per GDN layer. The paged transformation gave GDN an optimised kernel
   (`paged_gated_delta_net::opt`) and left the causal conv on the reference path.
3. **Reference eltwise adds up.** Add 4.7 + Swish 3.1 + StridedSlice 1.2 +
   Multiply 0.8 + Crop 0.7 + VariadicSplit 0.6 + Concat 0.4 ≈ 11.5% in
   `generic_eltwise_ref` / `activation_ref` / `*_ref` kernels.
4. Summing 1–3: **roughly half of a prefill chunk is spent in reference
   implementations.** DESIGN §5 made exactly this observation about decode on the
   stateful path ("roughly a third of decode time is in reference or host-side
   implementations") and the paged port fixed it for decode. Prefill never got
   the same look.

## 5. What is not measured

- The A770. Its last prefill number is 498.9 t/s at 29k, one lane, chunk 256.
  At two lanes and 40k context its chunk drops to 128, and nothing has been run
  there.
- Whether the stateful path also puts those 40 FC nodes on a reference kernel.
  `profile_step()` profiles a decode step only, so the comparison that would say
  whether the fallback is what costs the paged path its 22–30% does not exist yet.
- Prefill has no per-phase breakdown in `GenerationStats` the way decode does
  (`decode_forward/embed/sample/emit/wait`). `prefill_seconds` is a single number,
  so the embeddings gather, the graph and the cache snapshot are not separable at
  the request level.
- Prefix-cache hit behaviour under real agent traffic. The gates prove exactness,
  not hit rate, and the production unit runs `--prefix-cache-mib 8192` with 21532
  spare pages.


---

# Round two: two measurements, stated before they were run

## M1 — the attribution gap

**Question.** A past-0 profile of a 2048-token chunk accounts for ~100 us/token
of node time; the served rate at 15412 tokens is ~512 us/token. Where is the
rest? The MTP machinery would have been the obvious answer and it is dead for
this artifact: the b5 coder ships no MTP head, so `mtp_ready_` is false and
none of that code runs.

**Instrument.** `GenerationStats` gains the prefill breakdown decode already had
— `prefill_embed / prefill_forward / prefill_blocks / prefill_restore /
prefill_wait`, each net of the turnstile wait, with the remainder printed as
`other`. One line on the console, same shape as the decode line.

**Exit criterion, fixed in advance: >= 90% of served prefill wall time
attributed to named phases.** If one pass does not reach it, what is still dark
gets written down here and the profiler is not iterated on.

## M2 — a discriminating test of the shared-expert hypothesis

**The claim under test.** The 40 reference-kernel `FullyConnectedCompressed`
nodes are `mlp.shared_expert.down_proj`, one per layer, and the trigger is its
fused `sigmoid(shared_expert_gate)` epilogue turning into a per-row `[M,1]`
broadcast at M>1, which no optimised implementation accepts in combination with
i8 dynamic-quantised activations and u4 group-64 weights.

Both halves are currently inference. The identity comes from size-matching —
1.7 ms/node fits a 512-wide projection, `out_proj` would be an order of
magnitude more and the gate scalar an order less — and "nothing else is this
size" is a conjecture, not a name. The mechanism has never been made to fail.

**Instrument.** The profiler now prints `node_name` for reference-kernel rows,
and sweeps M ∈ {1, 2, depth} in one load, because kernel selection for a
dynamic-shape node happens per runtime shape and those are three different
questions asked of one graph.

**What the run must show if the claim holds:**

1. the reference node names are the shared-expert down projection, layers 0–39;
2. M=1 shows **zero** reference FC nodes (already observed 2026-08-29);
3. M=2 shows **40** reference FC nodes — the switch happens between one token
   and two.

**What kills it, stated in advance:**

- names other than `shared_expert.down_proj` → the identification is dead and
  whatever is named replaces it, size-matching having been a coincidence;
- M=2 shows no reference rows → "M>1" is not the trigger. The mechanism as
  stated is dead and what remains is a size or tile threshold, which is a
  different hypothesis needing a different test;
- M=1 shows reference rows → the prefill-specific framing is wrong outright;
- a count other than 40 → it is not one per layer across both layer types, so
  it is a GDN-only or attention-only projection and the shared-expert
  identification is dead.

If the claim survives, the deliverable is getting those nodes off the reference
path — not a fourth measurement of how slow they are.


---

# Round two: results

## M1 — the attribution gap does not exist

B60, b5 coder, u8 paged KV, one lane, `--n-ctx 262144`, chunk 2048:

```
prefill 14450 tok in  6.42 s (2252.4 t/s) | graph 6.37 s, embed 0.03 s, pages 0.00 s, restore 0.01 s, wait 0.00 s, other 0.00 s
prefill 57792 tok in 43.24 s (1336.6 t/s) | graph 43.14 s, embed 0.09 s, pages 0.00 s, restore 0.01 s, wait 0.00 s, other 0.00 s
```

**99.2% and 99.8% of served prefill is the graph.** The exit criterion was 90%.
Embeddings are 0.5% and 0.2%; page allocation, cache restore and device waiting
are zero to two decimal places. There is no host-side overhead, no dark time,
and nothing to chase in tokenizer, H2D copies or chunk scheduling.

**So the premise was wrong, and here is why it looked true.** The comparison was
between a profile of one chunk *at past 0* (100.7 us/token) and the *average
over a whole multi-chunk prefill* (~445 us/token at 15k). Those are not the same
quantity: chunk k attends to everything before it, so the first chunk of a 15k
prefill is the cheapest one in it. The graph really does cost ~445 us/token
averaged over that prefill. Nothing was hiding.

Consequence, and it points the other way from where the round started: **the
kernel findings are not second-order, they are the entire story**, because the
graph is essentially all of prefill. A share of the graph is a share of prefill.

## M2 — the node is named, and the mechanism is dead

The M sweep, one load, same graph, three token counts:

| M | FullyConnectedCompressed on `jit:gemm:any__i8` | on `ocl:ref:any__i8` |
|---|---|---|
| 1 | 371 | **0** |
| 2 | 371 | **0** |
| 2048 | 331 | **40** |

And the names, printed rather than inferred:

```
ref node: __module.model.model.language_model.layers.0.mlp.shared_expert_gate/ov_ext::linear/MatMul
ref node: __module.model.model.language_model.layers.1.mlp.shared_expert_gate/ov_ext::linear/MatMul
ref node: __module.model.model.language_model.layers.10.mlp.shared_expert_gate/ov_ext::linear/MatMul
ref node: __module.model.model.language_model.layers.11.mlp.shared_expert_gate/ov_ext::linear/MatMul
(40 node(s) on FullyConnectedCompressed  ocl:ref:any__i8)
```

**Against the criteria written down before the run:**

- *Identity: falsified.* The prediction was `mlp.shared_expert.down_proj`. It is
  `mlp.shared_expert_gate` — the one projection the size-matching argument
  explicitly excluded as "an order of magnitude less". Forty nodes, one per
  layer, layers 0–39, so the count and the per-layer structure held; the name
  did not. This is exactly what "identified by nothing else being this size"
  earns, and it is why the criterion was written first.
- *Trigger: falsified.* M=2 shows **zero** reference rows. The switch does not
  happen between one token and two, so "a per-row `[M,1]` broadcast epilogue at
  M>1" is dead as stated. What is left is a threshold somewhere between 2 and
  2048 — a different hypothesis, needing a different test.

**What the measurement replaced it with, and it is stranger than the guess.**
`shared_expert_gate` is `[1, 2048]`: it produces one scalar per token. At M=2048
that is 4.2 MMAC — arithmetically the cheapest FullyConnected in the layer by
three orders of magnitude. It costs **51.1 ms across 40 nodes, 1.28 ms each**,
against 30.1 ms for the *other 331* FC nodes together (0.09 ms each). A node
doing ~1/1000 of the work of a real projection takes 14x as long as the average
one. Whatever the reference kernel is doing for an N=1 output with u4 group-64
weights, it is not proportional to the arithmetic.

**Next test, named and not run in this round:** is it the N=1 output shape or the
u4 weight decompression? One A/B settles it — rewrite those 40 nodes to an
uncompressed f16 `MatMul` before compile (the weights are 2048 values x 40
layers = ~160 KB total, so materialising them costs nothing) and re-profile. If
the reference rows disappear, it is decompression; if they persist, it is the
shape. arcint already does this class of pre-compile surgery
(`store_kv_state_as`, `slice_logits_to_last_token`, `route_head_swap_permutes`).

## Not chased in this round, deliberately

- **The MTP prefill machinery.** The architect ranked it first: a per-chunk
  16 MB `hidden_states` read-back host-side, an `O(L x 2048)` f32 attention mask
  rebuilt per chunk, and a stateful `IndirectSDPA` head graph of the §5
  pathology class. All of that is real **for the dense Qwen3.8, which carries an
  MTP head**. The b5 coder ships none, so `mtp_ready_` is false and none of that
  code executes on the artifact every number here was taken from. Recorded so it
  is found when the 3.8 is profiled; not a lead for the coder.
- **"Attention prefill runs at f16 SIMD peak rather than XMX."** Inferred from
  the quadratic coefficient of the depth curve (~12 TFLOPS effective against a
  ~90 TFLOPS XMX f16 peak). A curve coefficient is evidence, not a mechanism.
  This stays a **conjecture** until an instruction-level check confirms or
  denies matrix-engine engagement — the same check §7.0.2b already queued and
  which `exec_type` strings cannot answer.
- Chunk 4096: measured upside is flattening (+22%, +15% for the previous two
  doublings), it coarsens prefix-cache granularity for the agent workload M6
  exists for, and it doubles the two-lane stall bound. Not shipped.
- The A770 has no prefill numbers past 29k, and none at two lanes.
- Whether the stateful path also puts those 40 nodes on a reference kernel:
  `profile_step()` still profiles a decode step only.


---

# Round three: the threshold, the sizing, and a fix not built

## The bisect — one load, eleven token counts

`ARCINT_PROFILE_SWEEP` sweeps M inside a single compile, so the whole search
costs one server start rather than six.

| M | FC nodes on `ocl:ref:any__i8` | their time | node total |
|---|---|---|---|
| 2, 4, 8, 16, 32, 64 | **0** | — | 148.5 → 86.8 ms |
| 128 | 40 | 32.3 ms | 86.2 ms |
| 256 | 40 | 34.3 ms | 87.5 ms |
| 512 | 40 | 36.1 ms | 90.6 ms |
| 1024 | 40 | 39.0 ms | 98.8 ms |
| 2048 | 40 | 41.4 ms | 115.0 ms |

**The threshold is in (64, 128].** Below it every one of the 371
`FullyConnectedCompressed` nodes is on `jit:gemm:any__i8`; at 128 and above,
exactly 40 fall back.

**And the cost is nearly flat in M.** Sixteen times the tokens — 128 to 2048 —
costs 28% more time. A `[M, 2048] x [2048, 1]` projection that scaled with its
arithmetic would cost 16x more. So what these nodes spend is **a fixed
per-invocation overhead, not per-token work.**

That is evidence about the mechanism, obtained without building either arm. A
degenerate N=1 GEMM would still do M x 2048 MACs and would scale with M; it does
not. A u4 group-64 weight decompression redone per invocation — 2048 values and
their scales, per node, per call — is constant in M, which is what the
measurement shows. **This points at decompression rather than the output
shape.** It is not isolated: the two variables have not been separated, and
until one of the arms below is run this stays a supported inference, not a
mechanism.

A corollary worth stating because it kills an attractive idea: you cannot dodge
the fallback with a small chunk. Chunk 64 avoids it, but the chunk sweep already
measured what small chunks cost — 1339 t/s at 512 against 1878 at 2048 — and a
64-token chunk multiplies the fixed per-chunk costs by 32. The fallback is
cheaper than the cure.

## Sizing, before deciding to fix

The 40 gate nodes are **depth-independent** — the projection attends to nothing —
and now also known to be **nearly chunk-size-independent**. Total cost over a
prefill is therefore `(N/C) x ~40 ms`:

| prompt tokens | chunks at 2048 | ref-FC total | prefill wall | share |
|---|---|---|---|---|
| 14450 | 7.06 | 292 ms | 6.42 s | **4.6%** |
| 57792 | 28.2 | 1.17 s | 43.24 s | **2.7%** |
| 115564 | 56.4 | 2.34 s | 135.89 s | **1.7%** |

The 33–39% figure is a share of a **past-0 chunk's node time**, and a past-0
chunk is the cheapest chunk in any real prefill. Against wall time the prize is
**1.7–4.6%, shrinking with depth**.

## Decision: the arms are not run and the fix is not built this round

Both arms require the pre-compile graph surgery that *is* the fix — rewriting 40
weight subgraphs before compile. Running an arm is therefore not a cheap
measurement that justifies a fix; it costs what the fix costs. Against a
1.7–4.6% payoff, and with `PagedCausalConv1D` sitting at **6.3% of both phases**
as the larger named cost, the effort goes there first.

Recorded rather than done, with the predictions fixed in advance so whoever runs
them is held to the same standard:

**Arm A — decompression, shape held.** Replace the 40 weight subgraphs with an
uncompressed f16 constant, output still `[1, 2048]`. ~160 KB of weights total.
*Prediction, given the flat-in-M cost:* the reference rows disappear. If they
**persist**, decompression is exonerated and the N=1 shape is the trigger —
which would also contradict the flat-cost evidence and should be reported as
such rather than explained away.

**Arm B — shape, compression held.** Keep u4 group-64, pad the output to N=8 (or
the smallest N above the threshold that a second bisect finds), slice the extra
columns away. *Prediction:* the reference rows persist, because the cost does not
scale with work and padding the output adds work without removing decompression.
If they **disappear**, the shape is the trigger and the flat-cost inference was
wrong.

Arm A is the easier build — a constant-fold of an existing subgraph, against
constructing new bit-packed u4 constants and matching scales for arm B. If only
one is ever run, it should be A, and B should be recorded as unrun rather than
inferred from A's result.

## The list, unchanged and still unchased

- `PagedCausalConv1D` on a reference kernel, 6.3% of both phases, 30 nodes —
  **the next candidate**, and larger than the one just sized.
- Chunk 4096.
- The A770 past 29k, and at two lanes.
- Whether the stateful path shares the FC fallback (`profile_step()` still
  profiles a decode step only).


---

# Round four: two corrections from the deployment side

Not my measurements. Taken on the coder endpoint (B60, one lane, `--n-ctx
262144`, arcint 0.2.3) with cold prefixes — a unique salt per request, or the
8 GiB prefix cache answers instead of the card — and depths read off
`usage.prompt_tokens` rather than planned. Verified from this side where it was
free to do so.

## 1. "u8 is never slower on decode" fails at depth

Decode at **53.5k** prompt tokens, warm prefix so the number is decode and not a
mixture, three runs each:

| | mean | spread |
|---|---|---|
| u8 | 49.41 t/s | 0.24% |
| f16 | **53.25 t/s** | 1.3% |

**f16 is 7.8% faster**, an order of magnitude outside either spread. §7.0.3's
table measured 512 / 4096 / 32768 and at 32768 u8 leads by 2.5%. Both hold:
**u8 leads at 32k, f16 leads at 53.5k, and the crossover is between them** —
unlocated, and three depths of a decode sweep away. DESIGN now carries the
qualifier; the unqualified sentence read as a property of the precision when it
is a property of a point on a curve.

This matters beyond the wording: it removes the last dimension in which u8 was
ahead. Prefill favours f16 at depth (mine), decode favours f16 at depth (theirs).
What is left for u8 is capability alone — two lanes at depth, and the A770 — and
that is exactly why the default stays where it is: **the default has to be the
setting that does not refuse**, not the setting that wins a benchmark.

## 2. Their prefill numbers extend my table one depth further

| prompt tokens | u8 | f16 | delta |
|---|---|---|---|
| 25 842 | 1710.4 t/s | 1843.4 t/s | +7.8% |
| 103 242 | 915.9 t/s | 1166.7 t/s | +27.4% |
| 206 393 | 539.6 t/s | 725.6 t/s | **+34.5%** — 98 s off a cold prompt |

My −22.1% at 115564 and their +27.4% at 103242 are the same fact from opposite
ends. Two instruments, two operators, agreeing before either is trusted.

## 3. The term the documentation was missing

Verified from the journal on this side rather than taken on trust — the endpoint
runs f16 now, so both load lines exist:

| | pool pages | live per lane | spare for cached prefixes |
|---|---|---|---|
| u8 | 37918 | 16386 | 21532 (~344k tokens) |
| f16 | 21477 | 16386 | **5091** (~81k tokens) |

The pool is sized in **bytes**; at the same 6.55 GiB an f16 page costs 1.77x a
u8 page (20.0 KiB/token against 11.3 — u8 carries per-block scales, so it is
0.565x and not a clean half) and the count falls 37918 → 21477. Live pages are a fixed count (n_ctx / 16), so the
entire halving comes out of the cached-prefix reserve: **4.2x less**. The
context ceiling (606688 → 343632) was documented; this was not, and for an agent
workload it is the term that can eat the win. Now in DESIGN §7.0.3.

## 4. The pattern, and where the warning now lives

Both retractions this round trace to one habit: reading node shares off a
profile taken at **past 0**. Chunk k attends to everything before it, so a
past-0 chunk is the cheapest chunk in any run, and every share taken from it
overstates that node's share of real prefill — for anything depth-independent,
by the ratio between that chunk and the average one. It produced the "invisible
two thirds" and the "33–39% ref-FC share", and it would have produced a third.

The warning now prints with every table the profiler emits, next to the numbers
rather than in a document beside them:

```
profile: prefill M=2048: 2048 token(s), node total 114.97 ms ...
profile:   (past 0: shares here overstate a real prefill's -- size against wall time)
```

## State

`--paged-kv f16` is live on the coder endpoint, committed in the unit repo, the
acceptance task scores 10/10 after the switch, and the revert is one flag. The
deployment runs what its own documentation points at.


## The mechanism behind the reserve, and how it scales

Added after the deployment side asked why the whole halving landed on the
reserve. Three facts compose, and they check to the page:

1. the pool is sized in **bytes** — whatever is left after weights, activations,
   GDN rows and margin — so the page count depends on what a page costs;
2. live pages are a fixed **count**, `n_ctx / 16 + 2` per lane, independent of
   precision;
3. the reserve is the difference.

`37918 − 16386 = 21532` and `21477 − 16386 = 5091`, with `262144 / 16 + 2 =
16386` in both. Everything that costs bytes therefore comes out of the reserve,
because the live side is a count and cannot absorb it.

The corollary is the part a deployer needs: **the context ceiling is not a
separate limit, it is the depth at which the reserve reaches zero.**
`606688 / 16 + 2 = 37920` against 37918 affordable pages; `343632 / 16 + 2 =
21479` against 21477. Raising `--n-ctx` spends the prefix cache first and
reaches the ceiling only when there is none left.

## The crossover: not requested

The disagreement was between a 32k measurement (u8 +2.5%) and a 53.5k one
(f16 +7.8%), and locating the crossover between them belongs to the instrument
that produced it. Asked whether the answer would change anything written here:
**no**, and the reason is worth recording so the question is not reopened.

- It does not move the default. That is chosen on what refuses.
- It does not change the deployment guidance, which is already depth-shaped:
  above ~50k both prefill and decode favour f16; below ~32k decode favours u8
  while prefill favours f16 from ~25k. A deployer who knows their workload depth
  can act on that today. Pinning the boundary to 38k or 44k changes a number in
  a sentence, not the shape of the advice.
- The band where the two axes disagree is already named as a band.

What *would* be worth the downtime is the ladder's **anchors**, not its middle:
32k and 53.5k reproducing the two existing numbers, three runs each. That is six
runs, and it tests something the crossover does not — whether the two
measurements, taken on different instruments weeks apart, agree when repeated.
If an anchor failed to reproduce, one of them is wrong and the curve between
them is not worth locating anyway. Worth folding into some other downtime;
not worth creating downtime for.


---

# The XMX check: the instrument cannot tell, and why that is the useful part

Taken first, ahead of `PagedCausalConv1D`, because it is minutes against an
order-of-magnitude question and a 6.3% grind should not start while that is
open. Rule for the check: **a marker, not a rate.** The ~150 int8-TOPS figure
and the depth curve's quadratic coefficient are both inferences from throughput,
and two inferences pointing the same way are still not an observation.

The question was already half-answered: 331 JIT nodes at ~150 int8-TOPS is not
reachable without a matrix unit, so the int8 GEMM path is very probably XMX-fed.
What was open is the **f16 attention** path.

**No card was taken. Three findings, all from stored profiles and the installed
plugin binary:**

1. **No `exec_type` carries an ISA marker**, for any kernel in any stored
   profile. The complete set is kernel-family-plus-dtype:
   `ocl::paged_attention::opt__f16`, `ocl::paged_gated_delta_net::opt___f16`,
   `jit:gemm:any__i8`, `ocl:ref:any__i8`, and so on. No `dpas`, `xmx`,
   `systolic`, `dp4a`, `mmad` anywhere.
2. **The plugin cannot be asked.** The GPU debug capabilities are compiled out
   of this build: zero occurrences of `OV_GPU_Verbose`, `DumpSources` or
   `ENABLE_DEBUG_CAPS`, against exactly two `OV_GPU_*` knobs present. There is
   no kernel-selection log to enable.
3. **The sources are not in the binary to grep.** No OpenCL source survives
   string extraction, so the XMX intrinsic cannot be searched for in the
   attention kernel.

**What the binary contains, as fact rather than as an answer.** DPAS material is
there — `__builtin_IB_sub_group_idpas_s8_s8_8_1` (int8), a `dpas.bf.bf` vISA
fragment, `CM_HAS_DPAS` — and it is oneDNN/CM microkernel infrastructure. That
is *consistent with* the int8 GEMM being matrix-fed and says **nothing either
way** about the f16 attention kernel, which is a different implementation family
(`ocl::` cldnn) from the one those microkernels serve. Recording the distinction
rather than the conclusion, because the conclusion would be a third inference.

Four apparent "SDPA + dpas" matches were false positives: `ov::pass::
SDPAScaleFusion` contains the letters D-P-A-S.

**Outcome three, and a next step that is now specific.** "Attention prefill runs
at SIMD peak" stays labelled a conjecture. What would settle it is no longer
"instruction-level tracing" in the abstract but a choice of two: a plugin build
with `ENABLE_DEBUG_CAPS=ON`, which makes `OV_GPU_Verbose` name the selected
implementation, or onetrace / Level-Zero PTI. The first is cheaper and answers
exactly this question.

**Correction carried in.** The standing note that the A770 never uses XMX is
about llama.cpp/Vulkan — ggml gates coopmat on `INTEL_XE2`, Alchemist fails the
check, DP4A is the path. It says nothing about OpenVINO and does not make the
A770 a control group here. A second card is a contrast only if measured on the
same stack.

`PagedCausalConv1D` at 6.3% of both phases resumes as the top item.

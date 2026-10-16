# 1.0.0 — feature complete, almost no gains by obvious changes

**Draft** (2026-10-14). The operator's definition: arcint is 1.0 when it is
feature complete and almost no gains are left in obvious changes. This
charter turns the two halves into checkable criteria. Each criterion names
its evidence and its state. The operator amends it; an agent does not.

## A. Feature complete

"Complete" means for the services arcint is built for: the dense agent and
the coder on the libllama engine, Flash-Next on one GPU plus host RAM, on the
two Arc cards. It does not mean every model or every engine feature.

| # | criterion | state (2026-10-15) |
|---|---|---|
| A1 | The three services run from the released package, on the engine the README names as the default, with answers right (facts, needle, the acceptance task) | **met** (0.8.0, 2026-10-15): the three services run libllama from the released package, the only engine; Paris, the needles and the acceptance task (agent 10/10, coder 9/10) through the deployed 0.8.0 (`measured-here`) |
| A2 | Flash-Next's serving shape is settled: one quant, one KV type, one context, one expert-cache size, with the quant inside the answer-level bar or the shortfall recorded as accepted | **decided** (operator, 2026-10-15): IQ3_XXS is the shape (made the default the same day for its accuracy; KL 0.218, top-1 89.7 % against unsloth's Q8_0, `measured-here`), the shortfall accepted; q8_0 KV, 131,072 tokens, 12,000 MiB of expert slots, the CPU misses (0056, fixed for its IQ4_NL banks in 0.8.0) and the Q4_K draft head, as the unit runs it. The other quants stay available as on-demand options on the same endpoint: IQ2_XS (KL 0.294, faster) and UD-Q3_K_XL (KL 0.100, inside the bar; its ~39 GiB host bank needs the host's planned 96 GB) (`docs/llama-engine.md`, "Flash-Next quant options") |
| A3 | Speculative decoding works on all three (MTP) with rollback of the recurrent state | met (`measured-here`, acceptance counts in every gate today) |
| A4 | Long context at the deployed sizes: the agent at 245,760, the coder at 98,304, Flash-Next at 131,072, with needles right and nothing evicted | met (`measured-here`, the 0045 and 0.7.8 gates) |
| A5 | Checkpoints and prefix reuse on the libllama engine (`--llama-checkpoints`) | **met** (operator, 2026-10-15): checkpoints for one lane are the 1.0 scope; no prefix cache shared across lanes (the two-lane test drive of 2026-10-08 ended because the lanes slowed each other) |
| A6 | The OpenVINO engine's role is decided: kept as the documented alternative with its own gates, or retired from the default build | **done** (0.8.0, 2026-10-15): the documentation and design overhaul first (README, DESIGN, llm.txt, DEVELOPMENT, `docs/history.md` linking the tag `openvino-history`), then the engine, the allowlist, marfrit-openvino, its patch series and recipe, the IR tools and the OpenVINO-only tests out of the tree and the package; the suites re-targeted to the libllama engine |
| A7 | Every switch that 1.0 keeps is documented; diagnostic and red-case switches are named as such | **met** (2026-10-16): `docs/switches.md` lists the 86 switches of the llama.cpp series in five classes (operational, tuning, off switch, red case, diagnostic) with default, patch and evidence; arcint itself reads none since 0.8.0 (the 38 were the OpenVINO engine's). The opt-in alternatives that measured slower or neutral, Infernix's LFRU policy (patch 0055) and the `FUSE_0049` alias were removed from the series (operator, 2026-10-16); `ARCINT_PA_HOST_INPUTS` went with the OpenVINO engine. **Operator:** which stay stable in 1.x (C4) |

## B. Almost no gains by obvious changes

"Obvious" means one of these: a mechanism one of the reference engines uses
for the same model or layer that arcint does not have; a lever an ablation
or a profile names with a price; or a known fault in a kernel. A gain counts
if it moves served tokens per second of a deployed service with the answers
unchanged.

| # | criterion | state (2026-10-15) |
|---|---|---|
| B1 | No priced lever of 2 % or more of served decode or prefill is left unbuilt on any service | **met** (operator, 2026-10-16): no single lever of 2 % or more is left; the decode launch count below is a 1.x campaign. The record: the expert cache's misses on the CPU with Strata's multi-token kernels: built as 0056 (Flash-Next decode +15.2 % on six answers, KL +0.0028 nats, `measured-here`), on the Flash-Next unit since 0.7.11; Flash-Next's prompt buffers: built as 0052 (reserve to a cell budget, +10-13 % decode at 131K, `measured-here`), on the unit since 0.7.9; the low-bit expert decode by multiplies (0053, +0.9 %, `measured-here`), and what is left of that kernel is integer and 64-bit address work with no single named lever; the GDN stage's conv-stage fusion (vllm-xpu-kernels' conv stage; the delta stage is built: 0050 +1.2-5.3 %, 0051's gating +0-1.2 % by service, Flash-Next neutral), priced 2026-10-15 by a ceiling at ~1.1 % of the agent's decode and ~1.4 % of the coder's (`measured-here`), below 2 %; the activation quantization folded into its producers (priced 2026-10-15 by a ceiling: at most ~1.4 % of the agent's round, ~1.1 % of Flash-Next's by launch count, `measured-here`; below 2 %); the rest of the decode kernel count (~13 % of a Flash-Next round, the sum of many small fusions, none obvious alone) |
| B2 | Every reference engine's mechanism for the served models is either built, measured as not paying here, or set aside by the operator. The references are Strata, FreeToken and Infernix for Flash-Next; llama.cpp's CUDA and Vulkan backends; vllm-xpu-kernels and sycl-tla on Xe2; and OpenVINO's GPU plugin | mostly met: this week's audits built or measured the reference's prompt attention, its wave split, CUDA's per-call K/V conversion, MMQ's shared tile, sycl-tla's payloads and Strata's tables. OpenVINO's post-op fusion, as the activation quantize folded into its producer: priced below 2 % (B1). vllm-xpu-kernels' GDN spec-decode rollback (2026-10-15): the state read from the accepted token's slot with no copy back is what the libllama engine already does (`code`, `llama-memory-recurrent.cpp` `s_copy`: a rejected draft sets the snapshot plane the next step reads); what differs is the conv state's K snapshot copies, priced by a ceiling at ~1.1 % (agent) and ~1.4 % (coder) of decode (`measured-here`, B1). Strata's and Infernix's CPU miss service: built as 0056 (Strata's doorbell tier and kernels, +15.2 % Flash-Next decode, `measured-here`), on the Flash-Next unit since 0.7.11. Infernix's expert-cache policy: built per layer as 0055, +1.6 % Flash-Next decode against Strata's adapt on the gate set and -2.1 % on six other prompts (`measured-here`): not a general gain, and removed from the series with the other opt-ins (operator, 2026-10-16; the port is in 0.8.0's patches); Strata's adapt rule is the policy. Its global pool, priced by replaying a routing trace offline: +0.4 points of hits, ~0.5 % of decode (`measured-here`), not built; the warm start from saved state, tried, seeded worse than the calibration profile (`measured-here`). DFlash2 drafting for the dense agent (llama.cpp at the pin carries it, `code`; 44.8 against MTP's 33.0 t/s on the retired OpenVINO engine, `measured-here` 2026-09): built before 1.0 as the operator asked (2026-10-16) and measured as not paying for the agent's long sessions: +27 % on code, -5 to -8 % on prose and at 4k, -33 to -36 % at 130-200k against MTP 5, the reference's own server likewise past ~60k (`measured-here`); opt-in, the unit keeps MTP 5 (operator, 2026-10-16) |
| B3 | Flash-Next is at parity with Strata on the same card, or the gap is explained by a measured cause | 2026-10-15, with 0056's CPU misses (opt-in): Strata's request decodes 43.8-44.0 t/s against Strata's 37.8 of 2026-10-14 on the B60 (`measured-here`, Strata not re-run). Before: prefill: met (677 t/s at the 20k needle against Strata's 620, `measured-here`). Decode on Strata's own request (`measured-here`, 2026-10-14): a round costs the same (63.1 against 63.4 ms at Strata's draft setting); with 3 drafts and Strata's stop at 0.5 arcint decodes 37.9 t/s against Strata's 37.8 (warm run; the first run 35.1 against 37.2). **Met** on the warm run; the Flash-Next unit has taken `--llama-mtp 3 --llama-mtp-min-p 0.5` since 2026-10-14; the first run's gap is the expert cache's warm-up |
| B4 | The kernels that dominate device time are at their measured ceiling for their design | met for the B60's prompt attention, the A770's quantized attention staging and the int8 Q4_K GEMM (`measured-here`, the campaign docs); Q5_K/Q6_K GEMMs measured (2026-10-15, `measured-here`, B60, test-backend-ops perf at 512 columns over the 27B's shapes): Q4_K 73-83 TFLOPS on the int8 path (0013), Q5_K 51-70, Q6_K 51-66 on the fp16 path. The dense agent's Q4_K_M holds 76.5 % of its block weights in Q4_K, 17.1 % in Q6_K (half of `ffn_down`, `attn_qkv`, `attn_v`), 6.1 % in Q5_K (`ssm_out`) (`measured-here`, gguf-py). In the served 58k profile Q6_K takes 7.5 s against Q4_K's 26.3 s, 1.28x Q4_K's time a weight: at Q4_K's rate it would save ~1.6 s, ~2 % of that prefill. That is an upper bound for a form this ISA does not offer: CUDA's int8 Q6_K applies one MMA a 16-weight sub-block (m16n8k16, `code`, `mmq-vec-dot.cuh`), and the Intel int8 DPAS is k32 only (`code`, the extension builtins), so per-16 scales cost two DPAS a 32-block with half the operand zero, the fp16 rate. The fp16 Q6_K kernel's shape was searched in 0014. **Met** for Q5_K/Q6_K: measured, no int8 form (corrects the first entry of this row, which counted ~11 % of the weights) |

B is met when B1 has no unbuilt entry of 2 % or more, B2 has no untried
reference mechanism, and B3's gap is explained or closed.

## C. Fit to release as 1.0

| # | criterion | state |
|---|---|---|
| C1 | DESIGN.md §7 and the README describe what ships: engines, numbers, the default | **met** by the overhaul of 2026-10-15: DESIGN §7 has 0.8.0's served numbers per service and the engine (Flash-Next's from IQ3_XXS since it became the default); the README the same (`measured-here`) |
| C2 | A soak: several days of the three services under real use on both cards with no wedge, no eviction and no restart, logged | **dropped** (operator, 2026-10-16): not a 1.0 criterion. The record keeps the B60's wedge after a long idle, xe eviction at thin VRAM margins and copy-engine resets, each recovered |
| C3 | The dependency base is a stated choice: llama.cpp at a pin with N patches and how a rebase is done; marfrit-openvino +pN and its role | **met** (step 3 done 2026-10-16: every patch's red case and the KL arms of one and four sequences a batch on the new base, `contrib/llama.cpp/README.md`, "Moving the pin"; 2026-10-15): the pin moved to upstream master `3e4dfa8` (288 commits on) with the 56 patches ported, the procedure and the move recorded (`contrib/llama.cpp/README.md`, "Moving the pin"; `docs/campaigns/research-upstream-llama-2026-10.md`), suites, KL and the three services measured on the new base (`measured-here`); marfrit-openvino: retired with the OpenVINO engine (A6); removed from the dev host 2026-10-16, the published package kept as 0.7.x's dependency |
| C4 | The interface promise: the HTTP surface, the unit flags and the supported environment switches that stay stable within 1.x | **decided** (operator, 2026-10-16): the HTTP surface, the command-line flags and the operational switches (`docs/switches.md`, the class a unit file may set) are stable within 1.x; tuning, off-switch, red-case and diagnostic switches may change without notice |
| C5 | The release procedure as used for 0.7.x (pre-tag served check, second-model review, signed repository, fresh container) | met |

## Order of work, proposed

1. B1 and B2's open references, by price: the GDN stage (patch 0050, the
   snapshots into the cache, done: +1.2-5.3 % decode by service; 0051, the
   gating, done: +0-1.2 %; the conv stage next), then OpenVINO-style post-op fusion of the
   activation quantization.
2. B3: measure Strata's tokens per round on the same texts; build only what
   that measurement prices.
3. A2, A6, A5: operator decisions, priced where possible.
4. C1, A7 and C4 together: the re-baselined DESIGN/README and the list of
   supported switches.
5. C3 as the last gate before the tag (C2, the soak, dropped by the operator, 2026-10-16).

Release plan (operator, 2026-10-16): 0.8.1 once C3's red cases and KL arms
pass (the opt-in switches removed); then DFlash2 on the agent (B2); then
1.0.0.

# 1.0.0 — feature complete, almost no gains by obvious changes

**Draft** (2026-10-14). The operator's definition: arcint is 1.0 when it is
feature complete and almost no gains are left in obvious changes. This
charter turns the two halves into checkable criteria. Each criterion names
its evidence and its state. The operator amends it; an agent does not.

## A. Feature complete

"Complete" means for the services arcint is built for: the dense agent and
the coder on the libllama engine, Flash-Next on one GPU plus host RAM, on the
two Arc cards. It does not mean every model or every engine feature.

| # | criterion | state (2026-10-14) |
|---|---|---|
| A1 | The three services run from the released package, on the engine the README names as the default, with answers right (facts, needle, the acceptance task) | met in practice: all three run libllama from 0.7.8 (`measured-here`, the 0.7.8 pre-tag check). **Open:** the README still calls OpenVINO "the default" |
| A2 | Flash-Next's serving shape is settled: one quant, one KV type, one context, one expert-cache size, with the quant inside the answer-level bar or the shortfall recorded as accepted | **open:** IQ2_XS misses the bar against unsloth's Q8_0 (KL 0.294, README); UD-Q3_K_XL meets it but decodes at 12-13 t/s. Accept IQ2_XS, or name what replaces it |
| A3 | Speculative decoding works on all three (MTP) with rollback of the recurrent state | met (`measured-here`, acceptance counts in every gate today) |
| A4 | Long context at the deployed sizes: the agent at 245,760, the coder at 98,304, Flash-Next at 131,072, with needles right and nothing evicted | met (`measured-here`, the 0045 and 0.7.8 gates) |
| A5 | Checkpoints and prefix reuse on the libllama engine (`--llama-checkpoints`) | met for one lane; **decide** whether a shared prefix cache across lanes belongs in 1.0 (not built) |
| A6 | The OpenVINO engine's role is decided: kept as the documented alternative with its own gates, or retired from the default build | **open** |
| A7 | Every switch that 1.0 keeps is documented; diagnostic and red-case switches are named as such | **drafted** (2026-10-15): `docs/switches.md` lists the 90 switches of the llama.cpp series and arcint's 38 in six classes (operational, tuning, opt-in alternative, off switch, red case, diagnostic) with default, patch and evidence. **Operator:** which stay stable in 1.x (C4), and whether to remove the seven opt-in alternatives that measured slower or neutral, `GGML_OPENCL_FUSE_RMS_SCALE` (an alias of `FUSE_0049`) and `ARCINT_PA_HOST_INPUTS` (no recorded result) |

## B. Almost no gains by obvious changes

"Obvious" means one of these: a mechanism one of the reference engines uses
for the same model or layer that arcint does not have; a lever an ablation
or a profile names with a price; or a known fault in a kernel. A gain counts
if it moves served tokens per second of a deployed service with the answers
unchanged.

| # | criterion | state (2026-10-14) |
|---|---|---|
| B1 | No priced lever of 2 % or more of served decode or prefill is left unbuilt on any service | **open:** Flash-Next's prompt buffers: built as 0052 (reserve to a cell budget, +10-13 % decode at 131K, `measured-here`), on the unit since 0.7.9; the low-bit expert decode by multiplies (0053, +0.9 %, `measured-here`), and what is left of that kernel is integer and 64-bit address work with no single named lever; the GDN stage's conv-stage fusion (vllm-xpu-kernels' conv stage; the delta stage is built: 0050 +1.2-5.3 %, 0051's gating +0-1.2 % by service, Flash-Next neutral), priced 2026-10-15 by a ceiling at ~1.1 % of the agent's decode and ~1.4 % of the coder's (`measured-here`), below 2 %; the activation quantization folded into its producers (priced 2026-10-15 by a ceiling: at most ~1.4 % of the agent's round, ~1.1 % of Flash-Next's by launch count, `measured-here`; below 2 %); the rest of the decode kernel count (~13 % of a Flash-Next round, the sum of many small fusions, none obvious alone) |
| B2 | Every reference engine's mechanism for the served models is either built, measured as not paying here, or set aside by the operator. The references are Strata, FreeToken and Infernix for Flash-Next; llama.cpp's CUDA and Vulkan backends; vllm-xpu-kernels and sycl-tla on Xe2; and OpenVINO's GPU plugin | mostly met: this week's audits built or measured the reference's prompt attention, its wave split, CUDA's per-call K/V conversion, MMQ's shared tile, sycl-tla's payloads and Strata's tables. OpenVINO's post-op fusion, as the activation quantize folded into its producer: priced below 2 % (B1). vllm-xpu-kernels' GDN spec-decode rollback (2026-10-15): the state read from the accepted token's slot with no copy back is what the libllama engine already does (`code`, `llama-memory-recurrent.cpp` `s_copy`: a rejected draft sets the snapshot plane the next step reads); what differs is the conv state's K snapshot copies, priced by a ceiling at ~1.1 % (agent) and ~1.4 % (coder) of decode (`measured-here`, B1). Infernix's expert-cache policy: built per layer as 0055 (opt-in), +1.3 % Flash-Next decode against Strata's adapt (`measured-here`); **operator:** which policy the unit runs. Its global pool (per-layer slot tensors) is not built; the warm start from saved state, tried, seeded worse than the calibration profile (`measured-here`) |
| B3 | Flash-Next is at parity with Strata on the same card, or the gap is explained by a measured cause | prefill: met (677 t/s at the 20k needle against Strata's 620, `measured-here`). Decode on Strata's own request (`measured-here`, 2026-10-14): a round costs the same (63.1 against 63.4 ms at Strata's draft setting); with 3 drafts and Strata's stop at 0.5 arcint decodes 37.9 t/s against Strata's 37.8 (warm run; the first run 35.1 against 37.2). **Met** on the warm run once the Flash-Next unit takes `--llama-mtp 3 --llama-mtp-min-p 0.5` (operator's call); the first run's gap is the expert cache's warm-up |
| B4 | The kernels that dominate device time are at their measured ceiling for their design | met for the B60's prompt attention, the A770's quantized attention staging and the int8 Q4_K GEMM (`measured-here`, the campaign docs); Q5_K/Q6_K GEMMs measured (2026-10-15, `measured-here`, B60, test-backend-ops perf at 512 columns over the 27B's shapes): Q4_K 73-83 TFLOPS on the int8 path (0013), Q5_K 51-70, Q6_K 51-66 on the fp16 path. The dense agent's Q4_K_M holds 76.5 % of its block weights in Q4_K, 17.1 % in Q6_K (half of `ffn_down`, `attn_qkv`, `attn_v`), 6.1 % in Q5_K (`ssm_out`) (`measured-here`, gguf-py). In the served 58k profile Q6_K takes 7.5 s against Q4_K's 26.3 s, 1.28x Q4_K's time a weight: at Q4_K's rate it would save ~1.6 s, ~2 % of that prefill. That is an upper bound for a form this ISA does not offer: CUDA's int8 Q6_K applies one MMA a 16-weight sub-block (m16n8k16, `code`, `mmq-vec-dot.cuh`), and the Intel int8 DPAS is k32 only (`code`, the extension builtins), so per-16 scales cost two DPAS a 32-block with half the operand zero, the fp16 rate. The fp16 Q6_K kernel's shape was searched in 0014. **Met** for Q5_K/Q6_K: measured, no int8 form (corrects the first entry of this row, which counted ~11 % of the weights) |

B is met when B1 has no unbuilt entry of 2 % or more, B2 has no untried
reference mechanism, and B3's gap is explained or closed.

## C. Fit to release as 1.0

| # | criterion | state |
|---|---|---|
| C1 | DESIGN.md §7 and the README describe what ships: engines, numbers, the default | **open:** §7.0's milestone table and numbers predate the libllama engine (Flash-Next at 6.5 t/s decode) |
| C2 | A soak: several days of the three services under real use on both cards with no wedge, no eviction and no restart, logged | **open:** the record has the B60's wedge after a long idle, xe eviction at thin VRAM margins and copy-engine resets, each recovered, none soaked |
| C3 | The dependency base is a stated choice: llama.cpp at a pin with N patches and how a rebase is done; marfrit-openvino +pN and its role | **open:** 53 patches over llama.cpp `bed0a85` (2026-10-02); no rebase procedure recorded. A dry run (`measured-here`, 2026-10-15): upstream master `a0e6dc7` (2026-10-14) is 269 commits on, 8 of them in `ggml-opencl`, and patch 0001 already conflicts in `load_cl_kernels`, which upstream's `f2cc728` split (`ggml_opencl_make_compile_opts`) and which most of the series extends. A rebase is a port of the backend part, not a re-apply |
| C4 | The interface promise: the HTTP surface, the unit flags and the supported environment switches that stay stable within 1.x | **open** (see A7) |
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
5. C3, then C2, the soak, as the last gate before the tag.

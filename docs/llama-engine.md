# The libllama engine (since 0.5.5)

arcint serves without OpenVINO: `--engine llama --gguf FILE` runs the GGUF
through libllama with ggml's OpenCL backend on the Arc card
(`src/exec/backend_llama.cpp`). arcint keeps the HTTP server, the chat
template rendering (the GGUF's own template), the sampler, stop handling and
the lanes; llama.cpp keeps the weights, tokenizer, attention KV and the gated
delta-net state, one sequence per lane.

Build: `-DARCINT_LLAMA=ON -DARCINT_LLAMA_DIR=<llama.cpp tree>`, the tree at
commit `bed0a85` (ggml-org/llama.cpp, 2026-10-02) with
`contrib/llama.cpp/patches` applied (`contrib/llama.cpp/README.md`; configure
warns when they are missing). It builds as a static subproject with
`GGML_OPENCL=ON`, Adreno kernels off, kernels embedded. `--device GPU.N` is
the N-th GPU in OpenCL's enumeration, or a substring of the device name
(`B60`, `A770`).

## Kernels

ggml's OpenCL kernels are tuned for Adreno, and on Intel they decided both
models' speed: the K-quant matrix products were the whole cost; K-quant
`MUL_MAT_ID` was unsupported, so the coder's experts ran on the CPU; and a
stock `supports_op` rule for Adreno (decline K-quant `MUL_MAT` at 512 or more
columns unless the Adreno GEMM applies) made llama.cpp place the coder's 60
small K-quant tensors of the gated delta-net layers on the CPU, a GPU-CPU-GPU
round trip in every layer of every token. Patch 0001 adds Intel kernels for
Q4_K/Q5_K/Q6_K -- an integer-dot matrix-vector product for 1 to 4 columns
with a two-term int8 activation, and an fp16 DPAS GEMM on the XMX units with
the weights dequantized per work-group tile, for `MUL_MAT` and `MUL_MAT_ID`
-- and exempts the shapes these take from that rule
(`contrib/llama.cpp/README.md`).

Patches 0002-0008 add the IQ types (Flash-Next), the gated delta-net,
the decode's small kernels, the few-row F32 projections, a MoE GEMM in the
shape of arcint's OpenVINO patch 0064, decode attention that does not
slow with context, and prompt attention on the XMX units
(`contrib/llama.cpp/README.md`, one section per patch).

`measured-here`, 2026-10-03, llama-bench, one sequence (A770 GT clock pinned
at 2000 MHz):

| model, card | prefill 512: stock / 0001 / 0001-0007 / 0001-0008 | decode: stock / 0001 / 0001-0007 |
|---|---|---|
| Qwen3.8-27B Q4_K_M, B60 | 71.1 / 417.9 / 521 / 633 t/s | 10.3 / 18.0 / 19.75 t/s |
| Qwen3.6-35B-A3B coder Q4_K_M, A770 | 111.2 / 605.0 / 1,292 / 1,608 t/s | 7.7 / 37.7 / 47.8 t/s |

0008 takes only prompts (more than 8 query rows); decode is 0001-0007's.
The 0001-0008 column and the figures below are from 0008 before its review
(the reviewed kernel is 5-7 % faster at test-backend-ops). Prefill at
depth (0001-0008): dense 540 t/s at 512 tokens after 4,096, 590
at 4,096 tokens, 363 at 512 after 16,384; coder 1,032 at 512 after 4,096,
1,322 at 4,096 tokens, 889 at 4,096 after 4,096.

Dense prefill on the B60 after the 2D-block GEMMs (`measured-here`,
2026-10-03, llama-bench): 590 t/s at 4,096 tokens and 633 at 512 with
0001-0008. Then 0011 702 / 758, 0012 816 / 892, and 0013 931 / 1,030.
0013 decodes the weights two to an instruction (inline vISA) and runs Q4_K
on int8 DPAS (`contrib/llama.cpp/README.md`); its int8 activations cost
+0.0005 nats of KL. The coder's prefill on the A770 is unchanged by
0011-0013 (B60-only kernels).

0014 sets tiles found by a genetic search scored at the endpoint
(`docs/campaigns/kernel-autotune-ga.md`, `measured-here`, llama-bench):
- coder, A770: 1,613 -> 1,740 t/s at 512 tokens, 1,345 -> 1,431 at 4,096,
  KL unchanged;
- dense, B60: 930 -> 935 at 4,096 (the same build pair, interleaved).

Decode with context (0001-0007; 0001-0006 in brackets): coder 47.3 (13.6)
t/s at 4,096 tokens and 43.5 (4.2) at 16,384; dense 19.39 (12.8) and 17.95
(6.4).

The coder's step from 20.9 to 37.7 t/s decode in 0001 was the placement
fix alone: the same kernels, the 60 small tensors of its gated delta-net
layers on the card.

For scale, the OpenVINO path (README): coder 1,379 prefill (4k) / 43.9
decode, dense 1,141 / 24.6 (MTP on), 24.0 plain. What is left
(`measured-here`, profiles with `GGML_OPENCL_PROFILING`; the ranked levers
are in `docs/campaigns/llama-engine-kernel-gap.md`):

- dense decode, B60: the K-quant matvecs, 22.7 of 26.4 ms device time a
  token at ~400 GB/s; the OpenVINO export's int4 is ~18 % fewer bytes than
  Q4_K_M;
- dense prefill: the K-quant GEMMs (Q4_K 464, Q6_K 128 ms of 983 per 512
  tokens, profiled before 0008; prefill attention was 188 ms of it, and
  0008's kernel is 18x faster at test-backend-ops). After 0013 the int8
  Q4_K kernel waits on load latency (~44 % of the int8 rate), and Q6_K has
  no int8 form yet (`docs/campaigns/llama-engine-kernel-gap.md`, lever 7);
- coder prefill: the MoE gate/up GEMM (117 ms of 389, before 0008;
  prefill attention was 97 ms);
- decode, both: ~1,100 kernel launches a token; NEO exposes no
  `cl_khr_command_buffer`, so fusion is the lever.

## Answers

Served (`--engine llama`, one lane, n_ctx 32,768, arcint's sampler,
`measured-here`, final build): the acceptance task at temperature 0 scores
10/10 on both models. Sampled at the card temperature 0.7, six runs per arm:
coder 6/6 at 10/10; dense 9, 0, 10, 10, 0, 10 against 10, 10, 8, 10, 9, 10
with ggml's float kernels; twenty runs per arm, 11 of 20 at 10/10 against
10 of 20 (mean 7.0 against 7.4, a standard error of ~1.3 on the difference:
the arms do not separate). The two zeros of the six are slips of the sampled answer --
a `local function` declared below its first use, and an answer that rewrote
itself into a second code block (the scorer reads the first) -- and six
runs do not separate the arms (one-sided Fisher p ~ 0.25 for 0-2 point
answers, 4 of 12 over the two candidate builds against 0 of 6). At
temperature 0 the dense answer depends on the last bits of the forward:
builds whose KL agrees to four places scored 10/10 and 0/10, the 0/10 ones
an answer that rewrote itself or left its code fence open.

KL against the same GGUF on the CPU backend (16 chunks of 512 tokens, English
and C++), baseline arm ggml's float kernels on the same card:

| | baseline | patched |
|---|---|---|
| dense, prefill (batched) | 0.003560 nats, top-1 97.84 % | 0.003560, 97.82 % |
| coder, prefill (batched) | 0.00728, 95.93 % | 0.00696, 96.10 % |
| dense, decode (1 token per ubatch) | 0.003560, 97.84 % | 0.003561, 97.82 % |
| coder, decode (1 token per ubatch) | 0.00759, 95.74 % | 0.00714, 96.05 % |

The single-term q8_1 activation of the first matvec moved the dense decode
by 0.0026 nats and 0.93 points of top-1 agreement; the two-term activation
replaced it at no measured decode cost on either card.

## Flash-Next

`qwen4exp` with `--llama-cpu-moe N`: the first N layers' experts stay in host
memory, memory-mapped, computed by llama.cpp's CPU backend; the IQ kernels
of patch 0002 run the rest on the card. Served on the B60 with 16 expert
layers on the card: the needle answered, 49.2 t/s prefill at 20k, 10.9 t/s
decode (`docs/campaigns/flash-next-llama-engine.md`).

## MTP (`--llama-mtp N`)

The GGUF's own MTP head drafts up to N tokens; the target verifies them in
one decode, and arcint's sampler walks the verified rows in order, keeping
the draft while it agrees -- the emitted tokens are what the plain loop
would sample. The drafter is the single-head case of llama.cpp's `draft-mtp`
(`common/speculative.cpp` at the pin) rebuilt on libllama in
`src/exec/llama_spec.cpp`: llama.cpp's common library compiles another
cpp-httplib (0.58) than arcint's (0.18) into the binary, and the server
then failed to bind. A rejected draft rolls back on the device through the
target's per-token recurrent snapshots (`n_rs_seq` = N). The logits rows
are bounded to 1 + N per lane: unbounded, llama.cpp reserved them for a
whole ubatch, vocabulary-wide, in both contexts, and the coder with its MTP
layer hung the A770 (two engine resets).

Served (`measured-here`, 2026-10-03, the acceptance task through arcint):

| model, card, drafts | decode, plain -> MTP | accepted | acceptance task |
|---|---|---|---|
| dense 27B, B60, 3 | 18.0-19.5 -> 30.8-35.7 t/s | 69-88 % | 10/10 at temperature 0; 13 of 20 sampled at 10/10 (plain arms 10, 11, 15 of 20) |
| coder, A770, 2 | 47.5 -> 67.1-68.9 t/s | 86-91 % | 10/10 at temperature 0; 9 of 9 sampled |

The verify walk (`src/exec/verify_walk.h`) is a pure function with its own
tests (`tests/test_verify_walk.cpp`: full and partial acceptance, a stop
token, the token budget, cancellation, and seeded sampled walks that draw
exactly what the plain loop draws; dropping its `observe()` fails them). A
review found no blocker; its fixes are in: the lane cleared when a request
fails mid-step, the drafter's carried row reset after a trimmed prefix, the
context budget of the plain loop (one token more at the limit before),
`/props` reporting MTP.

The greedy text with MTP matched the plain path's for the first 2,585
characters of the dense answer, then diverged at one line (the verify's
batched numerics; plain builds differ from each other the same way).

The coder with its MTP layer fits the A770 up to a 16,384-token context
(prefill 1,070 t/s); at 24,576 prefill drops to 585 t/s and at 32,768 to 67
with decode at 22 t/s: VRAM paged over the link. The dense model serves
122,880 tokens on the B60 with f16 KV (`--n-ctx 122880`, 2026-10-04,
`measured-here`). 24.4 of 25.7 GB of VRAM are in use. A 30,065-token prompt
prefills at 528 t/s at both 122,880 and 32,768, and a 120,945-token prompt
in 579 s.

## Quantized KV (`--llama-kv K[:V]`)

`q8_0` and `q8_0:q4_0` run on the Intel attention kernels of
`contrib/llama.cpp` 0015. 0015 also has a 4:4 kernel; arcint refuses 4:4
because it drops the dense 27B's top-1 agreement by 1.2 points, past the
answer-level bar. Any other pair has no kernel and would dequantize
all of K and V to f32 on every call, so arcint refuses it. A quantized V
makes llama.cpp force flash attention on (no automatic fallback). A layer
whose head size the OpenCL backend does not take then runs its attention
on the CPU. Measured on the head-size-256 Qwens only (the dense 27B and the
coder). The MTP draft context takes the same types.

The prices, KL and speed are in the 0015 section of
`contrib/llama.cpp/README.md`. On the dense 27B:
- 8:8 costs 4 % of prefill at 4k, 8 % at 16k depth;
- it costs 4-5 % of decode;
- its KL is within the noise of f16's.

**Dense 27B, B60, with MTP** (5 drafts, the 40,960-id head) and
`--llama-kv q8_0`:
- serves `--n-ctx 131072` (peak VRAM 23.06 of 25.7 GB). With f16 KV, MTP
  at 131,072 overcommitted the card: a copy-engine reset;
- a 128,133-token prompt prefills in 772 s (166 t/s), and decodes at
  7.5 t/s at that depth (38 % draft acceptance, a repetitive prompt);
- the acceptance task scores 10/10 at temperature 0, decode 47.1 t/s.
  The 52.6 t/s in the table above is the same prompt and drafts with f16 KV
  at the default 32,768 context. The difference, 10 %, is more than
  llama-bench's 4-5 % for 8:8. The cause is not separated: the KV type
  against the context size (the verify rows run through the quantized
  split decode).

**The deployed agent (0.5.6, 2026-10-04, `measured-here`).** The unit as
shipped (`--llama-kv q8_0`, 131,072 tokens, MTP 5) scores 8/10 at
temperature 0, four runs out of four. The two lost cases are CRLF and LF
row handling. The probe with the same flags had scored 10/10 once.

Greedy runs at `--n-ctx 65536` on the same binary and prompt, two each:

| KV | MTP 5 | plain |
|---|---|---|
| q8_0 | 10, 10 | 2, 2 |
| f16 | 10, 10 | 10, 10 |

A greedy score is one trajectory. It repeats within a configuration and
flips between configurations (KV type, MTP's verify batches, context
size). It cannot separate the KV types. The sampled task can, 30 runs
each, MTP 5, `--n-ctx 65536`:

| KV | mean | runs at 10/10 |
|---|---|---|
| q8_0 | 7.4 | 15 of 30 |
| f16 | 8.13 | 19 of 30 |

- The gap of 0.73 is under one standard error (0.87, per-run spread
  ±3.4).
- The second batch of 20 alone is reversed: q8_0 8.3, f16 8.0.
- The earlier f16 arm at 32,768 had a mean of 7.4.

With KL equal (0.003966 against 0.004034), 8:8 holds the answer-level bar,
and the agent stays on this configuration. The deployed unit's own sampled
runs: mean 7.2 over 10.

**DFlash on this engine.** The pinned llama.cpp carries DFlash and DFlash2
drafting (`code`):
- `common/speculative.cpp`, type `draft-dflash`, which turns DFlash2 on when
  the draft GGUF has a selector top-k;
- `src/models/dflash.cpp`;
- the converter in `convert_hf_to_gguf.py`.

arcint's engine wires only the GGUF's MTP head (`src/exec/llama_spec.cpp`).
DFlash2 runs on the OpenVINO engine (`--dflash`). Wiring llama.cpp's
DFlash2 into this engine is open and unmeasured.

**Verify attention at depth (0.5.7, `contrib/llama.cpp` 0016).** On the
B60, MTP verify calls (4-8 rows) take a kernel that reads K/V once per KV
head for all its query heads and rows. With the agent's flags at 62,597
tokens of depth, decode went from 11.0 to 15.6 t/s (`measured-here`); the
campaign is `docs/campaigns/gqa-small-t-decode.md`.

**Mistral Small 3.2 24B / Cydonia 24B (0.5.8), creative writing on the
B60.** The engine admits the `llama` architecture at that geometry only.
`contrib/llama.cpp` 0017 builds the XMX attention kernels at head size 128.
The README section of 0017 and the CHANGELOG have the numbers. The model has
no MTP head, so decode is plain: 25.2 t/s short, 18.2 at 16k and 8.4 at 89k
depth (`measured-here`). All 40 layers keep KV (85 KiB a
token at q8_0), which puts the ceiling at 98,304 tokens on the B60. Since 0.5.9 the writer
serves bartowski's imatrix Q4_K_M: KL against the model's Q8_0 is 0.016918
with q8_0 KV, against the static quant's 0.020316, top-1 equal
(`measured-here`; the 0017 section of `contrib/llama.cpp/README.md`).

**Coder, A770: smaller GGUFs for its context** (`measured-here`,
2026-10-04). Its Q4_K_M weights (16.06 GB) leave ~16k tokens with MTP. Two
mixes were made from the F16 GGUF with the model's imatrix. Expert gate/up
went to IQ3_XXS and expert down to IQ4_XS; the rest, the MTP layer
included, stayed as Q4_K_M has them. In mix2, layers 0-5 and 34-39 keep
gate/up at IQ4_XS.

KL is against Q8_0 logits on the CPU. A Q4_K_M reference would favour
Q4_K_M itself.

| GGUF | size | KL | top-1 |
|---|---|---|---|
| Q4_K_M (baseline arm) | 16.06 GB | 0.0212 | 93.43 % |
| IQ3_XXS mix | 9.92 GB | 0.0488 | 90.22 % |
| mix2 | 12.69 GB | 0.0402 | 90.86 % |

Both mixes serve `--n-ctx 98304` with MTP (4 drafts) and
`--llama-kv q8_0:q4_0` (14.2 GB of VRAM for the IQ3_XXS mix):
- a 95,323-token prompt prefills at 177-178 t/s;
- decode is 20.7-22.1 t/s at that depth and ~55 t/s shallow;
- the acceptance task scores 10/10 at temperature 0 and 3 of 3 sampled.

Both pass the KL bar and miss the top-1 bar, by 3.2 and 2.6 points against
1. Neither is adopted.

llama.cpp's own loop (`llama-speculative-simple`, raw prompt) for
reference: coder 68.1 t/s with 2 drafts, dense 31.9 with 3; four drafts
fall to 46.0 and 10.7 (a 5-row verify leaves the 4-column K-quant matvec).

Since then (2026-10-03, `docs/campaigns/mtp-cycle-wall.md` has the
references and every number):

- `contrib/llama.cpp` 0009 multiplies K-quants by 2 to 16 tokens on XMX,
  reading each weight once. An 8-row dense step went from 6.0x to 1.21x a
  1-row step.
- The drafter takes draft 0 from the catch-up batch, as Strata's does. That
  needed 0010, a masked-nextn row fix in the MTP graph.
- `--llama-mtp-vocab FILE` drafts from a subset of the vocabulary: the
  draft steps read those rows of the head instead of all 248k.

Served, the same prompt at temperature 0, 10/10 in every run
(`measured-here`):

| model, card | drafts | decode |
|---|---|---|
| dense 27B, B60 | 5, 40,960-id head | 52.6 t/s |
| coder, A770 | 4, 40,960-id head | 79.3 t/s |

The dense model's 20-run sampled arm with that configuration scored 11 of
20 at 10/10 (mean 7.4). The plain arm of the same binary scored 12 of 20
(mean 6.9), and the record's earlier plain arms 10, 11 and 15.

## Not yet on this engine

Flash-Next's MTP (`qwen4exp` is in llama.cpp at the pin), conversation
state (P6) and GPU prefill from the pinned bank (P5) are open.

Open observation: on the B60, xe logged GPU page faults with compute-engine
resets in windows where `test-backend-ops` ran (2026-10-03); a run of every
`MUL_MAT`/`MUL_MAT_ID` case set reproduced none, and every result in those
windows was correct. Not attributed.

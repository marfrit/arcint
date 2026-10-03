# The libllama engine (0.5.3.1459)

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
  0008's kernel is 18x faster at test-backend-ops);
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
with decode at 22 t/s: VRAM paged over the link. The dense model fits the
B60 at 32,768.

llama.cpp's own loop (`llama-speculative-simple`, raw prompt) for
reference: coder 68.1 t/s with 2 drafts, dense 31.9 with 3; four drafts
fall to 46.0 and 10.7 (a 5-row verify leaves the 4-column K-quant matvec).

## Not yet on this engine

Flash-Next's MTP (`qwen4exp` is in llama.cpp at the pin), conversation
state (P6) and GPU prefill from the pinned bank (P5) are open.

Open observation: on the B60, xe logged GPU page faults with compute-engine
resets in windows where `test-backend-ops` ran (2026-10-03); a run of every
`MUL_MAT`/`MUL_MAT_ID` case set reproduced none, and every result in those
windows was correct. Not attributed.

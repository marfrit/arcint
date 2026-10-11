# agent-context — the dense agent's context past 131,072 tokens on the 24 GB card

**Closed 2026-10-12** (operator: "work on extending the agent context"): the agent serves 229,376 tokens with q8_0:q4_0 KV (196,608 on 2026-10-11, before the card's real size was known).

## Charter

The dense agent (Qwen3.8-27B Q4_K_M GGUF on the libllama engine, MTP 5,
the B60) serves 131,072 tokens with a q8_0 KV cache. The model is trained
for 262,144 (`qwen35.context_length`). Serve as much more as the card
holds, with the answers right at the new depth.

## Reference to follow

- **llama.cpp's KV types** (`-ctk` / `-ctv`, `code`): the KV cache in
  q8_0 for K and q4_0 for V. arcint already ships the pair as
  `--llama-kv q8_0:q4_0` on its own kernels (`contrib/llama.cpp` 0015) and
  the coder serves with it.
- **Its price on this model** (`measured-here`, 0015's section of
  `contrib/llama.cpp/README.md`, B60, KL against the CPU reference):

  | KV | KL | top-1 | prefill 4k / decode (llama-bench, no MTP) |
  |---|---|---|---|
  | 8:8 (the unit today) | 0.003966 | 97.65 % | 896 / 18.9 t/s |
  | 8:4 | 0.005756 | 97.33 % | 893 / 18.9 t/s |
  | 4:4 (refused) | 0.007800 | 96.52 % | |

  8:4 is inside the answer-level bar against 8:8 (+0.0018 nats, -0.32
  points); 4:4 is not (-1.13 points).

## The budget (`code` and `measured-here`)

- **KV per token:** 16 full-attention layers (`full_attention_interval`
  4 of 64), 4 KV heads, K and V of 256: 32,768 values a token. q8_0 is 34
  KiB a token (4.25 GiB at 131,072; the lanes campaign measured 5.44 GiB for
  163,840 cells), 8:4 26 KiB (41 % of f16). The MTP layer adds its own KV in
  the draft context.
- **The card:** 24,480 MiB, 24,433 free when idle (xe debugfs `vram_mm`; the
  "22.71 GiB usable" this section first used was wrong, see below). The unit holds 21,867 MiB resident after
  load and three answers (2026-10-11, fdinfo, a check before the windows); its
  record has a 23.06 GB peak at a 128k prompt.
- **Arithmetic, not a measurement:** with 8:4, 196,608 tokens take ~0.6 GiB
  more than today (~22.0 GiB); 262,144 would need ~2 GiB more than the card
  holds with these weights (against the wrong 22.71 GiB; corrected below). The compute buffer's growth with n_ctx comes on
  top and is measured below.

## Gate

At the new size, on the B60, with the unit's flags otherwise unchanged:
- **Answers right:** the capital; a needle at 40 % depth of a prompt within
  2k tokens of the new cap; the acceptance task greedy not below the unit's
  record (8/10).
- **Quality:** 8:4 against 8:8 on today's build inside the answer-level bar
  (KL +0.03 nats, top-1 -1 point), dense-ref.kld, 16 x 512.
- **No eviction:** the process's GTT stays flat through the deepest prefill
  and decode (fdinfo every second); a verify at depth in milliseconds, not
  seconds.
- **Speed per phase:** decode at a 4k prompt and at ~130k within the
  run-to-run spread of the 8:8 unit's record (36.6 and 19.3 t/s with MTP 5);
  prefill at 4k and 130k likewise (824-910, 252-272 t/s).

## Current state

Two windows on the B60 (`measured-here`, 2026-10-11, the packaged arcint
0.7.3, the unit's flags, MTP 5; the prompts are the release tree's sources,
the needle a sentence at 40 % depth).

**Load probes** (VRAM after load, and the peak after one short answer):

| KV, n_ctx | KV buffer (main context; the MTP draft context adds 208-338 MiB) | after load | peak |
|---|---|---|---|
| q8_0, 131,072 (the unit before) | 4,352 MiB | 21,389 | 21,926 |
| 8:4, 131,072 | 3,328 | 20,301 | 20,557 |
| q8_0, 163,840 | 5,440 | 22,545 | 22,854 |
| 8:4, 196,608 | 4,992 | 22,069 | 22,673 |
| 8:4, 212,992 | 5,408 | 22,511 | 23,145 |

- The recurrent state takes 897.75 MiB whatever the context: six copies of
  the GDN state, one per MTP draft position plus one (`n_rs_seq` = the draft
  count; `code`, llama.cpp `src/llama-memory-recurrent.cpp:101`).
- A deep prompt adds 704-736 MiB over the load (runtime OpenCL buffers).

**8:8 against 8:4 at n_ctx 131,072** (same prompts, one process each, one run per cell):

| | 8:8 | 8:4 |
|---|---|---|
| 4,027-token prompt: prefill / decode | 786.8 / 37.8 t/s | 804.0 / 36.6 t/s |
| 128,839-token prompt: prefill / decode | 256.1 / 16.6 t/s | 256.7 / 17.0 t/s |
| peak VRAM | 21,953 MiB | 20,865 MiB |

**At depth, 8:4:**

| n_ctx | prompt | prefill | needle | decode at depth | peak VRAM / GTT |
|---|---|---|---|---|---|
| 212,992 | 209,291 | 1,199 s (174.5 t/s) | right | 13.0 t/s (300 tokens) | 23,247 / 26 MiB |
| 196,608 | 195,451 | 1,059 s (184.6 t/s) | right | 12.0 t/s (274 tokens) | 22,773 / 26 MiB |

**Corrected 2026-10-11:** the margins were first computed against "22.71 GiB usable". The
card has 24,480 MiB (24,433 free when idle, xe debugfs `vram_mm`; Flash-Next runs at a
24,259 MiB peak without eviction). 196,608 keeps ~1,660 MiB at the deepest prompt, 212,992
~1,190 MiB: both are deployable, and larger sizes are open (being measured).

**The gate at 196,608, 8:4:**
- the capital and the needle right;
- the acceptance task 10/10 greedy, one run (the unit's record 8/10);
- KL on today's tree against `dense-ref.kld`, 16 x 512: 8:8 0.004099 /
  97.745 %, 8:4 0.005950 / 97.304 % (+0.0019 nats, -0.44 points: inside the
  bar);
- GTT flat at both depths;
- speed: **amended after the measurement.** The gate named the 8:8 unit's
  2026-10-08 record (36.6 / 19.3 t/s decode, 824-910 / 252-272 t/s
  prefill). This session's own 8:8 arm on these prompts falls below that
  record too (4k prefill 786.8; 128k decode 16.6), so the record's prompts
  and build are not comparable. The comparison made is 8:4 against the
  same-session 8:8 arm, one run per cell: equal within one run's spread.
  The same two prompts on the deployed unit at 196,608 (8:4, after the
  windows): 4k prefill 836.9 t/s, decode 36.6; 128,840 tokens prefill
  256.8 t/s, decode 17.7 -- against 8:8's 786.8 / 37.8 and 256.1 / 16.6.

**Passed** on that comparison.

**Deployed** 2026-10-11: the agent unit and `contrib/systemd/arcint-agent.service`
take `--n-ctx 196608 --llama-kv q8_0:q4_0`. Checked after the deployment,
outside the windows: the endpoint and the proxy report n_ctx 196608, the
capital is right.

**Not reached** (arithmetic from the measured probes, corrected for the
card's real size): the model's 262,144 would peak ~24,540 MiB, ~110 MiB above
the idle free memory, with these weights and MTP
5 (+442 MiB after load per 16,384 tokens, times four, on 196,608's deep
peak). The levers would be smaller weights (a quality question) or fewer MTP
drafts (~150 MiB of recurrent state each, decode speed).

**Larger sizes** (`measured-here`, 2026-10-12, the same build, flags and prompts; margins
against the 24,433 MiB free on the idle card):

| n_ctx | prompt | prefill | needle | decode at depth | peak VRAM | margin |
|---|---|---|---|---|---|---|
| 229,376 | 223,628 | 1,356 s (164.9 t/s) | right | 13.3 t/s | 23,721 MiB | 712 MiB |
| 245,760 | 236,938 | 1,507 s (157.2 t/s) | right | 11.5 t/s | 24,195 MiB | 238 MiB |

GTT flat in both. At 229,376 the acceptance task scores 10/10 greedy (one run).
**Deployed 229,376** (2026-10-12): the endpoint reports n_ctx 229376 and answers right.
245,760 runs too, with a 238 MiB margin (the Flash-Next unit runs at 174 MiB); an eviction
there would make the agent crawl rather than fail, so the deployed size keeps 712 MiB.

## Where it lives

- `--n-ctx`, `--llama-kv` (`src/config.cpp`, `src/exec/backend_llama.cpp`).
- The unit's flags: `contrib/systemd/arcint-agent.service`.

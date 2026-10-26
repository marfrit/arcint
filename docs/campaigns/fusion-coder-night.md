# fusion-coder-night — the coder's small ops, the draft head, the prompt (2026-10-25/26)

One night of served A/Bs on the coder (Qwen3.6-35B-A3B, iwolucion v3 shq8 GGUF,
A770, 98,304 ctx, MTP 4, q8_0:q4_0 KV), with checks on the agent (dense 27B,
B60) and Flash-Next (IQ3_XXS, B60). Starting point: the coder re-count of
`research-strata-qwen36.md` (a token: 1,635 launches, 19.25 ms of device time;
a 5-token verify step: 1,962 launches, 39.1 ms, `measured-here`) and the leads
of `research-arc-community.md`.

Every speed row is `tests/acceptance/ab.py` (the release gate, majority-answer
decode reading) unless it says otherwise; "per verify step" is the served
log's verify time over its verify count. Every candidate is one binary with a
switch, so old and new differ only by an environment variable or a flag.

## Dispositions

| change | where | result | class |
|---|---|---|---|
| **0064** the fused top-k MoE router on one sub-group a token, register buckets by expert count | llama.cpp patch (OpenCL) | router 34.1 -> 13.5 us a launch (1.364 -> 0.542 ms a token); coder per verify step -1.4 to -2.5 %; Flash-Next PASS (+0.7 % prefill, +0.4 % decode); KL 0.023656 -> 0.023594, same-top 95.42 -> 95.37 % (bar PASS); the task 10/10 x 4. The coder A/B FAILs on two answers: the softmax sum order changes each greedy text and its draft acceptance | measured-here |
| **0065** one launch for a run of narrow strided copies (the conv-state rollback snapshots: K copies a delta-net layer) | patch | coder PASS, decode +1.8 to +1.9 %, same answers; snapshots 150 launches / 0.90 ms -> 30 / 0.175 ms a step | measured-here |
| **0066** the Intel GDN generation kernel for batches up to N tokens (`GGML_OPENCL_GDN_TG_MAX=8`: a verify) | patch | coder PASS, decode +1.1 to +1.2 %, same answers; GDN 1.044 -> 0.674 ms a step | measured-here |
| 0065 + 0066 on the agent | | prefill PASS (55k 710.6 -> 711.4, 138k 499.4 -> 499.9); decode +2.3 to +2.4 %, same answers and acceptance (a separate run: the served check has no decode metric) | measured-here |
| 0064 + 0065 + 0066 on Flash-Next | | PASS, prefill-20k +0.7 %, decode +1.2 %, per verify step -1.5 % | measured-here |
| **`--llama-mtp-vocab`** on the coder (HyperQwen's 40,960 rows; measured on the coder 2026-10-03, never on its unit) | unit flag | decode +10.1 to +12.6 %, the task 10/10 x 4; the draft head's full 248,320-row matvec was 4.2 ms of a 36.7 ms step | measured-here |
| **0070** `llama_set_embeddings_nextn_skip`: with the draft head the MTP context runs unmasked, and every prompt row's MTP output was copied to the host unread | patch + `llama_spec.cpp` | with the vocabulary, prefill-41k 723.25 -> 729.20 t/s (the -0.8 % the vocabulary cost, gone); Flash-Next's -0.9 % gone too | code + measured-here |
| `--llama-mtp-vocab` + 0070 on Flash-Next | unit flag | PASS, decode +2.9 %, prefill equal | measured-here |
| **ubatch 1024** for the coder's prefill (`ARCINT_LLAMA_UBATCH`; today min(n_batch, 512)) | experiment switch | prefill-41k +15.2 %, decode equal, peak VRAM +219 MiB, no GTT; KL identical; **2048**: the compute buffer spills 4.1-4.3 GiB to GTT, prefill swings 9 % -> busted on the 16 GB card | measured-here |
| **the stack** (0064 + 0065 + 0066 + 0070 + ubatch 1024 + the vocabulary) vs the released 1.2.6 | | PASS: decode 51.07 / 47.57 / 44.13 -> 58.03 / 57.97 / 50.63 t/s (+13.6 / +21.9 / +14.7 %), prefill-41k 725.97 -> 839.73 (+15.7 %), peak VRAM 15,886 -> 16,172 MiB; the task 10/10 x 2 | measured-here |
| MTP 5 instead of 4 with the stack (cheap draft steps) | unit flag | busted: decode -3.6 to -9.2 %, acceptance 37.9 -> 33.6 % (decode-a), per verify step 34.7 -> 37.0 ms | measured-here |
| MTP head over the prompt's tail only (a770-qwen-xe1's `LLAMA_MTP_PREFILL_TAIL`) | experiment | prefill +4 to +7.5 %, but the decode right after a 23k prompt loses 5-11 points of acceptance; ours first dropped the MTP context before a cut batch (the pin's batch check wants consecutive positions), then the reference's form with gaps (llama patch for the check): the follow-up turn keeps its acceptance, the first answer does not. Busted by the 1-point bar; end to end the first turn is still ~1.5 % faster | code (reference) + measured-here |
| 0069 a work-group a weight row for the small-N F32 matvec | patch | busted: per verify step +0.7 % (32 rows leave 32 work-groups) | measured-here |
| 0071 the fused norm writes the K-quant matvec's q8_1 (and skips the F32 router in its consumer scan) | patch | correct on both cards; as served it fuses ~7 of ~159 norms a step (rows <= 4, a verify has 5): no measurable gain; widening it to verify rows is open | measured-here |

## Open

- 0064 and 0071 change the greedy texts (summation order); the gate's
  tie rule then gates on acceptance. The per-verify device time and the KL
  bar say 0064 is faster and equal in quality. The operator decides how such a
  change is gated.
- The ubatch's VRAM margin on the A770 at a prompt deeper than 41k.
- After the stack, the step's top kernels: the experts' IQ4_XS matvec
  (6.8 ms), the K-quant few-token GEMMs (4.1 + 2.5 ms), flash attention
  (4.0 ms), the F32 matvecs (0.9 ms), the norms (0.87 ms).

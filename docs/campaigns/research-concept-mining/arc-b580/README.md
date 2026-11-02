# arc-b580 — one fork's 208 commits, read for what ports to arcint (2026-11-02)

Material: `~/research/arc-b580/arc-b580` (head `e04b0cd4`, based on upstream `5ea87ddad2`) and its
own commit file, read against arcint at `qfndev` 0fb156e (patches 0001-0079) and llama.cpp at the
pin `3e4dfa8` with the served series applied. Target differs from ours: one B580 12 GB, a ternary
27B and Gemma 4 12B, q4_0 KV, 128K, SYCL under xe. Its numbers are `paper` for us; source readings
are `code`; nothing here is `measured-here`.

Method: pass 1 classified all 208 commits in chunks (area, kind, status, concept, the stated
numbers, an arcint score 0-3); pass 2 merged the 43 at 2-3 into 20 concepts, re-read the top 10's
evidence and corrected five records; pass 3 read the mechanism in the fork's tree; pass 4 audited
every kept candidate against our own source, with file:line and the patch that already holds it.

Counts: **113 / 42 / 32 / 21** at score 0 / 1 / 2 / 3 (208 commits). Of the 20 merged concepts, 10
are in the top list and 9 are audited in full (the next two are ruled out on shape); the ternary/PTQ1_0/PQ2_0 formats and the non-Intel backends
score 0.

Files: `records.md` (one record a commit), `candidates.md` (the ranked list), `audit.md` (against
our code).

## Top 5

1. **int8-DPAS Q-K straight from the quantized cache, and a staged DPAS kernel for 1-3 rows** —
   the fork's decode/verify attention is opt-in and buys 16K 1 token 266 -> 123 us, 48K 1 token
   433 -> 244, MTP +4-7 % at 115K; arcint has the GQA-shared DPAS form and the f16 path, but
   dequantizes K before the XMX product and keeps 1-3 rows on the ALU kernel.
2. **GDN rollback by replay** — one base state plus three-vector records instead of K full states
   a verify: bit-identical state and 432 MiB less VRAM at n_rs_seq 4 on the 27B; arcint writes
   per-token snapshots into the cache (0050).
3. **The MTP head's prompt pass computes only K/V** — the fork stops the draft graph after the
   cache store, bit-identical rows, long prompts ~5 % faster; our MTP graph builds attention, FFN
   and the full head over every prompt row.
4. **Round-trip discipline** — skip a context synchronize when nothing queued, keep the KV
   rotation matrices on the device, read all draft ids with one copy: 2 syncs + ~272 KB per
   attention graph and ~0.2 ms per small D2H read; arcint has none of the three.
5. **The GDN state gather folded into the kernel** — +3.8 % TG on the fork's card, and a 3 MB/layer
   dirty temp that slowed the next GEMV ~15 %; our graph still gathers at
   `llama-graph.cpp:3959`.

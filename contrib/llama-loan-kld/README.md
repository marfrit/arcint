# llama-loan-kld — the loan-on side of the KL gate

The prefill-slot-loan's KL gate (docs/campaigns/prefill-slot-loan.md, round 13)
needs a run whose prefill goes through the lent layers' experts from the host
lend bank — the third `MUL_MAT_ID` branch. No llama.cpp tool takes a loan:
the loan is begun through `llama_context_slot_loan_begin` / `_end` (patch
0068's public API), and the only caller in the series is arcint's
`src/exec/backend_llama.cpp` (`code`: no call site under `tools/`, `common/`
or `examples/`; `LLAMA_SLOT_LOAN_MIB` arms the load, not a loan). Arcint's
endpoint returns no logprobs (`src/api/handlers.cpp`: `"logprobs", nullptr`),
so the served route cannot take the KL side either.

This driver is the third route. It reads the **loan-off** capture
(`llama-perplexity --kl-divergence-base`), which holds the context size and
the exact token chunks, decodes those same tokens with the loan armed, and
writes a capture of llama's own shape. `tools/capture_vs_capture.py` then
compares the two files row for row.

## Build (round 14)

The recipe builds arcint's deb with `LLAMA_BUILD_TOOLS OFF` (`CMakeLists.txt`),
so no tool of the series is built. `build-tools.sh` builds both tools the gate
needs from the recipe's patched llama tree -- the same patch series, the same
OpenCL settings, the kernels embedded from that tree:

    # the recipe leaves no llama tree (build-deb.sh: mktemp -d, trap rm -rf),
    # so assemble one exactly as it does, or pass a patched tree that exists.
    # --fresh needs an EMPTY work dir (an existing tree is refused).
    contrib/llama-loan-kld/build-tools.sh --fresh /tmp/loan-tools
    TREE=/tmp/loan-tools/llama.cpp-3e4dfa8f39f904e681737d9f512a6e6338d848aa

It copies the driver to `<tree>/tools/loan-kld`, registers `add_subdirectory`
once, configures `<tree>/build-tools` with `-DGGML_OPENCL=ON
-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DGGML_OPENCL_EMBED_KERNELS=ON
-DGGML_NATIVE=ON -DLLAMA_CURL=OFF`, builds `llama-perplexity` and
`llama-loan-kld`, and prints `TOOL <name>: <sha256>` for each. The script's log
is `<tree>/build-tools/build.log`. The two binaries are not standalone: both
resolve their shared libs by RPATH into `<tree>/build-tools/bin` (libllama,
libggml, the OpenCL backend), so run them there -- copying a binary alone copies
nothing it links against.

## Run (one fresh process an arm; the plugin and the card's flags are the unit's)

The on arm runs the served shape: one 8,192-token ubatch under the unit's
2,048-token resident reserve and the loan's 8,192-token reserve (`-ub` = the
chunk, `--ubatch-reserve 2048`, which is REQUIRED; patch 0068's
`llama_context_params.n_ubatch_reserve`, which common has no option for, so the
driver builds the context from common's own loaders). Round 15: the 4,096-row
capture window is scored in disjoint row windows of one loan -- each a full
8,192-token graph from an empty KV -- because the loan's sizing takes
`n_outputs = min(measure_ub, n_outputs_max)` and a 4,096-row window's ~4.07 GB of
logits is more than the 6.20 GiB lend's plan allows. `--rows-per-decode` picks
the window width (default 2,048: two decodes, 2,048 + 2,047 rows); the rows equal
`llama-perplexity`'s because a causal row depends only on its own prefix.

    LOAN=1 LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_SLOT_LOAN_MIB=6400 \
      llama-loan-kld -m <flash-next-iq3_s-00001-of-00002>.gguf -c 8192 -b 8192 -ub 8192 \
        --ubatch-reserve 2048 -ngl 99 -lm mmap -nr --base off.kld --out on.kld --loan

Without `--loan` it is the **control arm**: the same scorer and quantiser as
`llama-perplexity` (`nearest_int`, perplexity.cpp:72-77, half-to-even), so its
rows against the loan-off reference must sit at the twin's floor. The off arm
keeps the perplexity tool's own split (`-b 4096 -ub 2048`): the bar is
answer-level, and at this vocab `-b 8192` would reserve 8.14 GB of output where
4.07 GB suffices.

    llama-loan-kld -m <flash-next-iq3_s-00001-of-00002>.gguf -c 8192 -b 8192 -ub 2048 \
      --ubatch-reserve 2048 -ngl 99 -lm mmap -nr --base off.kld --out ctl.kld

## Memory (round 14)

Rows are written straight from `llama_get_logits(ctx)` before the loan's end
(`--ubatch-reserve` makes the end's resident re-reserve keep a buffer the logits
are read from); the driver holds ONE row (0.47 MiB at Flash-Next's 248,320-token
vocab) and the context's logits reservation is one row window
(`--rows-per-decode`), not the capture's 4,096 rows and not `n_batch` (round 13
held an 8.14 GB copy, a 2.03 GB row buffer and an 8.14 GB reservation). On
OpenCL the output buffer is a plain CPU buffer, so the anon RSS is
`width x n_vocab x 4` (2.03 GB at 2,048; 1.02 GB at 1,024; 0.51 GB at 512) and the
lent graph's compute buffer holds the same rows. The forecast for both arms is in
the campaign document's KL gate section and design §22b/§23.

## The red case (`code`)

`LLAMA_SLOT_LOAN_MUTATE=5` moves one lent layer's lend-bank row index by one in
the host table before its non-blocking write, so the card reads the neighbour's
rows during the prefill. No verify compares a table row (the refill's check
compares slot bytes with the bank's), so the run refills, returns and scores --
and the WARN line names layer, expert and both rows. It is run first in the KL
commands and must fail the bar; every earlier red arm exited before scoring,
and `MUTATE=1` with `VERIFY=off` was a false red (the corruption lands after the
prefill).

The driver's flags taken out of argv before `common_params_parse` sees them:
`--base FILE`, `--out FILE`, `--ubatch-reserve N`, `--loan`. Everything else is
llama.cpp's own (`-m -c -b -ub -ngl -lm -nr -t -np`).

What it refuses, and why the gate depends on it:

- `-c` must equal the capture's context size and `-b` must equal `-c` (one
  logical batch a chunk);
- a `--loan` run whose context size does not beat the resident chunk (2,048)
  exits 2 — a 512-token window would never cross the lend threshold and would
  measure nothing;
- a `--loan` run with an empty lend set (no `LLAMA_EXPERT_CACHE_MIB` /
  `LLAMA_SLOT_LOAN_MIB`, no expert profile) exits 2;
- a `--loan` run where any chunk's `llama_context_slot_loan_begin` did not
  return a loan out, or its `_end` did not come back, exits 2;
- a `--loan` at `-ub` below the chunk logs a WARN: the lent prefill is not the
  served shape.

Every exit 2 after the capture's header is written unlinks the partial `--out`
(an `out_file` guard), so the next window does not compare a truncated capture.
The first chunk's prefill rate is cold (no warmup decode ran); the driver logs
it.

The loan's own trace lines (`lend`, `refill ... verify`, `loan returned`, the
load's `lend bank check`) come through libllama's log, so the log proves the
branch ran; the driver's last line states how many chunks took a loan.

The rules are plain data in `loan_kld_book.h`, and arcint's device-free unit
test (`tests/test_loan_book.cpp`, R47-R53) reads them and the driver's order and
memory shape.

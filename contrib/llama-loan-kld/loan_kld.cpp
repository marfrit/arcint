// llama-loan-kld: decode a capture's exact token chunks with the prefill-slot
// loan armed, and write a second capture in llama.cpp's own format. The KL gate
// of docs/campaigns/prefill-slot-loan.md (rounds 13-14) needs a run whose prefill
// goes through the loaned layers' experts from the host lend bank -- the lent
// MUL_MAT_ID branch -- with the same tokens and rows as the loan-off arm.
//
// Why not llama-perplexity: the loan is taken through the public API
// llama_context_slot_loan_begin/end (patch 0068), and no tool in the series
// calls it (`code`: no call site under tools/, common/ or examples/; the only
// caller is arcint's src/exec/backend_llama.cpp). LLAMA_SLOT_LOAN_MIB arms the
// *load* (which layers and slots are lendable) but nothing begins a loan, so a
// perplexity run's "loan on" arm would be byte-identical to its off arm and
// would prove nothing. Arcint's served endpoint has no logprobs either
// (`code`, src/api/handlers.cpp: `"logprobs", nullptr`), so the KL side cannot
// come from it. This driver is the third route: it reads the loan-off capture
// (llama-perplexity --save-all-logits) for the context size and the exact
// tokens, decodes those tokens, and writes a capture of the same shape that
// tools/capture_vs_capture.py compares with the off arm.
//
// It mirrors llama.cpp's perplexity capture loop (`code`,
// tools/perplexity/perplexity.cpp: the KV clear a chunk, `first = n_ctx/2`, rows
// `n_ctx - 1 - first`, the log-prob writer and its half-to-even quantiser
// `nearest_int`), so rows line up byte for byte. Without --loan the driver is the
// control arm: the same scoring as llama-perplexity, and the capture's rows must
// sit at the twin's own floor.
//
//   LOAN=1 LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_SLOT_LOAN_MIB=6400
//   llama-loan-kld -m <flash-next-iq3_s>.gguf -c 8192 -b 8192 -ub 8192
//     --ubatch-reserve 2048 -t 8 -ngl 99 -lm mmap -nr
//     --base off-ref.kld --out on.kld --loan
//
// Round 14: the on arm runs the served shape (`-ub` = the chunk, one ubatch of
// 8,192 under a 2,048-token resident reserve and the loan's 8,192-token reserve;
// patch 0068's llama_context_params.n_ubatch_reserve, which common has no option
// for, so the context is built here, and --ubatch-reserve is REQUIRED). The off
// arm keeps llama-perplexity's own split (answer-level bar).
//
// Round 15, the lent graph's logits: the loan's sizing takes n_outputs =
// min(measure_ub, n_outputs_max), so a 4,096-row window puts ~4.07 GB of logits
// in the lent graph's compute buffer and the 6.20 GiB lend's plan refuses it; the
// same rows are the context's output reservation, a plain CPU buffer on OpenCL
// (~4.07 GB anon RSS). The driver now caps n_outputs_max at `--rows-per-decode`
// (default 2,048) and scores the served 8,192-token ubatch in disjoint row
// windows under ONE loan: each window is the full 8,192-token graph from an empty
// KV, and a causal row at p depends only on tokens 0..p, so the rows equal
// llama-perplexity's. Host memory: one row held, no n_ctx x n_vocab copy, and the
// logits reservation is one row window (2.03 GB at the default), not the capture.
//
// The loan's own trace lines come through libllama's log: `lend`, `refill`,
// `loan returned` and the load's `lend bank check`. A run whose begin never
// succeeded is refused (exit 2) -- it measured nothing. A refused exit unlinks a
// partial --out: the next window must not compare a truncated capture.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "loan_kld_book.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// the driver's own flags, taken out before common_params_parse sees argv
struct own_args {
    std::string base;
    std::string out;
    uint32_t    ubatch_reserve = 0;   // llama_context_params.n_ubatch_reserve, required (the resident chunk)
    uint32_t    rows_per_decode = 2048;   // round 15: the logits row window a decode asks (n_outputs_max)
    bool        loan = false;
};

static bool take_arg(int & argc, char ** argv, const char * name, std::string * value) {
    const std::string flag = name;
    for (int i = 1; i < argc; ++i) {
        if (flag != argv[i]) {
            continue;
        }
        const char * v = value ? (i + 1 < argc ? argv[i + 1] : nullptr) : nullptr;
        if (value) {
            if (v == nullptr) {
                LOG_ERR("%s: %s needs a value\n", __func__, name);
                return false;
            }
            *value = v;
        }
        const int drop = value ? 2 : 1;
        for (int j = i; j + drop < argc; ++j) {
            argv[j] = argv[j + drop];
        }
        argc -= drop;
        return true;
    }
    return value ? false : true;   // absent is fine; a missing value is the caller's error
}

static bool take_own_args(int & argc, char ** argv, own_args & own) {
    if (!take_arg(argc, argv, "--base", &own.base)) {
        LOG_ERR("%s: --base FILE (the loan-off capture) is required\n", __func__);
        return false;
    }
    if (!take_arg(argc, argv, "--out", &own.out)) {
        LOG_ERR("%s: --out FILE (the loan-on capture) is required\n", __func__);
        return false;
    }
    if (true) {
        std::string reserve;
        if (!take_arg(argc, argv, "--ubatch-reserve", &reserve)) {
            LOG_ERR("%s: --ubatch-reserve N is required (the resident token count the buffers are kept for; it is "
                    "the resident re-reserve the loan's return runs and the shape the unit runs)\n", __func__);
            return false;
        }
        if (!reserve.empty()) {
            char * end = nullptr;
            const long r = std::strtol(reserve.c_str(), &end, 10);
            if (end == nullptr || *end != '\0' || r <= 0 || r > 131072) {
                LOG_ERR("%s: --ubatch-reserve %s is not a positive token count\n", __func__, reserve.c_str());
                return false;
            }
            own.ubatch_reserve = (uint32_t) r;
        }
    }
    if (true) {
        std::string rows;
        if (!take_arg(argc, argv, "--rows-per-decode", &rows)) {
            LOG_ERR("%s: --rows-per-decode needs a positive row count\n", __func__);
            return false;
        }
        if (!rows.empty()) {
            char * end = nullptr;
            const long r = std::strtol(rows.c_str(), &end, 10);
            if (end == nullptr || *end != '\0' || r <= 0 || r > 131072) {
                LOG_ERR("%s: --rows-per-decode %s is not a positive row count\n", __func__, rows.c_str());
                return false;
            }
            own.rows_per_decode = (uint32_t) r;
        }
    }
    if (const int before = argc; true) {
        take_arg(argc, argv, "--loan", nullptr);
        own.loan = before != argc;
    }
    return true;
}

// Round 14: a partial --out must not survive an exit-2 refusal -- the next
// window would compare a truncated capture and see a KL that is an artefact of
// the read. The guard holds the file open for the whole run and removes it
// unless the run commits at its end.
struct out_file {
    std::string path;
    std::ofstream stream;
    bool committed = false;
    ~out_file() {
        if (!committed) {
            stream.close();
            std::remove(path.c_str());
        }
    }
};

// the capture's writer, transcribed from perplexity.cpp's log_softmax(int, ...):
// 16-bit quantised log-probs, scale and min_log_prob as the row's first 4 words,
// the quantiser being perplexity's own `nearest_int` (half-to-even, :72-77).
static double write_row(uint16_t * row, int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += std::exp(logits[i] - max_logit);
    }
    const float log_sum_exp = (float) std::log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit) / 65535.f;
    float * d = (float *) row;
    d[0] = scale;
    d[1] = min_log_prob;
    uint16_t * lp = row + 4;
    if (scale) {
        const float inv_scale = 1 / scale;
        for (int i = 0; i < n_vocab; ++i) {
            lp[i] = logits[i] > min_logit
                        ? (uint16_t) loan_kld_book::nearest_int(inv_scale * (logits[i] - min_logit))
                        : 0;
        }
    } else {
        std::memset(lp, 0, (size_t) n_vocab * sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

// the rows of one chunk, each written from the context's logits as it is made
// (perplexity's row order: row i is the logits of position first+i, the token it
// predicts being tokens[i + 1]). One row is held, not n_rows.
static void write_chunk_rows(std::ofstream & out, int n_vocab, const float * logits, const int * tokens, int n_rows,
                             std::vector<uint16_t> & row) {
    const size_t nv = (size_t) loan_kld_book::row_width(n_vocab);
    row.resize(nv);
    for (int i = 0; i < n_rows; ++i) {
        write_row(row.data(), n_vocab, logits + (size_t) i * n_vocab, tokens[i + 1]);
        out.write((const char *) row.data(), (std::streamsize) (nv * sizeof(uint16_t)));
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();

    own_args own;
    if (!take_own_args(argc, argv, own)) {
        return 2;
    }

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }
    // no warmup decode: the first chunk's prefill rate is cold -- chunk 0 pays the
    // first reservation and the kernels' first builds. The gate reads chunk 1 of 1,
    // so its rate is a cold rate and is compared as one.
    LOG_INF("%s: warmup off: the first chunk's prefill rate is cold (no warmup decode ran)\n", __func__);

    // the loan-off capture: the context size and the exact tokens, read first so
    // a mismatching -c is refused before the model load
    std::ifstream in(own.base.c_str(), std::ios::binary);
    if (!in) {
        LOG_ERR("%s: cannot open %s\n", __func__, own.base.c_str());
        return 2;
    }
    char magic[9] = {0};
    in.read(magic, 8);
    if (std::strncmp("_logits_", magic, 8) != 0) {
        LOG_ERR("%s: %s is not a llama.cpp capture\n", __func__, own.base.c_str());
        return 2;
    }
    int32_t base_ctx = 0, base_vocab = 0, base_chunk = 0;
    in.read((char *) &base_ctx, sizeof(base_ctx));
    in.read((char *) &base_vocab, sizeof(base_vocab));
    in.read((char *) &base_chunk, sizeof(base_chunk));
    if (in.fail() || base_ctx <= 0 || base_chunk <= 0) {
        LOG_ERR("%s: %s has no header\n", __func__, own.base.c_str());
        return 2;
    }
    std::vector<int32_t> tokens((size_t) base_chunk * base_ctx);
    in.read((char *) tokens.data(), (std::streamsize) (tokens.size() * sizeof(tokens[0])));
    if (in.fail()) {
        LOG_ERR("%s: %s is short: %d chunks of %d tokens\n", __func__, own.base.c_str(), base_chunk, base_ctx);
        return 2;
    }
    // the loan threshold, from the capture's own context size: the gate's point
    const bool wanted = loan_kld_book::loan_wanted((uint32_t) base_ctx);
    if (own.loan && !wanted) {
        LOG_ERR("%s: a %d-token chunk does not cross the lend threshold (%u); the loan would not be taken "
                "and the run would measure nothing -- use -c above %u\n",
                __func__, base_ctx, loan_kld_book::kResidentChunk, loan_kld_book::kResidentChunk);
        return 2;
    }

    // The context's logits reservation, and the loan's sizing: a row window, not
    // the capture's 4,096 rows and not n_batch. Round 15: the loan's sizing takes
    // n_outputs = min(measure_ub, n_outputs_max), so 4,096 rows put ~4.07 GB of
    // logits in the lent graph's compute buffer -- the 6.20 GiB lend's plan refuses
    // it -- and the same rows are the context's output buffer, a plain CPU buffer
    // on OpenCL (~4.07 GB anon). Each decode is one full 8,192-token ubatch that
    // asks `width` rows; the scored rows are split over n_decodes passes under one
    // loan (see loan_kld_book::decode_width).
    const uint32_t width = loan_kld_book::decode_width(loan_kld_book::n_rows(base_ctx), own.rows_per_decode);
    params.n_outputs_max = width;
    LOG_INF("%s: %d scored rows in %d row windows of %u (n_outputs_max %u)\n", __func__,
            loan_kld_book::n_rows(base_ctx), loan_kld_book::n_decodes(loan_kld_book::n_rows(base_ctx), width),
            width, width);

    // the model and the context: common's own loaders, with the resident ubatch
    // reserve set directly (llama_context_params.n_ubatch_reserve, patch 0068;
    // common has no option for it, and common_init_from_params cannot take one).
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ubatch_reserve = own.ubatch_reserve;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(params.model.path.c_str(), mparams), &llama_model_free);
    if (model == nullptr) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 2;
    }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cparams), &llama_free);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to init the context\n", __func__);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int n_ctx   = (int) llama_n_ctx(ctx.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_ctx != base_ctx || n_vocab != base_vocab) {
        LOG_ERR("%s: the context is %d tokens / %d vocab, the capture %d / %d\n", __func__, n_ctx, n_vocab, base_ctx,
                base_vocab);
        return 2;
    }
    if (llama_n_seq_max(ctx.get()) != 1) {
        LOG_ERR("%s: the slot loan needs one sequence (n_seq_max %u)\n", __func__, llama_n_seq_max(ctx.get()));
        return 2;
    }
    llama_slot_loan_info sl{};
    if (llama_model_slot_loan_info(model.get(), &sl)) {
        LOG_INF("%s: the cache's lend set: %d layers, %lld slots, %.2f GiB\n", __func__, (int) sl.layers,
                (long long) sl.slots, sl.bytes / 1073741824.0);
    }
    if (own.loan && sl.bytes == 0) {
        LOG_ERR("%s: --loan with no lendable slots -- set LLAMA_EXPERT_CACHE_MIB and LLAMA_SLOT_LOAN_MIB at load "
                "(and the expert profile)\n", __func__);
        return 2;
    }

    const int n_batch = params.n_batch;
    if (n_batch != n_ctx) {
        LOG_ERR("%s: this driver decodes a chunk in one batch; -b %d against -c %d\n", __func__, n_batch, n_ctx);
        return 2;
    }
    // the shape each arm runs (round 14). The on arm must be the served one: one
    // ubatch of the chunk; -ub below it splits the lent prefill and measures a
    // graph the unit does not run. The off arm's split is llama-perplexity's and
    // is allowed (the bar is answer-level).
    if (!loan_kld_book::served_shape((uint32_t) params.n_ubatch, (uint32_t) n_ctx)) {
        if (own.loan) {
            LOG_WRN("%s: --loan at -ub %u under an %d-token chunk runs %d ubatches of %u -- not the served shape "
                    "(the unit runs one ubatch of the chunk); set -ub %d\n",
                    __func__, params.n_ubatch, n_ctx, (n_ctx + (int) params.n_ubatch - 1) / (int) params.n_ubatch,
                    params.n_ubatch, n_ctx);
        } else {
            LOG_INF("%s: %d ubatches of %u under a %u-token reserve (the perplexity tool's split; allowed for the "
                    "off arm)\n", __func__, (n_ctx + (int) params.n_ubatch - 1) / (int) params.n_ubatch, params.n_ubatch,
                    own.ubatch_reserve);
        }
    } else {
        LOG_INF("%s: one ubatch of %d tokens under a %u-token resident reserve%s\n", __func__, n_ctx,
                own.ubatch_reserve, own.loan ? " (the served shape, lent)" : " (one batch)");
    }

    out_file out;
    out.path = own.out;
    out.stream.open(own.out.c_str(), std::ios::binary);
    if (!out.stream) {
        LOG_ERR("%s: cannot open %s\n", __func__, own.out.c_str());
        return 2;
    }
    out.stream.write("_logits_", 8);
    out.stream.write((const char *) &n_ctx, sizeof(int32_t));
    out.stream.write((const char *) &n_vocab, sizeof(int32_t));
    out.stream.write((const char *) &base_chunk, sizeof(int32_t));
    out.stream.write((const char *) tokens.data(), (std::streamsize) (tokens.size() * sizeof(tokens[0])));

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int first = loan_kld_book::first_row(n_ctx);
    const int n_rows = loan_kld_book::n_rows(n_ctx);
    std::vector<uint16_t> row;
    common_batch batch(ctx.get());

    int loans = 0, refused = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < base_chunk; ++c) {
        const int start = c * n_ctx;

        // the loan's scope: begin BEFORE the chunk's first row window, end AFTER its
        // last -- arcint's Scope around its prefill chunk (code,
        // src/exec/slot_loan.h). One loan covers every row window of the chunk; the
        // end's refill and resident re-reserve precede the next chunk's decode.
        bool out_loan = false;
        if (own.loan) {
            if (llama_context_slot_loan_begin(ctx.get(), loan_kld_book::chunk_tokens((uint32_t) n_ctx))) {
                out_loan = llama_context_slot_loan_out(ctx.get());
                ++loans;
            } else {
                ++refused;
                LOG_ERR("%s: chunk %d: the context refused the loan (the log line above says why)\n", __func__, c);
            }
        }

        double sec = 0.0;
        const int passes = loan_kld_book::n_decodes(n_rows, width);
        for (int p = 0; p < passes; ++p) {
            // each pass replays the chunk from an empty KV: the same 8,192-token
            // graph, the served shape. A causal row at position k depends only on
            // tokens 0..k, so a row scored in any pass equals llama-perplexity's row
            // at that position; the passes score disjoint ranges.
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            const int off = p * (int) width;
            const int w = loan_kld_book::rows_in_decode(n_rows, width, p);
            int n_outputs = 0;
            batch.clear();
            for (int k = 0; k < n_ctx; ++k) {
                // perplexity's window: the logits of positions first..n_ctx-2 (the
                // last output, position n_ctx-1, has no next token in the chunk and
                // is dropped), split over the row windows
                const bool need_logits = k >= first + off && k < first + off + w;
                // perplexity's own first-batch BOS: the model's vocab adds BOS, so
                // the decoded token at position 0 is BOS while the capture stores
                // the corpus's token (`code`, perplexity.cpp's `add_bos && j == 0`)
                const llama_token t = (add_bos && k == 0) ? llama_vocab_bos(vocab) : tokens[start + k];
                batch.add(t, k, 0, need_logits);
                n_outputs += need_logits;
            }
            const auto tc0 = std::chrono::steady_clock::now();
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: chunk %d window %d: decode failed\n", __func__, c, p);
                return 2;
            }
            llama_synchronize(ctx.get());
            sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
            if (n_outputs != w) {
                LOG_ERR("%s: chunk %d window %d: %d logit rows against %d asked\n", __func__, c, p, n_outputs, w);
                return 2;
            }
            const auto * batch_logits = llama_get_logits(ctx.get());
            // the rows are read BEFORE the loan's end: --ubatch-reserve makes the
            // end's resident re-reserve keep a buffer the logits are read from, so
            // the read is here; the round-14 comment's "the end frees the buffer the
            // logits live in" was false (--ubatch-reserve is required).
            write_chunk_rows(out.stream, n_vocab, batch_logits, tokens.data() + start + first + off, w, row);
        }

        if (own.loan) {
            const bool back = llama_context_slot_loan_end(ctx.get());
            if (!out_loan || !back) {
                LOG_ERR("%s: chunk %d: the loan did not come back exactly\n", __func__, c);
                return 2;
            }
        }

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        LOG_INF("%s: chunk %d/%d: prefill %.1f t/s (loan %s, %d row windows of %u), %lld ms elapsed\n", __func__,
                c + 1, base_chunk, n_ctx / std::max(sec, 1e-6), out_loan ? "out" : "off", passes, width,
                (long long) ms);
    }

    out.stream.flush();
    if (!out.stream) {
        LOG_ERR("%s: %s could not be written\n", __func__, out.path.c_str());
        return 2;
    }
    out.stream.close();

    if (own.loan) {
        LOG_INF("%s: %d chunks, %d loans taken, %d refused\n", __func__, base_chunk, loans, refused);
        if (loans != base_chunk) {
            LOG_ERR("%s: %d of %d chunks ran without a loan; the capture is not a loan arm\n", __func__,
                    base_chunk - loans, base_chunk);
            return 2;
        }
    }
    out.committed = true;   // every refusal above unlinks the partial --out
    LOG_INF("%s: wrote %s (%d chunks, %d rows a chunk)\n", __func__, out.path.c_str(), base_chunk, n_rows);
    ctx.reset();             // the backend shuts down after the context and the model are freed
    model.reset();
    llama_backend_free();
    return 0;
}

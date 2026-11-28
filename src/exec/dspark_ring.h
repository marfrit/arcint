// The DSpark drafter's own bookkeeping (the libllama engine's --llama-dflash-gguf
// with an Inkling 2 DSpark drafter, src/exec/llama_dspark.cpp), without a context:
// the Markov context ring, the block's shape, the drafts read from the block's
// logits, and the ring's part of a checkpoint. Pure, so every rule is testable on
// synthetic rows (tests/test_dspark.cpp), as dflash_walk.h and verify_walk.h are.
//
// The reference is the vLLM-XPU Inkling port (inkling_xpu/dspark.py at 6b71b909,
// `code`):
// - every target forward stores, per position, the Markov head's context row
//   f_rnn = norm(sum_i mix[-1, i] * tap_i) in a ring keyed by position
//   (`store`, f_ring[positions % ring]); rejected rows stay in it and are
//   overwritten by the next verify ("rejected rows are past the next committed
//   boundary");
// - a draft at `start` (the anchor's position) is one block of `block` rows
//   [anchor, mask x (block - 1)] at start..start+block-1, all of them, whatever
//   the width (`_block`); the drafts are rows 1..steps, `steps` the configured
//   width (num_speculative_tokens, 1..7);
// - the Markov head reads f_ring[(start - 1) % ring], the row of the last
//   committed token, never the anchor's own;
// - greedy proposals are the argmax of each row (logits.argmax(dim=-1): the
//   lowest id on a tie).
// The llama.cpp port (#32218, contrib/llama.cpp 0215/0216) keeps the same ring
// (block + 1 rows a lane, keyed by position) and drafts n_max + 1 rows; this
// driver drafts the reference's whole block (0218 caps the in-graph Markov chain
// at the width instead), see llama_dspark.cpp.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifndef ARCINT_DSPARK_RED
#define ARCINT_DSPARK_RED 0
#endif

namespace lgc {

// The Markov context rows of one lane: `rows` slots of `width` floats, slot
// p % rows holding position p (or nothing). A lookup of p answers only when its
// slot holds p itself, so a row of another position (an older pass round the
// ring, a rejected tail cut by trim) is never read as p's.
class DsparkRing {
public:
    DsparkRing() = default;
    DsparkRing(size_t rows, size_t width) { init(rows, width); }

    void init(size_t rows, size_t width) {
        rows_  = rows;
        width_ = width;
        data_.assign(rows * width, 0.0f);
        pos_.assign(rows, -1);
    }

    size_t rows() const { return rows_; }
    size_t width() const { return width_; }

    void put(int64_t p, const float* row) {
        if (rows_ == 0 || p < 0) return;
        const size_t s = static_cast<size_t>(p) % rows_;
        std::memcpy(data_.data() + s * width_, row, width_ * sizeof(float));
        pos_[s] = p;
    }

    // p's row, or null when the ring does not hold p
    const float* get(int64_t p) const {
        if (rows_ == 0 || p < 0) return nullptr;
        const size_t s = static_cast<size_t>(p) % rows_;
        return pos_[s] == p ? data_.data() + s * width_ : nullptr;
    }

    // a rollback of the sequence to `from` tokens: positions >= from leave
    void trim(int64_t from) {
#if ARCINT_DSPARK_RED == 2
        (void) from;   // red case: the rejected rows stay in the ring
#else
        for (int64_t& p : pos_)
            if (p >= from) p = -1;
#endif
    }

    void clear() { std::fill(pos_.begin(), pos_.end(), int64_t{-1}); }

    // the positions held, ascending
    std::vector<int64_t> held() const {
        std::vector<int64_t> h;
        for (int64_t p : pos_)
            if (p >= 0) h.push_back(p);
        std::sort(h.begin(), h.end());
        return h;
    }

    // The ring's part of a checkpoint of a sequence that holds n tokens: the
    // rows of positions n - rows .. n - 1 it holds, as [u64 count][u64 width]
    // then count x ([i64 pos][width floats]). Rows at n or later belong to
    // tokens the checkpoint does not have.
    std::vector<uint8_t> save(int64_t n) const {
        std::vector<std::pair<int64_t, size_t>> keep;
        for (size_t s = 0; s < rows_; ++s) {
            const int64_t p = pos_[s];
#if ARCINT_DSPARK_RED == 5
            if (p >= 0)   // red case: every row, those of the later tokens too
#else
            if (p >= 0 && p < n && p + static_cast<int64_t>(rows_) >= n)
#endif
                keep.emplace_back(p, s);
        }
        std::sort(keep.begin(), keep.end());
        const uint64_t cnt = keep.size(), w = width_;
        std::vector<uint8_t> out(16 + cnt * (8 + width_ * sizeof(float)));
        std::memcpy(out.data(), &cnt, 8);
        std::memcpy(out.data() + 8, &w, 8);
        size_t at = 16;
        for (const auto& [p, s] : keep) {
            std::memcpy(out.data() + at, &p, 8);
            std::memcpy(out.data() + at + 8, data_.data() + s * width_, width_ * sizeof(float));
            at += 8 + width_ * sizeof(float);
        }
        return out;
    }

    // a saved part back, for a sequence cut to n tokens: false (and the ring
    // cleared) for a buffer of another width or size, or one that carries a
    // row at n or later
    bool load(int64_t n, const uint8_t* data, size_t size) {
        clear();
        if (size < 16) return false;
        uint64_t cnt = 0, w = 0;
        std::memcpy(&cnt, data, 8);
        std::memcpy(&w, data + 8, 8);
        const size_t one = 8 + width_ * sizeof(float);
        if (w != width_ || cnt > rows_ || size != 16 + cnt * one) return false;
        for (size_t k = 0; k < cnt; ++k) {
            int64_t p = 0;
            std::memcpy(&p, data + 16 + k * one, 8);
            if (p < 0 || p >= n) {
                clear();
                return false;
            }
            put(p, reinterpret_cast<const float*>(data + 16 + k * one + 8));
        }
        return true;
    }

private:
    size_t               rows_  = 0;
    size_t               width_ = 0;
    std::vector<float>   data_;
    std::vector<int64_t> pos_;
};

// The position whose Markov context row a draft at pos0 (the anchor's
// position) reads: the last committed token's, pos0 - 1. -1: none (pos0 0).
inline int64_t dspark_anchor_pos(int64_t pos0) {
#if ARCINT_DSPARK_RED == 1
    return pos0;   // red case: the anchor's own row (one position late; cut by the rollback, so no draft)
#elif ARCINT_DSPARK_RED == 7
    return pos0 - 2;   // red case: the row one position early (a draft from the wrong context)
#else
    return pos0 - 1;
#endif
}

// The rows of one lane's draft block: the trained block, whatever the width
// (the reference's _block); the width only sets the chain (0218) and how many
// rows are read
inline int dspark_block_rows(int block_size, int n_draft) {
#if ARCINT_DSPARK_RED == 4
    (void) block_size;
    return n_draft + 1;   // red case: the llama.cpp port's n_max + 1 rows
#else
    (void) n_draft;
    return block_size;
#endif
}

// The drafts a lane may take this step: the configured width, the step's
// budget (max_tokens and the context, exec/spec_depth.h's spec_width), and the
// block's draft rows (block - 1), whichever is least; never negative
inline int dspark_step_drafts(int n_draft, int n_max_step, int block_size) {
#if ARCINT_DSPARK_RED == 6
    (void) n_max_step;   // red case: the step's budget ignored
    return std::max(0, std::min(n_draft, block_size - 1));
#else
    return std::max(0, std::min({n_draft, n_max_step, block_size - 1}));
#endif
}

// Rows of the verify a lane's step decodes: the anchor (id_last) and its drafts
inline int dspark_verify_rows(size_t n_drafts) { return 1 + static_cast<int>(n_drafts); }

// The shortlist path's drafts (patch 0241, --llama-dflash-shortlist): the ids the drafter's graph returned for a
// lane (llama_get_dspark_ids), the first min(n, n_take); none when the graph returned none (n < 0) or any id lies
// outside [0, vocab) (a broken graph must not draft: the target would verify garbage at the cost of a verify).
// Red 12: the vocabulary bound dropped.
inline std::vector<int> dspark_ids_take(const int32_t* ids, int n, int n_take, int vocab) {
    std::vector<int> out;
    if (ids == nullptr || n <= 0 || n_take <= 0) return out;
    const int m = std::min(n, n_take);
    for (int i = 0; i < m; ++i) {
#if ARCINT_DSPARK_RED == 12
        (void) vocab;
#else
        if (ids[i] < 0 || ids[i] >= vocab) return {};
#endif
        out.push_back(ids[i]);
    }
    return out;
}

// The greedy drafts of one block: the argmax of draft rows 1..n of the block's
// logits (`n_rows` rows of `vocab` floats; row 0 is the anchor's and drafts
// nothing), n = min(n_take, n_rows - 1). On a tie the HIGHEST id: ggml's ARGMAX
// rule (ggml-cpu/vec.h ggml_vec_argmax_f32, and the OpenCL op of patch 0240),
// which the drafter's chain conditions on; torch's argmax (the reference) takes
// the first, which only an exact tie can tell apart, and the target verifies
// every draft either way. Red 11: the first maximum.
inline std::vector<int> dspark_pick(const float* logits, size_t n_rows, size_t vocab, int n_take) {
    std::vector<int> out;
    if (logits == nullptr || vocab == 0 || n_take <= 0 || n_rows < 2) return out;
#if ARCINT_DSPARK_RED == 3
    const size_t first = 0;   // red case: the anchor's row drafted
#else
    const size_t first = 1;
#endif
    for (size_t i = first; i < n_rows && static_cast<int>(out.size()) < n_take; ++i) {
        const float* r    = logits + i * vocab;
        size_t       best = 0;
        for (size_t v = 1; v < vocab; ++v)
#if ARCINT_DSPARK_RED == 11
            if (r[v] > r[best]) best = v;
#else
            if (r[v] >= r[best]) best = v;
#endif
        out.push_back(static_cast<int>(best));
    }
    return out;
}

// Whether an injected chunk at positions [lo, hi) holds a row the ring must
// keep: a verify's rows always (the next draft reads one of them); a prompt's
// only within `rows` of its end or of a checkpoint the same prefill saves (its
// state carries the ring). The others' rows need not come back to the host.
inline bool dspark_chunk_needs_rows(int64_t lo, int64_t hi, bool verify, int64_t n_end,
                                    const std::vector<int64_t>& ckpts, int64_t rows) {
    if (verify || n_end < 0) return true;
    const auto meets = [&](int64_t end) { return hi > end - rows && lo < end; };
    if (meets(n_end)) return true;
    for (int64_t c : ckpts)
        if (meets(c)) return true;
    return false;
}

// The targets this drafter is for: Inkling 2 Mini as one pass (inkling2) or with its loop (inkling2_loop). Its taps are
// that graph's layer visits (0214); another model of the same width and vocabulary size would reach the graph's tap
// assert at its first decode instead of a refusal at load. Empty: allowed; else the refusal.
inline std::string dspark_target_refusal(const std::string& arch) {
#if ARCINT_DSPARK_RED == 8
    (void) arch;
    return {};   // red case: any target
#else
    if (arch == "inkling2" || arch == "inkling2_loop") return {};
    return "the DSpark drafter's taps are Inkling 2's layer visits; the model is '" + arch +
           "', not inkling2 / inkling2_loop";
#endif
}

// The parts of a DSpark checkpoint payload (SpecStateKind::DsparkWindowRing): [u64 window bytes][the windowed layers'
// K/V][the ring's rows]. False for a payload shorter than its header or a window past its end.
struct DsparkStateParts {
    size_t win_off = 0, win_size = 0, ring_off = 0, ring_size = 0;
};
inline bool dspark_state_split(const uint8_t* p, size_t size, DsparkStateParts& out) {
    out = {};
#if ARCINT_DSPARK_RED == 9
    if (p == nullptr) return false;   // red case: no bounds read (a short payload reads as an empty window)
    uint64_t w = 0;
    if (size >= 8) std::memcpy(&w, p, 8);
#else
    if (p == nullptr || size < 8) return false;
    uint64_t w = 0;
    std::memcpy(&w, p, 8);
    if (w > size - 8) return false;
#endif
    out.win_off   = 8;
    out.win_size  = static_cast<size_t>(w);
    out.ring_off  = 8 + out.win_size;
    out.ring_size = size >= out.ring_off ? size - out.ring_off : 0;
    return true;
}

// A lane's restore failures: a failed restore leaves the drafter's global layer without the cells below the cut for
// the rest of the lane's life (acceptance falls, the output stays the target's). Counted, cumulative, and the caller
// warns: never silent. on_restore says whether to warn.
struct DsparkLaneHealth {
    uint64_t restore_failures = 0;
    bool on_restore(bool ok) {
#if ARCINT_DSPARK_RED == 10
        (void) ok;
        return false;   // red case: failures neither counted nor warned
#else
        if (ok) return false;
        ++restore_failures;
        return true;
#endif
    }
};

// --llama-dflash-kv into the draft context: its K/V types, and flash attention on for a quantized cache (llama.cpp
// refuses a quantized V without it, so a context that cannot take the type refuses at load). config.cpp admits
// f16, q8_0 and q8_0:q4_0, the pairs of --llama-kv.
struct DsparkKvPlan {
    std::string type_k = "f16", type_v = "f16";
    bool        flash_attn_on = false;
};
inline DsparkKvPlan dspark_kv_plan(const std::string& k, const std::string& v) {
    DsparkKvPlan p;
    p.type_k        = k;
    p.type_v        = v;
    p.flash_attn_on = k != "f16" || v != "f16";
    return p;
}

// --llama-dflash-shortlist-probe FILE (patch 0243, a measurement aid): a lane's offered drafts, held from the draft
// to its verify, then one JSON line with the verify's kept count (tools/dspark_probe_read.py reads them). With the
// shortlist (sl = K > 0) f is the full chain's pick at each position on the same state and in whether it was among
// the position's K candidates (llama_get_dspark_probe); without it f = d and in is empty.
struct DsparkProbePending {
    bool             set     = false;
    int64_t          pos0    = 0;
    int              id_last = 0;
    int              sl      = 0;
    std::vector<int> d, f, in;
};

inline std::string dspark_probe_line(int seq, const DsparkProbePending& p, int a) {
    const auto list = [](const std::vector<int>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
        return s + "]";
    };
    return "{\"seq\": " + std::to_string(seq) + ", \"pos0\": " + std::to_string(p.pos0) + ", \"id_last\": " +
           std::to_string(p.id_last) + ", \"a\": " + std::to_string(a) + ", \"sl\": " + std::to_string(p.sl) +
           ", \"d\": " + list(p.d) + ", \"f\": " + list(p.sl > 0 ? p.f : p.d) + ", \"in\": " +
           list(p.sl > 0 ? p.in : std::vector<int>{}) + "}\n";
}

// One probe line to its file, flushed: false when the write or the flush fell short (a full disk: the log would be
// short while the line count still grew). Red 13: the results ignored.
inline bool dspark_probe_write(std::FILE* f, const std::string& line) {
    const size_t n  = std::fwrite(line.data(), 1, line.size(), f);
    const int    fl = std::fflush(f);
#if ARCINT_DSPARK_RED == 13
    (void) n;
    (void) fl;
    return true;
#else
    return n == line.size() && fl == 0;
#endif
}

}  // namespace lgc

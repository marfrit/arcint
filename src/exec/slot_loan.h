#pragma once

// The prefill-slot-loan's rules, without a card (docs/campaigns/prefill-slot-loan.md,
// docs/campaigns/prefill-slot-loan-design.md §3, §6). The mechanism lives in
// llama.cpp (the expert cache's lend/refill, the context's re-reserve and
// rebind); what is here is what decides *whether* a request borrows, and owns
// the scope that gives the slots back on every exit -- including the throw
// prefill_locked takes and the cancellation wave2a adds at ubatch boundaries.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace lgc {
namespace slot_loan {

// The unit's resident chunk: the webui's prefill chunk is what a loan may use,
// the 2,048 is what the buffer stays reserved for when no loan is out. A
// prompt of 1,000 tokens must not pay two reserves and a refill (finding 6):
// the sizing measured ~1.9 s for a short follow-up at a large chunk (patch
// 0043's record), which is the price this rule refuses to pay.
constexpr uint32_t kResidentChunk = 2048;

struct Rule {
    bool    enabled    = false;   // --llama-slot-loan
    size_t  lend_bytes = 0;       // what the cache set up at load (--llama-slot-loan N)
    uint32_t resident_chunk = kResidentChunk;
};

// The resident chunk the context reserves when no loan is out -- the floor a
// loan's chunk must beat.
inline uint32_t floor_chunk(uint32_t prefill_chunk, const Rule& r) {
    return r.enabled ? std::min(prefill_chunk, r.resident_chunk) : prefill_chunk;
}

// The chunk a loan asks the context to reserve: the request's own chunk, never
// more than the tokens still to run (a 3,000-token prompt at a chunk of 8,192
// asks for 3,000 -- no reserve larger than the prompt), and never below the
// resident floor (a loan of less than the floor is not worth a refill).
inline uint32_t loan_chunk(uint32_t prefill_chunk, size_t remaining_prompt) {
    return static_cast<uint32_t>(remaining_prompt < prefill_chunk ? remaining_prompt : prefill_chunk);
}

// Lend only when the prompt still to run exceeds the resident chunk, and only
// when the cache actually has lendable bytes. Everything else runs in the
// resident reserve the previous request left behind.
inline bool wanted(const Rule& r, uint32_t prefill_chunk, size_t remaining_prompt) {
    return r.enabled && r.lend_bytes > 0 && remaining_prompt > floor_chunk(prefill_chunk, r);
}

// A request may only start when no loan is out and no loan's state was left
// unusable: a previous request's scope failed, a `lend_undo`'s tables did not
// reach the card, or the resident re-reserve after a refused/failed loan failed
// (llama-context.cpp's slot_loan_out reports all three). A lent slot must never
// be read as an expert. The caller throws.
inline bool request_start_ok(bool loan_stuck) {
    return !loan_stuck;
}

// The lend budget from the flag: 0 is an explicit off, never the sentinel that
// makes llama.cpp read LLAMA_SLOT_LOAN_MIB (finding 8: the operator's flag wins
// over the environment; a stray LLAMA_SLOT_LOAN_MIB cannot enable a loan that
// --llama-slot-loan left off). This is arcint's half of the rule -- what it
// passes; the llama side (init and llama_model_default_params) is tested in
// tests/test_loan_book.cpp against patch 0068's src/llama-loan-book.h.
inline size_t lend_param(uint32_t mib) { return static_cast<size_t>(mib) << 20; }

// The scope: begin once before the prompt's first ubatch, end once after its
// last, and on EVERY exit path -- the destructor ends if begin succeeded and
// end was not called. end() is idempotent and reports what the refill said.
class Scope {
public:
    using Begin = std::function<bool(uint32_t chunk)>;
    using End   = std::function<bool(void)>;

    Scope(Rule rule, uint32_t prefill_chunk, Begin begin, End end)
        : rule_(rule), prefill_chunk_(prefill_chunk), begin_(std::move(begin)), end_(std::move(end)) {}

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    // finding 4: the destructor runs inside unwinding (prefill_locked's throw, a
    // cancellation). end_ is llama_context_slot_loan_end, and its re-reserves
    // throw on a failed allocation: a throw from a destructor during unwinding is
    // std::terminate and takes the server with it. Swallow it and record failure:
    // ok() is false, the caller fails the request, and the loan -- which the
    // context left out -- is refused by request_start_ok at the next request.
    ~Scope() {
        try {
            end();
        } catch (...) {
            ended_ = true;
            ok_    = false;
        }
    }

    // true when the loan is out. false: not wanted, or the context refused it
    // (the prompt's buffers need more than the lend bytes); then end() is never
    // called and the resident reserve stays.
    bool begin(size_t remaining_prompt) {
        if (begun_ || ended_) {
            return false;
        }
        if (!wanted(rule_, prefill_chunk_, remaining_prompt)) {
            return false;
        }
        const uint32_t chunk = loan_chunk(prefill_chunk_, remaining_prompt);
        bool ok_begin = false;
        try {
            ok_begin = begin_ && begin_(chunk);
        } catch (...) {
            // the context threw before it could report; it restores what it lent
            // itself, and a loan it left out is refused by request_start_ok
            ok_begin = false;
            if (end_) {
                try {
                    end_();
                } catch (...) {
                }
            }
            ended_ = true;
            ok_    = false;
            return false;
        }
        if (!ok_begin) {
            return false;
        }
        begun_ = true;
        ok_    = true;
        return true;
    }

    // false when the refill could not get the slots back or a row differed from
    // the file: the caller fails the request, the answer would be wrong.
    bool end() {
        if (!begun_) {
            return ok_;
        }
        if (ended_) {
            return ok_;
        }
        ended_ = true;
        if (end_) {
            try {
                ok_ = end_();
            } catch (...) {
                ok_ = false;   // the refill or its re-reserve failed: the loan stays out
            }
        }
        return ok_;
    }

    bool begun() const { return begun_; }
    bool ended() const { return ended_; }
    bool ok()    const { return ok_; }

private:
    Rule     rule_;
    uint32_t prefill_chunk_;
    Begin    begin_;
    End      end_;
    bool     begun_ = false;
    bool     ended_ = false;
    bool     ok_    = true;
};

}  // namespace slot_loan
}  // namespace lgc

// The verify walk of speculative decoding (the libllama engine's --llama-mtp):
// one target decode has produced logits for [last, draft_0, .., draft_{n-1}];
// the sampler draws those rows in order, from the history it has itself
// accepted, and keeps a draft token while it draws that same token. Every
// sample() is the one the plain loop would make at that point, so the
// emitted tokens are distributed as the plain loop's. Pure: no context, no
// lock, so the walk is testable on synthetic logits (tests/test_verify_walk.cpp).
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "core/sampler.h"
#include "exec/backend.h"

namespace lgc {

struct VerifyWalk {
    int          accepted = 0;     // draft tokens kept
    int          emitted  = 0;     // tokens handed to `emit`
    int          last     = -1;    // the last token drawn (the next step's first, unless done)
    bool         done     = false;
    FinishReason reason   = FinishReason::Stop;
};

// rows: n_rows x vocab logits (sample() may rewrite them); draft: the
// n_rows - 1 drafted tokens; budget: how many tokens may still be emitted
// (max_tokens and the context, as the plain loop checks them); emit hands a
// token on and says whether to go on.
inline VerifyWalk walk_verify(float* rows, size_t n_rows, size_t vocab, const std::vector<int>& draft,
                              Sampler& sampler, int budget, const std::function<bool(int)>& is_stop,
                              const std::function<Control(int)>& emit) {
    VerifyWalk w;
    for (size_t i = 0; i < n_rows; ++i) {
        const int tok = sampler.sample(rows + i * vocab, vocab);
        w.last        = tok;
        if (is_stop(tok)) {
            w.done = true;
            break;
        }
        if (w.emitted >= budget) {
            w.done   = true;
            w.reason = FinishReason::Length;
            break;
        }
        const Control ctl = emit(tok);
        ++w.emitted;
        sampler.observe(tok);
        if (ctl != Control::Continue) {
            w.done   = true;
            w.reason = ctl == Control::Cancel ? FinishReason::Abort : FinishReason::Stop;
            break;
        }
        if (i < draft.size() && tok == draft[i]) {
            ++w.accepted;
            continue;
        }
        break;
    }
    return w;
}

}  // namespace lgc

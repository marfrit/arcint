#pragma once

#include <cstddef>
#include <vector>

// Speculative decoding needs something to guess the next few tokens. DESIGN.md
// §3.5 plans to use Qwen3.8's native MTP head and leaves "a hook for external
// drafters"; no export currently carries that head, so the hook is what exists
// first — and it is the half that holds all the machinery anyway. Swapping in
// an MTP head later changes only where the draft comes from.
//
// The drafter here needs no weights at all: it looks for the most recent
// earlier occurrence of the last few tokens and proposes whatever followed. On
// code and on any prompt the model is quoting back, that is right surprisingly
// often; on free prose it is right rarely, and the acceptance figure on the
// console says which case you are in.
namespace lgc {

class Drafter {
public:
    virtual ~Drafter() = default;

    // Proposes up to `max_tokens` continuations of `tokens`. An empty result
    // means "no guess", and the caller decodes normally.
    virtual std::vector<int> draft(const std::vector<int>& tokens, size_t max_tokens) = 0;

    virtual const char* name() const = 0;
};

// Longest-suffix match over the sequence so far ("prompt lookup").
//
// `ngram` is how many trailing tokens must match for a hit. Too small and the
// guesses are noise that costs a rollback; too large and it never fires.
class NgramDrafter final : public Drafter {
public:
    NgramDrafter(size_t ngram, size_t max_draft) : ngram_(ngram), max_draft_(max_draft) {}

    std::vector<int> draft(const std::vector<int>& tokens, size_t max_tokens) override;
    const char*      name() const override { return "ngram"; }

    size_t ngram() const { return ngram_; }

private:
    size_t ngram_;
    size_t max_draft_;
};

}  // namespace lgc

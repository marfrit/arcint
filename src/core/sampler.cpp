#include "core/sampler.h"

#include "core/reasoning.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>

namespace lgc {
namespace {

// How many candidates to pull before falling back to a full sort. Softmax mass
// concentrates hard on a 248k vocabulary, so this covers top_p in practice
// while avoiding a quarter-million-element sort per token. Correctness does not
// depend on it: the probe's probabilities are normalised over the whole
// vocabulary, so a probe short of the top_p mass is seen and falls back.
constexpr size_t kCandidateProbe = 2048;

}  // namespace

Sampler::Sampler(const SamplerParams& params, uint64_t seed)
    : params_(params), seed_(seed), rng_(seed) {}

void Sampler::set_prompt(const std::vector<int>& prompt) {
    seen_.clear();
    generated_.clear();
    seen_.reserve(prompt.size());
    for (int id : prompt) {
        if (id >= 0) ++seen_[id];
    }
}

void TokenConstraint::note_desync(int token, const std::string& why) {
    ++desyncs_;
    log::error("grammar", "desync: %s (token %d); the rest of this request is unconstrained", why.c_str(), token);
}

void Sampler::observe(int token) {
    if (token < 0) return;
    ++seen_[token];
    ++generated_[token];
    // A token inside an open think block is reasoning, not a call: arcint
    // parses calls from the content only (split_reasoning), so the grammar is
    // not fed it -- a call opener the model mentions while thinking does not
    // trigger it. observe_think_text has already seen this token's text, so a
    // closer is fed (harmless: the lazy grammar is still waiting) and the
    // first token after it is the first the grammar can trigger on.
    if (constraint_ != nullptr && think_state_ != ThinkState::thinking) constraint_->accept(token);
}

void Sampler::set_think_mask_ids(ThinkBlockIds ids) {
    params_.think_ids = std::move(ids);
    measure_think_tags();
}

// The tail kept to notice a closer that straddles two pieces is bounded by the
// longest tag, so the check costs nothing per step. Both entry points measure:
// the ids may arrive in the params (a test, or a backend that already knows
// them) or through set_think_mask_ids.
void Sampler::measure_think_tags() {
    think_max_tag_ = 0;
    for (const std::string& t : params_.think_ids.close_tags) think_max_tag_ = std::max(think_max_tag_, t.size());
    for (const std::string& t : params_.think_ids.open_tags) think_max_tag_ = std::max(think_max_tag_, t.size());
}

void Sampler::start_in_think_block(bool open) {
    measure_think_tags();
    // The template wrote the opener as the lead of the turn: the block is
    // already open before the first token. Otherwise the first tokens decide.
    think_state_ = (tracks_think() && open) ? ThinkState::thinking : ThinkState::undecided;
    think_tail_.clear();
    think_lead_.clear();
}

// Feed the generated text, in generation order. The state is undecided
// (the opening is not settled yet), thinking (the block is open) or content
// (settled: the block closed, or never opened). The lead is examined only
// while undecided, and stripped of leading spaces and newlines with the same
// lead_of split_reasoning uses, so the mask and the reasoning split agree
// about where the block is. Once content the state is final for the turn: the
// lead is never examined again and nothing re-opens the block.
void Sampler::observe_think_text(std::string_view piece) {
    if (!tracks_think()) return;

    if (think_state_ == ThinkState::undecided) {
        // Before the block's opening is settled, only the generation's first
        // non-blank characters matter. The rule is deliberately one-sided, and
        // errs towards masking: the mask engages as soon as the lead matches an
        // opener or could still grow into one, and only lets go when the lead
        // has provably become something else. Masking the turn-ender for the
        // two or three tokens of a partial opener costs nothing; releasing a
        // turn that the client still sees as reasoning is the defect.
        think_lead_.append(piece);
        // Strip the lead exactly as lead_of does, so a pad of spaces and
        // newlines before a self-opened block is not part of the opener and
        // the mask engages where the split would. Keep only as many lead
        // characters as the longest tag: the decision needs no more, so a
        // blank run costs nothing and the lead stays bounded.
        const std::string_view lead = lead_of(think_lead_);
        if (lead.size() != think_lead_.size()) think_lead_.erase(0, think_lead_.size() - lead.size());
        if (think_lead_.size() > think_max_tag_) think_lead_.resize(think_max_tag_);

        bool matched = false;
        bool loomed  = false;
        for (const std::string& t : params_.think_ids.open_tags) {
            if (think_lead_.size() >= t.size()) {
                if (think_lead_.compare(0, t.size(), t) == 0) { matched = true; break; }
            } else if (!think_lead_.empty() && t.compare(0, think_lead_.size(), think_lead_) == 0) {
                loomed = true;
            }
        }
        if (matched || loomed) {
            // Open from here, and the rest of this piece is inside the block:
            // the closer scan below must run on it.
            think_state_ = ThinkState::thinking;
            think_lead_.clear();
        } else if (!think_lead_.empty()) {
            // No opener matched or loomed: the answer begins as content, and
            // the lead is settled and never examined again.
            think_state_ = ThinkState::content;
            think_lead_.clear();
            return;
        }
        // A blank-only lead could still grow into an opener: stay undecided.
    }

    // thinking
    think_tail_.append(piece);
    // Keep enough of the tail to see a closer that straddles two pieces, and
    // never less than the piece just appended: a closer wholly inside one long
    // piece must still be seen.
    const size_t keep = std::max(think_max_tag_ * 2 + 8, piece.size());
    if (think_tail_.size() > keep) think_tail_.erase(0, think_tail_.size() - keep);
    for (const std::string& t : params_.think_ids.close_tags) {
        if (think_tail_.find(t) != std::string::npos) {
            // Settled: content. Both windows are cleared, so neither the
            // stale lead nor a tail remnant can re-open the block later.
            think_state_ = ThinkState::content;
            think_tail_.clear();
            think_lead_.clear();
            return;
        }
    }
}

// While the block is open the end-of-turn token and a second opener are not
// legal next tokens. Applied to the logits, not to the stop check: a filter in
// the stop check would let the token be sampled, take its probability mass, and
// then be ignored, so the sampled token would not be the one the distribution
// describes.
void Sampler::mask_illegal_while_thinking(float* logits, size_t vocab) const {
    if (!params_.think_eog_mask || think_state_ != ThinkState::thinking) return;
    auto kill = [&](const std::vector<int>& ids) {
        for (int id : ids)
            if (id >= 0 && static_cast<size_t>(id) < vocab) logits[static_cast<size_t>(id)] = -INFINITY;
    };
    kill(params_.think_ids.eog);
    kill(params_.think_ids.open_ids);
}

int Sampler::argmax(const float* logits, size_t vocab) {
    size_t best = 0;
    float  top  = logits[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (logits[i] > top) {
            top  = logits[i];
            best = i;
        }
    }
    return static_cast<int>(best);
}

void Sampler::apply_penalties(float* logits, size_t vocab) const {
    const bool rep  = params_.repetition_penalty != 1.0f;
    const bool pres = params_.presence_penalty != 0.0f;
    const bool freq = params_.frequency_penalty != 0.0f;

    if (rep) {
        for (const auto& [id, n] : seen_) {
            (void)n;
            if (static_cast<size_t>(id) >= vocab) continue;
            float& l = logits[static_cast<size_t>(id)];
            // llama.cpp's convention: dividing a positive logit and multiplying
            // a negative one both push the token down.
            l = l > 0.0f ? l / params_.repetition_penalty : l * params_.repetition_penalty;
        }
    }
    if (pres || freq) {
        for (const auto& [id, n] : generated_) {
            if (static_cast<size_t>(id) >= vocab) continue;
            float& l = logits[static_cast<size_t>(id)];
            if (pres) l -= params_.presence_penalty;
            if (freq) l -= params_.frequency_penalty * static_cast<float>(n);
        }
    }
}

// Fills scratch_ with the candidates worth considering, largest logit first.
// Returns how many are usable.
size_t Sampler::collect_candidates(const float* logits, size_t vocab, bool full) {
    const size_t want = full ? vocab
                        : params_.top_k > 0
                            ? std::min(static_cast<size_t>(params_.top_k), vocab)
                            : std::min(kCandidateProbe, vocab);

    scratch_.clear();
    scratch_.reserve(vocab);
    for (size_t i = 0; i < vocab; ++i) scratch_.emplace_back(logits[i], static_cast<int>(i));

    auto by_logit = [](const auto& a, const auto& b) { return a.first > b.first; };
    if (want < vocab) {
        std::partial_sort(scratch_.begin(), scratch_.begin() + static_cast<long>(want),
                          scratch_.end(), by_logit);
        scratch_.resize(want);
    } else {
        std::sort(scratch_.begin(), scratch_.end(), by_logit);
    }
    return scratch_.size();
}

int Sampler::sample(float* logits, size_t vocab) {
    if (vocab == 0) return 0;

    mask_illegal_while_thinking(logits, vocab);
    apply_penalties(logits, vocab);

    const int tok = draw(logits, vocab);
    if (constraint_ == nullptr || think_state_ == ThinkState::thinking || constraint_->allows(tok)) return tok;
    // Grammar-based rejection sampling, as the reference's
    // common_sampler_sample: the drawn token does not fit, so mask the row
    // (after the think mask and the penalties, as the reference re-runs its
    // chain over the masked candidates) and draw again. A rejection costs one
    // full-vocabulary grammar pass; a token that fits costs one token's test.
    constraint_->mask(logits, vocab);
    return draw(logits, vocab);
}

int Sampler::draw(float* logits, size_t vocab) {
    // Penalties can reorder the distribution, so greedy is decided after them.
    if (params_.greedy()) return argmax(logits, vocab);

    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool   probing = attempt == 0 && params_.top_k <= 0 && kCandidateProbe < vocab;
        const size_t n       = collect_candidates(logits, vocab, attempt > 0);
        if (n == 0) return argmax(logits, vocab);

        const float temp = params_.temperature > 0.0f ? params_.temperature : 1.0f;
        const float max  = scratch_.front().first;

        double              sum = 0.0;
        std::vector<double> probs(scratch_.size());
        for (size_t i = 0; i < scratch_.size(); ++i) {
            probs[i] = std::exp(static_cast<double>(scratch_[i].first - max) / temp);
            sum += probs[i];
        }
        // A probe holds part of the vocabulary: normalise over all of it, so
        // top_p measures the true mass and a short probe is noticed
        if (probing) {
            sum = 0.0;
            for (size_t i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(logits[i] - max) / temp);
        }
        for (double& p : probs) p /= sum;

        // min-p as llama.cpp applies it (src/llama-sampler.cpp,
        // llama_sampler_min_p_apply): keep the logits at or above
        // max + ln(min_p), i.e. p >= min_p * p_max before temperature. The
        // candidates are sorted, so it keeps a prefix; once a candidate falls
        // below the floor, so does the rest of the vocabulary, and the probe
        // needs no full sort.
        size_t min_p_cut    = probs.size();
        bool   min_p_closed = false;
        if (params_.min_p > 0.0f) {
            const float floor = max + std::log(params_.min_p);
            min_p_cut         = 1;
            while (min_p_cut < scratch_.size() && scratch_[min_p_cut].first >= floor) ++min_p_cut;
            min_p_closed = min_p_cut < scratch_.size();
        }

        // top-p over the already-sorted order. At least one candidate always
        // survives, so a tiny top_p degenerates to greedy rather than nothing.
        size_t cutoff = probs.size();
        bool   short_of_mass = false;
        if (params_.top_p < 1.0f) {
            double acc = 0.0;
            cutoff     = probs.size();
            for (size_t i = 0; i < probs.size(); ++i) {
                acc += probs[i];
                if (acc >= static_cast<double>(params_.top_p)) {
                    cutoff = i + 1;
                    break;
                }
            }
            // The probe held less mass than top_p asked for; only then is the
            // full sort actually needed.
            if (acc < static_cast<double>(params_.top_p) && probing && !min_p_closed) short_of_mass = true;
        } else if (probing && scratch_.size() < vocab && !min_p_closed) {
            short_of_mass = true;
        }
        cutoff = std::min(cutoff, min_p_cut);

        if (short_of_mass) continue;   // the full set next pass (this token only)

        double renorm = 0.0;
        for (size_t i = 0; i < cutoff; ++i) renorm += probs[i];

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double                                 r = dist(rng_) * renorm;
        for (size_t i = 0; i < cutoff; ++i) {
            r -= probs[i];
            if (r <= 0.0) return scratch_[i].second;
        }
        return scratch_[cutoff - 1].second;
    }
    return argmax(logits, vocab);
}

}  // namespace lgc

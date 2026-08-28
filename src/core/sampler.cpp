#include "core/sampler.h"

#include <algorithm>
#include <cmath>

namespace lgc {
namespace {

// How many candidates to pull before falling back to a full sort. Softmax mass
// concentrates hard on a 248k vocabulary, so this covers top_p in practice
// while avoiding a quarter-million-element sort per token; correctness does not
// depend on it, because a shortfall falls back.
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

void Sampler::observe(int token) {
    if (token < 0) return;
    ++seen_[token];
    ++generated_[token];
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
size_t Sampler::collect_candidates(const float* logits, size_t vocab) {
    const size_t want = params_.top_k > 0
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

    apply_penalties(logits, vocab);

    // Penalties can reorder the distribution, so greedy is decided after them.
    if (params_.greedy()) return argmax(logits, vocab);

    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool   probing = attempt == 0 && params_.top_k <= 0 && kCandidateProbe < vocab;
        const size_t n       = probing ? collect_candidates(logits, vocab)
                                       : collect_candidates(logits, vocab);
        if (n == 0) return argmax(logits, vocab);

        const float temp = params_.temperature > 0.0f ? params_.temperature : 1.0f;
        const float max  = scratch_.front().first;

        double              sum = 0.0;
        std::vector<double> probs(scratch_.size());
        for (size_t i = 0; i < scratch_.size(); ++i) {
            probs[i] = std::exp(static_cast<double>(scratch_[i].first - max) / temp);
            sum += probs[i];
        }
        for (double& p : probs) p /= sum;

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
            if (acc < static_cast<double>(params_.top_p) && probing) short_of_mass = true;
        } else if (probing && scratch_.size() < vocab) {
            short_of_mass = true;
        }

        if (short_of_mass) {
            params_.top_k = static_cast<int>(vocab);  // force the full set next pass
            continue;
        }

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

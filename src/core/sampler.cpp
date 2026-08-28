#include "core/sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace lgc {

Sampler::Sampler(const SamplerParams& params, uint64_t seed)
    : params_(params), seed_(seed), rng_(seed) {}

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

void Sampler::apply_penalties(float* logits, size_t vocab,
                              const std::vector<int>& history) const {
    const bool rep  = params_.repetition_penalty != 1.0f;
    const bool pres = params_.presence_penalty != 0.0f;
    const bool freq = params_.frequency_penalty != 0.0f;
    if (!rep && !pres && !freq) return;
    if (history.empty()) return;

    std::unordered_map<int, int> counts;
    counts.reserve(history.size());
    for (int id : history) {
        if (id >= 0 && static_cast<size_t>(id) < vocab) ++counts[id];
    }

    for (const auto& [id, n] : counts) {
        float& l = logits[static_cast<size_t>(id)];
        if (rep) {
            // llama.cpp's convention: dividing a positive logit and multiplying
            // a negative one both push the token down.
            l = l > 0.0f ? l / params_.repetition_penalty : l * params_.repetition_penalty;
        }
        if (pres) l -= params_.presence_penalty;
        if (freq) l -= params_.frequency_penalty * static_cast<float>(n);
    }
}

int Sampler::sample(float* logits, size_t vocab, const std::vector<int>& history) {
    if (vocab == 0) return 0;

    apply_penalties(logits, vocab, history);

    // Penalties can reorder the distribution, so greedy is decided after them.
    if (params_.greedy()) return argmax(logits, vocab);

    scratch_.clear();
    scratch_.reserve(vocab);
    for (size_t i = 0; i < vocab; ++i) {
        scratch_.emplace_back(logits[i], static_cast<int>(i));
    }

    // top-k first: it is a pure prefix of the sorted order, so doing it before
    // the softmax keeps the expensive exp() count down.
    size_t keep = vocab;
    if (params_.top_k > 0 && static_cast<size_t>(params_.top_k) < vocab) {
        keep = static_cast<size_t>(params_.top_k);
        std::partial_sort(scratch_.begin(), scratch_.begin() + static_cast<long>(keep),
                          scratch_.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        scratch_.resize(keep);
    } else {
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    const float temp = params_.temperature > 0.0f ? params_.temperature : 1.0f;
    const float max  = scratch_.front().first;

    double sum = 0.0;
    std::vector<double> probs(scratch_.size());
    for (size_t i = 0; i < scratch_.size(); ++i) {
        probs[i] = std::exp(static_cast<double>(scratch_[i].first - max) / temp);
        sum += probs[i];
    }
    for (double& p : probs) p /= sum;

    // top-p over the already-sorted order. At least one candidate always
    // survives, so a tiny top_p degenerates to greedy rather than to nothing.
    size_t cutoff = probs.size();
    if (params_.top_p < 1.0f) {
        double acc = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            acc += probs[i];
            if (acc >= static_cast<double>(params_.top_p)) {
                cutoff = i + 1;
                break;
            }
        }
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

}  // namespace lgc

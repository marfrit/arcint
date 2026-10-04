#include "core/draft_policy.h"

#include <algorithm>

// Constants as in Strata's src/spec/draft_policy.cpp; see the header.
namespace lgc {
namespace {

// A round's cost by window size relative to one token, used only for sizes
// not measured yet (measured round times replace it).
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
constexpr double kCostAlpha = 0.1;  // EMA weight of a new round time
constexpr double kTokAlpha = 0.05;  // EMA weight of a new base-window outcome
constexpr double kDecay = 0.97;     // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its
// continuation; worth 4 observations, so a few real windows override it.
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;  // a lookup size is tried this often before a guessed cost can veto it

}  // namespace

DraftPolicy::DraftPolicy(int max_t, double margin) : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[t] > 0)
        return cost_[t];
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u)
        if (cost_n_[u] > 0) {
            const double w = std::min(cost_n_[u], 20.0);
            num += w * cost_[u] * kShape[t] / kShape[u];
            den += w;
        }
    return den > 0 ? num / den : kShape[t];
}

double DraftPolicy::base_tokens(int t) const {
    if (base_n_[t] > 0)
        return base_tok_[t];
    return 1.0 + 0.7 * (t - 1);
}

DraftPolicy::Pick DraftPolicy::choose(int t_base, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_base, 1, max_t_);
    if (lookup_k <= 0)
        return p;
    const double base = base_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) {
            best = r;
            best_t = k + 1;
        }
    }
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        return p;
    }
    // A guessed cost must not keep a size from ever being measured: the first
    // few times a confident lookup needs an unmeasured size, it is tried.
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (t_full > p.t && cost_n_[t_full] < kProbes && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

int DraftPolicy::chain_bucket(float p) {
    return p < 0.5f ? 0 : p < 0.7f ? 1 : p < 0.85f ? 2 : p < 0.95f ? 3 : 4;
}

double DraftPolicy::chain_rate(float p) const {
    const int b = chain_bucket(p);
    // The prior is the head's own probability, worth kPriorN observations.
    return (chain_ok_[b] + kPriorN * static_cast<double>(p)) / (chain_ok_[b] + chain_bad_[b] + kPriorN);
}

int DraftPolicy::choose_chain(const float* p, int n) const {
    n = std::clamp(n, 0, max_t_ - 1);
    double best = 1.0 / cost_ms(1), e = 1.0, a = 1.0;
    int    best_k = 0;
    for (int k = 1; k <= n; ++k) {
        a *= chain_rate(p[k - 1]);
        e += a;
        const double r = e / cost_ms(k + 1);
        if (r > best * (1.0 + margin_)) {
            best   = r;
            best_k = k;
        }
    }
    // Probe an unmeasured size while the chain is confident: every draft up
    // to k at least 0.85 likely, as the lookup side asks of its rate.
    float lo = 1.0f;
    int   confident = 0;
    for (int k = 1; k <= n; ++k) {
        lo = std::min(lo, p[k - 1]);
        if (lo < 0.85f) break;
        confident = k;
    }
    for (int k = confident; k > best_k; --k)
        if (cost_n_[k + 1] < kProbes)
            return k;
    return best_k;
}

void DraftPolicy::observe_chain(int t, const float* p, int n, int accepted, double round_ms) {
    observe(false, t, accepted, 0, round_ms);
    for (int j = 0; j < n && j <= accepted; ++j) {
        const int b = chain_bucket(p[j]);
        chain_ok_[b]  = kDecay * chain_ok_[b] + (j < accepted ? 1.0 : 0.0);
        chain_bad_[b] = kDecay * chain_bad_[b] + (j == accepted ? 1.0 : 0.0);
    }
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    if (round_ms > 0) {
        cost_[t] = cost_n_[t] > 0 ? (1.0 - kCostAlpha) * cost_[t] + kCostAlpha * round_ms : round_ms;
        cost_n_[t] += 1.0;
    }
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        base_tok_[t] = base_n_[t] > 0 ? (1.0 - kTokAlpha) * base_tok_[t] + kTokAlpha * got : got;
        base_n_[t] += 1.0;
    }
}

}  // namespace lgc

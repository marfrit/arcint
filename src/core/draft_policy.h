#pragma once

#include <array>

// Per verify round: verify a lookup (prompt-lookup) window, or decode plain?
// A port of Strata's DraftPolicy (github.com/Niko1221/Strata, MIT,
// src/spec/draft_policy.cpp): prompt lookup taken whenever it proposes lost
// on free text (measured here on Flash-Next, B60: a free-form answer 13.2 ->
// 11.4 t/s, a quoting answer 12.8 -> 17.9), because a wrong window costs more
// to verify than a plain step. The policy learns both sides online and
// compares expected committed tokens per millisecond:
//
//   plain   1 token at the measured cost of a 1-token round (and, once an MTP
//           head drafts for this model, the MTP window's committed tokens at
//           its measured cost -- Strata's other side)
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the
//           acceptance of lookup drafts whose match was about as long (4
//           buckets of match length, decayed counts), at the measured cost of
//           a k + 1 window
//
// and takes the lookup window only when its best E/cost beats the other side
// by `margin`. It chooses which drafts to verify; the output is unchanged.
namespace lgc {

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;

    explicit DraftPolicy(int max_t = kMaxT, double margin = 0.03);

    struct Pick {
        bool lookup = false;
        int t = 1;  // window size (1 + drafts)
    };
    // `t_base`: the other side's window (1 = plain decode); `lookup_k`: the
    // lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_base, int lookup_k, int match) const;
    // After the round: the window used, the drafts accepted, the round time.
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    double lookup_rate(int match) const;
    double cost_ms(int t) const;

    // An MTP draft chain (Strata's MTP side of the same comparison): `p[j]`
    // is draft j's probability under the draft head. Returns how many of the
    // n drafts to verify (0 = decode plain): the k with the best expected
    // committed tokens per ms, E(k) = 1 + a1 + a1 a2 + ..., a_j draft j's
    // acceptance estimate (the head's probability, calibrated online per
    // probability bucket), against the measured cost of a k + 1 window. A
    // window size not measured yet is tried kProbes times while the chain is
    // confident, so a guessed cost cannot veto it for good.
    int choose_chain(const float* p, int n) const;
    // After an MTP round: the window used, the drafts' probabilities, the
    // drafts accepted, the round time.
    void observe_chain(int t, const float* p, int n, int accepted, double round_ms);
    double chain_rate(float p) const;

private:
    static int bucket(int match);
    double base_tokens(int t) const;

    int max_t_;
    double margin_;
    std::array<double, kMaxT + 1> cost_{}, cost_n_{};
    std::array<double, kMaxT + 1> base_tok_{}, base_n_{};
    std::array<double, kBuckets> ok_{}, bad_{};
    static constexpr int kChainBuckets = 5;
    static int chain_bucket(float p);
    std::array<double, kChainBuckets> chain_ok_{}, chain_bad_{};
};

}  // namespace lgc

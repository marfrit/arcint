// Strata's cost-aware MTP window rule, as a pure class (no libllama): the MTP
// half of DraftPolicy, which decides whether the draft chain goes on and how
// many rows it verifies. Campaign research-mtp-width-policy, section 5.2.
//
// Ported from Strata-ref:src/spec/draft_policy.cpp (:40-53 cost_ms, :55-65
// clamp_sample, :66-74 seed_costs, :76-93 set_context, :118-121 mtp_tokens,
// :154-170 observe, :182-188 draft_rate, :190-196 observe_probs, :199-224
// keep_draft, :98-113 cost_table_text) and Strata-ref:include/strata/spec/
// draft_policy.hpp (:25-27, :57-62 the chain state). tools/mtp_width_replay.py
// ports the same rule; tests/test_draft_policy.cpp pins both sides round by
// round against one decisions fixture, so the class has ONE behaviour with the
// replay.
//
// What differs from the reference on purpose, each recorded:
//  - ChainState carries `best_e` beside `best_rows` (Strata does not): the
//    replay reports E at the width it would verify, so the class keeps it.
//  - observe_round also learns the MTP window-token EMA, as the reference's
//    observe(t, accepted, ms) does; the MTP rule itself does not read it.
//  - No lookahead drafter (choose/chain/stale), no chain_rate: the MTP rule
//    does not call them (generate.cpp:16513 is the only call site of the MTP
//    rule; :11221 is the suffix path's).
//
// The rule: E_j = 1 + q_0 + .. + q_j, q_i = prod a(p_k); the window of j + 2
// rows is worth E_j / cost(j + 2) tokens per ms; the chain goes on while one
// or two more drafts could beat the best width by margin/3. A draft below the
// min-p floor is never kept (the floor stays ours; the reference's cost arm
// has no floor - limitation recorded in the campaign doc, section 7.1).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace lgc {

// The MTP rule's chain: one round's drafting state (draft_policy.hpp:57-62).
struct PolicyChain {
    double q        = 1.0;
    double e        = 1.0;
    double best_v   = 0.0;
    double best_e   = 1.0;     // E at best_rows (the replay's own field, see above)
    int    best_rows = 1;
    int    kept      = 0;
};

class DraftPolicy {
public:
    static constexpr int    kMaxT        = 8;     // draft_policy.hpp:25, the widest window in rows
    static constexpr int    kCtxBuckets  = 3;     // draft_policy.hpp:27, <6K / <24K / longer
    static constexpr int    kPBuckets    = 8;     // draft_policy.cpp:174-179
    static constexpr double kMargin      = 0.03;  // draft_policy.hpp:32
    static constexpr double kCostAlpha   = 0.1;   // draft_policy.cpp:14
    static constexpr double kTokAlpha    = 0.05;  // draft_policy.cpp:15
    static constexpr double kCalDecay    = 0.995; // draft_policy.cpp:16
    static constexpr double kPriorN      = 4.0;   // draft_policy.cpp:21
    static constexpr int    kProbes      = 3;     // draft_policy.cpp:22
    static constexpr double kSeedWeight  = 6.0;   // generate.cpp's cost table seeding
    // draft_policy.cpp:13: the shape guess, rows 1..8
    static constexpr double kShape[kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};

    DraftPolicy() = default;

    // draft_policy.cpp:40-53: the measured value, else the kShape scaling of
    // the measured widths, weighted by how often each was seen (at most 20).
    double cost_ms(int t) const {
        t = std::max(1, std::min(t, kMaxT));
        if (cost_n_[bk_][t] > 0) return cost_[bk_][t];
        double num = 0.0, den = 0.0;
        for (int u = 1; u <= kMaxT; ++u)
            if (cost_n_[bk_][u] > 0) {
                const double w = std::min(cost_n_[bk_][u], 20.0);
                num += w * cost_[bk_][u] * kShape[t] / kShape[u];
                den += w;
            }
        return den > 0 ? num / den : kShape[t];
    }

    // draft_policy.cpp:55-65 with robust on (STRATA_SPEC_COST=1 sets it,
    // generate.cpp:16509): a width's FIRST sample is replaced by the guess from
    // the other widths (a first use captures its graph); later ones cap at 2x.
    double clamp_sample(int t, double ms) const {
        const double known = cost_ms(t);
        if (cost_n_[bk_][t] == 0) {
            for (int u = 1; u <= kMaxT; ++u)
                if (cost_n_[bk_][u] > 0) return known;
            return ms;
        }
        return std::min(ms, 2.0 * known);
    }

    // draft_policy.cpp:66-74: a per-width table counted as `weight` rounds.
    // A width already measured or seeded is not replaced.
    void seed_costs(const double* ms, int n, double weight = kSeedWeight, int bucket = 0) {
        bucket = std::max(0, std::min(bucket, kCtxBuckets - 1));
        for (int t = 1; t <= std::min(n, kMaxT); ++t)
            if (ms[t - 1] > 0.0 && cost_n_[bucket][t] == 0) {
                cost_[bucket][t]   = ms[t - 1];
                cost_n_[bucket][t] = weight;
            }
    }

    // Strata's STRATA_SPEC_COST_TABLE form (generate.cpp:16495-16511):
    // "T1,T2,..;T1,..;T1,..", the round ms per width, one list a context
    // bucket. False with `err` set on malformed input.
    bool seed_from_text(const std::string& text, std::string& err) {
        int bucket = 0;
        size_t at = 0;
        while (at <= text.size() && bucket < kCtxBuckets) {
            const size_t semi = text.find(';', at);
            const std::string part = text.substr(at, semi == std::string::npos ? std::string::npos : semi - at);
            std::vector<double> vals;
            size_t p = 0;
            while (p <= part.size()) {
                const size_t comma = part.find(',', p);
                std::string item = part.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
                // trim spaces
                const size_t b = item.find_first_not_of(" \t");
                const size_t e = item.find_last_not_of(" \t");
                item = b == std::string::npos ? std::string() : item.substr(b, e - b + 1);
                if (!item.empty()) {
                    char* end = nullptr;
                    const double v = std::strtod(item.c_str(), &end);
                    if (end == nullptr || *end != '\0') {
                        err = "--llama-mtp-cost-seed: " + item + " is not a number";
                        return false;
                    }
                    vals.push_back(v);
                }
                if (comma == std::string::npos) break;
                p = comma + 1;
            }
            if (!vals.empty()) seed_costs(vals.data(), static_cast<int>(vals.size()), kSeedWeight, bucket);
            ++bucket;
            if (semi == std::string::npos) break;
            at = semi + 1;
        }
        if (bucket == 0) {
            err = "--llama-mtp-cost-seed: no lists";
            return false;
        }
        return true;
    }

    // draft_policy.cpp:76-93: the context bucket of the next round's rows; an
    // empty bucket takes its nearest non-empty one at weight 1.
    void set_context(long long pos) {
        const int nb = pos < 6000 ? 0 : pos < 24000 ? 1 : 2;
        if (nb == bk_) return;
        bool has = false;
        for (int t = 1; t <= kMaxT; ++t) has = has || cost_n_[nb][t] > 0;
        if (!has) {
            for (int d = 1; d < kCtxBuckets && !has; ++d) {
                for (int src : {nb - d, nb + d}) {
                    if (src < 0 || src >= kCtxBuckets || has) continue;
                    for (int t = 1; t <= kMaxT; ++t)
                        if (cost_n_[src][t] > 0) {
                            cost_[nb][t]   = cost_[src][t];
                            cost_n_[nb][t] = 1.0;
                            has            = true;
                        }
                }
            }
        }
        bk_ = nb;
    }

    // draft_policy.cpp:118-121.
    double mtp_tokens(int t) const {
        if (mtp_n_[t] > 0) return mtp_tok_[t];
        return 1.0 + 0.7 * (t - 1);
    }

    // draft_policy.cpp:154-170 with lookup=false: the measured round cost of a
    // window of `rows` and the tokens it committed. `ms` <= 0 learns no cost
    // (the trace's own rule for an unknown round time).
    void observe_round(int rows, int accepted, double ms) {
        rows = std::max(1, std::min(rows, kMaxT));
        if (ms > 0.0) {
            const double m     = clamp_sample(rows, ms);
            const double alpha = std::max(kCostAlpha, 1.0 / (cost_n_[bk_][rows] + 1.0));
            const double cn    = cost_n_[bk_][rows];
            cost_[bk_][rows]   = cn > 0 ? (1.0 - alpha) * cost_[bk_][rows] + alpha * m : m;
            cost_n_[bk_][rows] += 1.0;
        }
        ++rounds_;
        const double got = accepted + 1.0;
        mtp_tok_[rows]   = mtp_n_[rows] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[rows] + kTokAlpha * got : got;
        mtp_n_[rows] += 1.0;
    }

    // draft_policy.cpp:182-188: until a bucket has data, a draft holds as often
    // as the MTP says (the prior is `p` itself, worth kPriorN).
    double draft_rate(double p) const {
        const int    b      = p_bucket(p);
        const double centre = std::max(0.0, std::min(p, 1.0));
        return (pa_ok_[b] + kPriorN * centre) / (pa_n_[b] + kPriorN);
    }

    // draft_policy.cpp:190-196: draft j was reached when the drafts before it
    // held (j <= accepted) and held when j < accepted.
    void observe_probs(const float* probs, int n_window_drafts, int accepted) {
        for (int j = 0; j < n_window_drafts; ++j) {
            if (j > accepted) break;
            const int b   = p_bucket(static_cast<double>(probs[j]));
            pa_ok_[b]     = kCalDecay * pa_ok_[b] + (j < accepted ? 1.0 : 0.0);
            pa_n_[b]      = kCalDecay * pa_n_[b] + 1.0;
        }
    }

    // draft_policy.cpp:199-224, verbatim, plus best_e. Returns whether to
    // compute the next draft; st.best_rows is the window to verify.
    bool keep_draft(PolicyChain& st, int idx, double pf, int cap) const {
        cap = std::max(1, std::min(cap, kMaxT));
        const double p = pf;
        if (idx == 0) {
            st = PolicyChain{};
            st.best_v = 1.0 / cost_ms(1);
        }
        const double a = draft_rate(p);
        st.q *= a;
        st.e += st.q;
        const int rows = idx + 2;
        if (rows <= cap) {
            const double v = st.e / cost_ms(rows);
            if (v > st.best_v * (1.0 + kMargin / 3.0)) {
                st.best_v   = v;
                st.best_e   = st.e;
                st.best_rows = rows;
            }
            if (cost_n_[bk_][rows] < kProbes && st.q >= 0.15) {
                st.best_e    = st.e;
                st.best_rows = rows;
            }
        }
        st.kept = st.best_rows - 1;
        if (rows + 1 > cap) return false;
        const double v1 = (st.e + st.q * a) / cost_ms(rows + 1);
        const double v2 = rows + 2 <= cap ? (st.e + st.q * a + st.q * a * a) / cost_ms(rows + 2) : 0.0;
        return std::max(v1, v2) > st.best_v * (1.0 + kMargin / 3.0) ||
               (cost_n_[bk_][rows + 1] < kProbes && st.q * a >= 0.15);
    }

    // draft_policy.cpp:98-113: the bucket named once, then a field a width.
    std::string cost_table_text() const {
        std::string out;
        char        b[64];
        for (int k = 0; k < kCtxBuckets; ++k) {
            bool any = false;
            for (int t = 1; t <= kMaxT; ++t) any = any || cost_n_[k][t] > 0;
            if (!any) continue;
            out += out.empty() ? "" : " | ";
            out += k == 0 ? "ctx<6K:" : k == 1 ? "ctx<24K:" : "ctx>=24K:";
            for (int t = 1; t <= kMaxT; ++t)
                if (cost_n_[k][t] > 0) {
                    std::snprintf(b, sizeof b, " T%d:%.2f(n=%.0f)", t, cost_[k][t], cost_n_[k][t]);
                    out += b;
                }
        }
        return out;
    }

    int bucket() const { return bk_; }
    double cost_count(int bucket, int t) const { return cost_n_[bucket][t]; }
    double cost_value(int bucket, int t) const { return cost_[bucket][t]; }
    double rounds() const { return rounds_; }

private:
    static int p_bucket(double p) {
        const int b = static_cast<int>(p * 8.0);
        return b < 0 ? 0 : b > 7 ? 7 : b;
    }

    double cost_[kCtxBuckets][kMaxT + 1]   = {};
    double cost_n_[kCtxBuckets][kMaxT + 1] = {};
    int    bk_        = 0;
    double mtp_tok_[kMaxT + 1] = {};
    double mtp_n_[kMaxT + 1]   = {};
    double pa_ok_[kPBuckets]   = {};
    double pa_n_[kPBuckets]    = {};
    double rounds_             = 0.0;
};

}  // namespace lgc

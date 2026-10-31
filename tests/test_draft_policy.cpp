// Strata's cost-aware MTP window rule (src/exec/draft_policy.h), device-free:
// one fixture pins the C++ class and tools/mtp_width_replay.py round by round.
//
// tests/python/fixtures/mtp_policy_decisions.txt is the replay's own
// `--decisions` output over tests/python/fixtures/mtp_trace_lines.txt (one line
// a round: the policy's cap, chosen width, E at that width, its rate, and the
// draft count the trace holds), then its learned cost table. Both sides compare
// their bytes to this file, so a divergence in either parser, the rule, the
// observer or the table's format fails one of them.
//
// Red mutations seen failing while this was written:
// (a) in keep_draft, `st.best_e = st.e` on every improvement -> the fixture's
//     second round reads E at the chain's end (4.189...) instead of at its
//     chosen width (3.843...); (b) observe_round's cost EMA incremented before
//     the alpha -> the table's later T counts differ and the last line fails;
// (c) the truncation helper below verifying the chain's `made` count instead
//     of the policy's `best_rows - 1` -> the 0.9/0.01 case holds 2 drafts where
//     its chosen width of 2 rows can hold 1; (d) kShape's T3 changed to 1.5 ->
//     the second round's chosen width and rate move and the fixture fails.
#include "exec/draft_policy.h"
#include "harness.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::string kSourceDir = ARCINT_SOURCE_DIR;

std::vector<std::string> lines_at(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) out.push_back(line);
    return out;
}

// an `mtp2` fixture line -> the fields the rule reads (the replay's parse_line
// is the other side of the same fixture; a field this parser misreads changes
// a decision line)
struct Round {
    long long req = 0, seq = 0, pos = 0;
    int       rows = 0, acc = 0, nmax = 0, minp = 0;
    double    ms = 0.0;
    std::vector<double> probs;   // double: the fixture prints %.6f, the replay parses double
};

bool parse_round(const std::string& line, Round& r) {
    std::istringstream ss(line);
    std::string        tok;
    if (!(ss >> tok) || tok != "mtp2") return false;
    while (ss >> tok) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) return false;
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "req") r.req = std::stoll(v);
        else if (k == "seq") r.seq = std::stoll(v);
        else if (k == "pos") r.pos = std::stoll(v);
        else if (k == "rows") r.rows = std::stoi(v);
        else if (k == "acc") r.acc = std::stoi(v);
        else if (k == "nmax") r.nmax = std::stoi(v);
        else if (k == "minp") r.minp = static_cast<int>(std::stof(v) * 1000.0 + 0.5);
        else if (k == "ms") r.ms = std::stod(v);
        else if (k == "p") {
            size_t at = 0;
            while (at <= v.size()) {
                const size_t comma = v.find(',', at);
                const std::string item = v.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                if (!item.empty()) r.probs.push_back(std::stod(item));
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
        }
    }
    return true;
}

// One round as the serving path runs it (src/exec/llama_spec.cpp's draft_multi
// truncation + backend's observe): the chain through keep_draft until it says
// stop, the verify of best_rows rows, then the round's outcome observed.
struct Outcome {
    int    cap      = 0;
    int    chosen   = 0;
    int    drafted  = 0;   // drafts the chain made
    int    drafts   = 0;   // drafts the verify holds = chosen - 1
    double e        = 0.0;
    double v        = 0.0;
};

Outcome run_round(lgc::DraftPolicy& policy, const Round& r) {
    policy.set_context(r.pos);
    const int cap = std::max(1, std::min(r.nmax + 1, lgc::DraftPolicy::kMaxT));
    lgc::PolicyChain st;
    int made = 0;
    for (size_t idx = 0; idx < r.probs.size(); ++idx) {
        const bool go = policy.keep_draft(st, static_cast<int>(idx), r.probs[idx], cap);
        ++made;
        if (!go) break;
    }
    Outcome o;
    o.cap     = cap;
    o.chosen  = std::max(1, st.best_rows);
    o.drafted = made;
    o.drafts  = o.chosen - 1;   // the truncation: never the whole drafted chain
    o.e       = st.best_e;
    o.v       = st.best_v;
    policy.observe_round(r.rows, r.acc, r.ms);
    std::vector<float> pf;
    for (double p : r.probs) pf.push_back(static_cast<float>(p));
    policy.observe_probs(pf.data(), std::max(0, std::min(r.rows - 1, static_cast<int>(pf.size()))),
                         r.acc);
    return o;
}

std::string decision_line(const Round& r, const Outcome& o) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "mtpdec req=%lld seq=%lld pos=%lld rows=%d acc=%d cap=%d chosen=%d e=%.9f v=%.9f n=%d", r.req,
                  r.seq, r.pos, r.rows, r.acc, o.cap, o.chosen, o.e, o.v, static_cast<int>(r.probs.size()));
    return buf;
}

std::string decisions_for(const std::vector<std::string>& trace_paths) {
    std::string      out;
    lgc::DraftPolicy policy;
    for (const std::string& path : trace_paths)
        for (const std::string& line : lines_at(path)) {
        Round r;
        if (!parse_round(line, r)) {
            t::fail(__FILE__, __LINE__, "unparsed fixture line: " + line);
            return out;
        }
        out += decision_line(r, run_round(policy, r)) + "\n";
    }
    out += "table " + (policy.cost_table_text().empty() ? std::string("(nothing measured)")
                                                        : policy.cost_table_text()) + "\n";
    return out;
}

std::string fixture_decisions() {
    return decisions_for({kSourceDir + "/tests/python/fixtures/mtp_trace_lines.txt",
                          kSourceDir + "/tests/python/fixtures/mtp_policy_trace.txt"});
}

std::string fixture_expected() {
    std::string out;
    for (const std::string& l : lines_at(kSourceDir + "/tests/python/fixtures/mtp_policy_decisions.txt")) out += l + "\n";
    return out;
}

}  // namespace

TEST(draft_policy_agrees_with_the_replay) {
    // red: see (a) and (b) in the header comment.
    const std::string got = fixture_decisions(), want = fixture_expected();
    CHECK_EQ(got, want);
}

TEST(draft_policy_constant_cost_reproduces_fixed_k) {
    // red: see (d). With every width costing the same, E/cost rises with every
    // draft the chain can make, so the chosen width is the cap - the fixed-K
    // arm - and a low probability keeps the chain at its first shorter width.
    lgc::DraftPolicy fixed;
    for (int b = 0; b < lgc::DraftPolicy::kCtxBuckets; ++b)
        fixed.seed_costs(std::vector<double>(lgc::DraftPolicy::kMaxT, 10.0).data(), lgc::DraftPolicy::kMaxT,
                         6.0, b);
    lgc::PolicyChain st;
    const int cap = 4;
    for (int idx = 0; idx < 2; ++idx) CHECK(fixed.keep_draft(st, idx, 0.9, cap));
    // the last draft is the chain's end: rows + 1 > cap, so no next draft
    CHECK(!fixed.keep_draft(st, 2, 0.9, cap));
    CHECK_EQ(st.best_rows, cap);   // the ceiling+1 rows the cap allows

    lgc::DraftPolicy low;
    for (int b = 0; b < lgc::DraftPolicy::kCtxBuckets; ++b)
        low.seed_costs(std::vector<double>(lgc::DraftPolicy::kMaxT, 10.0).data(), lgc::DraftPolicy::kMaxT, 6.0, b);
    lgc::PolicyChain slow;
    bool go = true;
    int idx = 0;
    while (go && idx < 3) go = low.keep_draft(slow, idx++, 0.2, cap);
    CHECK_EQ(slow.best_rows, 3);   // a(p) 0.2 does not pay for the fourth row
}

TEST(draft_policy_truncation_verifies_the_chosen_width) {
    // red: see (c). The hook truncates the drafted chain to best_rows - 1, so a
    // verify of best_rows rows never holds a draft the policy refused; a verify
    // of the whole chain would hold drafts past the chosen width.
    // the first draft's rate continues the chain (its look-ahead pays for row 3),
    // the second draft's near-zero rate does not beat the best width, so the chain
    // made two drafts and the policy verifies one: the hook's truncation drops
    // the draft past best_rows - 1.
    Round r; r.nmax = 3; r.pos = 0; r.probs = {0.9, 0.01};
    lgc::DraftPolicy policy;
    for (int b = 0; b < lgc::DraftPolicy::kCtxBuckets; ++b)
        policy.seed_costs(std::vector<double>(lgc::DraftPolicy::kMaxT, 10.0).data(), lgc::DraftPolicy::kMaxT,
                          6.0, b);
    const Outcome o = run_round(policy, r);
    CHECK_EQ(o.drafted, 2);    // the chain made both drafts
    CHECK_EQ(o.chosen, 2);     // the width the rule verifies
    CHECK_EQ(o.drafts, 1);     // rows - 1, not the chain
}

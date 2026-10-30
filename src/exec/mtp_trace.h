// The MTP window trace of a verify round: one line a round, off by default.
//
// Campaign research-mtp-width-policy, section 6 step 1: before any width
// policy is built into the serving path, a card run records what a round
// did, so `tools/mtp_width_replay.py` can replay Strata's cost-aware rule
// (Strata-ref:src/spec/draft_policy.cpp:199-224) over it offline.
//
// ARCINT_MTP_TRACE=<file> writes the lines; unset is OFF and costs one getenv
// a draft step plus a branch, with no probability computed beyond what
// --llama-mtp-min-p already asks for (llama_spec.cpp's DraftHead::pick). On,
// every draft step's head probability is computed even with min-p 0, because
// the trace's `p=` list IS the policy's input: one host pass over the head's
// logits a draft (`llama_spec.cpp:354-361`; the comment records a double `exp`
// over 248,320 logits at ~0.8 ms a draft, so a trace-on arm pays a per-row
// host cost the serving path does not - the replay states the bias, and the
// card window runs a trace-off arm with the same flags).
//
// One line a verify round, the fields in this fixed order (format `mtp2`):
//
//   mtp2 req=<request counter> seq=<lane index> pos=<context depth at the round's start>
//        rows=<rows verified, 1 + drafts> acc=<drafts the walk kept>
//        nmax=<the draft ceiling this round> minp=<the unit's min-p>
//        tp=<propose ms> tv=<verify ms> ms=<the round's own duration ms>
//        p=<the kept drafts' probabilities, then the floor's stopping draft>
//
// `req` counts the requests the process served (1, 2, ...), so a replay can
// split one process's history per request; `seq` is the lane's index, not a
// request id. `ms` is the round's own duration: the round's start to the next
// round's start, so a line carries the time of the window it names - the
// previous format (`mtp1`) carried the previous round's duration on this
// round's rows, which priced width_k with round_{k-1}'s time. The request's
// last round has no next start and measures start to the decode loop's exit.
// `ms` <= 0 means unknown (no cost), which the replay treats as such; Strata's
// own round time is start-to-end inside the round (Strata-ref:src/program/
// generate.cpp:16752-16767) and the research doc's "start-to-start" reading is
// answered by the replay printing (tp + tv) against `ms`.
//
// `p` holds the kept drafts first (rows - 1 of them) and then, when the
// min-p floor stopped the chain, the stopping draft's probability: its length
// is rows - 1 (chain ended at the ceiling) or rows (floor-stopped), which is
// how the replay tells the two apart without a flag of its own. A draft never
// made is absent, not 0 (Strata's own STRATA_SPEC_TRACE pads to the ceiling
// with 0.0, Strata-ref:src/program/generate.cpp:16768-16770 - its replay
// cannot tell "never drafted" from "probability 0", this one can).
//
// Device-free so the format is testable without a card and shared by both
// sides: tests/test_mtp_trace.cpp renders records and checks them against
// tests/python/fixtures/mtp_trace_lines.txt; tests/python/
// test_mtp_width_replay.py parses the same file with the replay's parser.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace lgc {

struct MtpTraceRound {
    unsigned long long req       = 0;     // the request counter, 1, 2, ...
    int                seq       = 0;     // the lane's index (not a request id)
    long long          pos       = 0;     // context depth at the round's start, tokens in the context
    int                rows      = 0;     // rows verified: 1 + drafts
    int                accepted  = 0;     // drafts the walk kept
    int                nmax      = 0;     // the draft ceiling this round (min(--llama-mtp, budget - 1))
    double             min_p     = 0.0;   // the unit's --llama-mtp-min-p
    std::vector<float> probs;             // kept drafts, then the floor's stopping draft
    double             t_propose_ms = 0.0;
    double             t_verify_ms  = 0.0;
    double             round_ms     = 0.0;   // this round's own duration, start to next start
};

// ARCINT_MTP_TRACE=<file>: the trace on. Read per call: the switch is an
// environment variable, and a test (or a long process) may set or clear it.
inline bool mtp_trace_on() {
    const char* f = std::getenv("ARCINT_MTP_TRACE");
    return f != nullptr && f[0] != '\0';
}

// The line without its newline, every value at the precision the parser reads
// back, so render(parse(line)) == line.
inline std::string mtp_trace_line(const MtpTraceRound& r) {
    char b[256];
    std::snprintf(b, sizeof b,
                  "mtp2 req=%llu seq=%d pos=%lld rows=%d acc=%d nmax=%d minp=%.6f tp=%.3f tv=%.3f ms=%.3f p=",
                  r.req, r.seq, r.pos, r.rows, r.accepted, r.nmax, r.min_p, r.t_propose_ms, r.t_verify_ms,
                  r.round_ms);
    std::string out(b);
    for (size_t i = 0; i < r.probs.size(); ++i) {
        std::snprintf(b, sizeof b, i == 0 ? "%.6f" : ",%.6f", static_cast<double>(r.probs[i]));
        out += b;
    }
    return out;
}

// One line a round to ARCINT_MTP_TRACE's file, appended and flushed (the
// writer buffers a record one round so `ms` is its own duration; a serving
// process appends across requests and the replay treats one file as one
// history). Off: nothing built, nothing opened. The file opens once a process:
// clearing or changing the variable later stops or does not redirect the
// writes, the open handle stays.
inline void mtp_trace_write(const MtpTraceRound& r) {
    if (!mtp_trace_on()) return;
    static std::mutex   mu;
    static std::FILE*   f   = nullptr;
    static bool         tried = false;
    std::lock_guard<std::mutex> lk(mu);
    if (f == nullptr && !tried) {
        tried = true;
        const char* name = std::getenv("ARCINT_MTP_TRACE");
        f = std::fopen(name, "a");
        if (f == nullptr) {
            std::fprintf(stderr, "arcint: cannot open ARCINT_MTP_TRACE=%s for the MTP trace\n", name);
            return;
        }
    }
    if (f == nullptr) return;
    const std::string line = mtp_trace_line(r);
    std::fprintf(f, "%s\n", line.c_str());
    std::fflush(f);
}

}  // namespace lgc

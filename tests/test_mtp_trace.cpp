// The MTP window trace's line format (src/exec/mtp_trace.h), device-free:
// the writer's own bytes are compared to the fixture the replay's parser is
// tested on, so one file pins both sides of the format.
//
// The fixture (tests/python/fixtures/mtp_trace_lines.txt) holds the shapes a
// round takes: the chain ending at the ceiling (`p` has rows - 1
// entries), the min-p floor stopping it (`p` has rows, the last the stopped
// draft's probability), and a round with no drafts at all (`rows=1`, `p=`
// empty). tests/python/test_mtp_width_replay.py parses the same file.
//
// Red mutations seen failing while this was written:
// (a) in mtp_trace_line, print `acc` before `rows` -> every line differs from
//     the fixture; (b) print the probabilities with %.3f -> the 0.812345 case
//     reads 0.812 and the fixture comparison fails; (c) drop the empty `p=`
//     for a zero-probability round -> the `rows=1` fixture line keeps a field
//     the writer no longer emits; (d) drop the `req=` field -> every fixture
//     line differs; (e) the off test without the unsetenv is vacuous when the
//     caller exported ARCINT_MTP_TRACE (seen failing with the variable set).
#include "exec/mtp_trace.h"
#include "harness.h"

#include <unistd.h>   // getpid, for the writer's own temp file

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> fixture_lines_at(const std::string& path) {
    std::ifstream f(path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) out.push_back(line);
    return out;
}

std::vector<std::string> fixture_lines() {
    return fixture_lines_at(std::string(ARCINT_SOURCE_DIR) + "/tests/python/fixtures/mtp_trace_lines.txt");
}

// the three shapes the fixture holds
std::vector<lgc::MtpTraceRound> fixture_rounds() {
    std::vector<lgc::MtpTraceRound> r(4);
    r[0].req = 1; r[0].seq = 7; r[0].pos = 4096; r[0].rows = 4; r[0].accepted = 3; r[0].nmax = 3; r[0].min_p = 0.5;
    r[0].t_propose_ms = 4.125; r[0].t_verify_ms = 36.5; r[0].round_ms = 41.875;
    r[0].probs = {0.812345f, 0.700001f, 0.659999f};
    r[1].req = 1; r[1].seq = 7; r[1].pos = 16384; r[1].rows = 4; r[1].accepted = 2; r[1].nmax = 5; r[1].min_p = 0.7;
    r[1].t_propose_ms = 3.75; r[1].t_verify_ms = 35.25; r[1].round_ms = 40.0;
    r[1].probs = {0.910000f, 0.840000f, 0.720000f, 0.650000f};
    r[2].req = 2; r[2].seq = 3; r[2].pos = 0; r[2].rows = 1; r[2].accepted = 0; r[2].nmax = 0; r[2].min_p = 0.0;
    r[2].t_propose_ms = 0.0; r[2].t_verify_ms = 0.0; r[2].round_ms = 0.125;
    r[3].req = 2; r[3].seq = 12; r[3].pos = 231422; r[3].rows = 8; r[3].accepted = 7; r[3].nmax = 7; r[3].min_p = 0.0;
    r[3].t_propose_ms = 6.1; r[3].t_verify_ms = 71.5; r[3].round_ms = 79.4;
    r[3].probs = {0.999001f, 0.998001f, 0.997001f, 0.996001f, 0.995001f, 0.994001f, 0.993001f};
    return r;
}

}  // namespace

TEST(mtp_trace_line_matches_the_shared_fixture) {
    const std::vector<std::string>        lines  = fixture_lines();
    const std::vector<lgc::MtpTraceRound> rounds = fixture_rounds();
    CHECK_EQ(lines.size(), rounds.size());
    for (size_t i = 0; i < lines.size() && i < rounds.size(); ++i) CHECK_EQ(lgc::mtp_trace_line(rounds[i]), lines[i]);
}

TEST(mtp_trace_line_one_line_and_its_field_order) {
    const lgc::MtpTraceRound r = fixture_rounds()[1];
    const std::string        l = lgc::mtp_trace_line(r);
    CHECK(l.find('\n') == std::string::npos);       // one record a line, the replay splits on \n
    CHECK_EQ(l.rfind("mtp2 ", 0), 0u);              // the replay rejects any other prefix
    // the fields in the order the replay's parser expects. Every key is
    // searched with its leading space: "p=" alone also occurs inside "minp=".
    const auto field = [&](const char* k) { return l.find(std::string(" ") + k + "="); };
    CHECK(field("req") < field("seq"));
    CHECK(field("seq") < field("pos"));
    CHECK(field("pos") < field("rows"));
    CHECK(field("rows") < field("acc"));
    CHECK(field("acc") < field("nmax"));
    CHECK(field("nmax") < field("minp"));
    CHECK(field("minp") < field("tp"));
    CHECK(field("tp") < field("tv"));
    CHECK(field("tv") < field("ms"));
    CHECK(field("ms") < field("p"));
}

TEST(mtp_trace_probs_tell_ceiling_end_from_floor_stop) {
    const auto rounds = fixture_rounds();
    // ceiling end: `p` holds the kept drafts only
    CHECK_EQ(rounds[0].probs.size(), static_cast<size_t>(rounds[0].rows - 1));
    // floor stop: `p` holds the kept drafts and the stopping draft
    CHECK_EQ(rounds[1].probs.size(), static_cast<size_t>(rounds[1].rows));
    CHECK(rounds[1].probs.back() < rounds[1].min_p);
    // no drafts: rows 1, an empty list, and the line still carries `p=`
    CHECK_EQ(rounds[2].rows, 1);
    CHECK(rounds[2].probs.empty());
    const std::string l = lgc::mtp_trace_line(rounds[2]);
    CHECK_EQ(l.substr(l.find(" p=") + 1), std::string("p="));
}

TEST(mtp_trace_off_by_default) {
    // the switch is the environment variable, read per call; clear it here so
    // the case is not vacuous when the caller exported ARCINT_MTP_TRACE
    unsetenv("ARCINT_MTP_TRACE");
    CHECK(!lgc::mtp_trace_on());
    lgc::mtp_trace_write(fixture_rounds()[0]);   // off: nothing built, nothing opened
}

TEST(mtp_trace_writes_on_appends_and_flushes) {
    // on: the file opens, one line a call, appended, flushed before the process
    // ends (the replay reads it while the serving process still runs)
    const std::string path = "/tmp/arcint-mtp-trace-" + std::to_string(getpid()) + ".txt";
    std::remove(path.c_str());
    setenv("ARCINT_MTP_TRACE", path.c_str(), 1);
    CHECK(lgc::mtp_trace_on());
    const lgc::MtpTraceRound r = fixture_rounds()[0];
    lgc::mtp_trace_write(r);
    lgc::mtp_trace_write(r);
    const std::vector<std::string> lines = fixture_lines_at(path);
    CHECK_EQ(lines.size(), 2u);
    if (lines.size() == 2) {   // after a failed CHECK, do not index past the vector
        CHECK_EQ(lines[0], lgc::mtp_trace_line(r));
        CHECK_EQ(lines[1], lgc::mtp_trace_line(r));
    }
    unsetenv("ARCINT_MTP_TRACE");
    CHECK(!lgc::mtp_trace_on());
    std::remove(path.c_str());
}

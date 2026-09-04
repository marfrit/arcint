#include "exec/backend.h"

#include <cstdlib>
#include <string>

#include "harness.h"

using lgc::format_profile_cycle_line;
using lgc::profile_cycle_enabled;

TEST(profile_cycle_off_by_default) {
    unsetenv("ARCINT_PROFILE_CYCLE");
    CHECK(!profile_cycle_enabled());
}

TEST(profile_cycle_on_when_set) {
    setenv("ARCINT_PROFILE_CYCLE", "1", 1);
    CHECK(profile_cycle_enabled());
    unsetenv("ARCINT_PROFILE_CYCLE");   // leave no trace for the next case
    CHECK(!profile_cycle_enabled());
}

TEST(profile_cycle_line_names_every_segment) {
    // Every argument gets its own distinguishable value so a swap anywhere
    // in the 17-argument signature (e.g. verify_infer_ms and
    // verify_hidden_ms trading places) fails a specific check below instead
    // of surviving because two segments happened to share a value.
    const std::string line = format_profile_cycle_line(
        /*past=*/76400, /*n=*/2, /*accepted=*/1, /*propose_ms=*/12.34,
        /*propose_embed_ms=*/0.10, /*propose_mask_ms=*/0.02, /*propose_layer_ms=*/1.50,
        /*propose_head_ms=*/0.30,
        /*verify_embed_ms=*/0.20, /*verify_index_ms=*/0.15, /*verify_infer_ms=*/430.0,
        /*verify_logits_ms=*/10.0, /*verify_hidden_ms=*/8.0,
        /*accept_ms=*/0.05, /*wait_ms=*/2.0, /*cycle_ms=*/455.0, /*frag=*/3);
    // Every segment M11 §1 named must be greppable in the one line an
    // operator reads off the log -- a field silently dropped from the
    // format string is a field nobody can attribute cost to again.
    for (const char* seg : {"past", "n", "accepted", "propose", "embed", "mask", "layer", "head",
                            "verify", "index", "infer", "logits", "hidden", "accept", "wait",
                            "total", "frag"}) {
        CHECK(line.find(seg) != std::string::npos);
    }
    // Value-anchored: catches an argument reordered against the format
    // string, not just a label that happens to still be present.
    CHECK(line.find("past 76400") != std::string::npos);
    CHECK(line.find("n 2 accepted 1") != std::string::npos);
    CHECK(line.find("propose 12.34") != std::string::npos);
    CHECK(line.find("embed 0.10") != std::string::npos);   // propose's own embed
    CHECK(line.find("mask 0.02") != std::string::npos);
    CHECK(line.find("layer 1.50") != std::string::npos);
    CHECK(line.find("head 0.30") != std::string::npos);
    CHECK(line.find("embed 0.20") != std::string::npos);   // verify's own embed
    CHECK(line.find("index 0.15") != std::string::npos);
    CHECK(line.find("infer 430.00") != std::string::npos);
    CHECK(line.find("logits 10.00") != std::string::npos);
    CHECK(line.find("hidden 8.00") != std::string::npos);
    CHECK(line.find("accept 0.05") != std::string::npos);
    CHECK(line.find("wait 2.00") != std::string::npos);
    CHECK(line.find("total 455.00") != std::string::npos);
    CHECK(line.find("frag 3") != std::string::npos);
}

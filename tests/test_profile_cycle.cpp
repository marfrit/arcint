#include "exec/backend.h"

#include <cstdlib>
#include <string>

#include "harness.h"

using lgc::draft_f32_enabled;
using lgc::draft_rope_f16_enabled;
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

// M11 §2 (DESIGN §7.0.2ag): ARCINT_DRAFT_F32 is presence-armed, the same
// idiom as ARCINT_PROFILE_CYCLE above -- the env VALUE is never read, only
// whether the variable is set. This is the pure half of the switch
// (draft_f32_enabled(), backend_stub.cpp); what backend_ov.cpp does with a
// `true` result (fold ov::hint::inference_precision(f32) into both drafter
// compiles' ov::AnyMap) needs an OpenVINO build and a card, and is not
// under test here -- see the design's own "test the helper that maps the
// env to the property set, not the compile."
TEST(draft_f32_off_by_default) {
    unsetenv("ARCINT_DRAFT_F32");
    CHECK(!draft_f32_enabled());
}

TEST(draft_f32_on_when_set) {
    setenv("ARCINT_DRAFT_F32", "1", 1);
    CHECK(draft_f32_enabled());
    unsetenv("ARCINT_DRAFT_F32");   // leave no trace for the next case
    CHECK(!draft_f32_enabled());
}

TEST(draft_f32_armed_regardless_of_value) {
    // Presence-armed: an empty value still counts as set (POSIX setenv with
    // an empty string still creates the variable, distinct from unset).
    setenv("ARCINT_DRAFT_F32", "", 1);
    CHECK(draft_f32_enabled());
    unsetenv("ARCINT_DRAFT_F32");
    CHECK(!draft_f32_enabled());
}

// M11 §2 RoPE follow-up (DESIGN §7.0.2ag): ARCINT_DRAFT_ROPE_F16 is also
// presence-armed, like every switch above -- but unlike them it is the
// lever back to the PRE-fix state (the RoPE marker is applied by default;
// setting this switch turns that back off for the A/B), which is why its
// own name names the state it restores rather than the fix itself.
TEST(draft_rope_f16_off_by_default) {
    unsetenv("ARCINT_DRAFT_ROPE_F16");
    CHECK(!draft_rope_f16_enabled());
}

TEST(draft_rope_f16_on_when_set) {
    setenv("ARCINT_DRAFT_ROPE_F16", "1", 1);
    CHECK(draft_rope_f16_enabled());
    unsetenv("ARCINT_DRAFT_ROPE_F16");
    CHECK(!draft_rope_f16_enabled());
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

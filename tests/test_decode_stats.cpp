#include "api/handlers.h"

#include "harness.h"

using lgc::GenerationStats;
using lgc::api::decode_other_seconds;

TEST(decode_other_seconds_zero_on_plain_decode) {
    // No drafting: the pre-M11 accounting already covered this case, and it
    // must keep doing so -- forward + embed + sample + emit + wait sum to
    // the whole decode wall.
    GenerationStats s;
    s.decode_forward_seconds = 0.40;
    s.decode_embed_seconds   = 0.05;
    s.decode_sample_seconds  = 0.02;
    s.decode_emit_seconds    = 0.01;
    s.decode_wait_seconds    = 0.00;
    s.decode_seconds = s.decode_forward_seconds + s.decode_embed_seconds +
                       s.decode_sample_seconds + s.decode_emit_seconds + s.decode_wait_seconds;
    CHECK_NEAR(decode_other_seconds(s), 0.0, 1e-9);
}

TEST(decode_other_seconds_zero_on_drafting_request) {
    // M11 (DESIGN §7.0.2aa row): a drafting cycle's own segments -- propose
    // (turnstile + embed + the drafter itself) and verify (embed + infer +
    // readback) -- are timed into draft_propose_seconds/draft_verify_seconds,
    // not decode_forward_seconds. Before the fix, `other` subtracted
    // neither, so it silently absorbed a real drafting request's whole
    // propose+verify cost -- this is the case that failed before the fix
    // landed (draft_verify_seconds was not subtracted).
    GenerationStats s;
    s.decode_embed_seconds   = 0.01;   // the plain-decode cycles this request also ran
    s.decode_forward_seconds = 0.05;
    s.decode_sample_seconds  = 0.02;
    s.decode_emit_seconds    = 0.03;
    s.decode_wait_seconds    = 0.00;
    s.draft_propose_seconds  = 0.11;
    s.draft_verify_seconds   = 0.44;
    s.decode_seconds = s.decode_embed_seconds + s.decode_forward_seconds +
                       s.decode_sample_seconds + s.decode_emit_seconds + s.decode_wait_seconds +
                       s.draft_propose_seconds + s.draft_verify_seconds;
    CHECK_NEAR(decode_other_seconds(s), 0.0, 1e-9);
}

TEST(decode_other_seconds_reports_the_untimed_remainder) {
    // A genuinely untimed segment (the accept loop and checkpoint-row
    // promotion, M11 §1 rows 12-15) must still show up as `other` -- the fix
    // subtracts the segments that now have their own timer, not everything.
    GenerationStats s;
    s.decode_forward_seconds = 0.10;
    s.draft_verify_seconds   = 0.20;
    s.draft_propose_seconds  = 0.05;
    s.decode_seconds         = 0.10 + 0.20 + 0.05 + 0.03;  // + 30 ms of untimed accept-loop work
    CHECK_NEAR(decode_other_seconds(s), 0.03, 1e-9);
}

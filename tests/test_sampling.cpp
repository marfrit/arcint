#include "core/sampling.h"
#include "harness.h"

using namespace lgc;

TEST(sampling_defaults_come_from_the_allowlist_entry) {
    SamplerDefaults d;
    d.temperature = 0.6f;
    d.top_p       = 0.95f;
    d.top_k       = 20;

    const SamplerParams p = sampler_from_defaults(d);
    CHECK_NEAR(p.temperature, 0.6, 1e-6);
    CHECK_NEAR(p.top_p, 0.95, 1e-6);
    CHECK_EQ(p.top_k, 20);
    CHECK(!p.seeded);
    CHECK_EQ(p.max_tokens, -1);
}

TEST(sampling_request_fields_win) {
    SamplerParams    p = sampler_from_defaults(SamplerDefaults{});
    SamplerOverrides o;
    o.temperature = 0.0f;
    o.max_tokens  = 16;
    o.seed        = 7u;

    CHECK(!sampler_apply(p, o).has_value());
    CHECK_NEAR(p.temperature, 0.0, 1e-6);
    CHECK_EQ(p.max_tokens, 16);
    CHECK(p.seeded);
    CHECK_EQ(static_cast<int>(p.seed), 7);
}

TEST(sampling_untouched_fields_keep_the_model_default) {
    SamplerDefaults d;
    d.top_k = 40;
    SamplerParams p = sampler_from_defaults(d);

    SamplerOverrides o;
    o.temperature = 1.0f;
    CHECK(!sampler_apply(p, o).has_value());
    CHECK_EQ(p.top_k, 40);
}

TEST(sampling_greedy_detection) {
    SamplerParams p;
    p.temperature = 0.0f;
    CHECK(p.greedy());

    p.temperature = 0.7f;
    CHECK(!p.greedy());

    // top_k 1 collapses the distribution just as temperature 0 does.
    p.top_k = 1;
    CHECK(p.greedy());
}

TEST(sampling_rejects_out_of_range) {
    SamplerParams base = sampler_from_defaults(SamplerDefaults{});

    auto rejects = [&](SamplerOverrides o) {
        SamplerParams p = base;
        return sampler_apply(p, o).has_value();
    };

    SamplerOverrides o;
    o.temperature = 2.5f;
    CHECK(rejects(o));

    o = {};
    o.top_p = 0.0f;
    CHECK(rejects(o));

    o = {};
    o.top_p = 1.5f;
    CHECK(rejects(o));

    o = {};
    o.top_k = -1;
    CHECK(rejects(o));

    o = {};
    o.max_tokens = 0;
    CHECK(rejects(o));

    o = {};
    o.repetition_penalty = 0.0f;
    CHECK(rejects(o));
}

TEST(sampling_rejection_leaves_params_untouched) {
    SamplerParams p = sampler_from_defaults(SamplerDefaults{});
    const float   before = p.temperature;

    SamplerOverrides o;
    o.temperature = 0.1f;   // valid
    o.max_tokens  = -5;     // invalid

    CHECK(sampler_apply(p, o).has_value());
    CHECK_NEAR(p.temperature, before, 1e-6);
    CHECK_EQ(p.max_tokens, -1);
}

TEST(sampling_overrides_any) {
    SamplerOverrides o;
    CHECK(!o.any());
    o.top_k = 5;
    CHECK(o.any());
}

TEST(sampling_operator_layer_overrides_only_what_it_sets) {
    SamplerDefaults d;  // family-card values, provenance "provisional"
    d.temperature = 0.7f;
    d.top_p       = 0.8f;
    d.top_k       = 20;
    d.provenance  = "artifact";

    SamplerOverrides o;
    o.temperature = 0.0f;
    CHECK(!sampler_defaults_apply(d, o));
    CHECK_NEAR(d.temperature, 0.0, 1e-6);
    CHECK_NEAR(d.top_p, 0.8, 1e-6);  // untouched
    CHECK_EQ(d.top_k, 20);           // untouched
    CHECK_EQ(d.provenance, std::string("operator"));
}

TEST(sampling_operator_layer_without_flags_changes_nothing) {
    SamplerDefaults d;
    d.provenance = "artifact";
    SamplerOverrides o;  // no flags set
    CHECK(!sampler_defaults_apply(d, o));
    CHECK_EQ(d.provenance, std::string("artifact"));
}

TEST(sampling_operator_layer_validates_like_a_request) {
    SamplerDefaults  d;
    SamplerOverrides o;
    o.temperature = 3.0f;  // out of [0, 2], same rule a request gets
    CHECK(sampler_defaults_apply(d, o).has_value());
}

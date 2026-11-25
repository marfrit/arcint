#include "core/admission.h"
#include "core/chat.h"
#include "core/toolcall.h"
#include "harness.h"

#include <string>

using namespace lgc;

// The admission tiers (operator 2026-11-23): the measured sizes keep their identity and tier,
// an unmeasured size of a family and every architecture not in the table are "runs". The pure
// decision and the "runs" rules, with no model.
namespace {
ModelGeometry gemma4_e4b() { return {42, 2560, 8, 2, 512, 262144}; }
ModelGeometry granite_8b() { return {40, 4096, 32, 8, 0, 100352}; }
ModelGeometry mistral24b() { return {40, 5120, 32, 8, 128, 131072}; }
ModelGeometry llama2_7b() { return {32, 4096, 32, 32, 0, 32000}; }
// Dory1.0-4B: the PHYSICAL block count (72), not the 112 or 192 libllama executes at 2 or 4 loops
ModelGeometry dory_4b() { return {72, 2560, 16, 4, 256, 131072}; }
// Inkling 2 Mini: the PHYSICAL block count (24), not the cache slots its three loop passes execute
ModelGeometry inkling2_mini() { return {24, 2304, 24, 8, 128, 200058}; }
// the measured qwen sizes (`code`: files' metadata)
ModelGeometry qwen35_agent() { return {64, 5120, 24, 4, 256, 248320}; }
// as libllama reports the file: 41 blocks minus the MTP layer
ModelGeometry qwen35moe_coder() { return {40, 2048, 16, 2, 256, 248320}; }
ModelGeometry qwen4exp_flash() { return {48, 2560, 24, 2, 256, 248320}; }
// as the served file reads: no key_length key (head_dim 0)
ModelGeometry qwen2vl_7b() { return {28, 3584, 28, 4, 0, 152064}; }

Admission a(std::string_view arch, const ModelGeometry& g, bool tts = false) { return admit_model(arch, g, tts); }
}  // namespace

TEST(admission_optimized_for_identities) {
    CHECK(a("inkling2_loop", inkling2_mini()).model == AdmittedModel::Inkling2Mini);
    CHECK(a("inkling2_loop", inkling2_mini()).tier == AdmissionTier::Optimized);
    CHECK(a("qwen35", qwen35_agent()).model == AdmittedModel::Qwen35);
    CHECK(a("qwen35", qwen35_agent()).tier == AdmissionTier::Optimized);
    CHECK(a("qwen4exp", qwen4exp_flash()).model == AdmittedModel::Qwen4Exp);
    CHECK(a("qwen4exp", qwen4exp_flash()).tier == AdmissionTier::Optimized);
    CHECK(std::string(admission_tier_name(AdmissionTier::Optimized)) == "optimized for");
}

TEST(admission_verified_identities) {
    for (auto [arch, g] : {std::pair<std::string_view, ModelGeometry>{"dory", dory_4b()},
                           {"llama", llama2_7b()},
                           {"qwen35moe", qwen35moe_coder()},
                           {"qwen2vl", qwen2vl_7b()}}) {
        const Admission d = a(arch, g);
        CHECK(d.tier == AdmissionTier::Verified);
        CHECK(std::string(admission_tier_name(AdmissionTier::Verified)) == "verified");
    }
    CHECK(a("qwen35moe", qwen35moe_coder()).model == AdmittedModel::Qwen35Moe);
    CHECK(a("qwen2vl", qwen2vl_7b()).model == AdmittedModel::Qwen2Vl);
    CHECK(a("qwen3tts", ModelGeometry{}, true).tier == AdmissionTier::Verified);
    CHECK(a("qwen3tts", ModelGeometry{}, true).model == AdmittedModel::Qwen3Tts);
    // no --tts: the audio backbone is refused, not loaded as a language model
    CHECK(!tts_backbone_refusal("qwen3tts", false).empty());
    CHECK(tts_backbone_refusal("qwen3tts", true).empty());
    CHECK(tts_backbone_refusal("qwen35", false).empty());
}

TEST(admission_the_measured_only_promises_are_runs) {
    CHECK(a("gemma4", gemma4_e4b()).model == AdmittedModel::Gemma4E4B);
    CHECK(a("gemma4", gemma4_e4b()).tier == AdmissionTier::Runs);
    CHECK(a("granite", granite_8b()).model == AdmittedModel::Granite8B);
    CHECK(a("granite", granite_8b()).tier == AdmissionTier::Runs);
    CHECK(a("llama", mistral24b()).model == AdmittedModel::MistralSmall24B);
    CHECK(a("llama", mistral24b()).tier == AdmissionTier::Runs);
    CHECK(std::string(admission_tier_name(AdmissionTier::Runs)) == "runs");
}

TEST(admission_every_unmeasured_qwen_size_is_runs_but_keeps_its_identity) {
    ModelGeometry g = qwen35_agent();
    g.n_layer = 32;
    CHECK(a("qwen35", g).model == AdmittedModel::Qwen35);
    CHECK(a("qwen35", g).tier == AdmissionTier::Runs);
    g = qwen35moe_coder();
    g.n_layer = 32;   // an unmeasured size (the coder reports 40)
    CHECK(a("qwen35moe", g).model == AdmittedModel::Qwen35Moe);
    CHECK(a("qwen35moe", g).tier == AdmissionTier::Runs);
    g = qwen4exp_flash();
    g.n_embd = 2048;
    CHECK(a("qwen4exp", g).model == AdmittedModel::Qwen4Exp);
    CHECK(a("qwen4exp", g).tier == AdmissionTier::Runs);
    g = qwen2vl_7b();
    g.n_layer = 24;   // a smaller Qwen2-VL
    CHECK(a("qwen2vl", g).model == AdmittedModel::Qwen2Vl);
    CHECK(a("qwen2vl", g).tier == AdmissionTier::Runs);
}

TEST(admission_a_near_miss_keeps_the_family_and_is_runs_not_refused) {
    // the hard refusal is gone: a near miss of a measured family is "runs", still identified
    for (int field = 0; field < 6; ++field) {
        ModelGeometry g = gemma4_e4b();
        switch (field) {
            case 0: g.n_layer = 35; break;
            case 1: g.n_embd = 1536; break;
            case 2: g.n_head = 16; break;
            case 3: g.n_head_kv = 4; break;
            case 4: g.head_dim = 256; break;
            default: g.n_vocab = 256000; break;
        }
        const Admission d = a("gemma4", g);
        CHECK(d.model == AdmittedModel::None);   // another Gemma 4 size has no identity
        CHECK(d.tier == AdmissionTier::Runs);
    }
    ModelGeometry wrong = mistral24b();
    wrong.head_dim = 0;
    CHECK(a("llama", wrong).model == AdmittedModel::None);
    CHECK(a("llama", wrong).tier == AdmissionTier::Runs);
    for (int executed : {72 + 40, 72 + 120, 72 + 80}) {
        ModelGeometry g = dory_4b();
        g.n_layer = executed;   // llama_model_n_layer's executed count, not the physical blocks
        CHECK(a("dory", g).model == AdmittedModel::None);
        CHECK(a("dory", g).tier == AdmissionTier::Runs);
    }
    CHECK(a("inkling2", inkling2_mini()).tier == AdmissionTier::Runs);   // the loop-less arch
}

TEST(admission_any_unknown_architecture_is_runs) {
    const ModelGeometry none{};
    for (const char* arch : {"mamba", "qwen3", "gemma3", "granitehybrid", "granitemoe", "whisper"}) {
        const Admission d = a(arch, none);
        CHECK(d.model == AdmittedModel::None);
        CHECK(d.tier == AdmissionTier::Runs);
    }
}

TEST(admission_dialects_and_the_runs_neutral_default) {
    CHECK(dialect_of(AdmittedModel::Gemma4E4B) == ChatDialect::Gemma4);
    CHECK(dialect_of(AdmittedModel::MistralSmall24B) == ChatDialect::Extended);
    CHECK(dialect_of(AdmittedModel::Granite8B) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Dory1_0_4B) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Dory1_0_4B).thinks_by_default());
    CHECK(!dialect_of(AdmittedModel::Dory1_0_4B).replays_arguments_as_object());
    CHECK(dialect_of(AdmittedModel::Qwen35) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen35Moe) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen4Exp) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen2Vl) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Llama2_7B) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen3Tts) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::None) == ChatDialect::Neutral);
    CHECK(dialect_of(AdmittedModel::None).neutral());
}

TEST(runs_the_neutral_dialect_becomes_qwen_only_from_the_template) {
    const std::string qwen_open = std::string(tool_call_open(ChatDialect(ChatDialect::Qwen)));
    const std::string plain     = "a template with no call form";
    CHECK(dialect_from_template(qwen_open, ChatDialect(ChatDialect::Neutral)) == ChatDialect::Qwen);
    CHECK(dialect_from_template(plain, ChatDialect(ChatDialect::Neutral)) == ChatDialect::Neutral);   // red: a guess would make it Qwen
    // a family that is Qwen by identity keeps it even without the form in this template
    CHECK(dialect_from_template(plain, ChatDialect(ChatDialect::Qwen)) == ChatDialect::Qwen);
}

TEST(runs_builds_no_renderer_material_and_no_token_mode) {
    CHECK(!runs_builds_renderer(AdmissionTier::Runs));
    CHECK(runs_builds_renderer(AdmissionTier::Verified));    // red: a "runs" build would advertise Qwen parser rows
    CHECK(runs_builds_renderer(AdmissionTier::Optimized));
}

TEST(runs_parses_no_tool_calls_and_refuses_the_grammar) {
    const std::string text = "an answer that writes the Qwen call form";
    const ToolCallParse p = parse_tool_calls(text, ChatDialect(ChatDialect::Neutral));
    CHECK(p.calls.empty());
    CHECK(p.content == text);   // no tag is eaten, unlike the Qwen parser (the red case)
    CHECK(!runs_may_use_tool_grammar(AdmissionTier::Runs));
    CHECK(runs_may_use_tool_grammar(AdmissionTier::Verified));
    CHECK(runs_may_use_tool_grammar(AdmissionTier::Optimized));
}

TEST(runs_a_chat_route_without_a_template_names_completions) {
    const std::string no = runs_no_chat_template_refusal();
    CHECK(no.find("no chat template") != std::string::npos);
    CHECK(no.find("/v1/completions") != std::string::npos);
    CHECK(no.find("ChatML") == std::string::npos);   // red: a guess would never be named as served
}

TEST(runs_the_start_warning_counts_the_downgrades_and_splits) {
    const std::string w = runs_start_warning(AdmissionTier::Runs, 3, 17, 4, 2);
    CHECK(w.find("3 downgrade key") != std::string::npos);
    CHECK(w.find("17 placement") != std::string::npos);
    CHECK(w.find("4 prompt / 2 token") != std::string::npos);
    CHECK(w.find("token mode off") != std::string::npos);
    CHECK(w.find("--tool-grammar refused") != std::string::npos);
    CHECK(runs_start_warning(AdmissionTier::Verified, 0, 0, 0, 0).empty());   // red: a promise tier warns nothing
}

TEST(checkpoints_the_flag_is_refused_where_nothing_can_be_snapshotted) {
    CHECK(!checkpoints_refusal(true, false, "gemma4").empty());
    CHECK(checkpoints_refusal(true, false, "gemma4").find("gemma4") != std::string::npos);
    CHECK(checkpoints_refusal(true, true, "qwen35moe").empty());
}

TEST(checkpoints_the_default_is_not_a_request) {
    CHECK(checkpoints_refusal(false, false, "llama").empty());
    CHECK(checkpoints_refusal(false, true, "qwen35").empty());
}

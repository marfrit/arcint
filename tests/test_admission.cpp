#include "core/admission.h"
#include "harness.h"

#include <string>

using namespace lgc;

// Admission by architecture AND geometry (DESIGN §2): the pure decision, with
// no model. Each admitted geometry has near misses that must be refused.
namespace {

ModelGeometry gemma4_e4b() { return {42, 2560, 8, 2, 512, 262144}; }
ModelGeometry granite_8b() { return {40, 4096, 32, 8, 0, 100352}; }
ModelGeometry mistral24b() { return {40, 5120, 32, 8, 128, 131072}; }
ModelGeometry llama2_7b() { return {32, 4096, 32, 32, 0, 32000}; }
// Dory1.0-4B: the PHYSICAL block count (72), not the 112 or 192 libllama executes at 2 or 4 loops
ModelGeometry dory_4b() { return {72, 2560, 16, 4, 256, 131072}; }

}  // namespace

TEST(admission_gemma4_e4b_at_its_geometry) {
    CHECK(admit_model("gemma4", gemma4_e4b(), false) == AdmittedModel::Gemma4E4B);
}

TEST(admission_gemma4_refuses_every_near_miss) {
    for (int field = 0; field < 6; ++field) {
        ModelGeometry g = gemma4_e4b();
        switch (field) {
            case 0: g.n_layer = 35; break;      // another size of the family
            case 1: g.n_embd = 1536; break;
            case 2: g.n_head = 16; break;
            case 3: g.n_head_kv = 4; break;
            case 4: g.head_dim = 256; break;
            default: g.n_vocab = 256000; break;
        }
        CHECK(admit_model("gemma4", g, false) == AdmittedModel::None);
    }
}

TEST(admission_granite_8b_at_its_geometry) {
    CHECK(admit_model("granite", granite_8b(), false) == AdmittedModel::Granite8B);
}

TEST(admission_granite_refuses_every_near_miss) {
    for (int field = 0; field < 5; ++field) {
        ModelGeometry g = granite_8b();
        switch (field) {
            case 0: g.n_layer = 32; break;      // right architecture, wrong layer count
            case 1: g.n_embd = 2048; break;
            case 2: g.n_head = 16; break;
            case 3: g.n_head_kv = 4; break;
            default: g.n_vocab = 49152; break;
        }
        CHECK(admit_model("granite", g, false) == AdmittedModel::None);
    }
}

TEST(admission_dory_4b_at_its_physical_geometry) {
    CHECK(admit_model("dory", dory_4b(), false) == AdmittedModel::Dory1_0_4B);
}

TEST(admission_dory_refuses_every_near_miss_and_the_executed_layer_counts) {
    for (int field = 0; field < 6; ++field) {
        ModelGeometry g = dory_4b();
        switch (field) {
            case 0: g.n_layer = 36; break;
            case 1: g.n_embd = 2048; break;
            case 2: g.n_head = 8; break;
            case 3: g.n_head_kv = 8; break;
            case 4: g.head_dim = 128; break;
            default: g.n_vocab = 151936; break;
        }
        CHECK(admit_model("dory", g, false) == AdmittedModel::None);
    }
    // llama_model_n_layer reports the executed count: 16 + 40 * loops + 16. Admission matches the
    // physical blocks, so a caller that passed the executed count would refuse the file at 2 or 4 loops.
    for (int executed : {72 + 40, 72 + 120, 72 + 80}) {
        ModelGeometry g = dory_4b();
        g.n_layer = executed;
        CHECK(admit_model("dory", g, false) == AdmittedModel::None);
    }
}

TEST(admission_a_geometry_is_not_admitted_under_another_architecture) {
    CHECK(admit_model("llama", dory_4b(), false) == AdmittedModel::None);
    CHECK(admit_model("qwen3", dory_4b(), false) == AdmittedModel::None);
    CHECK(admit_model("dory", gemma4_e4b(), false) == AdmittedModel::None);
    CHECK(admit_model("gemma4", granite_8b(), false) == AdmittedModel::None);
    CHECK(admit_model("granite", gemma4_e4b(), false) == AdmittedModel::None);
    CHECK(admit_model("llama", gemma4_e4b(), false) == AdmittedModel::None);
    CHECK(admit_model("llama", granite_8b(), false) == AdmittedModel::None);
    CHECK(admit_model("gemma3", gemma4_e4b(), false) == AdmittedModel::None);
    CHECK(admit_model("granitehybrid", granite_8b(), false) == AdmittedModel::None);
    CHECK(admit_model("granitemoe", granite_8b(), false) == AdmittedModel::None);
}

TEST(admission_the_earlier_geometries_are_unchanged) {
    CHECK(admit_model("llama", mistral24b(), false) == AdmittedModel::MistralSmall24B);
    CHECK(admit_model("llama", llama2_7b(), false) == AdmittedModel::Llama2_7B);
    ModelGeometry wrong = mistral24b();
    wrong.n_layer = 32;
    CHECK(admit_model("llama", wrong, false) == AdmittedModel::None);
    wrong = mistral24b();
    wrong.head_dim = 0;   // the key length absent: refused, as the string compare was
    CHECK(admit_model("llama", wrong, false) == AdmittedModel::None);
    wrong = llama2_7b();
    wrong.n_head_kv = 8;
    CHECK(admit_model("llama", wrong, false) == AdmittedModel::None);
    const ModelGeometry none{};
    CHECK(admit_model("qwen35", none, false) == AdmittedModel::Qwen35);
    CHECK(admit_model("qwen35moe", none, false) == AdmittedModel::Qwen35Moe);
    CHECK(admit_model("qwen4exp", none, false) == AdmittedModel::Qwen4Exp);
    CHECK(admit_model("qwen2vl", none, false) == AdmittedModel::Qwen2Vl);
    CHECK(admit_model("mamba", none, false) == AdmittedModel::None);
}

TEST(admission_qwen3tts_only_under_tts) {
    const ModelGeometry none{};
    CHECK(admit_model("qwen3tts", none, true) == AdmittedModel::Qwen3Tts);
    CHECK(admit_model("qwen3tts", none, false) == AdmittedModel::None);
    CHECK(admit_model("gemma4", gemma4_e4b(), true) == AdmittedModel::Gemma4E4B);
}

TEST(admission_dialects) {
    CHECK(dialect_of(AdmittedModel::Gemma4E4B) == ChatDialect::Gemma4);
    CHECK(dialect_of(AdmittedModel::MistralSmall24B) == ChatDialect::Extended);
    // every other admitted model keeps the Qwen dialect: no behaviour change
    CHECK(dialect_of(AdmittedModel::Granite8B) == ChatDialect::Qwen);
    // Dory1.0-4B: its template is ChatML with the plain think pair (the generation prompt ends inside
    // the open block when thinking is on) and the Qwen coder's function/parameter call form
    CHECK(dialect_of(AdmittedModel::Dory1_0_4B) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Dory1_0_4B).thinks_by_default());
    CHECK(!dialect_of(AdmittedModel::Dory1_0_4B).replays_arguments_as_object());   // the probe decides, as for the Qwens
    CHECK(dialect_of(AdmittedModel::Qwen35) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen35Moe) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen4Exp) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen2Vl) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Llama2_7B) == ChatDialect::Qwen);
    CHECK(dialect_of(AdmittedModel::Qwen3Tts) == ChatDialect::Qwen);
}

TEST(admission_refusal_names_the_admitted_sizes) {
    const std::string msg = admission_refusal("/m/x.gguf", "granite");
    CHECK(msg.find("/m/x.gguf") != std::string::npos);
    CHECK(msg.find("'granite'") != std::string::npos);
    CHECK(msg.find("not admitted") != std::string::npos);
    CHECK(msg.find("Gemma 4 E4B") != std::string::npos);
    CHECK(msg.find("42 x 2560") != std::string::npos);
    CHECK(msg.find("Granite 4.2 8B") != std::string::npos);
    CHECK(msg.find("40 x 4096") != std::string::npos);
    CHECK(msg.find("Mistral Small") != std::string::npos);
    CHECK(msg.find("Llama 2 7B") != std::string::npos);
    CHECK(msg.find("Dory1.0-4B") != std::string::npos);
    CHECK(msg.find("72 blocks") != std::string::npos);
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

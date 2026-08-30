#include "core/model_registry.h"
#include "harness.h"

using namespace lgc;

TEST(registry_holds_exactly_the_target_models) {
    // Three checkpoints, four artifacts: Intel's public export of the 3.8 is its
    // own entry with its own status, never an alias of our AWQ export.
    const auto ids = model_ids();
    CHECK_EQ(ids.size(), 5u);
    CHECK(find_model("qwen3.6-27b-a3b-coder") != nullptr);
    CHECK(find_model("qwen3.6-35b-a3b") != nullptr);
    CHECK(find_model("qwen3.8-27b") != nullptr);
    CHECK(find_model("qwen3.8-27b-intel-int4") != nullptr);
    CHECK(find_model("qwen3.6-35b-a3b-mtp") != nullptr);
}

TEST(registry_rejects_everything_else) {
    CHECK(find_model("llama-3-8b") == nullptr);
    CHECK(find_model("") == nullptr);
    CHECK(find_model("QWEN3.8-27B") == nullptr);  // ids are exact, not fuzzy
}

TEST(registry_matches_the_measured_ir_metadata) {
    // Transcribed from models/allowlist-raw.json (/models/ov/, 2026-08-28).
    // Note this contradicts the console sketch in DESIGN.md §4 ("41 GDN + 7
    // attn"): the artifacts say 40 layers with one in four full attention.
    const ModelEntry* coder = find_model("qwen3.6-27b-a3b-coder");
    CHECK(coder->moe);
    CHECK(!coder->has_mtp_head);  // the 3.6 pair ships none (§3.5)
    CHECK_EQ(coder->n_layer, 40);
    CHECK_EQ(coder->full_attention_interval, 4);
    CHECK_EQ(coder->n_attn_layer, 10);
    CHECK_EQ(coder->n_gdn_layer, 30);
    CHECK_EQ(coder->n_ctx_train, 262144);
    CHECK_EQ(coder->n_expert, 184);  // pruned from 256
    CHECK_EQ(coder->arch_hash, std::string("6745cfe3d57e3f0f"));
    CHECK_EQ(coder->template_hash, std::string("e84f32a23fdda276"));

    const ModelEntry* big = find_model("qwen3.6-35b-a3b");
    CHECK_EQ(big->n_expert, 256);
    CHECK_EQ(big->n_layer, 40);

    const ModelEntry* dense = find_model("qwen3.8-27b");
    CHECK(!dense->moe);
    // §3.5 claims a native MTP head here. The export does not contain one, and
    // the export is what can be served — see registry_mtp_reflects_the_export.
    // The dense export carries a head again once tools/export_mtp.py has run.
    CHECK(dense->has_mtp_head);
    CHECK(dense->mtp_in_checkpoint);
    CHECK_EQ(dense->n_layer, 64);
    CHECK_EQ(dense->n_attn_layer, 16);
    CHECK_EQ(dense->n_gdn_layer, 48);
    CHECK_EQ(dense->n_embd, 5120);
}

TEST(registry_layer_split_always_sums) {
    for (const ModelEntry& e : registry()) {
        CHECK_EQ(e.n_gdn_layer + e.n_attn_layer, e.n_layer);
        CHECK(e.n_attn_layer > 0);
        CHECK(e.n_gdn_layer > e.n_attn_layer);  // "most layers" carry GDN state (§2)
    }
}

TEST(registry_all_three_share_one_tokenizer) {
    // models/allowlist-raw.json: every artifact reports tokenizer 87a7830d63fcf43b.
    for (const ModelEntry& e : registry()) {
        CHECK_EQ(e.tokenizer_hash, std::string("87a7830d63fcf43b"));
    }
}

TEST(registry_maps_artifact_directory_names) {
    CHECK(find_by_artifact("qwen36-coder-b5-ov") == find_model("qwen3.6-27b-a3b-coder"));
    CHECK(find_by_artifact("qwen38-b7c1-ov") == find_model("qwen3.8-27b"));
    CHECK(find_by_artifact("qwen38-intel-int4-ov") == find_model("qwen3.8-27b-intel-int4"));
    CHECK(find_by_artifact("qwen38-intel-int4-ov") != find_by_artifact("qwen38-b7c1-ov"));
    CHECK(find_by_artifact("qwen36-35b-a3b-mtp-ov") == find_model("qwen3.6-35b-a3b-mtp"));
    CHECK(find_by_artifact("some-random-export") == nullptr);
}

TEST(registry_quants) {
    const ModelEntry* e = find_model("qwen3.8-27b");
    CHECK(e->accepts(Quant::Q4));
    CHECK(e->accepts(Quant::Q8));

    CHECK_EQ(std::string(quant_name(Quant::Q4)), std::string("q4"));
    CHECK(quant_parse("q8").has_value());
    CHECK(quant_parse("int4").has_value());
    CHECK(!quant_parse("fp16").has_value());
}

TEST(registry_hashes_are_pinned_but_sampler_defaults_are_not) {
    for (const ModelEntry& e : registry()) {
        // Pinned from a real IR read.
        CHECK(e.hashes_pinned());
        CHECK_EQ(e.arch_hash.size(), 16u);
        CHECK_EQ(e.template_hash.size(), 16u);
        CHECK(e.weights_bytes > 0);
        CHECK(!e.status.empty());

        // Not pinned: allowlist-raw.json carries no sampler settings, so these
        // are still inherited from the family card and must say so.
        CHECK_EQ(e.sampler.provenance, std::string("provisional"));
    }
}

namespace {
ArtifactInfo good_coder_artifact() {
    ArtifactInfo a;
    a.id             = "qwen3.6-27b-a3b-coder";
    a.quant          = Quant::Q4;
    a.n_layer        = 40;
    a.n_gdn_layer    = 30;
    a.n_attn_layer   = 10;
    a.n_ctx_train    = 262144;
    a.arch_hash      = "6745cfe3d57e3f0f";
    a.template_hash  = "e84f32a23fdda276";
    a.tokenizer_hash = "87a7830d63fcf43b";
    a.has_mtp_head   = false;
    return a;
}
}  // namespace

TEST(registry_validation_accepts_a_matching_artifact) {
    const ModelEntry*      e   = find_model("qwen3.6-27b-a3b-coder");
    const ValidationResult res = validate_artifact(*e, good_coder_artifact());

    CHECK(res.ok);
    CHECK(res.errors.empty());
    // Everything the artifact reports is pinned now, so nothing is merely
    // recorded.
    CHECK_EQ(res.warnings.size(), 0u);
}

TEST(registry_validation_rejects_a_wrong_template_hash) {
    // Template drift between exporter and server is a measured source of silent
    // quality loss (§3.7); it must be a hard refusal, not a warning.
    const ModelEntry* e = find_model("qwen3.6-27b-a3b-coder");
    ArtifactInfo      a = good_coder_artifact();
    a.template_hash     = "0000000000000000";

    const ValidationResult res = validate_artifact(*e, a);
    CHECK(!res.ok);
}

TEST(registry_validation_rejects_a_missing_hash) {
    const ModelEntry* e = find_model("qwen3.6-27b-a3b-coder");
    ArtifactInfo      a = good_coder_artifact();
    a.arch_hash.clear();

    CHECK(!validate_artifact(*e, a).ok);
}

TEST(registry_validation_rejects_a_layer_mismatch) {
    const ModelEntry* e = find_model("qwen3.6-27b-a3b-coder");
    ArtifactInfo      a = good_coder_artifact();
    a.n_gdn_layer       = 29;

    const ValidationResult res = validate_artifact(*e, a);
    CHECK(!res.ok);
    CHECK(!res.errors.empty());
}

TEST(registry_mtp_reflects_the_export_not_the_checkpoint) {
    // Every checkpoint declares mtp_num_hidden_layers 1, but that is a fact
    // about the checkpoint. What can be served is a fact about the export, and
    // the two must be tracked separately: the dense export carries a head
    // (reconstructed by tools/export_mtp.py), the MoE pair still does not.
    for (const ModelEntry& e : registry()) {
        CHECK(e.mtp_head_pinned);
        CHECK(e.mtp_in_checkpoint);
    }
    CHECK(find_model("qwen3.8-27b")->has_mtp_head);
    CHECK(!find_model("qwen3.6-27b-a3b-coder")->has_mtp_head);
    CHECK(!find_model("qwen3.6-35b-a3b")->has_mtp_head);
}

namespace {
// A dense-3.8 artifact that matches its entry in every respect, so a test can
// flip exactly one field and know that field caused the refusal.
ArtifactInfo good_dense_artifact() {
    const ModelEntry* e = find_model("qwen3.8-27b");
    ArtifactInfo      a;
    a.id             = e->id;
    a.quant          = Quant::Q4;
    a.n_layer        = e->n_layer;
    a.n_gdn_layer    = e->n_gdn_layer;
    a.n_attn_layer   = e->n_attn_layer;
    a.n_ctx_train    = e->n_ctx_train;
    a.arch_hash      = e->arch_hash;
    a.template_hash  = e->template_hash;
    a.tokenizer_hash = e->tokenizer_hash;
    a.has_mtp_head   = e->has_mtp_head;
    return a;
}
}  // namespace

TEST(registry_a_matching_dense_artifact_is_accepted) {
    // The control: without this, the refusal test below could pass for any
    // reason at all.
    const ModelEntry* e = find_model("qwen3.8-27b");
    CHECK(validate_artifact(*e, good_dense_artifact()).ok);
}

TEST(registry_pinned_mtp_mismatch_is_a_refusal) {
    const ModelEntry* e = find_model("qwen3.8-27b");
    ArtifactInfo      a = good_dense_artifact();
    a.has_mtp_head      = false;  // an export that suddenly lacks one

    const ValidationResult res = validate_artifact(*e, a);
    CHECK(!res.ok);
    CHECK_EQ(res.errors.size(), 1u);  // and only for the MTP flag
}

TEST(registry_validation_rejects_a_foreign_id) {
    const ModelEntry* e = find_model("qwen3.8-27b");
    CHECK(!validate_artifact(*e, good_coder_artifact()).ok);
}

#include "core/model_registry.h"
#include "harness.h"

using namespace lgc;

TEST(registry_holds_exactly_the_target_models) {
    // Six entries now: the four 3.6/3.8 artifacts, Intel's public export of the
    // 3.8 (its own entry with its own status, never an alias of our AWQ export),
    // and the dense qwen3.5-2b marker artifact (FIX 8.2, hashes pinned off the
    // dev-host IR 2026-09-10).
    // Seven since 2026-09-13: the Flash-Next serving-shape IR at depth 4, the
    // served path's first Flash-Next artifact (hashes pinned off the dev-host
    // directory the same day).
    // Eight with the full-depth (48-layer) serving-shape artifact, built the
    // same day; the two Flash-Next entries differ in depth and hashes only.
    // Nine with the 12-layer rung (0.5.1 WP3, the depth ladder's first
    // measured step between 4 and 48).
    // Ten with the segmented 48-layer chain (0.5.1 B.1) -- admitted by its
    // CHAIN hash, because it has no single language-model file to hash.
    // Eleven with the fused-MoE rewrite of the 12-layer rung (2026-09-17,
    // campaign sub4bit-vram-kernel): the same rung in the shape the GPU
    // plugin's tiled MoE matcher accepts, pinned by its own xml hash.
    // Twelve with the full-depth artifact in that shape: first d48f (the
    // same day; it served noise -- its fill was wrong, DESIGN 7.0.2bz), then
    // d48g in its slot (2026-09-18, the re-export through the corrected
    // fill, which serves the Paris line), pinned by its own xml hash.
    const auto ids = model_ids();
    CHECK_EQ(ids.size(), 14u);   // d48n beside d48g (2026-09-18), and the native qwen3_5_moe rung (2026-09-25)
    CHECK(find_model("qwen3.8-flash-next-d48g") != nullptr);
    CHECK(find_by_artifact("qwen38-flash-next-d48g-ov") == find_model("qwen3.8-flash-next-d48g"));
    CHECK(find_model("qwen3.8-flash-next-d48n") != nullptr);
    CHECK(find_by_artifact("qwen38-flash-next-d48n-ov") == find_model("qwen3.8-flash-next-d48n"));
    CHECK_EQ(find_model("qwen3.8-flash-next-d48n")->arch_hash, std::string("641fcb1863f83629"));
    CHECK(find_model("qwen3.8-flash-next-d48f") == nullptr);   // superseded, not admitted
    CHECK(find_model("qwen3.8-flash-next-d12r") != nullptr);
    CHECK(find_by_artifact("qwen38-flash-next-d12r-ov") == find_model("qwen3.8-flash-next-d12r"));
    CHECK(find_model("qwen3.8-flash-next-d12") != nullptr);
    CHECK(find_model("qwen3.8-flash-next-seg12") != nullptr);
    CHECK(find_model("qwen3.8-flash-next") != nullptr);
    CHECK(find_model("qwen3.6-35b-a3b-native-d4") != nullptr);
    CHECK(find_model("qwen3.6-27b-a3b-coder") != nullptr);
    CHECK(find_model("qwen3.6-35b-a3b") != nullptr);
    CHECK(find_model("qwen3.8-27b") != nullptr);
    CHECK(find_model("qwen3.8-27b-intel-int4") != nullptr);
    CHECK(find_model("qwen3.6-35b-a3b-mtp") != nullptr);
    CHECK(find_model("qwen3.5-2b") != nullptr);
    CHECK(find_model("qwen3.8-flash-next-d4") != nullptr);
}

// FULL-DEPTH (2026-09-13): the depth-4 serving-shape artifact is a MoE
// qwen4exp entry with one attention layer in four, the GGUF's own chat
// template and the passthrough tokenizer every other entry shares.
TEST(registry_flash_next_d4_is_the_serving_shape_at_depth_4) {
    const ModelEntry* e = find_model("qwen3.8-flash-next-d4");
    CHECK(e != nullptr);
    if (e == nullptr) return;
    CHECK(find_by_artifact("qwen38-flash-next-d4-ov") == e);
    CHECK(e->moe);
    CHECK_EQ(e->model_type, std::string("qwen4_exp"));
    CHECK_EQ(e->n_expert, 512);
    CHECK_EQ(e->n_embd, 2560);
    CHECK_EQ(e->n_layer, 4);
    CHECK_EQ(e->n_attn_layer, 1);
    CHECK_EQ(e->n_gdn_layer, 3);
    CHECK(!e->has_mtp_head);
    CHECK_EQ(e->template_hash, std::string("12827f24b742ea4e"));
    CHECK_EQ(e->arch_hash, std::string("2910a860bf9dc6bb"));

    const ModelEntry* full = find_model("qwen3.8-flash-next");
    CHECK(full != nullptr);
    if (full == nullptr) return;
    CHECK(find_by_artifact("qwen38-flash-next-ov") == full);
    CHECK(find_by_artifact("qwen38-flash-next-ov") != e);
    CHECK_EQ(full->n_layer, 48);
    CHECK_EQ(full->n_attn_layer, 12);
    CHECK_EQ(full->n_gdn_layer, 36);
    CHECK(full->arch_hash != e->arch_hash);
    CHECK_EQ(full->template_hash, e->template_hash);
    CHECK_EQ(full->tokenizer_hash, e->tokenizer_hash);

    // THE 12-LAYER RUNG (0.5.1 WP3): 3 attention + 9 GDN, its own XML hash,
    // the same template and tokenizer as the other two rungs.
    const ModelEntry* d12 = find_model("qwen3.8-flash-next-d12");
    CHECK(d12 != nullptr);
    if (d12 == nullptr) return;
    CHECK(find_by_artifact("qwen38-flash-next-d12-ov") == d12);
    CHECK_EQ(d12->n_layer, 12);
    CHECK_EQ(d12->n_attn_layer, 3);
    CHECK_EQ(d12->n_gdn_layer, 9);
    CHECK_EQ(d12->model_type, std::string("qwen4_exp"));
    CHECK(d12->arch_hash != e->arch_hash && d12->arch_hash != full->arch_hash);
    CHECK_EQ(d12->arch_hash, std::string("7738fa87cddca8e2"));
    CHECK_EQ(d12->template_hash, e->template_hash);
    CHECK_EQ(d12->tokenizer_hash, e->tokenizer_hash);
}

// THE SEGMENTED ARTIFACT (0.5.1 B.1): 48 layers over four 12-layer compiled
// models. It has no single language-model file, so its pin is the CHAIN hash
// (segplan::chain_arch_hash over the four segment xml digests, in segment
// order) -- read off the directory with `arcint --model <dir>
// --inspect-artifact`. The bent-field refusal below is why the accepting line
// above measures anything: without it, an empty check list would "pass" too.
TEST(registry_the_segmented_entry_is_admitted_by_its_chain_hash_not_a_file_digest) {
    const ModelEntry* e = find_model("qwen3.8-flash-next-seg12");
    CHECK(e != nullptr);
    if (e == nullptr) return;
    CHECK(find_by_artifact("qwen38-flash-next-seg12-ov") == e);
    CHECK(e->moe);
    CHECK_EQ(e->n_layer, 48);
    CHECK_EQ(e->n_attn_layer, 12);
    CHECK_EQ(e->n_gdn_layer, 36);
    CHECK_EQ(e->arch_hash, std::string("32d3060ca30238d1"));
    // Not the 12-layer rung's file digest, and not segment 0's own.
    CHECK(e->arch_hash != std::string("7738fa87cddca8e2"));
    CHECK_EQ(e->weights_bytes, 21953654588ull);  // the SUM over four segment bins

    ArtifactInfo a;
    a.id             = e->id;
    a.quant          = Quant::Q4;
    a.n_ctx_train    = 262144;
    a.n_layer        = 48;
    a.n_gdn_layer    = 36;
    a.n_attn_layer   = 12;
    a.arch_hash      = e->arch_hash;
    a.template_hash  = e->template_hash;
    a.tokenizer_hash = e->tokenizer_hash;
    a.weights_bytes  = e->weights_bytes;
    a.has_mtp_head   = false;
    CHECK(validate_artifact(*e, a).ok);

    ArtifactInfo bent   = a;
    bent.arch_hash      = "da8dfc2704891698";  // segment 0's OWN xml digest
    const ValidationResult res = validate_artifact(*e, bent);
    CHECK(!res.ok);
    CHECK(!res.errors.empty());
}

TEST(registry_the_native_qwen35moe_rung_is_admitted_without_a_ple) {
    // 2026-09-25: the qwen3_5_moe serving-shape rung at depth 4, the first
    // admitted artifact of this family that is NOT an HF export. It carries the
    // checkpoint's own IQ2_S/IQ3_XXS experts and NO PLE / n-gram table, which
    // is why its n-gram binding is inert and --ngram-gguf is not needed.
    const ModelEntry* e = find_model("qwen3.6-35b-a3b-native-d4");
    CHECK(e != nullptr);
    if (e == nullptr) return;
    CHECK(find_by_artifact("qwen36-35b-a3b-d4n-ov") == e);
    CHECK(find_by_artifact("qwen36-35b-a3b-d4n-ov") !=
          find_by_artifact("qwen36-35b-a3b-int4-ov"));
    CHECK(e->moe);
    CHECK_EQ(e->model_type, std::string("qwen3_5_moe"));
    CHECK_EQ(e->ov_arch, std::string("Qwen3_5MoeForConditionalGeneration"));
    CHECK_EQ(e->n_expert, 256);
    CHECK_EQ(e->n_embd, 2048);
    CHECK_EQ(e->n_layer, 4);  // of 40
    CHECK_EQ(e->n_attn_layer, 1);
    CHECK_EQ(e->n_gdn_layer, 3);
    CHECK(!e->has_mtp_head);
    CHECK_EQ(e->arch_hash, std::string("391bd21db6368d57"));
    CHECK_EQ(e->template_hash, std::string("55d4931433fe502b"));
    CHECK_EQ(e->weights_bytes, 4284499713ull);

    // The accepting control: an artifact matching the entry in every respect.
    ArtifactInfo a;
    a.id             = e->id;
    a.quant          = Quant::Q4;
    a.n_ctx_train    = e->n_ctx_train;
    a.n_layer        = e->n_layer;
    a.n_gdn_layer    = e->n_gdn_layer;
    a.n_attn_layer   = e->n_attn_layer;
    a.arch_hash      = e->arch_hash;
    a.template_hash  = e->template_hash;
    a.tokenizer_hash = e->tokenizer_hash;
    a.weights_bytes  = e->weights_bytes;
    a.has_mtp_head   = false;
    CHECK(validate_artifact(*e, a).ok);

    // RED FIRST: each pinned field, flipped alone, refuses. The weights_bytes
    // cell is the one this leg added -- before it, the allowlist pinned the
    // number and nothing read it, so a re-exported .bin passed on its xml hash.
    ArtifactInfo bad_arch = a;
    bad_arch.arch_hash    = "0000000000000000";
    CHECK(!validate_artifact(*e, bad_arch).ok);

    ArtifactInfo bad_bytes = a;
    bad_bytes.weights_bytes = e->weights_bytes + 1;
    CHECK(!validate_artifact(*e, bad_bytes).ok);

    ArtifactInfo missing_bytes = a;
    missing_bytes.weights_bytes = 0;  // an artifact that reports nothing
    CHECK(!validate_artifact(*e, missing_bytes).ok);

    ArtifactInfo bad_layers = a;  // the full-depth geometry, under a depth-4 pin
    bad_layers.n_layer      = 40;
    bad_layers.n_gdn_layer  = 30;
    bad_layers.n_attn_layer = 10;
    CHECK(!validate_artifact(*e, bad_layers).ok);
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
    a.weights_bytes  = 13760293946ull;
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
    a.weights_bytes  = e->weights_bytes;
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

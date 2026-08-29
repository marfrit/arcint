#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include "core/model_registry.h"
#include "harness.h"

// The compiled allowlist is transcribed by hand from models/allowlist-raw.json.
// Transcription drifts — that is exactly how "41 GDN + 7 attn" ended up quoted
// as fact. This test pins the two together so the raw IR read stays the single
// source of the numbers.
namespace {

using nlohmann::json;

json load_raw() {
    const std::string path = std::string(ARCINT_SOURCE_DIR) + "/models/allowlist-raw.json";
    std::ifstream     in(path);
    if (!in.good()) return json();
    json raw;
    in >> raw;
    return raw;
}

}  // namespace

using namespace lgc;

TEST(provenance_raw_metadata_is_readable) {
    const json raw = load_raw();
    CHECK(raw.is_object());
    CHECK(!raw.empty());
}

TEST(provenance_every_entry_matches_its_artifact) {
    const json raw = load_raw();
    CHECK(raw.is_object());

    for (const ModelEntry& e : registry()) {
        CHECK(!e.artifact_aliases.empty());
        if (e.artifact_aliases.empty()) continue;

        const std::string& alias = e.artifact_aliases.front();
        CHECK(raw.contains(alias));
        if (!raw.contains(alias)) continue;

        const json& m = raw.at(alias);

        CHECK_EQ(e.model_type, m.at("model_type").get<std::string>());
        CHECK_EQ(e.ov_arch, m.at("arch").at(0).get<std::string>());
        CHECK_EQ(e.n_layer, m.at("layers").get<int>());
        CHECK_EQ(e.n_embd, m.at("hidden").get<int>());
        CHECK_EQ(e.n_ctx_train, m.at("ctx").get<int>());
        CHECK_EQ(e.full_attention_interval, m.at("full_attention_interval").get<int>());
        CHECK_EQ(e.arch_hash, m.at("lm_xml_sha").get<std::string>());
        CHECK_EQ(e.template_hash, m.at("template_sha").get<std::string>());
        CHECK_EQ(e.tokenizer_hash, m.at("tokenizer_sha").get<std::string>());
        CHECK_EQ(e.weights_bytes, m.at("lm_bin_bytes").get<uint64_t>());

        // The split is derived, not transcribed: one layer in `interval` is
        // full attention, the rest carry GDN state.
        const int interval = m.at("full_attention_interval").get<int>();
        const int layers   = m.at("layers").get<int>();
        CHECK_EQ(e.n_attn_layer, layers / interval);
        CHECK_EQ(e.n_gdn_layer, layers - layers / interval);

        if (m.contains("mtp_head_exported")) {
            CHECK_EQ(e.has_mtp_head, m.at("mtp_head_exported").get<bool>());
            CHECK(e.mtp_head_pinned);
        }
        if (m.contains("mtp_num_hidden_layers")) {
            CHECK_EQ(e.mtp_in_checkpoint, m.at("mtp_num_hidden_layers").get<int>() > 0);
        }

        const json& experts = m.at("experts");
        if (experts.is_null()) {
            CHECK(!e.moe);
            CHECK_EQ(e.n_expert, 0);
        } else {
            CHECK(e.moe);
            CHECK_EQ(e.n_expert, experts.get<int>());
        }
    }
}

TEST(provenance_no_artifact_is_left_unmapped) {
    // An IR sitting in the raw metadata with no allowlist entry would be a
    // model the engine silently cannot serve.
    const json raw = load_raw();
    CHECK(raw.is_object());

    for (const auto& [key, value] : raw.items()) {
        if (!key.empty() && key.front() == '_') continue;  // "_comment"
        CHECK(find_by_artifact(key) != nullptr);
    }
}

TEST(provenance_calibration_lesson_is_carried_into_the_allowlist) {
    // DESIGN.md §3.1: the SE-calibration finding is part of the contract, not
    // folklore. If the raw metadata records it, the entry must too.
    const json raw = load_raw();
    CHECK(raw.is_object());

    for (const auto& [key, value] : raw.items()) {
        if (!key.empty() && key.front() == '_') continue;
        const ModelEntry* e = find_by_artifact(key);
        CHECK(e != nullptr);
        if (e == nullptr) continue;
        CHECK(!e->status.empty());

        const std::string raw_status = value.at("status").get<std::string>();
        if (raw_status.find("do not re-add SE") != std::string::npos) {
            CHECK(e->status.find("do not re-add SE") != std::string::npos);
        }
    }
}

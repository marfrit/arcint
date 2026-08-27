#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// The allowlist (DESIGN.md §3.1). A checkpoint outside this table is rejected
// at load time — that is the whole point of the "no model zoo" non-goal.
//
// The pinned numbers below are transcribed from models/allowlist-raw.json,
// which was read off the actual IR directories on dirac:/models/ov/ on
// 2026-08-28, and a test holds the two together. Where the artifacts and the
// prose disagree the artifacts win: the console sketch in DESIGN.md §4 quotes
// "41 GDN + 7 attn layers", and the IRs say 40 layers with one in four full
// attention.
//
// One field is the exception and is marked as such: `has_mtp_head` comes from
// DESIGN.md §3.5, not from any artifact, because the raw metadata records no
// MTP field at all. It is therefore carried with `mtp_head_pinned = false`,
// which downgrades a mismatch from a load refusal to a warning. Refusing to
// load a real artifact over an unsourced claim would be the tail wagging the
// dog.
//
// Provenance is part of the contract: the campaign showed scale-estimation
// calibration degenerating greedy decoding on the dense 3.8 (0/10) where
// AWQ-only stayed healthy (7/10), so which artifact produced an IR is not the
// user's problem to remember.
namespace lgc {

enum class Quant { Q4, Q8 };

const char*          quant_name(Quant q);
std::optional<Quant> quant_parse(std::string_view s);

struct SamplerDefaults {
    float temperature        = 0.7f;
    float top_p              = 0.8f;
    int   top_k              = 20;
    float repetition_penalty = 1.05f;
    float presence_penalty   = 0.0f;

    // "artifact"    — read from the IR's generation_config.json (M1+)
    // "provisional" — the Qwen3-family card values, carried until an artifact
    //                 is on disk to read. models/allowlist-raw.json carries no
    //                 sampler settings, so all three entries are still
    //                 provisional. /props reports this so nobody mistakes an
    //                 inherited default for a measured one.
    std::string provenance = "provisional";
};

struct ModelEntry {
    std::string id;
    std::string family;

    // Artifact directory names under /models/ov/ that map onto this entry.
    std::vector<std::string> artifact_aliases;

    // Structure, as reported by the IR.
    std::string ov_arch;     // e.g. "Qwen3_5MoeForConditionalGeneration"
    std::string model_type;  // e.g. "qwen3_5_moe"
    bool moe = false;

    // From DESIGN.md §3.5, not from the IR — see the header comment. Until an
    // artifact confirms it, `mtp_head_pinned` stays false and validation only
    // warns on a mismatch.
    bool has_mtp_head    = false;
    bool mtp_head_pinned = false;
    int         n_embd       = 0;
    int         n_expert     = 0;  // 0 for the dense model

    // One layer in `full_attention_interval` is full attention; the rest carry
    // GDN recurrent state. n_attn_layer and n_gdn_layer are derived from it and
    // stored so validation compares concrete numbers.
    int full_attention_interval = 0;
    int n_layer                 = 0;
    int n_gdn_layer             = 0;
    int n_attn_layer            = 0;
    int n_ctx_train             = 0;

    std::vector<Quant> quants;

    // sha256 prefixes, 16 hex chars, as recorded in models/allowlist-raw.json.
    // An empty hash means "not pinned": the backend's observed value is
    // recorded and warned about rather than enforced.
    std::string arch_hash;       // lm_xml_sha
    std::string template_hash;   // chat template — stops exporter/server drift (§3.7)
    std::string tokenizer_hash;

    uint64_t weights_bytes = 0;

    // What the Prüfstand actually says about this artifact, verbatim enough to
    // be useful in a bug report.
    std::string status;

    SamplerDefaults sampler;

    bool accepts(Quant q) const;
    bool matches_alias(std::string_view artifact_dir) const;
    bool layers_pinned() const { return n_layer > 0; }
    bool hashes_pinned() const { return !arch_hash.empty() && !template_hash.empty(); }
};

// What the backend actually read out of an IR directory.
struct ArtifactInfo {
    std::string id;
    Quant       quant        = Quant::Q4;
    int         n_ctx_train  = 0;
    int         n_layer      = 0;
    int         n_gdn_layer  = 0;
    int         n_attn_layer = 0;
    std::string arch_hash;
    std::string template_hash;
    std::string tokenizer_hash;
    bool        has_mtp_head = false;
};

struct ValidationResult {
    bool                     ok = false;
    std::vector<std::string> errors;    // load is refused
    std::vector<std::string> warnings;  // unpinned fields, recorded not enforced
};

const std::vector<ModelEntry>& registry();
const ModelEntry*              find_model(std::string_view id);
const ModelEntry*              find_by_artifact(std::string_view artifact_dir);
std::vector<std::string>       model_ids();

ValidationResult validate_artifact(const ModelEntry& entry, const ArtifactInfo& seen);

}  // namespace lgc

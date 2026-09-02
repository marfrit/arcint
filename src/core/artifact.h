#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/model_registry.h"

// Reading an OpenVINO IR directory (DESIGN.md §3.1). Everything the engine
// needs to know about an artifact comes from the artifact: geometry from
// config.json, sampler defaults from generation_config.json, the chat template
// and the tokenizer from their own files. The allowlist's job is to say whether
// what was read is allowed, not to supply it.
namespace lgc {

// An IR file that sits in the artifact directory but is never read by the
// loader -- the vision tower and projector of a `*ForConditionalGeneration`
// export (M13, docs/milestone-0.3.0.md). Reported so the load-time log can
// say what disk space (not VRAM: it is never compiled) is going unused for
// the modality v1 does not serve, without pretending vision is loaded.
struct UnloadedIr {
    std::string name;         // basename, e.g. "openvino_vision_embeddings_merger_model.bin"
    uint64_t    bytes = 0;
};

struct Artifact {
    std::string dir;
    std::string directory_name;  // basename, matched against the allowlist aliases
    std::string id;              // resolved allowlist id

    nlohmann::json config;      // config.json
    nlohmann::json generation;  // generation_config.json (may be null)

    std::string chat_template;  // chat_template.jinja, verbatim
    std::string bos_token;
    std::string eos_token;
    std::vector<int> eos_ids;

    // Paths, resolved and checked for existence.
    std::string language_model_xml;
    std::string language_model_bin;
    std::string text_embeddings_xml;
    std::string tokenizer_xml;
    std::string detokenizer_xml;

    // Present on disk (stat'd, not opened) but never loaded: a VLM export's
    // vision tower and projector (M13). Empty for a text-only export.
    std::vector<UnloadedIr> unloaded_vision_irs;

    // Geometry, from config.json (text_config when the export is a VLM).
    std::string model_type;
    std::string ov_arch;
    int n_layer                 = 0;
    int n_gdn_layer             = 0;
    int n_attn_layer            = 0;
    int full_attention_interval = 0;
    int n_embd                  = 0;
    int n_ctx_train             = 0;
    int n_expert                = 0;
    bool moe                    = false;
    std::vector<std::string> layer_types;

    // sha256 prefixes, in the allowlist's pinned form.
    std::string arch_hash;       // openvino_language_model.xml
    std::string template_hash;   // chat_template.jinja
    std::string tokenizer_hash;  // tokenizer.json

    uint64_t weights_bytes = 0;

    // Read from generation_config.json, so provenance is "artifact" — this is
    // what retires the "provisional" family-card values (§3.6).
    SamplerDefaults sampler;

    bool has_mtp_head = false;
    // Written by tools/export_mtp.py beside the model; empty when absent.
    std::string mtp_layer_xml;           // the reconstructed layer (tools/export_mtp.py)
    std::string mtp_exported_layer_xml;  // optimum-intel's own export of the layer, when present
    std::string mtp_lm_head_xml;

    ArtifactInfo to_info(Quant quant) const;
};

// Returns an error message on failure. The directory basename decides which
// allowlist entry the artifact claims to be; a name outside the allowlist is
// refused here rather than after a two-minute compile.
std::optional<std::string> load_artifact(const std::string& dir, Artifact& out);

}  // namespace lgc

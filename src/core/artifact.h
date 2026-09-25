#pragma once

#include <cstdint>
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

// SEGMENTED (0.5.1, window-051 §2 / src/exec/segment_plan.h): one compiled
// sub-model of a segmented artifact, as `serving-shape.json`'s `segments[]`
// row describes it, plus its own resolved, existence-checked file paths.
// A non-segmented artifact (no `serving-shape.json`, or one whose
// `segment_layers` is null) still gets exactly one of these -- index 0,
// dir ".", the same paths `language_model_xml`/`_bin` already carry -- so
// callers can always iterate `Artifact::segments` rather than branching on
// whether the artifact is segmented.
struct ArtifactSegment {
    int index = 0;
    std::string dir;  // "." for a non-segmented artifact, "segmentK" otherwise
    std::string language_model_xml;
    std::string language_model_bin;
    int  layer_lo = 0, layer_hi = 0;
    bool first = false;
    bool last  = false;
    int  inputs_embeds_width = 0;
    bool has_ple    = false;
    int  attn_layers = 0;
    int  gdn_layers  = 0;
    uint64_t    lm_bin_bytes = 0;
    std::string xml_sha;  // full 64-hex sha256 of language_model_xml
};

// One expert body of a segmented artifact's `expert_bodies.u8` blob, as
// `serving-shape.json`'s `expert_bodies.entries[]` row describes it.
struct ExpertBodyEntry {
    std::string name;  // "layerI/moe/experts_{gate,up,down}/weight_u8", I global
    int segment = 0;
    int layer   = 0;
    std::string kind;  // "gate" | "up" | "down"
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t bytes  = 0;
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

    // SEGMENTED (0.5.1): always >= 1 after a successful load_artifact. A
    // non-segmented artifact (no serving-shape.json, or segment_layers ==
    // null) gets exactly one entry, index 0, dir ".", mirroring
    // language_model_xml/_bin above -- today's behaviour unchanged.
    std::vector<ArtifactSegment> segments;
    // Empty when the artifact carries no expert_bodies.u8 blob (a
    // non-segmented artifact, or a segmented one with no MoE layers).
    std::string expert_bodies_path;
    uint64_t    expert_bodies_bytes = 0;
    std::vector<ExpertBodyEntry> expert_bodies;
    // The emitter's declared expert format from serving-shape.json's
    // `expert_fill.format` ("native" for the checkpoint's own blocks, "u4"
    // for the grouped-affine repack); empty when the manifest omits it (an
    // HF export, or an artifact with no serving-shape.json). The all-resident
    // native pool at --offload-ratio 0 is only meaningful for "native": an
    // affine model stays on the direct resident Constants.
    std::string expert_format;

    // True iff serving-shape.json declared segment_layers != null, i.e. the
    // artifact was produced (and must be driven) as a chain of segments,
    // even a chain of one. `segments.size()` alone does not carry this: a
    // depth shorter than one segment_layers step still produces exactly one
    // segment row, and its arch_hash is still the chain form (segplan::
    // chain_arch_hash), never the plain single-file sha256.
    bool from_segmented_manifest = false;
    bool segmented() const { return segments.size() > 1 || from_segmented_manifest; }

    // Geometry, from config.json (text_config when the export is a VLM).
    std::string model_type;
    std::string ov_arch;
    int n_layer                 = 0;
    int n_gdn_layer             = 0;
    int n_attn_layer            = 0;
    int full_attention_interval = 0;
    int n_embd                  = 0;
    // The hyper-connection width term: the hidden state carried between two
    // segments is hc_count * hidden_size wide (window-051 §2), and
    // segplan::plan_segments refuses a segment whose inputs_embeds_width
    // disagrees with it. Zero when the checkpoint declares no hyper-connection
    // (every non-qwen4exp export).
    int hc_count                = 0;
    int n_ctx_train             = 0;
    int n_expert                = 0;
    bool moe                    = false;
    std::vector<std::string> layer_types;

    // The artifact's own serving-shape.json, verbatim (null when the directory
    // carries none). The segment rows are transcribed into `segments` above for
    // the file contract; this is what segplan::plan_segments reads for the
    // graph contract, so the loader stays the only reader of the file.
    nlohmann::json serving_shape;

    // sha256 prefixes, in the allowlist's pinned form. `arch_hash` is
    // segment 0's xml sha256 for a non-segmented artifact; for a segmented
    // one it is segplan::chain_arch_hash over every segment's xml sha256,
    // in segment order.
    std::string arch_hash;       // openvino_language_model.xml
    std::string template_hash;   // chat_template.jinja
    std::string tokenizer_hash;  // tokenizer.json

    // Sum of every segment's language_model_bin size (one term for a
    // non-segmented artifact).
    uint64_t weights_bytes = 0;

    // Read from generation_config.json, so provenance is "artifact" — this is
    // what retires the "provisional" family-card values (§3.6).
    SamplerDefaults sampler;

    bool has_mtp_head = false;
    // Written by tools/export_mtp.py beside the model; empty when absent.
    std::string mtp_layer_xml;           // the reconstructed layer (tools/export_mtp.py)
    std::string mtp_exported_layer_xml;  // optimum-intel's own export of the layer, when present
    std::string mtp_lm_head_xml;

    // FIX D Link 2 (docs/design-qwen-flash-next.md §"Link 1"/"Links 2 and 3"):
    // Flash-Next declares an n-gram embedding table in its config via
    // `ngram_size`, `ngram_vocab_size_base`, `heads_per_ngram`,
    // `ple_embed_dim` and `ple_layer_ids`. `load_artifact` reads them into
    // the struct below if they appear in `text_config`. Zero fields
    // mean "the checkpoint does not declare an n-gram table" -- the
    // dense qwen35 checkpoints on the current allowlist land there, and
    // no admission path fires. When any of the below is non-zero, the
    // artifact carries an n-gram component, and load-time refusal or
    // admission is `admit_ngram_table_from_disk`'s job.
    struct NGramConfig {
        int  ngram_size            = 0;   // 0 = no n-gram declared
        int  ngram_vocab_size_base = 0;
        int  heads_per_ngram       = 0;
        int  ple_embed_dim         = 0;
        std::vector<int> ple_layer_ids;
        // Needed by FIX D Link 3's dummy-weight hash-constant derivation
        // (derive_hash_constants, exec/ngram_row_ids.h): vocab_size bounds the
        // multipliers, eos_token_id (the config's ngram_boundary_token_id)
        // bounds every hash window. Both read from text_config; zero when the
        // checkpoint declares no n-gram table.
        int  vocab_size            = 0;
        int  ngram_boundary_token_id = -1;   // -1 = no eos_token_id found in the config chain
    };
    NGramConfig ngram_config;

    ArtifactInfo to_info(Quant quant) const;
};

// FIX D Link 2 (docs/design-qwen-flash-next.md §"Links 2 and 3"): the
// on-disk `--flash-next-ngram` admission. Reads the 24-byte ARCINGRM
// header from `path`, cross-checks the header's `(ggml_type, n_cols,
// n_rows)` against `artifact.ngram_config`, and reports the
// `payload_bytes` the loader must add to a host_ram_fit refusal.
//
// Returns:
//   - empty string + non-zero `out_payload_bytes`: admission passed,
//     shape and size are consistent.
//   - non-empty error string: the file was refused, `out_payload_bytes`
//     is zero, and the error names the exact failure (bad magic, wrong
//     type, shape mismatch against the config, host-RAM refusal).
//
// The function does not open the language-model IR or the checkpoint;
// its inputs are Artifact's config fields (already parsed by
// load_artifact) and the file path. This keeps admission testable
// without a full artifact directory on disk.
std::string admit_ngram_table_from_disk(const Artifact& artifact,
                                        const std::string& path,
                                        uint64_t host_ram_bytes,
                                        uint64_t other_resident_bytes,
                                        uint64_t margin_bytes,
                                        uint64_t& out_payload_bytes);

// Returns an error message on failure. The directory basename decides which
// allowlist entry the artifact claims to be; a name outside the allowlist is
// refused here rather than after a two-minute compile.
//
// `require_allowlisted` false is --inspect-artifact's reading: report what the
// directory IS (geometry, segments, hashes, blob) even when no allowlist entry
// claims that name, because reading a new artifact's contract is exactly the
// work that happens before its pin exists. `id` stays empty then, and nothing
// about the file's own validation changes.
std::optional<std::string> load_artifact(const std::string& dir, Artifact& out,
                                        bool require_allowlisted = true);

// Whether today's single-graph backend can SERVE this artifact at all.
//
// A segmented artifact (window-051 §2) is loadable, hashable and inspectable,
// and it is NOT servable until a segmented forward exists: the single-graph
// path reads `language_model_xml`, which for a chain resolves to segment 0's
// file -- so the served binary would compile ONE segment (its own layer range,
// say layers 0..11 of 48), answer tokens from it, and do so under an allowlist
// entry that says 48 layers. That is a wrong answer with a valid pin on it, so
// it refuses by name instead.
//
// Returns the refusal, naming the artifact and what to run instead; empty when
// the artifact is servable as it stands.
std::string serve_refusal_for(const Artifact& artifact);

}  // namespace lgc

#include "core/artifact.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "core/ngram_header.h"
#include "exec/fit.h"
#include "exec/segment_plan.h"
#include "exec/ngram_row_ids.h"
#include "util/log.h"
#include "util/sha256.h"

namespace lgc {
namespace {

using json = nlohmann::json;

bool file_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

uint64_t file_size(const std::string& p) {
    struct stat st{};
    if (::stat(p.c_str(), &st) != 0) return 0;
    return static_cast<uint64_t>(st.st_size);
}

std::string read_file(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The vision tower and projector of a `*ForConditionalGeneration` export
// (M13): every checkpoint arcint serves ships these, but the loader below
// resolves only the language model and text embeddings, and backend_ov.cpp
// compiles only those two -- so these three are always present-and-unread on
// the artifacts this repository actually loads. Each name is checked with
// both extensions independently: a checkpoint can ship the graph (.xml)
// without its weights (.bin) mid-export, and the inventory should say exactly
// what is on disk, not assume a pair. The base names are spelled out from an
// artifact listing, not abbreviated: a first version dropped the "embeddings_"
// infix from the pos/merger names and the load-time line reported "2 files,
// 1.7 MiB" against a 457 MB merger on disk -- the runtime line caught what a
// test mirroring the constant could not.
constexpr std::array<const char*, 3> kUnloadedVisionIrBaseNames = {
    "openvino_vision_embeddings_model",
    "openvino_vision_embeddings_pos_model",
    "openvino_vision_embeddings_merger_model",
};

std::vector<UnloadedIr> scan_unloaded_vision_irs(const std::string& dir) {
    std::vector<UnloadedIr> out;
    for (const char* base : kUnloadedVisionIrBaseNames) {
        for (const char* ext : {".xml", ".bin"}) {
            const std::string path = dir + "/" + base + ext;
            if (file_exists(path)) {
                out.push_back(UnloadedIr{base + std::string(ext), file_size(path)});
            }
        }
    }
    return out;
}

std::string basename_of(const std::string& path) {
    std::string s = path;
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    const size_t slash = s.find_last_of('/');
    return slash == std::string::npos ? s : s.substr(slash + 1);
}

// A VLM export nests the language-model geometry under "text_config"; a plain
// text export puts it at the top level.
const json& text_config(const json& config) {
    if (config.contains("text_config") && config["text_config"].is_object()) {
        return config["text_config"];
    }
    return config;
}

int int_or(const json& j, const char* key, int fallback) {
    if (j.contains(key) && j[key].is_number_integer()) return j[key].get<int>();
    return fallback;
}

uint64_t uint64_or(const json& j, const char* key, uint64_t fallback) {
    if (j.contains(key) && j[key].is_number_integer() && !j[key].is_number_float()) {
        return j[key].get<uint64_t>();
    }
    return fallback;
}

// Resolve one `segments[]` row of serving-shape.json into paths under `dir`
// and the file contract the loader needs (both files present, the row's own
// geometry). Returns an empty string on success, else the refusal naming the
// segment. The row's SEMANTICS -- whether the chain tiles the model, whether
// a segment holds a full-attention layer, whether the expert bodies' shapes
// agree across segments -- belong to segplan::plan_segments, the runtime's
// first call; the loader refuses a directory whose files are not there, not a
// graph it has not read.
std::string resolve_segment(const std::string& dir, const json& row, int position,
                            ArtifactSegment& out) {
    const int index = int_or(row, "index", -1);
    if (index != position) {
        return log::format("serving-shape.json segment at position %d has index %d; "
                           "segments must be listed in index order with no gap",
                           position, index);
    }
    out.index = index;
    out.dir   = row.value("dir", std::string("."));
    if (out.dir.empty()) {
        return log::format("segment %d has an empty 'dir'", index);
    }
    if (!row.contains("layers") || !row.at("layers").is_array() ||
        row.at("layers").size() != 2 || !row.at("layers")[0].is_number_integer() ||
        !row.at("layers")[1].is_number_integer()) {
        return log::format("segment %d: 'layers' must be [lo, hi]", index);
    }
    out.layer_lo            = row.at("layers")[0].get<int>();
    out.layer_hi            = row.at("layers")[1].get<int>();
    out.first               = row.value("first", false);
    out.last                = row.value("last", false);
    out.inputs_embeds_width = int_or(row, "inputs_embeds_width", 0);
    out.has_ple             = row.value("has_ple", false);
    out.attn_layers         = int_or(row, "attn_layers", 0);
    out.gdn_layers          = int_or(row, "gdn_layers", 0);

    const std::string sub = out.dir == "." ? dir : dir + "/" + out.dir;
    out.language_model_xml = sub + "/openvino_language_model.xml";
    out.language_model_bin = sub + "/openvino_language_model.bin";
    // A subdirectory segment is checked here; the top-level pair goes through
    // the required-files loop below, keeping today's "artifact is missing"
    // wording for a non-segmented artifact.
    if (out.dir != ".") {
        if (!file_exists(out.language_model_xml)) {
            return log::format("segment %d is missing %s", index,
                               out.language_model_xml.c_str());
        }
        if (!file_exists(out.language_model_bin)) {
            return log::format("segment %d is missing %s", index,
                               out.language_model_bin.c_str());
        }
    }
    return {};
}

void collect_eos(const json& j, std::vector<int>& out) {
    if (!j.contains("eos_token_id")) return;
    const json& e = j["eos_token_id"];
    if (e.is_number_integer()) {
        out.push_back(e.get<int>());
    } else if (e.is_array()) {
        for (const json& v : e) {
            if (v.is_number_integer()) out.push_back(v.get<int>());
        }
    }
}

}  // namespace

ArtifactInfo Artifact::to_info(Quant quant) const {
    ArtifactInfo info;
    info.id             = id;
    info.quant          = quant;
    info.n_ctx_train    = n_ctx_train;
    info.n_layer        = n_layer;
    info.n_gdn_layer    = n_gdn_layer;
    info.n_attn_layer   = n_attn_layer;
    info.arch_hash      = arch_hash;
    info.template_hash  = template_hash;
    info.tokenizer_hash = tokenizer_hash;
    info.weights_bytes  = weights_bytes;
    info.has_mtp_head   = has_mtp_head;
    return info;
}

std::optional<std::string> load_artifact(const std::string& dir, Artifact& out,
                                       bool require_allowlisted) {
    Artifact          a;
    a.dir            = dir;
    a.directory_name = basename_of(dir);

    const ModelEntry* entry = find_by_artifact(a.directory_name);
    if (entry == nullptr && require_allowlisted) {
        return log::format(
            "'%s' is not an allowlisted artifact directory (see models/allowlist-raw.json)",
            a.directory_name.c_str());
    }
    // --inspect-artifact reaches here with no entry: the id stays empty and the
    // sampler keeps its unset family-card state rather than claiming a family.
    if (entry != nullptr) a.id = entry->id;

    a.language_model_xml  = dir + "/openvino_language_model.xml";
    a.language_model_bin  = dir + "/openvino_language_model.bin";
    a.text_embeddings_xml = dir + "/openvino_text_embeddings_model.xml";
    a.tokenizer_xml       = dir + "/openvino_tokenizer.xml";
    a.detokenizer_xml     = dir + "/openvino_detokenizer.xml";

    a.unloaded_vision_irs = scan_unloaded_vision_irs(dir);

    const std::string config_path     = dir + "/config.json";
    const std::string generation_path = dir + "/generation_config.json";
    const std::string template_path   = dir + "/chat_template.jinja";
    const std::string tokenizer_json  = dir + "/tokenizer.json";
    const std::string tokenizer_cfg   = dir + "/tokenizer_config.json";
    const std::string shape_path      = dir + "/serving-shape.json";

    // ------------------------------------------- segments (0.5.1, window-051 §2)
    // A segmented export writes its language model as segmentK/openvino_
    // language_model.{xml,bin} and its expert bodies into one expert_bodies.u8
    // blob; serving-shape.json's `segments[]` says which directory carries
    // which layer range. It is read BEFORE the required-files loop, because a
    // segmented artifact has no top-level openvino_language_model.* to require,
    // and requiring one would refuse a good artifact for a file it never had.
    // An artifact with no manifest at all -- every artifact on the allowlist
    // today -- still gets exactly one segment (index 0, dir ".", the paths
    // above), so callers iterate `segments` instead of branching.
    json shape;
    if (file_exists(shape_path)) {
        try {
            shape = json::parse(read_file(shape_path));
        } catch (const json::exception& e) {
            return log::format("serving-shape.json is not valid JSON: %s", e.what());
        }
        a.serving_shape = shape;  // plan_segments reads this; the file is read once
    }
    if (shape.is_object() && shape.contains("segments") && shape.at("segments").is_array()) {
        int position = 0;
        for (const json& row : shape.at("segments")) {
            ArtifactSegment seg;
            if (auto err = resolve_segment(dir, row, position++, seg); !err.empty()) {
                return err;
            }
            a.segments.push_back(std::move(seg));
        }
    }
    // "declared as a chain", not "has more than one segment": an export with
    // segment_layers set to the whole depth is one segment and must still be
    // hashed (and driven) as a chain of one.
    a.from_segmented_manifest =
        shape.is_object() && !shape.value("segment_layers", json()).is_null();
    if (a.segments.empty()) {
        ArtifactSegment seg;
        seg.index              = 0;
        seg.dir                = ".";
        seg.language_model_xml = a.language_model_xml;
        seg.language_model_bin = a.language_model_bin;
        a.segments.push_back(std::move(seg));
    }
    // Segment 0 IS the language model for every caller that does not know
    // about segments (the compile, the allowlist check, the load-time log).
    a.language_model_xml = a.segments.front().language_model_xml;
    a.language_model_bin = a.segments.front().language_model_bin;

    // ------------------------------------------- expert bodies (window-051 §2)
    // The blob the runtime refills one segment at a time from. Its `entries[]`
    // is the index (global layer, kind, offset, bytes); the loader transcribes
    // it and checks the blob against the manifest's own size claim.
    if (shape.is_object() && shape.contains("expert_bodies") &&
        shape.at("expert_bodies").is_object()) {
        const json& eb = shape.at("expert_bodies");
        a.expert_bodies_path = dir + "/" + eb.value("file", std::string("expert_bodies.u8"));
        if (!file_exists(a.expert_bodies_path)) {
            return log::format("artifact is missing %s (serving-shape.json's "
                               "expert_bodies.file)",
                               a.expert_bodies_path.c_str());
        }
        a.expert_bodies_bytes = file_size(a.expert_bodies_path);
        const uint64_t claimed = uint64_or(eb, "bytes", a.expert_bodies_bytes);
        if (claimed != a.expert_bodies_bytes) {
            return log::format("%s is %llu bytes, serving-shape.json claims %llu",
                               a.expert_bodies_path.c_str(),
                               static_cast<unsigned long long>(a.expert_bodies_bytes),
                               static_cast<unsigned long long>(claimed));
        }
        if (eb.contains("entries") && eb.at("entries").is_array()) {
            for (const json& e : eb.at("entries")) {
                ExpertBodyEntry entry;
                entry.name    = e.value("name", std::string());
                entry.segment = int_or(e, "segment", -1);
                entry.layer   = int_or(e, "layer", -1);
                entry.kind    = e.value("kind", std::string());
                if (e.contains("shape") && e.at("shape").is_array()) {
                    for (const json& d : e.at("shape")) {
                        if (d.is_number_integer()) entry.shape.push_back(d.get<int64_t>());
                    }
                }
                entry.offset = uint64_or(e, "offset", 0);
                entry.bytes  = uint64_or(e, "bytes", 0);
                a.expert_bodies.push_back(std::move(entry));
            }
        }
    }

    for (const std::string& required :
         {a.language_model_xml, a.language_model_bin, a.text_embeddings_xml, a.tokenizer_xml,
          a.detokenizer_xml, config_path, template_path, tokenizer_json}) {
        if (!file_exists(required)) {
            return log::format("artifact is missing %s", required.c_str());
        }
    }

    try {
        a.config = json::parse(read_file(config_path));
    } catch (const json::exception& e) {
        return log::format("config.json is not valid JSON: %s", e.what());
    }
    if (file_exists(generation_path)) {
        try {
            a.generation = json::parse(read_file(generation_path));
        } catch (const json::exception& e) {
            return log::format("generation_config.json is not valid JSON: %s", e.what());
        }
    }

    a.chat_template = read_file(template_path);
    if (a.chat_template.empty()) return "chat_template.jinja is empty";

    // ------------------------------------------------------------- geometry
    const json& tc = text_config(a.config);
    a.model_type   = a.config.value("model_type", tc.value("model_type", std::string()));
    if (a.config.contains("architectures") && a.config["architectures"].is_array() &&
        !a.config["architectures"].empty() && a.config["architectures"][0].is_string()) {
        a.ov_arch = a.config["architectures"][0].get<std::string>();
    }

    a.n_layer                 = int_or(tc, "num_hidden_layers", 0);
    a.n_embd                  = int_or(tc, "hidden_size", 0);
    a.hc_count                = int_or(tc, "hc_count", 0);
    a.n_ctx_train             = int_or(tc, "max_position_embeddings", 0);
    a.n_expert                = int_or(tc, "num_experts", 0);
    a.moe                     = a.n_expert > 0;
    a.full_attention_interval = int_or(tc, "full_attention_interval", 0);

    if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
        for (const json& t : tc["layer_types"]) {
            if (t.is_string()) a.layer_types.push_back(t.get<std::string>());
        }
        // "full_attention" is the qwen3.5/3.6 exports' name; "qwen_sparse_
        // attention" is the Flash-Next pin's (layer_types at layer_idx % 4 ==
        // 3), served dense-causal -- the selection branch is the indexer,
        // which is not emitted (window-050 §8: price 0.0 to T=2051). Either
        // way the layer carries a KV cache and one ScaledDotProductAttention,
        // which is what n_attn_layer counts.
        for (const std::string& t : a.layer_types) {
            if (t == "full_attention" || t == "qwen_sparse_attention") ++a.n_attn_layer;
        }
        a.n_gdn_layer = static_cast<int>(a.layer_types.size()) - a.n_attn_layer;
        if (!a.layer_types.empty() && static_cast<int>(a.layer_types.size()) != a.n_layer) {
            return log::format("config.json disagrees with itself: %zu layer_types but "
                               "num_hidden_layers %d",
                               a.layer_types.size(), a.n_layer);
        }
    } else if (a.full_attention_interval > 0 && a.n_layer > 0) {
        // Derived, and only when the explicit list is absent.
        a.n_attn_layer = a.n_layer / a.full_attention_interval;
        a.n_gdn_layer  = a.n_layer - a.n_attn_layer;
    }

    // ------------------------------------------------------ n-gram (FIX D)
    // The Flash-Next checkpoint declares its n-gram embedding table via
    // five text_config keys (docs/design-qwen-flash-next.md FIX B delta
    // table; the RED-B-01 red case sits on the loader having no field
    // for them). All zeros means "no n-gram declared" -- the dense
    // qwen35 checkpoints on the current allowlist land there and the
    // admission path below never fires.
    a.ngram_config.ngram_size            = int_or(tc, "ngram_size", 0);
    a.ngram_config.ngram_vocab_size_base = int_or(tc, "ngram_vocab_size_base", 0);
    a.ngram_config.heads_per_ngram       = int_or(tc, "heads_per_ngram", 0);
    a.ngram_config.ple_embed_dim         = int_or(tc, "ple_embed_dim", 0);
    if (tc.contains("ple_layer_ids") && tc["ple_layer_ids"].is_array()) {
        for (const json& v : tc["ple_layer_ids"]) {
            if (v.is_number_integer()) {
                a.ngram_config.ple_layer_ids.push_back(v.get<int>());
            }
        }
    }
    a.ngram_config.vocab_size = int_or(tc, "vocab_size", 0);
    // The n-gram hash boundary is the checkpoint's eos_token_id (reference
    // config.py:195-197). It is set from `a.eos_ids` below, once the full
    // generation_config -> text_config -> config fallback chain has been read --
    // a top-level `eos_token_id` (the common HF layout) must not be missed, or
    // token 0 would silently act as a boundary.

    // ------------------------------------------------------ tokens, sampler
    // The family-card defaults are the fallback; the artifact's own
    // generation_config.json overrides them below and marks the provenance
    // "artifact". No entry (--inspect-artifact on a directory the allowlist
    // does not claim) leaves them as the struct's own zeros.
    if (entry != nullptr) a.sampler = entry->sampler;
    if (!a.generation.is_null()) {
        collect_eos(a.generation, a.eos_ids);
        if (a.generation.contains("temperature") && a.generation["temperature"].is_number()) {
            a.sampler.temperature = a.generation["temperature"].get<float>();
        }
        if (a.generation.contains("top_p") && a.generation["top_p"].is_number()) {
            a.sampler.top_p = a.generation["top_p"].get<float>();
        }
        if (a.generation.contains("top_k") && a.generation["top_k"].is_number_integer()) {
            a.sampler.top_k = a.generation["top_k"].get<int>();
        }
        if (a.generation.contains("repetition_penalty") &&
            a.generation["repetition_penalty"].is_number()) {
            a.sampler.repetition_penalty = a.generation["repetition_penalty"].get<float>();
        }
        if (a.generation.contains("presence_penalty") &&
            a.generation["presence_penalty"].is_number()) {
            a.sampler.presence_penalty = a.generation["presence_penalty"].get<float>();
        }
        a.sampler.provenance = "artifact";
    }
    if (a.eos_ids.empty()) collect_eos(tc, a.eos_ids);
    if (a.eos_ids.empty()) collect_eos(a.config, a.eos_ids);
    // The n-gram hash boundary follows the same eos chain; -1 stays when the
    // config carries no eos_token_id at all (admission refuses a declared table
    // then, rather than hashing with a legal token id as the boundary).
    a.ngram_config.ngram_boundary_token_id =
        a.eos_ids.empty() ? -1 : a.eos_ids.front();

    if (file_exists(tokenizer_cfg)) {
        try {
            const json tcfg = json::parse(read_file(tokenizer_cfg));
            if (tcfg.contains("eos_token") && tcfg["eos_token"].is_string()) {
                a.eos_token = tcfg["eos_token"].get<std::string>();
            }
            if (tcfg.contains("bos_token") && tcfg["bos_token"].is_string()) {
                a.bos_token = tcfg["bos_token"].get<std::string>();
            }
        } catch (const json::exception&) {
            // tokenizer_config.json is advisory here; the tokenizer itself is
            // the authority and it is validated by hash.
        }
    }

    // ---------------------------------------------------------------- hashes
    // Does the export actually carry an MTP head? The checkpoints all declare
    // mtp_num_hidden_layers, but optimum-intel drops the graph, and only the
    // graph can be served. Detect it rather than trusting either the config or
    // DESIGN.md §3.5.
    // Both halves are needed: the head's own layer, and the LM head extracted
    // from the base model so the draft can be turned into a token.
    // Two layers can serve: the reconstructed one, and -- since optimum-intel's
    // development branch started exporting it (seen 2026-08-30 in Intel's
    // public Qwen3.8 IR) -- the exporter's own openvino_mtp_model. Neither
    // carries the lm_head the draft is decoded with; that is always ours.
    a.mtp_layer_xml          = dir + "/openvino_mtp_layer.xml";
    a.mtp_exported_layer_xml = dir + "/openvino_mtp_model.xml";
    a.mtp_lm_head_xml        = dir + "/openvino_mtp_lm_head.xml";
    if (!file_exists(a.mtp_layer_xml)) a.mtp_layer_xml.clear();
    if (!file_exists(a.mtp_exported_layer_xml)) a.mtp_exported_layer_xml.clear();
    a.has_mtp_head = (!a.mtp_layer_xml.empty() || !a.mtp_exported_layer_xml.empty()) &&
                     file_exists(a.mtp_lm_head_xml);
    if (!a.has_mtp_head) {
        a.mtp_layer_xml.clear();
        a.mtp_exported_layer_xml.clear();
        a.mtp_lm_head_xml.clear();
    }

    // A non-segmented artifact's single segment spans the whole model; a
    // segmented one carries its own ranges in the manifest rows.
    if (!a.from_segmented_manifest) {
        for (ArtifactSegment& s : a.segments) {
            s.layer_lo = 0;
            s.layer_hi = a.n_layer;
        }
    }

    a.template_hash  = hash_prefix(sha256_hex(a.chat_template));
    a.tokenizer_hash = hash_prefix(sha256_file(tokenizer_json));
    // Every segment's xml is hashed (the chain hash is over those, in segment
    // order) and every segment's .bin counts toward weights_bytes: a
    // segmented artifact's resident set is the SUM, and reporting only
    // segment 0's would understate it by K-1 segments.
    {
        std::vector<std::string> xml_shas;
        uint64_t weights = 0;
        for (ArtifactSegment& s : a.segments) {
            s.xml_sha        = sha256_file(s.language_model_xml);
            s.lm_bin_bytes   = file_size(s.language_model_bin);
            xml_shas.push_back(s.xml_sha);
            weights += s.lm_bin_bytes;
        }
        a.weights_bytes = weights;
        a.arch_hash     = a.segmented() ? segplan::chain_arch_hash(xml_shas)
                                        : hash_prefix(xml_shas.front());
    }

    out = std::move(a);
    return std::nullopt;
}

std::string admit_ngram_table_from_disk(const Artifact& artifact,
                                        const std::string& path,
                                        uint64_t host_ram_bytes,
                                        uint64_t other_resident_bytes,
                                        uint64_t margin_bytes,
                                        uint64_t& out_payload_bytes) {
    out_payload_bytes = 0;
    const auto& nc = artifact.ngram_config;
    if (nc.ngram_size == 0 && nc.ple_embed_dim == 0) {
        return log::format(
            "the artifact does not declare an n-gram table (config.json has "
            "no non-zero ngram_size or ple_embed_dim); refusing to admit "
            "%s as a per_layer_token_embd for a checkpoint that does not "
            "have one",
            path.c_str());
    }

    // The table is declared, so the whole n-gram config must be present and
    // usable: the hash index and its dummy-weight constant derivation
    // (exec/ngram_row_ids.h) need every one of these, and a missing field would
    // otherwise be filled with a silent, wrong default (vocab_size 0 -> a
    // multiplier bound of 2^63; an empty ple_layer_ids -> a fabricated single
    // layer; a missing eos -> token 0 as the hash boundary). Refuse by name
    // instead. The reference treats an empty ple_layer_ids as "no PLE at all".
    if (nc.ngram_size < 2 || nc.heads_per_ngram < 1 || nc.ngram_vocab_size_base <= 0 ||
        nc.ple_embed_dim <= 0 || nc.vocab_size <= 0 || nc.ple_layer_ids.empty() ||
        nc.ngram_boundary_token_id < 0) {
        return log::format(
            "%s: the artifact declares an n-gram table but its config is "
            "incomplete (ngram_size=%d heads_per_ngram=%d ngram_vocab_size_base=%d "
            "ple_embed_dim=%d vocab_size=%d ple_layer_ids=%zu eos=%d); all must be "
            "present (ngram_size>=2, the rest > 0, ple_layer_ids non-empty, an "
            "eos_token_id found)",
            path.c_str(), nc.ngram_size, nc.heads_per_ngram, nc.ngram_vocab_size_base,
            nc.ple_embed_dim, nc.vocab_size, nc.ple_layer_ids.size(),
            nc.ngram_boundary_token_id);
    }

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return log::format("could not open %s for reading", path.c_str());
    }
    std::array<uint8_t, ngram::kHeaderBytes> hdr_bytes{};
    f.read(reinterpret_cast<char*>(hdr_bytes.data()), hdr_bytes.size());
    if (!f) {
        return log::format(
            "%s is shorter than the ARCINGRM 24-byte header (%zd bytes read)",
            path.c_str(), static_cast<ptrdiff_t>(f.gcount()));
    }
    ngram::Header header;
    if (auto err = ngram::parse_header(hdr_bytes.data(), hdr_bytes.size(), header);
        !err.empty()) {
        return log::format("%s: %s", path.c_str(), err.c_str());
    }

    // Shape cross-check: the file's (n_cols, n_rows) must be consistent
    // with the config's (ple_embed_dim, ngram_vocab_size_base) at
    // physical-row width 160 (design doc's own row-width convention).
    // n_cols must equal 160 today -- a non-160 row width is admitted
    // through the header parser but not by this admission surface,
    // because the AVX2 kernel and the fit arithmetic assume 160.
    constexpr uint32_t kPhysicalRowWidth = 160;
    if (header.n_cols != kPhysicalRowWidth) {
        return log::format(
            "%s: n_cols %u differs from the served row width %u "
            "(the AVX2 gather kernel and fit arithmetic assume 160)",
            path.c_str(), header.n_cols, kPhysicalRowWidth);
    }
    if (artifact.ngram_config.ple_embed_dim > 0 &&
        (artifact.ngram_config.ple_embed_dim % kPhysicalRowWidth) != 0) {
        return log::format(
            "%s: config's ple_embed_dim %d is not a multiple of the "
            "physical row width %u (a real spec would need this, and "
            "the current admission has no other place to catch it)",
            path.c_str(), artifact.ngram_config.ple_embed_dim,
            kPhysicalRowWidth);
    }
    // Row-count admission. The table must hold every row id the hashed-vocab
    // index (exec/ngram_row_ids.h) can produce: the concatenated per-head prime
    // vocab sizes, topped by the last PLE layer's band. This is a LOWER bound
    // (>=), not the pre-correction equality n_rows == base*(ple_embed_dim/160):
    // that formula assumed a plain "rows per vocab entry" table and is
    // inconsistent with the reference's own sizing (docs/research-freetoken.md
    // "Code-side ground truth"; the exact shipped size may be padded up via
    // split_ngram_parts and can only be pinned against a real artifact -- see
    // docs/design-qwen-flash-next.md FIX D "Links 2 and 3" reconcile flag).
    {
        // The completeness check above guarantees every field here is valid.
        const int num_ple_layers = static_cast<int>(artifact.ngram_config.ple_layer_ids.size());
        const uint64_t required_rows = static_cast<uint64_t>(ngram::ngram_required_rows(
            artifact.ngram_config.vocab_size, artifact.ngram_config.ngram_size,
            artifact.ngram_config.heads_per_ngram,
            artifact.ngram_config.ngram_vocab_size_base, num_ple_layers));
        if (header.n_rows < required_rows) {
            return log::format(
                "%s: n_rows %u is below the %llu rows the hashed n-gram index "
                "needs (%d heads x prime vocab >= ngram_vocab_size_base %d, "
                "last of %d PLE layer(s)); the table cannot hold every row id",
                path.c_str(), header.n_rows,
                static_cast<unsigned long long>(required_rows),
                (artifact.ngram_config.ngram_size - 1) * artifact.ngram_config.heads_per_ngram,
                artifact.ngram_config.ngram_vocab_size_base, num_ple_layers);
        }
    }

    // File size must equal header + payload; a truncated or over-sized
    // file (a common corruption on rsync interrupts) fails here rather
    // than deep in a decode.
    const uint64_t payload = ngram::payload_bytes(header);
    const uint64_t on_disk = file_size(path);
    if (on_disk != ngram::kHeaderBytes + payload) {
        return log::format(
            "%s: on-disk size %llu differs from header + payload %llu "
            "(24 + %u x %u x %zu)",
            path.c_str(),
            static_cast<unsigned long long>(on_disk),
            static_cast<unsigned long long>(ngram::kHeaderBytes + payload),
            header.n_rows, header.n_cols / ngram::kBlockElements,
            ngram::bytes_per_block(header.ggml_type));
    }

    // Host-RAM fit refusal. `host_ram_fit_must_refuse` is pure arithmetic
    // (`src/exec/fit.h`); this call site names the actual host numbers.
    if (host_ram_bytes > 0 &&
        host_ram_fit_must_refuse(payload, /*expert_pool_bytes=*/0,
                                 other_resident_bytes, host_ram_bytes,
                                 margin_bytes)) {
        return log::format(
            "%s (%llu payload bytes) does not fit the host RAM budget "
            "(host_ram %llu, other resident %llu, margin %llu)",
            path.c_str(),
            static_cast<unsigned long long>(payload),
            static_cast<unsigned long long>(host_ram_bytes),
            static_cast<unsigned long long>(other_resident_bytes),
            static_cast<unsigned long long>(margin_bytes));
    }

    out_payload_bytes = payload;
    return {};
}

std::string serve_refusal_for(const Artifact& artifact) {
    if (!artifact.segmented()) return {};
    const ArtifactSegment& first = artifact.segments.front();
    return log::format(
        "'%s' is a SEGMENTED artifact (%zu compiled segments, %d layers in all) and "
        "nothing in this build drives a chain: the single-graph path would open %s "
        "(its OWN range is layers %d..%d) and serve that under an allowlist entry "
        "that says %d layers -- a wrong answer with a valid pin on it. Read the "
        "contract with --inspect-artifact; the segmented forward is window-051 §2 "
        "and does not exist yet.",
        artifact.directory_name.c_str(), artifact.segments.size(), artifact.n_layer,
        first.language_model_xml.c_str(), first.layer_lo, first.layer_hi - 1,
        artifact.n_layer);
}
}  // namespace lgc

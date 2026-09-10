#include "core/artifact.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "core/ngram_header.h"
#include "exec/fit.h"
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
    info.has_mtp_head   = has_mtp_head;
    return info;
}

std::optional<std::string> load_artifact(const std::string& dir, Artifact& out) {
    Artifact          a;
    a.dir            = dir;
    a.directory_name = basename_of(dir);

    const ModelEntry* entry = find_by_artifact(a.directory_name);
    if (entry == nullptr) {
        return log::format(
            "'%s' is not an allowlisted artifact directory (see models/allowlist-raw.json)",
            a.directory_name.c_str());
    }
    a.id = entry->id;

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
    a.n_ctx_train             = int_or(tc, "max_position_embeddings", 0);
    a.n_expert                = int_or(tc, "num_experts", 0);
    a.moe                     = a.n_expert > 0;
    a.full_attention_interval = int_or(tc, "full_attention_interval", 0);

    if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
        for (const json& t : tc["layer_types"]) {
            if (t.is_string()) a.layer_types.push_back(t.get<std::string>());
        }
        for (const std::string& t : a.layer_types) {
            if (t == "full_attention") ++a.n_attn_layer;
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

    // ------------------------------------------------------ tokens, sampler
    a.sampler = entry->sampler;  // family-card fallback, marked provisional
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

    a.arch_hash      = hash_prefix(sha256_file(a.language_model_xml));
    a.template_hash  = hash_prefix(sha256_hex(a.chat_template));
    a.tokenizer_hash = hash_prefix(sha256_file(tokenizer_json));
    a.weights_bytes  = file_size(a.language_model_bin);

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
    if (artifact.ngram_config.ngram_size == 0 &&
        artifact.ngram_config.ple_embed_dim == 0) {
        return log::format(
            "the artifact does not declare an n-gram table (config.json has "
            "no non-zero ngram_size or ple_embed_dim); refusing to admit "
            "%s as a per_layer_token_embd for a checkpoint that does not "
            "have one",
            path.c_str());
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
    if (artifact.ngram_config.ngram_vocab_size_base > 0 &&
        artifact.ngram_config.ple_embed_dim > 0) {
        const uint64_t rows_per_vocab_entry =
            artifact.ngram_config.ple_embed_dim / kPhysicalRowWidth;
        const uint64_t expected_rows =
            static_cast<uint64_t>(artifact.ngram_config.ngram_vocab_size_base) *
            rows_per_vocab_entry;
        if (header.n_rows != expected_rows) {
            return log::format(
                "%s: n_rows %u differs from the config-implied %llu "
                "(ngram_vocab_size_base %d x ple_embed_dim/%u = %d)",
                path.c_str(), header.n_rows,
                static_cast<unsigned long long>(expected_rows),
                artifact.ngram_config.ngram_vocab_size_base,
                kPhysicalRowWidth,
                artifact.ngram_config.ple_embed_dim / kPhysicalRowWidth);
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

}  // namespace lgc

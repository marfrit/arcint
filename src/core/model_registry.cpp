#include "core/model_registry.h"

#include <algorithm>

#include "util/log.h"

namespace lgc {
namespace {

// The Qwen3-family model-card sampler recommendation. Provisional by
// construction — models/allowlist-raw.json carries no sampler settings, so
// these are inherited, not measured. See SamplerDefaults::provenance.
SamplerDefaults qwen_card_defaults() {
    SamplerDefaults d;
    d.temperature        = 0.7f;
    d.top_p              = 0.8f;
    d.top_k              = 20;
    d.repetition_penalty = 1.05f;
    d.presence_penalty   = 0.0f;
    d.provenance         = "provisional";
    return d;
}

// One layer in `interval` is full attention; the remainder carry GDN state.
void split_layers(ModelEntry& e) {
    e.n_attn_layer = e.full_attention_interval > 0 ? e.n_layer / e.full_attention_interval : 0;
    e.n_gdn_layer  = e.n_layer - e.n_attn_layer;
}

std::vector<ModelEntry> build_registry() {
    std::vector<ModelEntry> r;

    {
        ModelEntry e;
        e.id                      = "qwen3.6-27b-a3b-coder";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-coder-b5-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;  // no MTP graph in the export
        e.mtp_head_pinned         = true;   // inspected 2026-08-28
        e.mtp_in_checkpoint       = true;   // config: mtp_num_hidden_layers 1
        e.n_embd                  = 2048;
        e.n_expert                = 184;  // pruned from 256
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4, Quant::Q8};
        e.arch_hash               = "6745cfe3d57e3f0f";
        e.template_hash           = "e84f32a23fdda276";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 13760293946ull;
        e.status                  = "production, 10/10 on the Pruefstand (b5 artifact)";
        e.sampler                 = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-int4-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;  // no MTP graph in the export
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4, Quant::Q8};
        e.arch_hash               = "21fe4d57d6d016f5";
        e.template_hash           = "c3f7038f278583e1";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 18646558274ull;
        // Measured 2026-08-30 as the fleet's agent endpoint (arcint-agent.service,
        // GPU.0 = B60, 262144 context, the reservation admitting 377552): 10/10
        // greedy on the acceptance task, 3/3 on the tool probe, clean German with
        // enable_thinking:false, 62.7 t/s decode at 53.5k and 1584 t/s prefill.
        e.status                  = "10/10 greedy, 3/3 tool probe (B60, 262144, 2026-08-30); "
                                    "62.7 t/s decode at 53.5k, 1584 t/s prefill";
        e.sampler                 = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        // The same 35B artifact with the reconstructed MTP head beside it, as a
        // separate directory of symlinks, so the production agent's directory is
        // never touched: --mtp auto turns drafting on the moment a head exists,
        // and a head goes into production only after it is measured.
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b-mtp";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-mtp-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = true;   // the reconstructed head (tools/export_mtp.py, MoE)
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "21fe4d57d6d016f5";
        e.template_hash           = "c3f7038f278583e1";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 18646558274ull;
        e.status                  = "unmeasured against the current harness (the 35B with the "
                                    "reconstructed MoE MTP head; acceptance is the oracle)";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        ModelEntry e;
        e.id                      = "qwen3.8-27b";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-b7c1-ov"};
        e.ov_arch                 = "Qwen3_5ForConditionalGeneration";
        e.model_type              = "qwen3_5";
        e.moe                     = false;  // dense (§2)
        // §3.5 says this one ships a native MTP head. optimum-intel's export
        // drops it, so for a long time the answer here was "no". It is now
        // "yes": tools/export_mtp.py reconstructs the head from the weights in
        // the checkpoint and writes openvino_mtp_{layer,lm_head}.xml beside the
        // model. An artifact without those files still loads -- the check
        // against this flag is what tells the user their export lacks them.
        e.has_mtp_head            = true;
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 5120;
        e.n_expert                = 0;
        e.full_attention_interval = 4;
        e.n_layer                 = 64;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4, Quant::Q8};
        e.arch_hash               = "5892b9b333bf0ab3";
        e.template_hash           = "c3cf9e34abf4f9e3";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 14405167394ull;
        // Measured through arcint's paged executor with the MTP head drafting
        // (2026-08-29): 10/10 under greedy at 36.2 t/s on the B60, 93.2% draft
        // acceptance. The earlier stateful measurement was 8/10 greedy; greedy
        // is deterministic per configuration, so both numbers are real -- the
        // paged path's near-tie landings simply score better on this task.
        e.status = "10/10 greedy (paged+MTP, B60); AWQ-only - SE calibration degenerates "
                   "greedy, do not re-add SE";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        // Intel's public export of the same checkpoint -- OpenVINO/Qwen3.8-27B-
        // int4-ov -- as its own entry, not an alias of qwen38-b7c1-ov: that entry
        // carries a measured status for our AWQ-only export, and an alias would
        // make the allowlist assert a result nobody obtained for this file.
        // Same geometry (hidden 5120, 64 layers, 24/4 heads, vocab 248320, one
        // MTP layer, untied embeddings), same chat template and tokenizer,
        // different quantisation of the body. The reconstructed MTP head from
        // tools/export_mtp.py is placed beside it; whether it pairs is what the
        // draft-acceptance measurement decides (DESIGN 7.0.2n).
        ModelEntry e;
        e.id                      = "qwen3.8-27b-intel-int4";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-intel-int4-ov"};
        e.ov_arch                 = "Qwen3_5ForConditionalGeneration";
        e.model_type              = "qwen3_5";
        e.moe                     = false;
        e.has_mtp_head            = true;   // the reconstructed head, copied beside it
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 5120;
        e.n_expert                = 0;
        e.full_attention_interval = 4;
        e.n_layer                 = 64;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "ab94a08ce150de6a";
        e.template_hash           = "c3cf9e34abf4f9e3";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 13929177378ull;
        // The reconstructed head pairs with this export (2026-08-30, B60, greedy,
        // thinking off): draft acceptance 96.3% (157/163) on a code prompt and
        // 77.3% (140/181) on prose, 34.6-37.4 t/s -- next to the 93.2% the same
        // head measures on our own export. A wrong head cannot raise acceptance,
        // only depress it, so the number is the oracle (tools/export_mtp.py).
        e.status                  = "10/10 greedy (paged+MTP, B60, 2026-08-30), 36.3 t/s at 90.8% draft "
                                    "acceptance vs 25.0 t/s plain; Intel's own MTP layer + our lm_head 37.7-38.1 t/s";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    return r;
}

void check_int(ValidationResult& res, const char* field, int pinned, int seen) {
    if (pinned == 0) {
        res.warnings.push_back(
            log::format("%s not pinned in the allowlist; artifact reports %d", field, seen));
        return;
    }
    if (pinned != seen) {
        res.errors.push_back(
            log::format("%s mismatch: allowlist %d, artifact %d", field, pinned, seen));
    }
}

void check_hash(ValidationResult& res, const char* field, const std::string& pinned,
                const std::string& seen) {
    if (pinned.empty()) {
        res.warnings.push_back(log::format(
            "%s not pinned in the allowlist; artifact reports %s", field,
            seen.empty() ? "nothing" : seen.c_str()));
        return;
    }
    if (seen.empty()) {
        res.errors.push_back(log::format("%s missing from artifact, allowlist pins %s", field,
                                         pinned.c_str()));
        return;
    }
    if (pinned != seen) {
        res.errors.push_back(log::format("%s mismatch: allowlist %s, artifact %s", field,
                                         pinned.c_str(), seen.c_str()));
    }
}

}  // namespace

const char* quant_name(Quant q) {
    switch (q) {
        case Quant::Q4: return "q4";
        case Quant::Q8: return "q8";
    }
    return "?";
}

std::optional<Quant> quant_parse(std::string_view s) {
    if (s == "q4" || s == "int4") return Quant::Q4;
    if (s == "q8" || s == "int8") return Quant::Q8;
    return std::nullopt;
}

bool ModelEntry::accepts(Quant q) const {
    return std::find(quants.begin(), quants.end(), q) != quants.end();
}

bool ModelEntry::matches_alias(std::string_view artifact_dir) const {
    return std::find(artifact_aliases.begin(), artifact_aliases.end(), artifact_dir) !=
           artifact_aliases.end();
}

const std::vector<ModelEntry>& registry() {
    static const std::vector<ModelEntry> kRegistry = build_registry();
    return kRegistry;
}

const ModelEntry* find_model(std::string_view id) {
    for (const ModelEntry& e : registry()) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

const ModelEntry* find_by_artifact(std::string_view artifact_dir) {
    for (const ModelEntry& e : registry()) {
        if (e.matches_alias(artifact_dir)) return &e;
    }
    return nullptr;
}

std::vector<std::string> model_ids() {
    std::vector<std::string> ids;
    ids.reserve(registry().size());
    for (const ModelEntry& e : registry()) ids.push_back(e.id);
    return ids;
}

ValidationResult validate_artifact(const ModelEntry& entry, const ArtifactInfo& seen) {
    ValidationResult res;

    if (entry.id != seen.id) {
        res.errors.push_back(
            log::format("id mismatch: allowlist %s, artifact %s", entry.id.c_str(),
                        seen.id.c_str()));
    }
    if (!entry.accepts(seen.quant)) {
        res.errors.push_back(log::format("quant %s not allowed for %s", quant_name(seen.quant),
                                         entry.id.c_str()));
    }
    if (entry.has_mtp_head != seen.has_mtp_head) {
        const std::string msg = log::format("MTP head mismatch: allowlist %s, artifact %s",
                                            entry.has_mtp_head ? "present" : "absent",
                                            seen.has_mtp_head ? "present" : "absent");
        // Only a refusal when the claim has a source. The allowlist's MTP flag
        // is currently design prose, and prose must not block a real artifact.
        if (entry.mtp_head_pinned) {
            res.errors.push_back(msg);
        } else {
            res.warnings.push_back(msg + " (allowlist value is unpinned; artifact wins)");
        }
    }

    check_int(res, "n_layer", entry.n_layer, seen.n_layer);
    check_int(res, "n_gdn_layer", entry.n_gdn_layer, seen.n_gdn_layer);
    check_int(res, "n_attn_layer", entry.n_attn_layer, seen.n_attn_layer);
    check_int(res, "n_ctx_train", entry.n_ctx_train, seen.n_ctx_train);

    check_hash(res, "arch_hash", entry.arch_hash, seen.arch_hash);
    check_hash(res, "template_hash", entry.template_hash, seen.template_hash);
    check_hash(res, "tokenizer_hash", entry.tokenizer_hash, seen.tokenizer_hash);

    if (seen.n_layer > 0 && seen.n_gdn_layer + seen.n_attn_layer != seen.n_layer) {
        res.errors.push_back(log::format("layer split does not sum: %d GDN + %d attn != %d",
                                         seen.n_gdn_layer, seen.n_attn_layer, seen.n_layer));
    }

    res.ok = res.errors.empty();
    return res;
}

}  // namespace lgc

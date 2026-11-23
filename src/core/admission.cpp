#include "core/admission.h"

#include <cstdio>

namespace lgc {
namespace {

bool is_mistral_small_24b(std::string_view arch, const ModelGeometry& g) {
    return arch == "llama" && g.n_layer == 40 && g.n_embd == 5120 && g.n_head == 32 && g.n_head_kv == 8 &&
           g.head_dim == 128 && g.n_vocab == 131072;
}

bool is_llama2_7b(std::string_view arch, const ModelGeometry& g) {
    return arch == "llama" && g.n_layer == 32 && g.n_embd == 4096 && g.n_head == 32 && g.n_head_kv == 32 &&
           g.n_vocab == 32000;
}

// Gemma 4 E4B (the 4-billion-effective-parameter edition): 42 layers, 2560
// wide, 8 query heads over 2 KV heads, a global-attention key length of 512,
// 262,144 tokens (`code`: the GGUF's metadata). The other Gemma 4 sizes are "runs".
bool is_gemma4_e4b(std::string_view arch, const ModelGeometry& g) {
    return arch == "gemma4" && g.n_layer == 42 && g.n_embd == 2560 && g.n_head == 8 && g.n_head_kv == 2 &&
           g.head_dim == 512 && g.n_vocab == 262144;
}

// Granite 4.2 8B, the dense edition: 40 layers, 4096 wide, 32 query heads over
// 8 KV heads, 100,352 tokens (`code`: the GGUF's metadata; the file carries no
// key length, it is the 4096 / 32 = 128 of the head split). The 'granite'
// architecture also covers other sizes, which are "runs".
bool is_granite_8b(std::string_view arch, const ModelGeometry& g) {
    return arch == "granite" && g.n_layer == 40 && g.n_embd == 4096 && g.n_head == 32 && g.n_head_kv == 8 &&
           g.n_vocab == 100352;
}

// Dory1.0-4B (NVIDIA's looped nGPT, `dory`): 72 PHYSICAL blocks (16 input, 40 recurrent,
// 16 output), 2560 wide, 16 query heads over 4 KV heads, key length 256, 131,072 tokens
// (`code`: the GGUF's metadata). `llama_model_n_layer` reports the executed count (112 at
// 2 loops, 192 at 4), so the backend passes llama_model_n_layer_physical here. The loop
// count is not part of the identity: 1 to 4 run over the same weights (core/dory.h).
bool is_dory_4b(std::string_view arch, const ModelGeometry& g) {
    return arch == "dory" && g.n_layer == 72 && g.n_embd == 2560 && g.n_head == 16 && g.n_head_kv == 4 &&
           g.head_dim == 256 && g.n_vocab == 131072;
}

// Inkling 2 Mini (Thinking Machines, `inkling2_loop`): 24 PHYSICAL blocks, 2304 wide, 24 query heads
// over 8 KV heads, key length 128, 200,058 tokens (`code`: the GGUF's metadata; the vocabulary is the
// unpadded count llama_vocab_n_tokens reports, the file pads its embedding rows to 201,024). The core
// blocks 6-17 run three times through cache slots, so, as for Dory, the backend passes the physical count.
bool is_inkling2_mini(std::string_view arch, const ModelGeometry& g) {
    return arch == "inkling2_loop" && g.n_layer == 24 && g.n_embd == 2304 && g.n_head == 24 && g.n_head_kv == 8 &&
           g.head_dim == 128 && g.n_vocab == 200058;
}

// The measured sizes of the families admitted at any size today (`code`: the files' metadata;
// the tier decision 2026-11-23 keeps only these at "optimized for" or "verified", every other
// size becomes "runs").
bool is_qwen35_agent(std::string_view arch, const ModelGeometry& g) {
    return arch == "qwen35" && g.n_layer == 64 && g.n_embd == 5120 && g.n_head == 24 && g.n_head_kv == 4 &&
           g.head_dim == 256 && g.n_vocab == 248320;   // Qwen3.8-27B, the dense agent
}

bool is_qwen35moe_coder(std::string_view arch, const ModelGeometry& g) {
    return arch == "qwen35moe" && g.n_layer == 40 && g.n_embd == 2048 && g.n_head == 16 && g.n_head_kv == 2 &&
           g.head_dim == 256 && g.n_vocab == 248320;   // Qwen3.6-27B-A3B, the coder. llama_model_n_layer is
                                                        // block_count minus the MTP (nextn) layers: the
                                                        // file's 41 blocks report 40, as the agent's 65 report 64
}

bool is_qwen4exp_flash(std::string_view arch, const ModelGeometry& g) {
    return arch == "qwen4exp" && g.n_layer == 48 && g.n_embd == 2560 && g.n_head == 24 && g.n_head_kv == 2 &&
           g.head_dim == 256 && g.n_vocab == 248320;   // Qwen3.8-Flash-Next
}

bool is_qwen2vl_7b(std::string_view arch, const ModelGeometry& g) {
    // the file carries no qwen2vl.attention.key_length, so head_dim reads 0 (absent); 3584 / 28 = 128
    return arch == "qwen2vl" && g.n_layer == 28 && g.n_embd == 3584 && g.n_head == 28 && g.n_head_kv == 4 &&
           (g.head_dim == 0 || g.head_dim == 128) && g.n_vocab == 152064;   // Qwen2.5-VL-7B: verified (the C1/C2 gate)
}

}  // namespace

std::string checkpoints_refusal(bool asked, bool can_snapshot, std::string_view arch) {
    if (!asked || can_snapshot) return {};
    return "--llama-checkpoints: a '" + std::string(arch) +
           "' model has no recurrent memory to snapshot (its attention cache trims); drop the flag or pass 0";
}

Admission admit_model(std::string_view arch, const ModelGeometry& g, bool tts) {
    if (tts && arch == "qwen3tts") return {AdmittedModel::Qwen3Tts, AdmissionTier::Verified};
    if (arch == "qwen35")
        return {AdmittedModel::Qwen35, is_qwen35_agent(arch, g) ? AdmissionTier::Optimized : AdmissionTier::Runs};
    if (arch == "qwen35moe")
        return {AdmittedModel::Qwen35Moe, is_qwen35moe_coder(arch, g) ? AdmissionTier::Verified : AdmissionTier::Runs};
    if (arch == "qwen4exp")
        return {AdmittedModel::Qwen4Exp, is_qwen4exp_flash(arch, g) ? AdmissionTier::Optimized : AdmissionTier::Runs};
    if (arch == "qwen2vl")
        return {AdmittedModel::Qwen2Vl, is_qwen2vl_7b(arch, g) ? AdmissionTier::Verified : AdmissionTier::Runs};
    if (is_mistral_small_24b(arch, g)) return {AdmittedModel::MistralSmall24B, AdmissionTier::Runs};
    if (is_llama2_7b(arch, g)) return {AdmittedModel::Llama2_7B, AdmissionTier::Verified};
    if (is_gemma4_e4b(arch, g)) return {AdmittedModel::Gemma4E4B, AdmissionTier::Runs};
    if (is_granite_8b(arch, g)) return {AdmittedModel::Granite8B, AdmissionTier::Runs};
    if (is_dory_4b(arch, g)) return {AdmittedModel::Dory1_0_4B, AdmissionTier::Verified};
    if (is_inkling2_mini(arch, g)) return {AdmittedModel::Inkling2Mini, AdmissionTier::Optimized};
    // Anything libllama loads that the memory guard passed: no identity, neutral dialect.
    return {AdmittedModel::None, AdmissionTier::Runs};
}

const char* admission_tier_name(AdmissionTier t) {
    switch (t) {
        case AdmissionTier::Optimized: return "optimized for";
        case AdmissionTier::Verified: return "verified";
        default: return "runs";
    }
}

std::string tts_backbone_refusal(std::string_view arch, bool tts) {
    if (arch == "qwen3tts" && !tts)
        return "qwen3tts is an audio-generation backbone, not a language model: start with --tts for the speech routes";
    return {};
}

ChatDialect dialect_of(AdmittedModel m) {
    switch (m) {
        case AdmittedModel::MistralSmall24B: return ChatDialect::Extended;
        case AdmittedModel::Gemma4E4B: return ChatDialect::Gemma4;
        // Dory1.0-4B: ChatML framing, the plain think pair (the generation prompt ends inside the open
        // block when thinking is on, the template's default) and the Qwen coder's function/parameter
        // call form (`code`: the GGUF's template text; tests/test_chat_render.cpp, test_toolcall.cpp)
        case AdmittedModel::Dory1_0_4B: return ChatDialect::Qwen;
        // Inkling 2 Mini: its own message-block framing and JSON calls (core/dialect.h)
        case AdmittedModel::Inkling2Mini: return ChatDialect::Inkling;
        case AdmittedModel::None: return ChatDialect::Neutral;
        default: return ChatDialect::Qwen;
    }
}

bool runs_builds_renderer(AdmissionTier t) { return t != AdmissionTier::Runs; }

bool runs_may_use_tool_grammar(AdmissionTier t) { return t != AdmissionTier::Runs; }

std::string runs_no_chat_template_refusal() { return "no chat template: use /v1/completions"; }

std::string runs_start_warning(AdmissionTier t, int downgrade_keys, uint64_t downgrade_occurrences,
                              int splits_pp, int splits_tg) {
    if (t != AdmissionTier::Runs) return {};
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "admission: tier runs (no promise): %d downgrade key(s), %llu placement(s) on the CPU-side path, "
                  "graph splits %d prompt / %d token; renderer material and token mode off; the known dialect is kept "
                  "for a table-named model and the neutral one otherwise, which parses no tool calls unless the "
                  "template carries the Qwen form; --tool-grammar refused; a chat route without a template answers 400",
                  downgrade_keys, static_cast<unsigned long long>(downgrade_occurrences), splits_pp, splits_tg);
    return std::string(buf);
}

}  // namespace lgc

#include "core/admission.h"

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
// 262,144 tokens (`code`: the GGUF's metadata). The other Gemma 4 sizes are not
// admitted.
bool is_gemma4_e4b(std::string_view arch, const ModelGeometry& g) {
    return arch == "gemma4" && g.n_layer == 42 && g.n_embd == 2560 && g.n_head == 8 && g.n_head_kv == 2 &&
           g.head_dim == 512 && g.n_vocab == 262144;
}

// Granite 4.2 8B, the dense edition: 40 layers, 4096 wide, 32 query heads over
// 8 KV heads, 100,352 tokens (`code`: the GGUF's metadata; the file carries no
// key length, it is the 4096 / 32 = 128 of the head split). The 'granite'
// architecture also covers other sizes, which are not admitted.
bool is_granite_8b(std::string_view arch, const ModelGeometry& g) {
    return arch == "granite" && g.n_layer == 40 && g.n_embd == 4096 && g.n_head == 32 && g.n_head_kv == 8 &&
           g.n_vocab == 100352;
}

}  // namespace

std::string checkpoints_refusal(bool asked, bool can_snapshot, std::string_view arch) {
    if (!asked || can_snapshot) return {};
    return "--llama-checkpoints: a '" + std::string(arch) +
           "' model has no recurrent memory to snapshot (its attention cache trims); drop the flag or pass 0";
}

AdmittedModel admit_model(std::string_view arch, const ModelGeometry& g, bool tts) {
    if (arch == "qwen35") return AdmittedModel::Qwen35;
    if (arch == "qwen35moe") return AdmittedModel::Qwen35Moe;
    if (arch == "qwen4exp") return AdmittedModel::Qwen4Exp;
    if (arch == "qwen2vl") return AdmittedModel::Qwen2Vl;
    if (is_mistral_small_24b(arch, g)) return AdmittedModel::MistralSmall24B;
    if (is_llama2_7b(arch, g)) return AdmittedModel::Llama2_7B;
    if (is_gemma4_e4b(arch, g)) return AdmittedModel::Gemma4E4B;
    if (is_granite_8b(arch, g)) return AdmittedModel::Granite8B;
    if (tts && arch == "qwen3tts") return AdmittedModel::Qwen3Tts;
    return AdmittedModel::None;
}

ChatDialect dialect_of(AdmittedModel m) {
    switch (m) {
        case AdmittedModel::MistralSmall24B: return ChatDialect::Extended;
        case AdmittedModel::Gemma4E4B: return ChatDialect::Gemma4;
        default: return ChatDialect::Qwen;
    }
}

std::string admission_refusal(const std::string& gguf_path, std::string_view arch) {
    return gguf_path + " is a '" + std::string(arch) +
           "' model (or a size of one that is not admitted); --engine llama serves qwen35, qwen35moe, qwen4exp, "
           "qwen2vl (vision, with --mmproj), Mistral Small 3.2 24B ('llama', 40 x 5120, 32/8 heads), "
           "Llama 2 7B ('llama', 32 x 4096, 32/32 heads), Gemma 4 E4B ('gemma4', 42 x 2560, 8/2 heads, "
           "key length 512), Granite 4.2 8B ('granite', 40 x 4096, 32/8 heads) and, under --tts, qwen3tts";
}

}  // namespace lgc

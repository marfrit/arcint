#pragma once
#include <string>
#include <string_view>

#include "core/dialect.h"

namespace lgc {

// Which GGUFs the llama engine serves (DESIGN §2). Admission is by the GGUF's
// own `general.architecture` AND, for the families that are served at one size
// only, by the geometry: an untested size of an admitted architecture is
// refused, not guessed at. Pure: the backend reads the numbers from the loaded
// model and asks here, so every case is unit-tested with no model.
struct ModelGeometry {
    int n_layer  = 0;
    int n_embd   = 0;
    int n_head   = 0;
    int n_head_kv = 0;
    int head_dim = 0;    // the architecture's attention key length; 0 = not given
    int n_vocab  = 0;
};

enum class AdmittedModel {
    None,             // refused
    Qwen35,
    Qwen35Moe,
    Qwen4Exp,
    Qwen2Vl,
    MistralSmall24B,  // 'llama', 40 x 5120, 32/8 heads
    Llama2_7B,        // 'llama', 32 x 4096, 32/32 heads
    Gemma4E4B,        // 'gemma4', 42 x 2560, 8/2 heads, key length 512
    Granite8B,        // 'granite', see admission.cpp
    Qwen3Tts,         // only under --tts
};

AdmittedModel admit_model(std::string_view arch, const ModelGeometry& g, bool tts);

// --llama-checkpoints asked for on a model with nothing to snapshot: an attention
// cache trims, and only a recurrent memory (a hybrid's or a pure one) needs a
// snapshot, so the flag cannot do what it says. `asked` is the operator having
// passed a count above 0; the default of 32 is not a request. Empty when fine.
std::string checkpoints_refusal(bool asked, bool can_snapshot, std::string_view arch);

// The dialect a model speaks (think tags, tool-call form).
ChatDialect dialect_of(AdmittedModel m);

// The refusal text for a GGUF that is not admitted: names every admitted
// architecture and geometry, so an untested size reads as one.
std::string admission_refusal(const std::string& gguf_path, std::string_view arch);

}  // namespace lgc

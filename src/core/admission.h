#pragma once
#include <cstdint>
#include <string>
#include <string_view>

#include "core/dialect.h"

namespace lgc {

// Which GGUFs the llama engine serves, and with what promise (DESIGN §2, the
// admission tiers, operator 2026-11-23):
//
//   optimized for  arcint's own kernels or fast path, the release gates passed, numbers in the README.
//   verified       the generic path with KL, acceptance and a served check passed.
//   runs           anything libllama loads that passes the spill guard; no promise, a start warning.
//
// A measured size keeps its identity and its tier; an unmeasured size of a family, or any
// architecture not in the table, is "runs" (the hard refusal is gone). Admission is by the
// GGUF's own `general.architecture` AND, for the families served at one size only, by the
// geometry. Pure: the backend reads the numbers from the loaded model and asks here, so every
// case is unit-tested with no model.
struct ModelGeometry {
    int n_layer  = 0;
    int n_embd   = 0;
    int n_head   = 0;
    int n_head_kv = 0;
    int head_dim = 0;    // the architecture's attention key length; 0 = not given
    int n_vocab  = 0;
};

enum class AdmittedModel {
    None,             // no identity: an architecture not in the table (tier "runs", neutral dialect)
    Qwen35,
    Qwen35Moe,
    Qwen4Exp,
    Qwen2Vl,
    MistralSmall24B,  // 'llama', 40 x 5120, 32/8 heads
    Llama2_7B,        // 'llama', 32 x 4096, 32/32 heads
    Gemma4E4B,        // 'gemma4', 42 x 2560, 8/2 heads, key length 512
    Granite8B,        // 'granite', see admission.cpp
    Dory1_0_4B,       // 'dory', 72 physical blocks x 2560, 16/4 heads, key length 256 (core/dory.h)
    Inkling2Mini,     // 'inkling2_loop', 24 physical blocks x 2304, 24/8 heads, key length 128
    Qwen3Tts,         // only under --tts
};

enum class AdmissionTier {
    Runs,       // anything libllama loads that passes the memory guard; no promise
    Verified,   // the generic path, KL/acceptance/served check passed
    Optimized,  // arcint's own kernels or fast path, release gates passed
};

struct Admission {
    AdmittedModel model = AdmittedModel::None;
    AdmissionTier tier  = AdmissionTier::Runs;
};

Admission admit_model(std::string_view arch, const ModelGeometry& g, bool tts);

const char* admission_tier_name(AdmissionTier t);   // "optimized for", "verified", "runs"

// --llama-checkpoints asked for on a model with nothing to snapshot: an attention
// cache trims, and only a recurrent memory (a hybrid's or a pure one) needs a
// snapshot, so the flag cannot do what it says. `asked` is the operator having
// passed a count above 0; the default of 32 is not a request. Empty when fine.
std::string checkpoints_refusal(bool asked, bool can_snapshot, std::string_view arch);

// Qwen3-TTS is an audio-generation backbone, not a language model: it serves on the speech
// routes only under --tts, and without the flag the start is refused.
std::string tts_backbone_refusal(std::string_view arch, bool tts);

// The dialect a model speaks (think tags, tool-call form).
ChatDialect dialect_of(AdmittedModel m);

// The "runs" tier's refusals and neutral defaults, pure and unit-tested. They are one
// place so the load path and the tests cannot read the decision differently.
bool runs_builds_renderer(AdmissionTier t);   // false: no renderer material, token mode off
std::string runs_no_chat_template_refusal();   // "no chat template: use /v1/completions"
bool runs_may_use_tool_grammar(AdmissionTier t);   // only non-"runs" tiers may
// The start warning: the downgrade keys, their occurrences (the CPU-side placements), the
// graph splits of the prompt and token graphs, and the tier's own switches.
std::string runs_start_warning(AdmissionTier t, int downgrade_keys, uint64_t downgrade_occurrences,
                              int splits_pp, int splits_tg);

}  // namespace lgc

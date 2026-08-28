#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/chat.h"
#include "core/model_registry.h"
#include "core/sampling.h"

// The seam between "ligence owns the state" and "OV owns the math"
// (DESIGN.md §1). Everything above this interface — HTTP, scheduling, stop
// handling, tool-call parsing, usage accounting — is backend-agnostic and
// testable on a machine with no GPU in it.
namespace lgc {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int> encode(std::string_view text)        = 0;
    virtual std::string      decode(const std::vector<int>& ids)  = 0;
    virtual std::string      decode_one(int id)                   = 0;
    virtual int              eos_id() const                       = 0;
};

struct ModelStatus {
    std::string id;
    Quant       quant        = Quant::Q4;
    bool        loaded       = false;
    bool        stub         = false;
    int         n_ctx        = 0;
    int         n_layer      = 0;
    int         n_gdn_layer  = 0;
    int         n_attn_layer = 0;
    bool        mtp_enabled  = false;
    uint64_t    weights_bytes = 0;

    SamplerDefaults sampler_defaults;
};

// What the caller does with each emitted piece.
enum class Control {
    Continue,
    Stop,    // a stop sequence matched — finish_reason "stop"
    Cancel,  // the client is gone — abort at the next boundary (§3.7)
};

using TokenCallback = std::function<Control(std::string_view piece, int token_id)>;

enum class FinishReason { Stop, Length, Abort };

const char* finish_reason_name(FinishReason r);

struct GenerationInput {
    std::string   prompt;
    SamplerParams sampler;

    // Names of the tools this request declared. A real backend ignores this —
    // the tools already reached the model through the chat template — but the
    // stub backend needs them to synthesise a tool call so the parsing path can
    // be exercised without a GPU.
    std::vector<std::string> tool_names;
};

struct GenerationStats {
    int    prompt_tokens     = 0;
    int    completion_tokens = 0;
    int    cache_hit_tokens  = 0;
    int    draft_proposed    = 0;
    int    draft_accepted    = 0;
    // Wall time spent copying the whole model state so a rejected draft can be
    // rolled back. On the stateful path this is the price of speculation.
    double draft_rollback_seconds = 0.0;
    double draft_verify_seconds   = 0.0;   // the one pass that checks all drafts
    double draft_reforward_seconds = 0.0;  // re-running the accepted prefix after a reject
    double prefill_seconds   = 0.0;
    double decode_seconds    = 0.0;

    double prefill_rate() const {
        return prefill_seconds > 0.0 ? prompt_tokens / prefill_seconds : 0.0;
    }
    double decode_rate() const {
        return decode_seconds > 0.0 ? completion_tokens / decode_seconds : 0.0;
    }
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual const ModelStatus& status() const = 0;
    virtual Tokenizer&         tokenizer()    = 0;

    // Renders a chat request into the model's own prompt format. DESIGN.md §3.7
    // makes the artifact the single source of truth for the template, so this
    // is the backend's job: only it knows which template shipped with the
    // weights. The stub answers with ChatML because it has no artifact.
    virtual std::string render_chat(const ChatRequest& req) const = 0;

    virtual FinishReason generate(const GenerationInput& in, const TokenCallback& on_piece,
                                  GenerationStats& stats) = 0;
};

// M0: no OpenVINO, no weights, deterministic synthetic output.
// `delay_ms` inserts artificial per-token latency so that cancellation and
// streaming can be observed; zero for the fastest possible round-trip.
std::unique_ptr<Backend> make_stub_backend(const ModelEntry& entry, Quant quant, int n_ctx,
                                           int delay_ms = 0);

#ifdef LIGENCE_OPENVINO
struct Artifact;

// M1: the OpenVINO executor. Throws std::runtime_error with a readable message
// if the artifact cannot be compiled or validated.
struct Config;
std::unique_ptr<Backend> make_ov_backend(const Artifact& artifact, const Config& cfg, int n_ctx);
#endif

}  // namespace lgc

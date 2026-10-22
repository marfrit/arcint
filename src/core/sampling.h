#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// DESIGN.md §3.6. Greedy, temperature, top-k, top-p, min-p, repetition
// penalty (min-p since 0.5.9, for role play: operator, 2026-10-04). Defaults come from the served GGUF and the operator flags;
// explicit request fields always win.
namespace lgc {

// The sampler regime a request starts from, before its own fields apply.
struct SamplerDefaults {
    float temperature        = 0.7f;
    float top_p              = 0.8f;
    int   top_k              = 20;
    float min_p              = 0.0f;
    float repetition_penalty = 1.05f;
    float presence_penalty   = 0.0f;

    // clean-turns: the operator's switch for the think-block mask. Off keeps
    // today's behaviour; the artifact cannot decide it, so it is not read from
    // the GGUF.
    bool think_eog_mask = false;

    // clean-turns step 2: the operator's switch for the tool-call grammar.
    // Off keeps today's behaviour; on, a request that declared tools is
    // constrained to a well-formed call. A no-op when no tools are declared.
    bool tool_grammar = false;

    // "provisional" -- the Qwen3-family card values, carried because no
    //                  GGUF key is read yet; /props reports this so nobody
    //                  mistakes an inherited default for a measured one.
    // "operator"    -- at least one --temp/--top-p/... flag set it.
    std::string provenance = "provisional";
};

// What spells the think block, for the mask below. The ids come from the
// vocabulary; the tag strings come from think_tag_pairs() in reasoning.h,
// which is the only place they are written out anywhere in this tree.
struct ThinkBlockIds {
    std::vector<int> eog;        // the vocabulary's end-of-generation ids: the
                                 // block is not a place to end a turn. Together
                                 // with the request's own stop ids (SamplerParams::
                                 // stop_token_ids, same field the decode loop's
                                 // stop check reads) this is the set the mask
                                 // holds back while the block is open
    std::vector<int> open_ids;   // the block's opening marker id: a second one
                                 // while the block is open is the same defect
                                 // one step later
    std::vector<std::string> close_tags;  // the closers, matched as TEXT: a
                                          // model can spell them as ordinary
                                          // text (measured on the coder
                                          // artifact), so they cannot be
                                          // assumed to be single ids
    std::vector<std::string> open_tags;   // likewise the openers, for a model
                                          // that opens the block itself
};

struct SamplerParams {
    float    temperature        = 0.7f;
    float    top_p              = 0.8f;
    int      top_k              = 20;
    // llama.cpp's min-p: keep the tokens with p >= min_p * p_max, on the
    // logits before temperature (0 disables)
    float    min_p              = 0.0f;
    float    repetition_penalty = 1.05f;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
    uint64_t seed               = 0;
    bool     seeded             = false;
    int      max_tokens         = -1;  // -1: until EOS or the context limit
    bool     ignore_eos         = false;

    std::vector<std::string> stop;
    std::vector<int>         stop_token_ids;

    // clean-turns: while a think block is open, the end-of-turn token is not a
    // legal next token. Off keeps today's behaviour exactly. See ThinkBlockIds.
    bool think_eog_mask = false;
    ThinkBlockIds think_ids;

    // clean-turns step 2: constrain the tool-call form to the declared schemas
    // with a GBNF grammar, so a call the model starts is one it finishes. Off
    // keeps today's behaviour exactly; on, the backend builds the grammar from
    // the request's ToolSchemas and drives it around the sampler. The grammar
    // is a no-op when the request declared no tools.
    bool tool_grammar = false;

    // top_k == 1 collapses the distribution just as temperature 0 does; both
    // must take the argmax path, or "greedy" means two different things.
    bool greedy() const { return temperature <= 0.0f || top_k == 1; }
};

struct SamplerOverrides {
    std::optional<float>                    temperature;
    std::optional<float>                    top_p;
    std::optional<int>                      top_k;
    std::optional<float>                    min_p;
    std::optional<float>                    repetition_penalty;
    std::optional<float>                    presence_penalty;
    std::optional<float>                    frequency_penalty;
    std::optional<uint64_t>                 seed;
    std::optional<int>                      max_tokens;
    std::optional<bool>                     ignore_eos;
    std::optional<std::vector<std::string>> stop;
    std::optional<std::vector<int>>         stop_token_ids;
    std::optional<bool>                     think_eog_mask;
    std::optional<bool>                     tool_grammar;

    bool any() const;
};

SamplerParams sampler_from_defaults(const SamplerDefaults& d);

// The operator layer (config flags): overrides only the fields it sets and
// marks the defaults' provenance "operator". Same validation as a request,
// so an out-of-range flag fails at boot with the same message a client would
// get at 400.
std::optional<std::string> sampler_defaults_apply(SamplerDefaults& d, const SamplerOverrides& o);

// Applies the overrides in place. Returns an error message when a field is out
// of range, in which case `p` is left untouched.
std::optional<std::string> sampler_apply(SamplerParams& p, const SamplerOverrides& o);

}  // namespace lgc

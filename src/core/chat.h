#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/sampling.h"
#include "core/toolcall.h"

namespace lgc {

struct ToolSpec {
    std::string    name;
    std::string    description;
    nlohmann::json parameters;  // JSON Schema object, as declared by the client
};

// One media part of a message, as the OpenAI `image_url` part carries it: a
// `data:` URI (base64) or a local file path. The bytes are read in the
// backend, where the multimodal projector decides how to preprocess them
// (docs/milestone-1.2.0.md §3). More media types (audio input) are the
// transcription route's, not the chat's.
struct ImagePart {
    std::string url;
};

struct ChatMessage {
    std::string           role;
    std::string           content;   // text, with one media marker per image part
    std::vector<ImagePart> images;   // in marker order, across the whole request
    std::string           name;          // function name, for role == "tool"
    std::string           tool_call_id;  // for role == "tool"
    std::vector<ToolCall> tool_calls;    // assistant turns replayed back to us
};

struct ChatRequest {
    std::string              model;
    std::vector<ChatMessage> messages;
    std::vector<ToolSpec>    tools;
    std::string              tool_choice = "auto";

    // Set when tool_choice named one function explicitly; empty otherwise.
    std::string tool_choice_function;

    // OpenAI's parallel_tool_calls (default true). Read for the tool-call
    // grammar only (clean-turns step 2): false admits exactly one call.
    bool parallel_tool_calls = true;

    bool stream = false;

    // stream_options.include_usage. Defaults on: DESIGN.md §3.7 requires usage
    // in the final chunk, so opting out is the client's explicit choice.
    bool stream_include_usage = true;

    // chat_template_kwargs.enable_thinking — the fleet standard switch (§4).
    bool enable_thinking     = true;
    bool has_enable_thinking = false;

    SamplerOverrides sampler;
};

struct CompletionRequest {
    std::string      model;
    std::string      prompt;
    // A token-id prompt (`prompt: [760, 6511, ...]`, the OpenAI array form),
    // fed to the model as given, never re-tokenised. FULL-DEPTH (2026-09-13):
    // the KLD gate replays the reference capture's own token windows
    // (tools/kld_served.py), and a text round-trip through the tokenizer
    // cannot promise the same ids at a window boundary. One sequence, as
    // before; batches stay refused.
    std::vector<int> prompt_ids;
    bool             stream = false;
    bool             echo   = false;
    bool             stream_include_usage = true;
    SamplerOverrides sampler;
};

// One /v1/audio/speech body: OpenAI's shape (docs/speech-service.md). `voice`
// must be an enrolled name; the API-level refusal of any reference-audio field
// and the availability/unknown-voice checks live in prepare_speech (handlers),
// so this parser stays a body validator. `response_format` admits only "wav"
// today; a non-1 `speed` is refused rather than silently ignored (the served
// rate is the model's own).
struct SpeechRequest {
    std::string      model;
    std::string      text;                    // OpenAI's `input`
    std::string      voice;
    std::string      response_format = "wav";
    SamplerOverrides sampler;
};

// Both return an error message on rejection and leave the output untouched.
std::optional<std::string> parse_chat_request(const nlohmann::json& body, ChatRequest& out);
std::optional<std::string> parse_completion_request(const nlohmann::json& body,
                                                    CompletionRequest& out);
std::optional<std::string> parse_speech_request(const nlohmann::json& body,
                                                SpeechRequest& out);

// Parameter types from the declared tool schemas, for coercing the coder
// model's XML tool-call values (see core/toolcall.h).
ToolSchemas tool_schemas(const std::vector<ToolSpec>& tools);

// The media marker libmtmd replaces with the encoded media chunk
// (mtmd_default_marker(), mtmd.cpp). The renderer puts one into a message's
// content at each image part, so the model's own chat template carries it and
// mtmd_tokenize() finds it (`docs/milestone-1.2.0.md §3`).
inline constexpr const char* kMediaMarker = "<__media__>";

// M0 stub prompt rendering.
//
// This is NOT the model's chat template. DESIGN.md §3.7 makes the artifact the
// single source of truth for the template, and M1 replaces this with the jinja
// template shipped in the IR directory. Until then the stub backend needs some
// deterministic rendering to count tokens against, and ChatML is the family's
// shape. Never let this run against a real model.
std::string render_chatml_stub(const ChatRequest& req);
// The wire format carries a tool call's arguments as a JSON *string* (OpenAI
// does, and so does every client that follows it). Some templates iterate the
// arguments as a mapping -- Qwen3.6's does `tool_call.arguments|items` -- and
// fail on a string with "Can only get item pairs from a mapping". Handing the
// template what its contract asks for is input normalisation, not a polyfill:
// when the template requires an object and the string parses as one, the
// object goes in; anything else stays the string it was.
nlohmann::json tool_call_arguments_for_template(const std::string& arguments,
                                                bool template_wants_object);


}  // namespace lgc

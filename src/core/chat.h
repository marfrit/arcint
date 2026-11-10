#pragma once

#include <optional>
#include <string>
#include <string_view>
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
    // The assistant turn's reasoning (Open Responses reasoning items, or the
    // chat route's reasoning_content). Rendered only through the template's own
    // logic as `reasoning_content`; arcint applies no policy of its own
    // (docs/campaigns/feyntalk.md open question 2, operator decision 2026-10-31).
    std::string           reasoning;
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

    // arcint.timings: the request asks for the per-response timing object
    // (prompt_n/prompt_ms, predicted_n/predicted_ms, cache_n, draft_n,
    // draft_accepted) on the response. Absent keeps the body byte-identical.
    bool timings = false;

    // chat_template_kwargs.enable_thinking — the fleet standard switch (§4).
    bool enable_thinking     = true;
    bool has_enable_thinking = false;
    // chat_template_kwargs.preserve_thinking — the coder template replays an
    // assistant turn's thinking only after the last user query unless this is
    // true; a steering message moves that boundary and would otherwise drop
    // the running tool loop's thinking (measured in UAT: a 10.6 % cache hit).
    bool preserve_thinking     = false;
    bool has_preserve_thinking = false;
    // OpenAI's top-level reasoning_effort, as the client sent it (empty when it
    // sent none). template_context hands it to a template that reads the
    // `reasoning_effort` variable, rewritten only where the template cannot take it.
    std::string reasoning_effort;
    // The raw chat_template_kwargs object, passed through to the template (the
    // llama-server dialect: the keys are template-specific and a client may
    // send ones a given template ignores). `enable_thinking` and
    // `preserve_thinking` are type-checked above; every other key reaches
    // extra_context unchanged. Empty for a request that sent none, which keeps
    // the render byte-identical to before this field existed.
    nlohmann::json chat_template_kwargs = nlohmann::json::object();

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

// The names a chat_template_kwargs key must not take, because the vendored
// minja already owns them in the template context:
//   - the six keys apply() sets before it merges extra_context
//     (third_party/minja/chat-template.hpp): messages, add_generation_prompt,
//     bos_token, eos_token, strftime_now, tools;
//   - the 31 builtins Context::builtins() installs (third_party/minja/
//     minja.hpp), on a parent context, so a same-named key on the child
//     shadows them: ==, count, default, dictsort, e, equalto, escape, in,
//     indent, int, items, join, joiner, last, length, list, lower, map,
//     namespace, raise_exception, range, reject, rejectattr, safe, select,
//     selectattr, string, tojson, trim, unique, upper.
// A key that names one would let a client override the engine's value or break
// a template that calls the builtin, so it is refused by name.
bool reserved_template_context_key(std::string_view key);

// The `extra_context` a request produces: the typed switches
// (enable_thinking, preserve_thinking) plus every non-reserved
// chat_template_kwargs key, unchanged. Empty when the request set neither, so
// a render stays byte-identical to before the passthrough existed. `error`
// names the first reserved key found; the caller refuses with a 400.
// Pure: the one construction the llama render_chat and the unit test share.
//
// `template_src` is the model's chat template text. With it, a top-level
// reasoning_effort reaches a template that reads the variable as
// `reasoning_effort`: the client's level when the template quotes it, else the
// nearest it takes (minimal -> low, high -> xhigh, xhigh -> high), unless
// chat_template_kwargs sets the key itself; a null kwarg reads as absent and
// does not suppress the level. Without a source (the parse-time reserved-key
// check) no level is mapped. none/off and unknown levels pass no level.
// Enforced by test_chat.cpp's and test_chat_render.cpp's reasoning_effort_* cases.
nlohmann::json template_context(const ChatRequest& req, std::string& error,
                                std::string_view template_src = {});

// --chat-template-kwarg KEY=VALUE for a key other than enable_thinking: the
// unit default of one chat_template_kwargs key. VALUE is JSON when it parses as
// JSON (true, false, a number, a quoted string, an object or array) and the
// bare text as a string otherwise; `null` is refused (a null reads as absent
// everywhere else, so a default of null would mean nothing). A reserved name
// (reserved_template_context_key) is refused, and preserve_thinking must be a
// boolean, as in a request. Returns the error message, or nullopt with `key` and
// `value` set. Enforced by test_config.cpp's config_chat_template_kwarg_* cases.
std::optional<std::string> parse_template_kwarg_default(std::string_view kv, std::string& key,
                                                        nlohmann::json& value);

// Adds each configured default (key, JSON text) the request does not carry in
// its own chat_template_kwargs; the request's value always wins, an explicit
// false included, and a null in the request was dropped at parse time so it
// reads as absent. preserve_thinking lands in its typed field, the rest in the
// passthrough object. reasoning_effort is the exception: it is a level, so it
// lands in the typed ChatRequest::reasoning_effort (which template_context maps
// for the template, as it does a request's own) unless the request stated one by
// the top-level field, the Responses reasoning.effort or its kwargs. It also
// sets the thinking switch when no one else has, for every level: none/off turn
// thinking off, any other level turns it on (on a template that does not think by
// default, a non-off unit level makes every silent request a thinking turn, as a
// request's top-level reasoning_effort or Responses reasoning.effort does; a level
// a request sends inside chat_template_kwargs sets no switch). Enforced by test_chat.cpp's chat_template_kwarg_default_*.
void apply_template_kwarg_defaults(ChatRequest& req,
                                   const std::vector<std::pair<std::string, std::string>>& defaults);

// force_nonempty_content (NVIDIA's Dory1.0-4B model card tells agents to send
// chat_template_kwargs.force_nonempty_content=true; its chat template ignores
// the name, `measured-here`, so the server acts on it, as vLLM's Nemotron-V3
// reasoning parser does: `code`, vllm/parser/nemotron_v3.py
// _should_force_content / extract_reasoning, abstract_parser.py
// finalize_generation, main 1479deb0392f).
//
// The trigger is the effective kwarg (the request's, else the unit default of
// --chat-template-kwarg) being the JSON boolean true; the string "true" and the
// number 1 do not trigger (the reference tests `is True`).
bool force_nonempty_content_requested(const nlohmann::json& chat_template_kwargs);

// true when `text` is empty or only ASCII whitespace (space, \t \n \r \v \f): the
// ASCII part of Python's str.strip() set, the reference's blank test. A non-ASCII
// space (U+00A0, U+3000) counts as text here; a recorded difference (DESIGN 3.7).
bool text_is_blank(std::string_view text);

// The one decision all three response paths use (non-streaming, chat stream,
// Responses stream): true when the kwarg is set, the reasoning is not blank and
// the content is blank (text_is_blank). Then the answer is the reasoning text.
// Pure. Deviation from the reference: it also swaps when the reasoning is blank
// (content=None, reasoning=whitespace); there is nothing to promote then, so this
// does not. Enforced by test_force_content.cpp.
bool promote_reasoning_to_content(const nlohmann::json& chat_template_kwargs,
                                  std::string_view reasoning, std::string_view content);

// The levels of low/medium/high/xhigh a chat template takes for the
// reasoning_effort variable: the ones it quotes as a token, and only when it
// reads the variable at all (empty otherwise). /props publishes them.
std::vector<std::string> template_reasoning_effort_levels(std::string_view template_src);

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

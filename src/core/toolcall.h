#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/dialect.h"

// DESIGN.md §3.7: the models' native tool-call output is parsed server-side
// into OpenAI-shaped `tool_calls`. arcint parses, never executes.
//
// Two wire forms are handled, because the target set spans both:
//
//   <tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>   (Qwen chat)
//
//   <tool_call>                                                   (Qwen coder)
//   <function=f>
//   <parameter=a>
//   1
//   </parameter>
//   </function>
//   </tool_call>
//
// Callers must not invoke the parser at all when the request declared no
// tools: a parser that eats tags on tool-less requests makes the content
// silently vanish, which the fleet's proxy learned the expensive way.
namespace lgc {

struct ChatRequest;

struct ToolCall {
    // Deterministic ("call_N") rather than random, so the same request gives
    // the same ids. The parser numbers a turn's calls from call_0; a route that
    // answers a request renumbers them after its history (number_calls below),
    // so ids stay unique across the turns of a conversation.
    std::string id;
    std::string name;
    std::string arguments;  // compact JSON object, as OpenAI defines the field
};

// tool name -> (parameter name -> JSON type from the declared schema).
using ParamTypes  = std::map<std::string, std::string>;
using ToolSchemas = std::map<std::string, ParamTypes>;

struct ToolCallParse {
    // The text with the parsed blocks removed, untrimmed: it is exactly the
    // concatenation of the segments outside the <tool_call> blocks, so a
    // streaming caller that already sent the text before the first block can
    // send the remainder as a plain suffix.
    std::string content;
    std::vector<ToolCall> calls;

    // An opening <tool_call> was never closed. The unterminated remainder is
    // left in `content` verbatim rather than dropped.
    bool truncated = false;

    // Why a block was returned as raw content, for the log line
    // (tool_call_reject_message). Set by the Inkling parser, whose `truncated`
    // covers five different rejects of a block that may well be terminated:
    // no `{` after the call marker, an object that never closes (unbalanced
    // braces or an unclosed string), a JSON parse error (its message), a name /
    // args mismatch, no block closer right after the object. `reject_offset`
    // is the byte of the failure counted from the block's header. Empty / npos
    // for the other dialects, whose only reject is a missing closer.
    std::string reject_reason;
    size_t      reject_offset = std::string::npos;
};

// The log line's text for a parse with `truncated` set, without the
// "returned as raw content" tail the callers add: the Qwen and Gemma forms
// keep "unterminated <opener> block"; a parse that names its reason says
// "<opener> block rejected: <reason> (at byte N of the block)".
std::string tool_call_reject_message(const ToolCallParse& parsed, ChatDialect dialect);

// `schemas` is optional. When present, the XML form's string parameter values
// are coerced to the declared JSON types; without it every value stays a
// string, which is the honest reading of the wire format.
ToolCallParse parse_qwen_tool_calls(std::string_view text, const ToolSchemas* schemas = nullptr);

// Gemma 4's form (ChatDialect::Gemma4): its own tool-call marker pair around
// the word call, a colon, the function name and a brace-delimited list of
// key:value pairs. A string value sits between two string delimiters (single
// special tokens, core/markers.h), a number, a boolean or null stands bare, and
// an object or an array nests in the same form. Keys are bare words (a key in
// the string delimiter is read too). Where `schemas` declares a parameter a
// string, a bare value is kept as the string it was written as.
// Dispatches on the dialect: Gemma4 takes the form above, every other dialect
// the Qwen forms, so a caller passes the loaded model's dialect and nothing else.
ToolCallParse parse_tool_calls(std::string_view text, ChatDialect dialect, const ToolSchemas* schemas = nullptr);

// Tool call ids unique across turns. A template may look a tool result's
// function name up by its tool_call_id over the whole history (Inkling 2 Mini's
// does, and concatenates every match): two turns that both minted call_0 then
// render a header naming both functions, which rewrites earlier prompt text and
// defeats the lane's prefix reuse. The rule, one for every route that emits
// ids: a response's calls are numbered after the highest call_N already in the
// request's history (assistant tool_calls ids and tool messages' tool_call_id;
// the Responses route maps its function_call / function_call_output call_ids
// onto these fields). An id not of the form call_<decimal digits> is ignored
// for the offset. Deterministic: the same request gives the same ids.
//
// next_call_index: 0 for a history without any call_N id, else max(N) + 1.
size_t next_call_index(const ChatRequest& req);
// Sets calls[i].id = "call_{base + i}".
void number_calls(std::vector<ToolCall>& calls, size_t base);

// The marker that starts / ends a call in this dialect: the streaming callers
// hold the text back from the opener on, and need it spelled once, from here.
std::string_view tool_call_open(ChatDialect dialect);
std::string_view tool_call_close(ChatDialect dialect);

// The coder's call tags, written once here so the parser and the grammar
// (clean-turns step 2) cannot drift. The open/close are single special tokens
// on the coder vocabulary (measured); the inner tags are spelled as text.
struct ToolCallTags {
    std::string_view open;      // the tool-call opener
    std::string_view close;     // the tool-call closer
    std::string_view fn_open;   // the function opener prefix
    std::string_view fn_close;  // the function closer
    std::string_view param_open; // the parameter opener prefix
    std::string_view param_end;  // the parameter closer
};
const ToolCallTags& tool_call_tags();

}  // namespace lgc

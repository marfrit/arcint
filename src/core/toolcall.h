#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

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

struct ToolCall {
    // Deterministic ("call_0", "call_1", ...) rather than random. Greedy output
    // must be byte-identical run to run (§3.4); a random id would break that in
    // the response body even when the tokens matched.
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
};

// `schemas` is optional. When present, the XML form's string parameter values
// are coerced to the declared JSON types; without it every value stays a
// string, which is the honest reading of the wire format.
ToolCallParse parse_qwen_tool_calls(std::string_view text, const ToolSchemas* schemas = nullptr);

}  // namespace lgc

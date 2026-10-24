#pragma once

#include <string>
#include <vector>

#include "core/chat.h"

namespace lgc {

// The GBNF grammar that constrains the coder's tool-call form to the declared
// schemas (clean-turns step 2). Built from the pinned llama.cpp's PEG parser
// for exactly this model (common/parsers/qwen3-coder.cpp, vendored into
// third_party/llama-peg), so the call block is well-formed: every tag closed,
// the function name drawn from the declared set, required parameters present
// in any order, and each parameter value typed to its declared schema.
//
// The grammar is a LAZY trigger on the call opener: prose before the call is
// unconstrained, the call block is constrained, and the turn ends after the
// call (a completed GBNF rejects everything but the turn-ending token, which
// is the tool-calling protocol). The caller drives it as a lazy grammar
// sampler, triggered on the call opener.
//
// `parallel`: more than one call per turn (the request's parallel_tool_calls,
// the OpenAI default true), as the reference builds `calls`.
//
// Returns an empty string when no tools are declared: the caller then applies
// no grammar, and a tool-less request is never constrained. Throws
// std::invalid_argument when a declared schema cannot be constrained (not an
// object, or one the vendored closure refuses): the client's error.
std::string generate_tool_grammar(const std::vector<ToolSpec>& tools, bool parallel = true);

// The call opener: the lazy grammar's trigger word (the reference's
// tool_call_starts for a template with a think block). The caller makes it a
// trigger TOKEN when the vocabulary holds it as one token, else a regex-escaped
// pattern, as llama.cpp's common_sampling_add_grammar_triggers does.
std::string tool_grammar_trigger();

}  // namespace lgc

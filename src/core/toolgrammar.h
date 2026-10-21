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
// Returns an empty string when no tools are declared: the caller then applies
// no grammar, and a tool-less request is never constrained.
std::string generate_tool_grammar(const std::vector<ToolSpec>& tools);

// The call opener, as the lazy grammar's trigger pattern (regex-escaped, as
// llama.cpp's common_sampling_add_grammar_triggers does for a WORD trigger).
// Empty when generate_tool_grammar would return empty (no tools).
std::string tool_grammar_trigger();

}  // namespace lgc

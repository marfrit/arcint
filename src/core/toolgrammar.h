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
// One deviation from that reference: the parameter closer. The reference
// takes only the canonical "\n</parameter>\n"; here `</parameter>` closes a
// value with or without the newline before it, and an optional newline
// follows, so a free-text value ends at its first `</parameter>`, exactly
// where the parser (core/toolcall.h) ends it, and cannot carry that text.
// Reason (measured-here, a served raw trace, AREX-2 27B, --tool-grammar):
// the model closes short values inline (`finding</parameter>`); under the
// canonical-only closer the rest of the call stayed in the value, the grammar
// never completed, the end of generation stayed masked, and the turn ran to
// max_tokens or repeated the call. The canonical form is accepted as before.
// Cells: tests/test_toolgrammar.cpp, toolgrammar_llama_inline_closer_*,
// toolgrammar_llama_mixed_closers_in_one_call and
// toolgrammar_llama_canonical_closer_unchanged (ARCINT_LLAMA builds).
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

#pragma once

#include <functional>
#include <string>
#include <string_view>
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
// Two deviations from that reference. The first is the parameter closer. The
// reference takes only the canonical "\n</parameter>\n"; here `</parameter>` closes a
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
// The second, an operator decision 2026-11-09: a string parameter whose
// strings are listed takes only those strings. The reference's arg_string
// leaves every string value free text up to the closer, so a string `enum` is
// not enforced there. Here a value whose schema lists its strings -- an enum's
// string members, a const string, also inside anyOf / oneOf, a "type" array
// such as ["string","null"], or a $ref -- is exactly one of them, then the
// closer above. A listed string is raw text, as every value in this call
// form: `<` and newlines in it are written as is, a string ending in a
// newline takes the canonical closer only. Free stays: a schema with any
// unlisted string branch (a plain string, allOf, no type), a schema declared
// `nullable: true`, and a list holding the closer text itself (</parameter>,
// </function>, </tool_call>) or a NUL, which the parser could not read back or
// the grammar text cannot carry. Non-string alternatives (null, a number, ...)
// keep their JSON values; in a mixed enum only its string members are listed. Reason
// (measured-here, served, AREX-2, --tool-grammar): a call arrived as
// `"kind":"<parameter=class>\nunclear"` -- the model opened `kind`, wrote no
// value and opened the next parameter, which a free-text value takes as data.
// Cells: tests/test_toolgrammar.cpp, toolgrammar_llama_enum_* and the
// out-of-enum line of toolgrammar_llama_refuses_the_malformed_calls
// (ARCINT_LLAMA builds).
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

// Inkling 2 Mini's call form (ChatDialect::Inkling), modelled on mainline
// llama.cpp's common/parsers/inkling.cpp (commit 690b87d8, 2026-11-21:
// common_chat_params_init_inkling, `open` / `args` / the lazy trigger). One
// block per call:
//
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
//
// The grammar starts at NAME (the trigger below matches the header and the
// pattern's capture group starts at the name, as the reference's does): NAME
// is one of the declared tools' names, the object's prefix is the template's
// compact `{"name":<NAME as JSON>,"args":`, `args` is a JSON object, then `}`
// and the block closer. With `parallel` (the request's parallel_tool_calls)
// another `<|message_model|>` + call may follow, as the reference's `calls`;
// the turn then ends (a complete grammar admits only the end-of-generation
// token).
//
// Deviations from the reference:
//   - a JSON string admits no raw control byte 0x00-0x1F (`\t`, `\n` and the
//     other escapes stay legal). The reference builds args from its PEG
//     builder's p.json(), whose string rule in GBNF is `[^"\\]` plus the
//     escapes (common/peg-parser.cpp, the common_peg_string_parser case), so a
//     raw TAB passes its grammar and then fails every JSON parser.
//     `measured-here` (served, Inkling 2 Mini, twice on 2026-11-26): a bash
//     call whose "command" held tab-indented C copied raw into a heredoc was
//     correctly framed but returned as raw content ("Invalid control
//     character"). json-schema-to-grammar's own `char` rule excludes the
//     control bytes as here. A \uXXXX escape may not be a lone surrogate
//     (nlohmann refuses one); the reference admits any four hex digits.
//   - `args` is a JSON object, not any JSON value: the template writes an
//     object (it refuses anything else) and the parser (core/toolcall.cpp)
//     takes only an object.
//   - whitespace around the header's name: the parser trims it (text::trim),
//     so the trigger skips whitespace after the header and the grammar admits
//     it between the name and the call marker. The reference's trigger fires
//     on the first byte after the header, so a leading space starts its
//     grammar on a byte the grammar refuses, and the call runs unconstrained
//     (a desync here); a header followed by whitespace and then a content
//     marker is still no call.
// The reference also adds an anchored "[^<]+" trigger for Inkling 1, whose
// generation prompt ends with the header; Inkling 2 Mini's template has no
// generation prompt (the model writes the header), and the backend refuses
// the grammar for a prompt whose last tokens are the header
// (prompt_ends_with_inkling_header below), since the think-block tracker that
// keeps the grammar dormant while the model thinks (operator, 2026-11-26)
// does not see a header the prompt wrote.
//
// Known limit: the grammar is dormant inside the FIRST thinking block of a
// turn only. The think-block tracker (core/sampler.h) settles to content
// when that block closes and does not re-open, so a later thinking block's
// text reaches the trigger: a call the model spells there starts the
// grammar inside reasoning. Still stricter than the reference, which feeds
// every block. Counted, not changed (operator, 2026-11-26): the backend logs
// one line and counts it on the request's decode line when the grammar
// triggers in a turn that opened a thinking block after its content began
// (InklingTriggerWatch below).
//
// `reference_strings`: the reference's string rule instead of the strict one,
// for the red arm of the tests only.
//
// Throws std::invalid_argument for a tool name the form cannot carry: empty,
// starting with `<` (the trigger needs a character other than `<` after the
// header), with whitespace at either end (the parser reads the header name
// trimmed), or holding a NUL or the call marker. Returns an empty string when
// no tools are declared.
std::string generate_inkling_tool_grammar(const std::vector<ToolSpec>& tools, bool parallel = true,
                                          bool reference_strings = false);

// The lazy trigger, after the reference's COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
// `<\|message_model\|>([^<])`: the header, optional whitespace, then a byte
// that is neither `<` (a content marker after the header is a reasoning or
// text block, not a call) nor whitespace. The capture group marks where the
// grammar starts: the name.
std::string inkling_tool_grammar_trigger_pattern();

// Whether a prompt's last tokens spell the assistant header, layout after it
// ignored: `piece` renders one token with its special text (the header is one
// special token in Inkling's vocabulary, several text tokens elsewhere). Read
// from the token ids, so a prompt given as ids (no text) is judged too.
bool prompt_ends_with_inkling_header(const std::vector<int>& ids, const std::function<std::string(int)>& piece);

// The audit for the known limit above: fed the pieces the grammar is fed (the
// tokens outside the first thinking block), it follows the trigger's rule
// and remembers whether a thinking block opened after content began (an
// <|end_message|> already fed). late_trigger(): the trigger fired after such
// a block opened -- the case the dormancy does not cover.
class InklingTriggerWatch {
public:
    void feed(std::string_view piece);
    bool triggered() const { return triggered_; }
    bool late_think() const { return late_think_; }
    bool late_trigger() const { return late_trigger_; }

private:
    std::string tail_;
    bool        content_begun_ = false;
    bool        late_think_    = false;
    bool        after_header_  = false;
    bool        triggered_     = false;
    bool        late_trigger_  = false;
};

}  // namespace lgc

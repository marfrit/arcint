#pragma once
#include <string_view>

namespace lgc {

// The chat-template markers this engine parses or renders, in one place and
// assembled from parts. No source file spells one out: a marker typed as one
// string literal in a parser or a renderer is a defect class of its own (a
// test asserting it, a fixture carrying it, a comment teaching the next
// reader). tests/test_markers.cpp scans src/, tests/ and docs/ for a literal.
//
// Parts, not a full literal: "<" "thi" "nk" ">" compiles to the same bytes,
// while the source text never contains the marker. Keep this property when
// adding one.
namespace markers {

// The Qwen pair the split, the sampler's think mask and the stub template use.
inline constexpr std::string_view kThinkOpen  = "<" "thi" "nk" ">";
inline constexpr std::string_view kThinkClose = "</" "thi" "nk" ">";

// The Mistral-family pairs, honoured only under ModelStatus::dialect.
inline constexpr std::string_view kThinkOpenAngle  = "<" "think" "ing" ">";
inline constexpr std::string_view kThinkCloseAngle = "</" "think" "ing" ">";
inline constexpr std::string_view kThinkOpenSquare  = "[" "THINK" "]";
inline constexpr std::string_view kThinkCloseSquare = "[" "/THINK" "]";

// Gemma 4's reasoning channel, honoured only under ChatDialect::Gemma4. The
// opener is the channel-start marker, the word thought and a newline; the
// closer is the channel-end marker. Both markers are single special tokens in
// the vocabulary (`code`: the GGUF's token table), the word and the newline
// are ordinary text.
inline constexpr std::string_view kGemmaThoughtOpen  = "<" "|chan" "nel>" "thought" "\n";
inline constexpr std::string_view kGemmaThoughtClose = "<chan" "nel|>";

// Gemma 4's tool-call pair, and the delimiter its call form wraps a string
// value in: the call is the opener, the word call, a colon, the function name,
// a brace-delimited list of key:value pairs, and the closer.
inline constexpr std::string_view kGemmaToolOpen  = "<" "|tool" "_call>";
inline constexpr std::string_view kGemmaToolClose = "<tool" "_call|>";
inline constexpr std::string_view kGemmaStrDelim  = "<" "|\"" "|>";


// Inkling 2 Mini's message framing. Every block of a turn is a header, a
// content marker, the body and the shared closer; the assistant turn is one
// block per reasoning/text/tool-call part. `...Model` is the assistant header,
// `...Think` the reasoning content marker (its opener is that header and this
// marker together), `...Text` the answer marker, `...ToolJson` the call
// marker, and `...End` closes every block. `...ModelEnd` is the vocabulary's
// end-of-generation token.
inline constexpr std::string_view kInkMessageModel  = "<" "|message_" "model" "|>";
inline constexpr std::string_view kInkMessageUser   = "<" "|message_" "user" "|>";
inline constexpr std::string_view kInkMessageSystem = "<" "|message_" "system" "|>";
inline constexpr std::string_view kInkMessageTool   = "<" "|message_" "tool" "|>";
inline constexpr std::string_view kInkContentThink  = "<" "|content_" "thinking" "|>";
// The reasoning block's opener: the assistant header and the content marker in
// one string, because a model output spells both before its reasoning.
inline constexpr std::string_view kInkThinkOpen = "<" "|message_" "model|><|content_" "thinking" "|>";
inline constexpr std::string_view kInkContentText   = "<" "|content_" "text" "|>";
inline constexpr std::string_view kInkContentImage  = "<" "|content_" "image" "|>";
inline constexpr std::string_view kInkContentToolJson = "<" "|content_" "invoke_tool_json" "|>";
inline constexpr std::string_view kInkEndMessage    = "<" "|end_" "message" "|>";
inline constexpr std::string_view kInkModelEnd      = "<" "|content_" "model_end_sampling" "|>";

// ChatML's turn framing: the start-of-turn and end-of-turn markers.
inline constexpr std::string_view kTurnStart = "<" "|im_" "start" "|>";
inline constexpr std::string_view kTurnEnd   = "<" "|im_" "end" "|>";

}  // namespace markers
}  // namespace lgc

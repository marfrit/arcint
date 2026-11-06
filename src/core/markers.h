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

// ChatML's turn framing: the start-of-turn and end-of-turn markers.
inline constexpr std::string_view kTurnStart = "<" "|im_" "start" "|>";
inline constexpr std::string_view kTurnEnd   = "<" "|im_" "end" "|>";

}  // namespace markers
}  // namespace lgc

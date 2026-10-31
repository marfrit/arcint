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

// The Mistral-family pairs, honoured only under ModelStatus::think_tags_extended.
inline constexpr std::string_view kThinkOpenAngle  = "<" "think" "ing" ">";
inline constexpr std::string_view kThinkCloseAngle = "</" "think" "ing" ">";
inline constexpr std::string_view kThinkOpenSquare  = "[" "THINK" "]";
inline constexpr std::string_view kThinkCloseSquare = "[" "/THINK" "]";

// ChatML's turn framing: the start-of-turn and end-of-turn markers.
inline constexpr std::string_view kTurnStart = "<" "|im_" "start" "|>";
inline constexpr std::string_view kTurnEnd   = "<" "|im_" "end" "|>";

}  // namespace markers
}  // namespace lgc

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lgc::text {

std::string_view ltrim(std::string_view s);
std::string_view rtrim(std::string_view s);
std::string_view trim(std::string_view s);

bool iequals(std::string_view a, std::string_view b);

std::vector<std::string_view> split(std::string_view s, char sep);

// "15.9 GiB", "380 MiB" — one decimal below 100, none above, matching the
// memory-map lines in DESIGN.md §4.
std::string human_bytes(uint64_t bytes);

// Longest k with 0 < k < stop.size() such that `text` ends with the first k
// bytes of `stop`. Zero when no proper prefix of `stop` closes `text`.
//
// This is what makes streaming stop sequences correct: a chunk ending in the
// first half of a stop string must be held back, or the stop string leaks out
// one piece at a time.
size_t partial_stop_suffix(std::string_view text, std::string_view stop);

}  // namespace lgc::text

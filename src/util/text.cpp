#include "util/text.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace lgc::text {
namespace {

inline bool is_space(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

inline char lower(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

}  // namespace

std::string_view ltrim(std::string_view s) {
    size_t i = 0;
    while (i < s.size() && is_space(s[i])) ++i;
    return s.substr(i);
}

std::string_view rtrim(std::string_view s) {
    size_t n = s.size();
    while (n > 0 && is_space(s[n - 1])) --n;
    return s.substr(0, n);
}

std::string_view trim(std::string_view s) { return rtrim(ltrim(s)); }

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (lower(a[i]) != lower(b[i])) return false;
    }
    return true;
}

std::vector<std::string_view> split(std::string_view s, char sep) {
    std::vector<std::string_view> out;
    size_t start = 0;
    while (true) {
        const size_t pos = s.find(sep, start);
        if (pos == std::string_view::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
}

std::string human_bytes(uint64_t bytes) {
    static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};

    double value = static_cast<double>(bytes);
    size_t unit  = 0;
    while (value >= 1024.0 && unit + 1 < sizeof(kUnits) / sizeof(kUnits[0])) {
        value /= 1024.0;
        ++unit;
    }

    char buf[64];
    if (unit == 0 || value >= 100.0) {
        std::snprintf(buf, sizeof(buf), "%.0f %s", value, kUnits[unit]);
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f %s", value, kUnits[unit]);
    }
    return buf;
}

size_t partial_stop_suffix(std::string_view text, std::string_view stop) {
    if (stop.empty()) return 0;

    const size_t max_k = std::min(text.size(), stop.size() - 1);
    for (size_t k = max_k; k > 0; --k) {
        if (text.substr(text.size() - k) == stop.substr(0, k)) return k;
    }
    return 0;
}

}  // namespace lgc::text

#pragma once

// Self-contained shim for the vendored llama.cpp PEG grammar closure
// (json, json-schema, json-schema-to-grammar, peg-parser, trie, unicode).
//
// The closure is copied verbatim from the pinned llama.cpp tree (MIT, see
// THIRD_PARTY.md) so arcint can build the Qwen-coder tool-call GBNF grammar
// without the llama.cpp tree at build time. It takes three things from
// llama.cpp's common.h / log.h / ggml.h that arcint does not otherwise link:
//
//   - GGML_ASSERT          (a plain assert)
//   - the llama_token type (int32_t, from include/llama.h)
//   - string_join / string_split / string_repeat (from common/common.cpp)
//
// This header provides exactly those, so the closure compiles against
// arcint's vendored nlohmann/json and nothing else. The string helpers are
// copied verbatim from common/common.cpp (MIT).

#include <cassert>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#ifndef GGML_ASSERT
#define GGML_ASSERT(x) assert(x)
#endif

// The closure's one logging call site (peg-parser.cpp's debug dump path) uses
// llama.cpp's LOG_DBG. arcint has no reason to print PEG debug, so it is a
// no-op here.
#ifndef LOG_DBG
#define LOG_DBG(...) ((void) 0)
#endif

// include/llama.h: typedef int32_t llama_token;
// Redeclaring the same typedef is legal C++, so this is safe even in a
// translation unit that also includes the real llama.h.
#include <cstdint>
using llama_token = int32_t;

// ------------------------------------------------------------------ string
// verbatim from llama.cpp common/common.cpp (MIT)

inline std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

inline std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end   = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end   = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

inline std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

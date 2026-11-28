// A stand-in for llama.cpp's common/log.h, for building the pinned tree's tests/test-llama-archs.cpp
// inside arcint's unit tests. arcint does not build llama.cpp's common library (CMakeLists.txt,
// LLAMA_BUILD_COMMON), and the test needs only these names from it.
#pragma once

#include "ggml.h"

#include <cstdio>

#define LOG_LEVEL_DEBUG  5
#define LOG_LEVEL_TRACE  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0

struct common_log;

inline struct common_log * common_log_main() { return nullptr; }
inline void common_log_flush(struct common_log *) { fflush(stdout); fflush(stderr); }
inline void common_log_set_verbosity_thold(int) {}

inline int common_log_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_NONE:  return LOG_LEVEL_OUTPUT;
        default:                   return LOG_LEVEL_INFO;
    }
}

#define LOG(...)     do { printf(__VA_ARGS__); } while (0)
#define LOG_INF(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_WRN(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_DBG(...) do { } while (0)

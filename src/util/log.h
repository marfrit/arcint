#pragma once

#include <cstdarg>
#include <string>
#include <string_view>

// Console output in the llama.cpp tradition (DESIGN.md §4): one line per
// event, greppable, stderr only, no colors.
//
//     lgc  load: qwen3.6-27b-a3b-coder q4 | 41 GDN + 7 attn layers
//     lgc  mem:  kv pool 4.2 GiB (2688 blocks à 32 tok)
//     lgc  slot 0: decode 592 tok in 9.86 s ( 60.0 t/s)
//
// The tag is padded to 5 columns so that short tags line their content up;
// longer tags (`slot 0:`) simply run past it, as in the design sketch.
namespace lgc::log {

enum class Level : int {
    Error   = 0,
    Warn    = 1,
    Info    = 2,
    Verbose = 3,  // -v
    Debug   = 4,  // -vv
};

void  set_level(Level lvl);
Level level();

// True when `lvl` would be printed. Use to guard expensive message building.
bool enabled(Level lvl);

// Writes exactly one line: "lgc  <tag>: <msg>\n". `tag` carries no colon.
void line(Level lvl, std::string_view tag, std::string_view msg);

void vlinef(Level lvl, std::string_view tag, const char* fmt, va_list ap);

void error(std::string_view tag, const char* fmt, ...)   __attribute__((format(printf, 2, 3)));
void warn(std::string_view tag, const char* fmt, ...)     __attribute__((format(printf, 2, 3)));
void info(std::string_view tag, const char* fmt, ...)     __attribute__((format(printf, 2, 3)));
void verbose(std::string_view tag, const char* fmt, ...)  __attribute__((format(printf, 2, 3)));
void debug(std::string_view tag, const char* fmt, ...)    __attribute__((format(printf, 2, 3)));

// Formats into a std::string with printf semantics. Exposed because several
// call sites build message fragments before deciding on a tag.
std::string vformat(const char* fmt, va_list ap);
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

}  // namespace lgc::log

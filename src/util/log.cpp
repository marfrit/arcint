#include "util/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <vector>

namespace lgc::log {
namespace {

std::atomic<Level> g_level{Level::Info};

// stderr is shared with every request thread; one mutex keeps lines intact.
std::mutex g_mutex;

constexpr std::string_view kPrefix   = "lgc  ";
constexpr size_t           kTagWidth = 5;  // "load:", "mem: " — see header

}  // namespace

void set_level(Level lvl) { g_level.store(lvl, std::memory_order_relaxed); }

Level level() { return g_level.load(std::memory_order_relaxed); }

bool enabled(Level lvl) {
    return static_cast<int>(lvl) <= static_cast<int>(level());
}

std::string vformat(const char* fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (n < 0) return {};

    std::string out(static_cast<size_t>(n), '\0');
    std::vsnprintf(out.data(), static_cast<size_t>(n) + 1, fmt, ap);
    return out;
}

std::string format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::string out = vformat(fmt, ap);
    va_end(ap);
    return out;
}

void line(Level lvl, std::string_view tag, std::string_view msg) {
    if (!enabled(lvl)) return;

    std::string buf;
    buf.reserve(kPrefix.size() + tag.size() + msg.size() + 4);
    buf.append(kPrefix);
    buf.append(tag);
    buf.push_back(':');
    for (size_t w = tag.size() + 1; w < kTagWidth; ++w) buf.push_back(' ');
    buf.push_back(' ');
    buf.append(msg);
    buf.push_back('\n');

    std::lock_guard<std::mutex> guard(g_mutex);
    std::fwrite(buf.data(), 1, buf.size(), stderr);
    std::fflush(stderr);
}

void vlinef(Level lvl, std::string_view tag, const char* fmt, va_list ap) {
    if (!enabled(lvl)) return;
    line(lvl, tag, vformat(fmt, ap));
}

#define LIGENCE_LOG_FN(name, lvl)                          \
    void name(std::string_view tag, const char* fmt, ...) { \
        if (!enabled(lvl)) return;                          \
        va_list ap;                                         \
        va_start(ap, fmt);                                  \
        vlinef(lvl, tag, fmt, ap);                          \
        va_end(ap);                                         \
    }

LIGENCE_LOG_FN(error, Level::Error)
LIGENCE_LOG_FN(warn, Level::Warn)
LIGENCE_LOG_FN(info, Level::Info)
LIGENCE_LOG_FN(verbose, Level::Verbose)
LIGENCE_LOG_FN(debug, Level::Debug)

#undef LIGENCE_LOG_FN

}  // namespace lgc::log

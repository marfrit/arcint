#include "api/raw_trace.h"

#include "api/handlers.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>

#include <nlohmann/json.hpp>

namespace lgc::api {

bool raw_trace_on() {
    const char* f = std::getenv("ARCINT_RAW_TRACE");
    return f != nullptr && f[0] != '\0';
}

std::string raw_trace_line(const RawTraceRecord& r, unsigned long long n) {
    const char* start = r.think_start.open() ? "open" : r.think_start.closed() ? "closed" : "undecided";
    nlohmann::ordered_json calls = nlohmann::ordered_json::array();
    if (r.calls != nullptr) {
        for (const ToolCall& c : *r.calls) {
            nlohmann::ordered_json call;
            call["name"]      = c.name;
            call["arguments"] = c.arguments;
            calls.push_back(std::move(call));
        }
    }
    // Insertion order is the documented field order (ordered_json), so a line
    // reads the same way every time.
    nlohmann::ordered_json line;
    line["trace"]       = "raw1";
    line["n"]           = n;
    line["request_id"]  = r.request_id;
    line["route"]       = r.route;
    line["stream"]      = r.stream;
    line["token_mode"]  = r.token_mode;
    line["finish"]      = r.finish;
    line["think_start"] = start;
    line["raw"]         = r.raw != nullptr ? *r.raw : std::string();
    line["reasoning"]   = r.reasoning;
    line["tool_parse"]  = r.tool_parse;
    line["tool_calls"]  = calls;
    line["content"]     = r.content;
    return line.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);
}

RawTraceRecord raw_trace_record(const PreparedChat& prep, const char* route, bool stream, const char* finish,
                                const std::string& raw) {
    RawTraceRecord r;
    r.route       = route;
    r.request_id  = prep.request_id;
    r.stream      = stream;
    r.finish      = finish;
    r.think_start = prep.think_start();
    r.raw         = &raw;
    return r;
}

void raw_trace_write(const RawTraceRecord& r) {
    if (!raw_trace_on()) return;
    static std::mutex         mu;
    static std::FILE*         f     = nullptr;
    static bool               tried = false;
    static unsigned long long n     = 0;
    std::lock_guard<std::mutex> lk(mu);
    if (f == nullptr && !tried) {
        tried            = true;
        const char* name = std::getenv("ARCINT_RAW_TRACE");
        f                = std::fopen(name, "a");
        if (f == nullptr) {
            std::fprintf(stderr, "arcint: cannot open ARCINT_RAW_TRACE=%s for the raw trace\n", name);
            return;
        }
    }
    if (f == nullptr) return;
    const std::string line = raw_trace_line(r, ++n);
    std::fprintf(f, "%s\n", line.c_str());
    std::fflush(f);
}

}  // namespace lgc::api

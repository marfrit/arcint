#include "api/raw_trace.h"

#include "api/handlers.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>
#include <utility>

#include <nlohmann/json.hpp>

namespace lgc::api {
namespace {

// Everything of the line after the counter: `"request_id":...}`. Built outside
// the writer's lock; only the counter and the write are serialised.
std::string line_tail(const RawTraceRecord& r) {
    using ojson       = nlohmann::ordered_json;
    const char* start = r.think_start.open() ? "open" : r.think_start.closed() ? "closed" : "undecided";
    ojson       calls = ojson::array();
    if (r.calls != nullptr) {
        for (const ToolCall& c : *r.calls) {
            ojson call;
            call["name"]      = c.name;
            call["arguments"] = c.arguments;
            calls.push_back(std::move(call));
        }
    }
    // Insertion order is the documented field order (ordered_json), so a line
    // reads the same way every time.
    ojson line;
    line["request_id"]   = r.request_id;
    line["route"]        = r.route;
    line["stream"]       = r.stream;
    line["token_mode"]   = r.token_mode;
    line["finish"]       = r.finish;
    line["think_start"]  = start;
    line["raw"]          = r.raw != nullptr ? *r.raw : std::string();
    line["reasoning"]    = r.reasoning;
    line["tool_parse"]   = r.tool_parse;
    line["parsed_input"] = r.parsed_input != nullptr ? ojson(*r.parsed_input) : ojson(nullptr);
    line["tool_calls"]   = calls;
    line["content"]      = r.content;
    line["error"]        = r.error.empty() ? ojson(nullptr) : ojson(r.error);
    std::string out = line.dump(-1, ' ', false, ojson::error_handler_t::replace);
    out.erase(0, 1);   // the leading brace: the head carries it
    return out;
}

std::string with_head(unsigned long long n, const std::string& tail) {
    return "{\"trace\":\"raw1\",\"n\":" + std::to_string(n) + "," + tail;
}

}  // namespace

bool raw_trace_on() {
    const char* f = std::getenv("ARCINT_RAW_TRACE");
    return f != nullptr && f[0] != '\0';
}

std::string raw_trace_line(const RawTraceRecord& r, unsigned long long n) { return with_head(n, line_tail(r)); }

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

void raw_trace_error(const PreparedChat& prep, const char* route, bool stream, const std::string& raw,
                     const char* message) noexcept {
    if (!raw_trace_on()) return;
    try {
        RawTraceRecord r = raw_trace_record(prep, route, stream, "error", raw);
        r.error          = message != nullptr && message[0] != '\0' ? message : "error";
        raw_trace_write(r);
    } catch (...) {
        // the trace is a diagnostic: it must not replace the route's error
    }
}

namespace {

void write_line(const RawTraceRecord& r) {
    static std::mutex         mu;
    static int                fd    = -1;
    static bool               tried = false;
    static unsigned long long n     = 0;
    const std::string         tail  = line_tail(r);   // outside the lock
    std::lock_guard<std::mutex> lk(mu);
    if (fd < 0 && !tried) {
        tried            = true;
        const char* name = std::getenv("ARCINT_RAW_TRACE");
        // O_APPEND: every write lands at the end of the file, whoever else
        // appends, so two processes on one file interleave whole lines.
        // 0600: the file holds generated text and call arguments, so a file
        // this process creates is its owner's alone (an existing one keeps
        // its mode).
        fd = ::open(name, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (fd < 0) {
            std::fprintf(stderr, "arcint: cannot open ARCINT_RAW_TRACE=%s for the raw trace\n", name);
            return;
        }
    }
    if (fd < 0) return;
    // One write() a line: atomic against other appenders on a regular file
    // (Linux takes the inode lock for an O_APPEND write). A short write is
    // finished with more writes, which only another appender could split.
    const std::string line = with_head(++n, tail) + "\n";
    size_t            done = 0;
    while (done < line.size()) {
        const ssize_t w = ::write(fd, line.data() + done, line.size() - done);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) {
            // failed, or wrote nothing (it would only repeat while the lock
            // is held): the trace ends here for this process
            std::fprintf(stderr, "arcint: the raw trace's write failed (%s); ARCINT_RAW_TRACE is off from here\n",
                         w < 0 ? std::strerror(errno) : "wrote 0 bytes");
            ::close(fd);
            fd = -1;
            return;
        }
        done += static_cast<size_t>(w);
    }
}

}  // namespace

void raw_trace_write(const RawTraceRecord& r) noexcept {
    if (!raw_trace_on()) return;
    try {
        write_line(r);
    } catch (...) {
        // the trace is a diagnostic: a line that cannot be built (out of
        // memory) is dropped, the turn's answer goes out as it would have
    }
}

}  // namespace lgc::api

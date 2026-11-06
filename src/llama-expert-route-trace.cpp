#include "llama-expert-route-trace.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

uint8_t llama_expert_route_graph_kind(uint32_t n_tokens, uint32_t decode_max, bool all_outputs, bool has_tokens) {
    const bool prompt = !all_outputs || !has_tokens || n_tokens > decode_max;
    return prompt ? LLAMA_EXPERT_ROUTE_PROMPT : (n_tokens == 1 ? LLAMA_EXPERT_ROUTE_DECODE : LLAMA_EXPERT_ROUTE_VERIFY);
}

bool llama_expert_route_trace_enabled() {
    // once: the graph builder asks on every MTP MoE layer
    static const bool on = [] {
        const char * p = getenv("LLAMA_EXPERT_ROUTE_TRACE");
        return p != nullptr && *p != '\0';
    }();
    return on;
}

// one WARN a process: the writer is dependency-free (a device-free test
// compiles this file alone), so it cannot use llama's logger
static void trace_warn(const char * what, const char * detail) {
    std::fprintf(stderr, "WARN: expert route trace: %s%s%s\n", what,
                 detail != nullptr ? ": " : "", detail != nullptr ? detail : "");
}

static uint64_t trace_max_bytes(bool & warned) {
    const char * p = getenv("LLAMA_EXPERT_ROUTE_TRACE_MAX_MB");
    // a record is a few KB; 256 MiB is a long span. Fractional values are for
    // the device-free rotation test (a cap in bytes, not in a serving run).
    // A non-numeric or non-positive value falls back to the default; 0 used to
    // mean "rotate every record", the defect this check removes.
    const double def = 256.0;
    if (p == nullptr || *p == '\0') {
        return (uint64_t) (def * 1048576.0);
    }
    char * end = nullptr;
    errno = 0;
    const double mb = std::strtod(p, &end);
    if (end == p || *end != '\0' || errno != 0 || !(mb > 0)) {
        if (!warned) {
            warned = true;
            char msg[96];
            std::snprintf(msg, sizeof(msg), "LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=%s is not a positive number", p);
            trace_warn("falling back to 256 MB", msg);
        }
        return (uint64_t) (def * 1048576.0);
    }
    return (uint64_t) (mb * 1048576.0);
}

void llama_expert_route_trace::put_u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        buf.push_back((uint8_t) ((v >> (8 * i)) & 0xff));
    }
}

void llama_expert_route_trace::put_i32(int32_t v) {
    put_u32((uint32_t) v);
}

void llama_expert_route_trace::flush() {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!opened || buf.empty()) {
        buf.clear();
        return;
    }
    // one write a record; the file was opened O_WRONLY|O_EXCL with one writer
    // (the trace's mu), never O_APPEND, so a short write is resumed in the
    // loop below instead of trusting a page-atomic write
    size_t off = 0;
    while (off < buf.size()) {
        const ssize_t n = ::write(fd, buf.data() + off, buf.size() - off);
        if (n <= 0) {
            if (errno == EINTR) {
                continue;
            }
            const int err = errno;
            opened = false;   // a full disk or a closed fd: stop tracing, do not fail the run
            buf.clear();
            if (!warned) {
                warned = true;
                char msg[160];
                std::snprintf(msg, sizeof(msg), "cannot write %s: %s; tracing stops",
                              path.c_str(), std::strerror(err));
                trace_warn("the file can no longer be written", msg);
            }
            return;
        }
        off += (size_t) n;
        bytes_in_file += (uint64_t) n;
    }
    buf.clear();
}

static std::string trace_file_name(const std::string & base, bool dir_mode, uint32_t seq) {
    if (dir_mode) {
        char tail[32];
        std::snprintf(tail, sizeof(tail), ".%03u.ert1", seq);
        return base + tail;
    }
    if (seq == 0) {
        return base;
    }
    char tail[32];
    std::snprintf(tail, sizeof(tail), ".%03u", seq);
    return base + tail;
}

std::string llama_expert_route_trace_sidecar() {
    auto & t = llama_expert_route_trace::instance();
    std::lock_guard<std::recursive_mutex> lk(t.mu);
    if (!t.opened) {
        return {};
    }
    return trace_file_name(t.path, t.dir_mode, t.seq) + ".accept";
}

void llama_expert_route_trace::close() {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (fd >= 0) {
        ::close(fd);
    }
    fd = -1;
    opened = false;
    buf.clear();
}

// a fresh name only: O_EXCL, advance the sequence past anything already there.
// Nothing is ever appended, so a restart cannot put a second header in an old
// file and the replay never meets one mid-file.
static int trace_open_fresh(const std::string & base, bool dir_mode, uint32_t & seq, uint64_t & bytes) {
    for (uint32_t guard = 0; guard < 10000; ++guard) {
        const std::string f = trace_file_name(base, dir_mode, seq);
        const int fd = ::open(f.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd >= 0) {
            bytes = 0;
            return fd;
        }
        if (errno != EEXIST) {
            return -1;
        }
        ++seq;
    }
    errno = EEXIST;
    return -1;
}

bool llama_expert_route_trace::open(const char * p, uint32_t nl, uint32_t ne, uint32_t nu) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    close();
    path      = p;
    seq       = 0;
    bytes_in_file = 0;
    n_layer   = nl;
    n_expert  = ne;
    n_used    = nu;
    warned    = false;
    max_bytes = trace_max_bytes(warned);
    struct stat st;
    const bool exists = ::stat(p, &st) == 0;
    dir_mode = exists && S_ISDIR(st.st_mode);
    if (!exists) {
        // the flag is a DIR: a missing one is created, or refused loudly. It is
        // never silently treated as a base file path (the round-6 defect).
        if (::mkdir(p, 0755) != 0) {
            trace_warn("cannot create the trace directory", p);
            return false;
        }
        dir_mode = true;
    }
    if (dir_mode) {
        char base[4096];
        // the start time in the base: a reused pid after a restart opens a new
        // family, not the previous run's sequence
        std::snprintf(base, sizeof(base), "%s/expert-route-%d-%lld", p, (int) ::getpid(),
                      (long long) std::time(nullptr));
        path = base;
    }
    fd = trace_open_fresh(path, dir_mode, seq, bytes_in_file);
    if (fd < 0) {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "cannot open a fresh file for %s: %s", p, std::strerror(errno));
        trace_warn("tracing is off", msg);
        return false;
    }
    opened = true;
    buf.clear();
    buf.insert(buf.end(), {'E', 'R', 'T', '1'});
    put_u32(2);   // version: a graph carries its first position
    put_u32(n_layer);
    put_u32(n_expert);
    put_u32(n_used);
    flush();
    return opened;
}

bool llama_expert_route_trace_rotate_now(const llama_expert_route_trace & t) {
    if (!t.opened || t.pending_graphs > 0) {
        return false;
    }
    return t.at_cap();
}

bool llama_expert_route_trace::rotate_next() {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!opened) {
        return false;
    }
    close();
    fd = trace_open_fresh(path, dir_mode, seq, bytes_in_file);
    if (fd < 0) {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "cannot open the next file for %s: %s", path.c_str(), std::strerror(errno));
        trace_warn("tracing stops", msg);
        return false;
    }
    opened = true;
    buf.clear();
    buf.insert(buf.end(), {'E', 'R', 'T', '1'});
    put_u32(2);
    put_u32(n_layer);
    put_u32(n_expert);
    put_u32(n_used);
    flush();
    return opened;
}

void llama_expert_route_trace::event_layers(const std::vector<int> & il, const std::vector<int> & n_slots,
                                            const std::vector<std::vector<int32_t>> & residents) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!opened) {
        return;
    }
    put_u8(0);
    put_u32((uint32_t) il.size());
    for (size_t i = 0; i < il.size(); ++i) {
        put_i32(il[i]);
        put_i32(n_slots[i]);
        for (int s = 0; s < n_slots[i]; ++s) {
            put_i32(s < (int) residents[i].size() ? residents[i][s] : -1);
        }
    }
    flush();
}

void llama_expert_route_trace::set_residents(intptr_t key, const std::vector<int> & il, const std::vector<int> & n_slots,
                                             const std::vector<std::vector<int32_t>> & residents) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    for (auto & s : layer_sets) {
        if (s.key == key) {
            s.il = il;
            s.n_slots = n_slots;
            s.residents = residents;
            return;
        }
    }
    layer_sets.push_back({ key, il, n_slots, residents });
}

void llama_expert_route_trace::event_layers_all() {
    std::lock_guard<std::recursive_mutex> lk(mu);
    std::vector<int> il, ns;
    std::vector<std::vector<int32_t>> res;
    for (const auto & s : layer_sets) {
        il.insert(il.end(), s.il.begin(), s.il.end());
        ns.insert(ns.end(), s.n_slots.begin(), s.n_slots.end());
        res.insert(res.end(), s.residents.begin(), s.residents.end());
    }
    event_layers(il, ns, res);   // the lock is recursive; one record either way
}

void llama_expert_route_trace::pending_graph(bool pending) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (pending) {
        ++pending_graphs;
    } else if (pending_graphs > 0) {
        --pending_graphs;
    }
}

void llama_expert_route_trace::event_graph(uint8_t kind, int32_t first_pos, uint32_t n_tokens,
                                           const std::vector<int32_t> & tokens,
                                           const std::vector<int> & il, const std::vector<std::vector<int32_t>> & ids) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!opened) {
        return;
    }
    put_u8(kind);
    put_i32(first_pos);
    put_u32(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        put_i32(i < tokens.size() ? tokens[i] : -1);
    }
    put_u32((uint32_t) il.size());
    for (size_t i = 0; i < il.size(); ++i) {
        put_i32(il[i]);
        put_u32((uint32_t) ids[i].size());
        for (int32_t e : ids[i]) {
            put_i32(e);
        }
    }
    flush();
}

void llama_expert_route_trace::event_tables(const std::vector<llama_expert_route_trace::change> & changed) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!opened || changed.empty()) {
        buf.clear();
        return;
    }
    put_u8(5);
    put_u32((uint32_t) changed.size());
    for (const auto & c : changed) {
        put_i32(c.il);
        put_i32(c.expert);
        put_u8(c.resident);
    }
    flush();
}

llama_expert_route_trace & llama_expert_route_trace::instance() {
    static llama_expert_route_trace t;
    return t;
}

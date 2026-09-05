#include "harness.h"

#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>

namespace t {
namespace {
int g_failures = 0;
bool g_skipped = false;
std::string g_skip_reason;
}  // namespace

std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}

int register_case(const char* name, void (*fn)()) {
    cases().push_back(Case{name, fn});
    return 0;
}

void fail(const char* file, int line, const std::string& msg) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s:%d\n           %s\n", file, line, msg.c_str());
}

void skip(const std::string& reason) {
    g_skipped = true;
    g_skip_reason = reason;
    throw SkipSignal{};
}

int failures() { return g_failures; }
void reset_failures() { g_failures = 0; }

bool skipped() { return g_skipped; }
const std::string& skip_reason() { return g_skip_reason; }
void reset_skip() { g_skipped = false; g_skip_reason.clear(); }

}  // namespace t

namespace {

struct SkipRecord {
    std::string case_name;
    std::string reason;
};

void print_usage() {
    std::fprintf(stderr,
                  "usage: arcint-test [filter] [--max-skips N] [--allow-skip NAME]...\n");
}

}  // namespace

int main(int argc, char** argv) {
    const char* filter = nullptr;
    std::optional<long> max_skips;
    std::vector<std::string> allow_skip;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-skips") {
            if (i + 1 >= argc) { print_usage(); return 2; }
            char* end = nullptr;
            const long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0) {
                std::fprintf(stderr, "--max-skips needs a non-negative integer, got '%s'\n", argv[i]);
                return 2;
            }
            max_skips = v;
        } else if (arg == "--allow-skip") {
            if (i + 1 >= argc) { print_usage(); return 2; }
            allow_skip.emplace_back(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", arg.c_str());
            print_usage();
            return 2;
        } else {
            filter = argv[i];
        }
    }

    int run = 0;
    int failed_cases = 0;
    std::vector<SkipRecord> skips;

    for (const t::Case& c : t::cases()) {
        if (filter != nullptr && std::strstr(c.name, filter) == nullptr) continue;
        ++run;
        t::reset_failures();
        t::reset_skip();
        try {
            c.fn();
        } catch (const t::SkipSignal&) {
            // recorded via t::skipped()/t::skip_reason() below
        }
        // A failed CHECK is a failure whether or not the case then skipped:
        // otherwise a SKIP_UNLESS placed after an assertion would launder the
        // assertion's verdict (found in review, 2026-09-05).
        if (t::failures() > 0) {
            ++failed_cases;
            std::fprintf(stderr, "FAIL %s\n", c.name);
        }
        if (t::skipped()) {
            skips.push_back(SkipRecord{c.name, t::skip_reason()});
        }
    }

    for (const SkipRecord& s : skips) {
        std::fprintf(stderr, "SKIP %s: %s\n", s.case_name.c_str(), s.reason.c_str());
    }

    std::fprintf(stderr, "\n%d case%s run, %d failed, %zu skipped\n", run,
                 run == 1 ? "" : "s", failed_cases, skips.size());

    // The same rule the acceptance runner applies to its cells (DESIGN §5.1):
    // a name on --allow-skip must refer to a case that actually skipped, or
    // the allow-list itself is a lie and cannot be trusted to shrink over
    // time. A case named there that ran-and-passed, ran-and-failed, or does
    // not exist at all is an error, independent of --max-skips.
    std::set<std::string> skipped_names;
    for (const SkipRecord& s : skips) skipped_names.insert(s.case_name);

    bool bad_allow_skip = false;
    for (const std::string& name : allow_skip) {
        if (skipped_names.find(name) == skipped_names.end()) {
            bool known = false;
            for (const t::Case& c : t::cases()) known = known || name == c.name;
            if (!known) {
                std::fprintf(stderr, "--allow-skip %s: no such case\n", name.c_str());
            } else {
                std::fprintf(stderr, "--allow-skip %s: did not skip\n", name.c_str());
            }
            bad_allow_skip = true;
        }
    }

    int unnamed_skips = 0;
    for (const SkipRecord& s : skips) {
        bool named = false;
        for (const std::string& name : allow_skip) named = named || name == s.case_name;
        if (!named) ++unnamed_skips;
    }

    bool skip_cap_violated = false;
    if (max_skips.has_value() && unnamed_skips > *max_skips) {
        skip_cap_violated = true;
        std::fprintf(stderr, "%d unnamed skip%s exceeds --max-skips %ld\n", unnamed_skips,
                     unnamed_skips == 1 ? "" : "s", *max_skips);
    }

    if (failed_cases > 0 || bad_allow_skip || skip_cap_violated) return 1;
    return 0;
}

#include "harness.h"

#include <cstring>

namespace t {
namespace {
int g_failures = 0;
}

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

int failures() { return g_failures; }
void reset_failures() { g_failures = 0; }

}  // namespace t

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;

    int run = 0;
    int failed_cases = 0;

    for (const t::Case& c : t::cases()) {
        if (filter != nullptr && std::strstr(c.name, filter) == nullptr) continue;
        ++run;
        t::reset_failures();
        c.fn();
        if (t::failures() > 0) {
            ++failed_cases;
            std::fprintf(stderr, "FAIL %s\n", c.name);
        }
    }

    std::fprintf(stderr, "\n%d case%s run, %d failed\n", run, run == 1 ? "" : "s", failed_cases);
    return failed_cases == 0 ? 0 : 1;
}

#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

// A test harness small enough to read in one sitting. No framework, matching
// the engine's own "no framework" stance.
namespace t {

struct Case {
    const char* name;
    void (*fn)();
};

std::vector<Case>& cases();
int  register_case(const char* name, void (*fn)());
void fail(const char* file, int line, const std::string& msg);

template <typename T>
std::string show(const T& v) {
    std::ostringstream os;
    os << v;
    return os.str();
}
inline std::string show(bool v) { return v ? "true" : "false"; }
inline std::string show(const std::string& v) { return "\"" + v + "\""; }

}  // namespace t

#define TEST(name)                                                        \
    static void name();                                                   \
    [[maybe_unused]] static const int name##_registered =                 \
        t::register_case(#name, &name);                                   \
    static void name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) t::fail(__FILE__, __LINE__, "expected: " #cond);     \
    } while (0)

#define CHECK_EQ(a, b)                                                    \
    do {                                                                  \
        const auto& lhs_ = (a);                                           \
        const auto& rhs_ = (b);                                           \
        if (!(lhs_ == rhs_)) {                                            \
            t::fail(__FILE__, __LINE__,                                   \
                    std::string(#a " == " #b "\n           left:  ") +    \
                        t::show(lhs_) + "\n           right: " + t::show(rhs_)); \
        }                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                             \
    do {                                                                  \
        const double lhs_ = (a);                                          \
        const double rhs_ = (b);                                          \
        if (!((lhs_ - rhs_ < (eps)) && (rhs_ - lhs_ < (eps)))) {          \
            t::fail(__FILE__, __LINE__,                                   \
                    std::string(#a " ~= " #b "\n           left:  ") +    \
                        t::show(lhs_) + "\n           right: " + t::show(rhs_)); \
        }                                                                 \
    } while (0)

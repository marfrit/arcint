// The cost of the Inkling call grammar's dormant trigger, device-free.
//
// While a lazy grammar waits for its trigger, libllama appends every token it
// is fed to a buffer and runs the trigger's regex over the WHOLE buffer
// (src/llama-grammar.cpp at the pin: llama_grammar_accept_impl, the
// awaiting_trigger branch, and llama_grammar_trigger_pattern::find, which is
// std::regex_search unless the pattern is ^...$). arcint feeds it every token
// outside the first thinking block, so a long answer before (or without) a
// call costs a search of everything written so far, per token: quadratic.
//
// This tool times exactly that loop -- the buffer append, the positions
// vector, and find() copied from the pin -- with the trigger arcint uses
// (core/toolgrammar.cpp inkling_tool_grammar_trigger_pattern) and with
// mainline's, over synthetic non-thinking output of N tokens, and prints the
// per-token cost at the end of the run (mean of the last 200 tokens) and the
// total. Then the same with the search bounded to the buffer's tail from the
// last possible header start (the proposed mitigation; see its arm below).
//
// Not a ctest case (a timing). Build and run with the recipe's toolchain:
//   g++ -O3 -std=c++20 -o inkling_trigger_cost tools/inkling_trigger_cost.cpp
//   ./inkling_trigger_cost 2000 16000
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace {

// llama_grammar_trigger_pattern::find at the pin, verbatim in behaviour
struct Pattern {
    std::string pattern;
    std::regex  regex;
    size_t find(const std::string& input) const {
        auto find_start_pos = [](const std::smatch& match) {
            size_t start = std::string::npos;
            for (auto i = 1u; i < match.size(); i++) {
                if (match.length(i) > 0) {
                    start = match.position(i);
                    break;
                }
            }
            if (start == std::string::npos) start = match.position(0);
            return start;
        };
        if (!pattern.empty() && pattern.front() == '^' && pattern.back() == '$') {
            std::smatch match;
            if (std::regex_match(input, match, regex)) return find_start_pos(match);
        }
        std::smatch match;
        if (std::regex_search(input, match, regex)) return find_start_pos(match);
        return std::string::npos;
    }
};

// Non-thinking output: prose, code with indentation, and the block framing
// a text answer carries (headers followed by a content marker, never by a
// name), about 4 bytes a piece as a BPE vocabulary gives.
std::vector<std::string> pieces(size_t n) {
    static const char* const words[] = { "The", " parser", " reads", " each", " field", ",", " then", "\n",
                                         "    ", "if", " (", "n", " <", " 0", ")", " {", "\n", "\t",
                                         "return", " -", "1", ";", "\n", "}", " quoted", " value", ".",
                                         " A", " CSV", " line", " ends", " at", " CRLF", "\"\"", " x",
                                         "_count", " ==", " 3", " and", " the" };
    std::vector<std::string> out;
    out.reserve(n);
    unsigned s = 12345;
    for (size_t i = 0; i < n; ++i) {
        if (i % 400 == 0) {   // a text block's framing now and then
            out.push_back("<|end_message|>");
            out.push_back("<|message_model|>");
            out.push_back("<|content_text|>");
            continue;
        }
        s = s * 1103515245u + 12345u;
        out.push_back(words[(s >> 16) % (sizeof words / sizeof *words)]);
    }
    return out;
}

struct Result {
    double last_us = 0.0, total_ms = 0.0;
    bool   fired   = false;
};

// bounded: search only the tail that can still hold a match ending in the new
// piece -- from the last header start, or the last header-length bytes
Result run(const std::vector<std::string>& ps, const Pattern& pat, bool bounded) {
    using clock = std::chrono::steady_clock;
    static const std::string header = "<|message_model|>";
    std::string                                       buffer;
    std::vector<std::pair<int, std::pair<size_t, size_t>>> positions;
    std::vector<double>                               per;
    per.reserve(ps.size());
    size_t pending = std::string::npos;   // bounded: the start of the last header still undecided
    Result r;
    const auto t_all = clock::now();
    for (size_t i = 0; i < ps.size(); ++i) {
        const auto t0 = clock::now();
        positions.push_back({ static_cast<int>(i), { buffer.size(), buffer.size() + ps[i].size() } });
        const size_t before = buffer.size();
        buffer += ps[i];
        size_t at;
        if (!bounded) {
            at = pat.find(buffer);
        } else {
            const size_t from = before >= header.size() ? before - header.size() + 1 : 0;
            const size_t h    = buffer.find(header, from);
            if (h != std::string::npos) pending = h;
            const size_t start = pending != std::string::npos ? pending : from;
            at                 = pat.find(buffer.substr(start));
            if (at != std::string::npos) at += start;
            // a header followed by a decided byte is no longer pending
            if (pending != std::string::npos) {
                size_t k = pending + header.size();
                while (k < buffer.size() && std::isspace(static_cast<unsigned char>(buffer[k]))) ++k;
                if (k < buffer.size()) pending = std::string::npos;
            }
        }
        per.push_back(std::chrono::duration<double, std::micro>(clock::now() - t0).count());
        if (at != std::string::npos) r.fired = true;
    }
    r.total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_all).count();
    const size_t k = per.size() < 200 ? per.size() : 200;
    for (size_t i = per.size() - k; i < per.size(); ++i) r.last_us += per[i];
    r.last_us /= static_cast<double>(k);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const Pattern ours{ "<\\|message_model\\|>\\s*([^<\\s])", std::regex("<\\|message_model\\|>\\s*([^<\\s])") };
    const Pattern mainline{ "<\\|message_model\\|>([^<])", std::regex("<\\|message_model\\|>([^<])") };
    for (int a = 1; a < argc; ++a) {
        const size_t n  = static_cast<size_t>(std::atol(argv[a]));
        const auto   ps = pieces(n);
        size_t       bytes = 0;
        for (const auto& p : ps) bytes += p.size();
        for (const auto& [name, pat] : { std::pair<const char*, const Pattern*>{ "arcint ", &ours },
                                         std::pair<const char*, const Pattern*>{ "mainline", &mainline } }) {
            for (const bool bounded : { false, true }) {
                const Result r = run(ps, *pat, bounded);
                std::printf("%6zu tokens (%7zu bytes) %s %-9s: last 200 tokens %9.2f us/token, run total %9.1f ms%s\n",
                            n, bytes, name, bounded ? "bounded" : "full", r.last_us, r.total_ms,
                            r.fired ? "  [FIRED: the synthetic text holds no call]" : "");
            }
        }
    }
    return 0;
}

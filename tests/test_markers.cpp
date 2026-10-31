#include "harness.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// The house rule: a chat-template marker is never typed as one literal in a
// parser, a renderer, a test fixture or a comment. The markers live in
// src/core/markers.h, built from parts. This scan is the guard for that rule
// and is device-free: it reads text only.
//
// Scope: src/, tests/, docs/ and the top-level DESIGN.md and llm.txt. Vendored
// trees (third_party/, vendor/) and the patch files under
// contrib/llama.cpp/patches carry upstream text and are out of scope by design.
// CHANGELOG.md is excluded explicitly: its entries are the released history and
// stay as they shipped (a marker there is a record, not an instruction).
//
// The scan patterns themselves are built from parts, so this file passes its
// own scan. A marker added anywhere in scope fails with its file and line.

namespace {

const std::string kSourceDir = ARCINT_SOURCE_DIR;

struct Marker {
    const char* name;
    std::string text;
};

std::vector<Marker> markers() {
    return {
        {"think opener", "<" + std::string("thi") + "nk" + ">"},
        {"think closer", "</" + std::string("thi") + "nk" + ">"},
        {"angle think opener", "<" + std::string("think") + "ing" + ">"},
        {"angle think closer", "</" + std::string("think") + "ing" + ">"},
        {"square think opener", "[" + std::string("THINK") + "]"},
        {"square think closer", "[" + std::string("/THINK") + "]"},
        {"turn start", "<" + std::string("|im_") + "start" + "|" + ">"},
        {"turn end", "<" + std::string("|im_") + "end" + "|" + ">"},
    };
}

bool skipped(const std::string& rel) {
    // CHANGELOG.md's entries are released history: quoted markers stay as they
    // shipped, and rewriting them would falsify the record.
    if (rel == "CHANGELOG.md") return true;
    return rel.rfind("third_party/", 0) == 0 || rel.rfind("vendor/", 0) == 0 ||
           rel.rfind("contrib/llama.cpp/patches/", 0) == 0;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

}  // namespace

TEST(no_chat_template_marker_is_typed_literally_in_source_tests_or_docs) {
    const std::vector<Marker> marks = markers();
    std::vector<std::string> offenders;

    const auto scan_one = [&](const std::filesystem::path& path) {
        const std::string rel = std::filesystem::relative(path, kSourceDir).generic_string();
        if (skipped(rel)) return;
        const std::string text = read_file(path);
        if (text.find('\0') != std::string::npos) return;   // binary
        size_t line = 1;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\n') ++line;
            for (const Marker& m : marks) {
                if (text.compare(i, m.text.size(), m.text) != 0) continue;
                offenders.push_back(rel + ":" + std::to_string(line) + ": " + m.name);
                i += m.text.size() - 1;
                break;
            }
        }
    };

    std::vector<std::string> notes;

    for (const char* root : {"src", "tests", "docs", "DESIGN.md", "llm.txt"}) {
        const std::filesystem::path base = std::filesystem::path(kSourceDir) / root;
        if (!std::filesystem::exists(base)) {
            // A missing root is a note, not an early exit: offenders already
            // found in an earlier root must still fail the case.
            notes.push_back(std::string(root) + " is missing under the source tree (root skipped)");
            continue;
        }
        if (std::filesystem::is_regular_file(base)) {
            scan_one(base);
            continue;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(base)) {
            if (entry.is_regular_file()) scan_one(entry.path());
        }
    }

    for (const std::string& note : notes) std::fprintf(stderr, "  note %s\n", note.c_str());

    if (!offenders.empty()) {
        std::string msg = "chat-template markers typed literally (build them from parts in src/core/markers.h):";
        for (const std::string& o : offenders) msg += "\n  " + o;
        t::fail(__FILE__, __LINE__, msg);
    }
    CHECK(offenders.empty());
}

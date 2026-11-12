#include "config.h"
#include "core/chat.h"
#include "harness.h"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// --chat-template-file: the file reader/validator, the flag's parse_args
// behaviour and a source scan of the llama backend (it cannot be constructed
// without a card and a model).
using namespace lgc;

namespace {
ArgParse run(std::vector<const char*> args, Config& cfg) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("arcint"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    return parse_args(static_cast<int>(argv.size()), argv.data(), cfg);
}

bool rejected(std::vector<const char*> args) {
    Config cfg;
    return !run(std::move(args), cfg).ok;
}
}  // namespace

// ---------------------------------------------------------------------------
// --chat-template-file (load_chat_template_file): present, missing, empty,
// blank and unparsable files. What needs a GGUF (the template rendering for the
// model, the capabilities and the renderer material built from the file's text)
// is not covered here: the llama backend cannot be constructed without a card
// and a model; the card check is in the commit message.
namespace {
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& content) {
        static int n = 0;
        path = (std::filesystem::temp_directory_path() /
                ("arcint-tmpl-test-" + std::to_string(::getpid()) + "-" + std::to_string(n++) + ".jinja"))
                   .string();
        std::ofstream(path, std::ios::binary) << content;
    }
    ~TempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};
}  // namespace

TEST(chat_template_file_reads_a_valid_template_verbatim) {
    const std::string src = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}"
                            "{% if add_generation_prompt %}[assistant] {% endif %}\n\n";
    TempFile    f(src);
    std::string text;
    const auto  err = load_chat_template_file(f.path, text);
    CHECK(!err.has_value());
    CHECK_EQ(text, src);   // trailing newlines kept, as llama.cpp's read_file
}

TEST(chat_template_file_refuses_a_missing_or_unreadable_path) {
    std::string text = "untouched";
    const std::string missing = "/nonexistent-dir/arcint-no-such-template.jinja";
    auto err = load_chat_template_file(missing, text);
    CHECK(err.has_value());
    if (err) CHECK(err->find(missing) != std::string::npos);
    CHECK_EQ(text, std::string("untouched"));
    // a directory is not a template
    err = load_chat_template_file(std::filesystem::temp_directory_path().string(), text);
    CHECK(err.has_value());
    CHECK_EQ(text, std::string("untouched"));
}

TEST(chat_template_file_refuses_an_empty_or_blank_file) {
    std::string text = "untouched";
    TempFile    empty("");
    TempFile    blank(" \n\t\n");
    for (const TempFile* f : {&empty, &blank}) {
        const auto err = load_chat_template_file(f->path, text);
        CHECK(err.has_value());
        if (err) CHECK(err->find(f->path) != std::string::npos);
    }
    CHECK_EQ(text, std::string("untouched"));
}

TEST(chat_template_file_refuses_a_template_minja_cannot_parse) {
    std::string text = "untouched";
    for (const char* bad : {"{% for m in messages %}{{ m.content }}",       // no endfor
                            "{{ messages[ }}",                               // broken expression
                            "{% if %}x{% endif %}"}) {                       // no condition
        TempFile   f(bad);
        const auto err = load_chat_template_file(f.path, text);
        CHECK(err.has_value());
        if (err) CHECK(err->find(f.path) != std::string::npos);
    }
    CHECK_EQ(text, std::string("untouched"));
}

// --chat-template-file PATH: parse_args reads and validates the file, so a unit
// that names a missing, empty or unparsable template does not start. Needs a
// model (the stub has no template, and silently ignoring the flag would be a
// fallback).
TEST(config_chat_template_file) {
    // unique per process: two runs of the suite must not share the file
    const std::string path = (std::filesystem::temp_directory_path() /
                              ("arcint-cfg-tmpl-test-" + std::to_string(::getpid()) + ".jinja")).string();
    const std::string src  = "{% for m in messages %}{{ m.content }}{% endfor %}";
    std::ofstream(path) << src;

    Config cfg;
    CHECK(run({"--gguf", "/m/q.gguf", "--chat-template-file", path.c_str()}, cfg).ok);
    CHECK_EQ(cfg.chat_template_file, path);
    CHECK_EQ(cfg.chat_template_text, src);

    Config none;
    CHECK(run({"--gguf", "/m/q.gguf"}, none).ok);
    CHECK(none.chat_template_file.empty());
    CHECK(none.chat_template_text.empty());

    // the stub has no template to replace
    CHECK(rejected({"--stub", "--chat-template-file", path.c_str()}));
    // a missing file, a missing value
    CHECK(rejected({"--gguf", "/m/q.gguf", "--chat-template-file", "/nonexistent-dir/none.jinja"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--chat-template-file"}));
    // empty and unparsable
    std::ofstream(path) << "";
    CHECK(rejected({"--gguf", "/m/q.gguf", "--chat-template-file", path.c_str()}));
    std::ofstream(path) << "{% for m in messages %}";
    CHECK(rejected({"--gguf", "/m/q.gguf", "--chat-template-file", path.c_str()}));
    // --help does not read the file
    Config help;
    CHECK(run({"--help", "--chat-template-file", "/nonexistent-dir/none.jinja"}, help).ok);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// A repeated --chat-template-file is a start failure, as a repeated
// --chat-template-kwarg key is (review F9; it was silently last-wins). With --stub
// the later validation would refuse the flag too, so the message tells the two apart.
TEST(config_chat_template_file_given_twice_fails) {
    Config cfg;
    const ArgParse r = run({"--stub", "--chat-template-file", "/a.jinja", "--chat-template-file", "/b.jinja"}, cfg);
    CHECK(!r.ok);
    CHECK(r.error.find("given twice") != std::string::npos);
}

// --chat-template-file: the llama backend needs a card and a model to
// construct, so what it does with the file is pinned as a source scan, the way
// the KV-precision tripwire is: the embedded template is read in ONE place
// (llama_model_chat_template), the file replaces it there, the source reads
// "file", and /props' chat_template_caps publishes the source.
TEST(llama_backend_template_file_replaces_the_embedded_template_in_one_place) {
    std::ifstream in(std::string(ARCINT_SOURCE_DIR) + "/src/exec/backend_llama.cpp");
    CHECK(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    size_t reads = 0;
    for (size_t at = src.find("llama_model_chat_template("); at != std::string::npos;
         at = src.find("llama_model_chat_template(", at + 1))
        ++reads;
    CHECK_EQ(reads, static_cast<size_t>(1));
    CHECK(src.find("template_source_ = \"file\"") != std::string::npos);
    CHECK(src.find("cfg.chat_template_text") != std::string::npos);
    CHECK(src.find("{\"template_source\", template_source_}") != std::string::npos);
}

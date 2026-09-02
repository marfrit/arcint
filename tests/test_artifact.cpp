#include "core/artifact.h"
#include "harness.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace lgc;

namespace {

// A throwaway artifact directory under an allowlisted alias
// (models/allowlist-raw.json's "qwen36-coder-b5-ov"), holding just enough
// for load_artifact to succeed: the language model, text embeddings,
// tokenizer/detokenizer, config.json and chat_template.jinja. Content is
// throwaway bytes -- load_artifact only checks that these files exist and
// hashes them, it does not parse the IR graph.
class TempArtifactDir {
public:
    TempArtifactDir() {
        const char* tmpdir = std::getenv("TMPDIR");
        std::string tmpl_s = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/arcint-test-artifact-XXXXXX";
        std::vector<char> tmpl(tmpl_s.begin(), tmpl_s.end());
        tmpl.push_back('\0');
        const char* base = ::mkdtemp(tmpl.data());
        if (base == nullptr) throw std::runtime_error("mkdtemp failed");
        parent_ = base;
        dir_    = parent_ + "/qwen36-coder-b5-ov";
        if (::mkdir(dir_.c_str(), 0700) != 0) throw std::runtime_error("mkdir failed");

        write("openvino_language_model.xml", "<xml/>");
        write("openvino_language_model.bin", "weights");
        write("openvino_text_embeddings_model.xml", "<xml/>");
        write("openvino_tokenizer.xml", "<xml/>");
        write("openvino_detokenizer.xml", "<xml/>");
        write("config.json", "{}");
        write("chat_template.jinja", "{{ messages }}");
        write("tokenizer.json", "{}");
    }

    ~TempArtifactDir() {
        for (const char* name :
             {"openvino_language_model.xml", "openvino_language_model.bin",
              "openvino_text_embeddings_model.xml", "openvino_tokenizer.xml",
              "openvino_detokenizer.xml", "config.json", "chat_template.jinja",
              "tokenizer.json", "openvino_vision_embeddings_model.xml",
              "openvino_vision_embeddings_model.bin", "openvino_vision_embeddings_pos_model.xml",
              "openvino_vision_embeddings_pos_model.bin", "openvino_vision_embeddings_merger_model.xml",
              "openvino_vision_embeddings_merger_model.bin"}) {
            ::unlink((dir_ + "/" + name).c_str());
        }
        ::rmdir(dir_.c_str());
        ::rmdir(parent_.c_str());
    }

    const std::string& dir() const { return dir_; }

    void write(const std::string& name, const std::string& content) {
        std::ofstream out(dir_ + "/" + name, std::ios::binary);
        out << content;
    }

    // Writes exactly `bytes` bytes -- the vision-file-size case needs the
    // reported size to match precisely, not just be nonzero.
    void write_sized(const std::string& name, uint64_t bytes) {
        std::ofstream out(dir_ + "/" + name, std::ios::binary);
        const std::string content(bytes, 'v');
        out << content;
    }

private:
    std::string parent_;
    std::string dir_;
};

}  // namespace

// M13 (docs/milestone-0.3.0.md): a *ForConditionalGeneration export carries
// vision-tower/projector IRs the loader never reads (src/core/artifact.cpp
// resolves only the language model and text embeddings). A plain text-only
// export has none of them, and the inventory must say exactly that -- no
// entries, not "we didn't look".
TEST(artifact_reports_no_unloaded_vision_irs_without_them) {
    TempArtifactDir d;
    Artifact         a;
    const auto       err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK(a.unloaded_vision_irs.empty());
}

// The red case this milestone cares about: a vision IR present on disk (here,
// just the merger's .bin -- the loader never asked for its .xml either) is
// inventoried with its exact name and byte size, so backend_ov.cpp can log
// "N files, X MiB on disk" truthfully.
TEST(artifact_reports_a_present_vision_ir_with_its_exact_size) {
    TempArtifactDir d;
    d.write_sized("openvino_vision_embeddings_merger_model.bin", 12345);
    Artifact   a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK_EQ(a.unloaded_vision_irs.size(), static_cast<size_t>(1));
    if (a.unloaded_vision_irs.size() == 1) {
        CHECK_EQ(a.unloaded_vision_irs[0].name,
                 std::string("openvino_vision_embeddings_merger_model.bin"));
        CHECK_EQ(a.unloaded_vision_irs[0].bytes, static_cast<uint64_t>(12345));
    }
}

// All three named IRs (embeddings, pos, merger), both xml and bin, are
// checked independently -- a checkpoint that ships every half of every
// vision file must report all six, not just the first one found.
TEST(artifact_reports_every_present_vision_ir_file) {
    TempArtifactDir d;
    d.write("openvino_vision_embeddings_model.xml", "<xml/>");
    d.write_sized("openvino_vision_embeddings_model.bin", 100);
    d.write("openvino_vision_embeddings_pos_model.xml", "<xml/>");
    d.write_sized("openvino_vision_embeddings_pos_model.bin", 200);
    d.write("openvino_vision_embeddings_merger_model.xml", "<xml/>");
    d.write_sized("openvino_vision_embeddings_merger_model.bin", 300);

    Artifact   a;
    const auto err = load_artifact(d.dir(), a);
    CHECK(!err.has_value());
    CHECK_EQ(a.unloaded_vision_irs.size(), static_cast<size_t>(6));

    uint64_t total = 0;
    for (const auto& ir : a.unloaded_vision_irs) total += ir.bytes;
    // Three ".xml" files at 6 bytes ("<xml/>") each, plus the three sized
    // ".bin" files.
    CHECK_EQ(total, static_cast<uint64_t>(3 * 6 + 100 + 200 + 300));
}

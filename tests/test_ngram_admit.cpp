// FIX D Link 2 (docs/design-qwen-flash-next.md §"Links 2 and 3"):
// red-before-green companion of `admit_ngram_table_from_disk` in
// src/core/artifact.cpp. Every case here exercises one of the
// refusal reasons the admission function returns; deleting any
// production line the function relies on makes at least one of these
// cases fail. That is the deletion test the roadmap requires.
#include "core/artifact.h"
#include "core/ngram_header.h"
#include "harness.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lgc;

namespace {

std::string temp_path(const char* stem) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "/tmp/arcint-ngram-admit-%s-%d.bin",
                  stem, ::getpid());
    return std::string(buf);
}

// Write a well-formed synthetic ARCINGRM file at `path` with the given
// header fields. Payload is zero-filled to header + n_rows * blocks *
// bytes_per_block bytes.
void write_ngram_file(const std::string& path, uint32_t ggml_type,
                      uint32_t n_cols, uint32_t n_rows) {
    std::ofstream f(path, std::ios::binary);
    f.write("ARCINGRM", 8);
    const uint32_t reserved = 0;
    f.write(reinterpret_cast<const char*>(&ggml_type), 4);
    f.write(reinterpret_cast<const char*>(&n_cols), 4);
    f.write(reinterpret_cast<const char*>(&n_rows), 4);
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    const uint64_t blocks = n_cols / ngram::kBlockElements;
    const uint64_t bpb    = ngram::bytes_per_block(ggml_type);
    std::vector<uint8_t> payload(n_rows * blocks * bpb, 0);
    f.write(reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
}

Artifact synth_artifact_with_ngram(int ngram_vocab_size_base, int ple_embed_dim) {
    Artifact a;
    a.model_type = "qwen4_exp";
    a.ov_arch    = "Qwen4ExpForConditionalGeneration";
    a.ngram_config.ngram_size            = 3;
    a.ngram_config.ngram_vocab_size_base = ngram_vocab_size_base;
    a.ngram_config.heads_per_ngram       = 8;
    a.ngram_config.ple_embed_dim         = ple_embed_dim;
    return a;
}

}  // namespace

TEST(ngram_admit_refuses_a_valid_file_when_the_artifact_declares_no_ngram_table) {
    // A dense qwen35 artifact (like the served 2B) does not declare an
    // n-gram table; admitting a file for it is a category error the
    // loader must catch by name, not silently drop.
    Artifact a;
    a.model_type = "qwen3_5";
    // ngram_config left default (all zero)

    const auto path = temp_path("no-config");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, /*n_rows*/ 16);
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("does not declare") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_accepts_a_matching_file_and_reports_payload_bytes) {
    // Config declares 20 vocab entries at ple_embed_dim 320 => 40
    // physical rows at width 160. A file of the right shape must be
    // admitted; payload bytes reported so the loader can plumb them
    // into the fit refusal.
    Artifact a = synth_artifact_with_ngram(/*vocab*/ 20, /*ple_embed_dim*/ 320);
    const auto path = temp_path("match");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, /*n_rows*/ 40);
    uint64_t payload = 0;
    const std::string err = admit_ngram_table_from_disk(
        a, path, /*host_ram*/ (1ull << 30), /*other*/ 0, /*margin*/ 0, payload);
    ::unlink(path.c_str());
    CHECK(err.empty());
    // 40 rows * 5 blocks/row * 18 bytes/block = 3600
    CHECK_EQ(payload, uint64_t{40 * 5 * 18});
}

TEST(ngram_admit_refuses_a_shape_mismatch_against_the_config) {
    Artifact a = synth_artifact_with_ngram(/*vocab*/ 20, /*ple_embed_dim*/ 320);
    // File has n_rows 42 instead of 40 -- config-derived expected is 40.
    const auto path = temp_path("mismatch");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, /*n_rows*/ 42);
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("n_rows") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_bad_magic) {
    Artifact a = synth_artifact_with_ngram(20, 320);
    const auto path = temp_path("bad-magic");
    // Write a file that starts with the wrong magic but the right shape.
    {
        std::ofstream f(path, std::ios::binary);
        f.write("NOTAGRAM", 8);
        const uint32_t vals[4] = {2, 160, 40, 0};
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
        std::vector<uint8_t> pad(40 * 5 * 18, 0);
        f.write(reinterpret_cast<const char*>(pad.data()),
                static_cast<std::streamsize>(pad.size()));
    }
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("magic") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_unknown_type) {
    Artifact a = synth_artifact_with_ngram(20, 320);
    const auto path = temp_path("bad-type");
    // Ggml_type 99 is not in {Q4_0=2, Q4_1=3, Q8_0=8}; header parser
    // must refuse.
    {
        std::ofstream f(path, std::ios::binary);
        f.write("ARCINGRM", 8);
        const uint32_t vals[4] = {99, 160, 40, 0};
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("ggml_type") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_wrong_row_width) {
    Artifact a = synth_artifact_with_ngram(20, 320);
    const auto path = temp_path("wrong-width");
    // n_cols 128 (a valid multiple of 32) is admitted by the header
    // parser but not by the admission surface: the AVX2 kernel and fit
    // arithmetic assume 160.
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 128, /*n_rows*/ 40);
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("row width") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_on_host_ram_shortfall) {
    // The fit arithmetic must catch a table that fits the disk but not
    // the declared host RAM. Reuse the 40-row valid file, hand it a
    // host_ram_bytes below the payload; refusal names the budget.
    Artifact a = synth_artifact_with_ngram(20, 320);
    const auto path = temp_path("no-ram");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, /*n_rows*/ 40);
    uint64_t payload = 999;
    // Table is 3600 bytes; host_ram = 100 forces the refusal.
    const std::string err = admit_ngram_table_from_disk(
        a, path, /*host_ram*/ 100, /*other*/ 0, /*margin*/ 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("host RAM budget") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_a_truncated_file) {
    // Header advertises 40 rows but the file only has 20 rows of
    // payload -- the on-disk-size check must catch this.
    Artifact a = synth_artifact_with_ngram(20, 320);
    const auto path = temp_path("truncated");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, /*n_rows*/ 40);
    // Truncate the file to header + 20 rows.
    ::truncate(path.c_str(), ngram::kHeaderBytes + 20 * 5 * 18);
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("on-disk size") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_missing_file) {
    Artifact a = synth_artifact_with_ngram(20, 320);
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(
        a, "/tmp/arcint-nonexistent-ngram-file-do-not-create.bin",
        0, 0, 0, payload);
    CHECK(!err.empty());
    CHECK_EQ(payload, uint64_t{0});
}

// FIX D Link 2 (docs/design-qwen-flash-next.md §"Links 2 and 3"):
// red-before-green companion of `admit_ngram_table_from_disk` in
// src/core/artifact.cpp. Every case here exercises one of the
// refusal reasons the admission function returns; deleting any
// production line the function relies on makes at least one of these
// cases fail. That is the deletion test the roadmap requires.
#include "core/artifact.h"
#include "core/ngram_header.h"
#include "exec/ngram_row_ids.h"
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
    a.ngram_config.vocab_size            = 4096;   // model vocab (bounds the hash multipliers)
    a.ngram_config.ple_layer_ids         = {1};    // one PLE layer (non-empty is required)
    a.ngram_config.ngram_boundary_token_id = 2;    // a found eos id
    return a;
}

// The minimum admissible n_rows for the corrected hashed-vocab spec: the
// concatenated per-head prime vocab sizes, topped by the last PLE layer's band.
// A file must have at least this many rows to hold every hashed row id.
uint32_t required_rows(const Artifact& a) {
    const int num_ple = std::max<int>(1, static_cast<int>(a.ngram_config.ple_layer_ids.size()));
    return static_cast<uint32_t>(ngram::ngram_required_rows(
        a.ngram_config.vocab_size, a.ngram_config.ngram_size,
        a.ngram_config.heads_per_ngram, a.ngram_config.ngram_vocab_size_base, num_ple));
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

TEST(ngram_admit_refuses_a_declared_but_incomplete_config) {
    // A table declared (ple_embed_dim non-zero) but missing a required field
    // must be refused by name, not admitted with a silent wrong default.
    Artifact a = synth_artifact_with_ngram(/*vocab*/ 20, /*ple_embed_dim*/ 320);
    a.ngram_config.ple_layer_ids.clear();        // empty => "no PLE" per the reference
    const auto path = temp_path("incomplete");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, /*n_rows*/ 1000);
    uint64_t payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("incomplete") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});

    // A missing eos (no boundary id found) is likewise refused.
    Artifact b = synth_artifact_with_ngram(20, 320);
    b.ngram_config.ngram_boundary_token_id = -1;
    const auto pathb = temp_path("incomplete-eos");
    write_ngram_file(pathb, 2, 160, 1000);
    uint64_t payloadb = 999;
    const std::string errb = admit_ngram_table_from_disk(b, pathb, 0, 0, 0, payloadb);
    ::unlink(pathb.c_str());
    CHECK(!errb.empty());
    CHECK(errb.find("incomplete") != std::string::npos);
}

TEST(ngram_admit_accepts_a_file_large_enough_for_the_hashed_row_space) {
    // A file must have at least required_rows(a) rows -- the concatenated
    // per-head prime vocab sizes (corrected hashed-vocab spec), not the old
    // base*(ple_embed_dim/160). Exactly required_rows is admitted; payload
    // bytes reported so the loader can plumb them into the fit refusal.
    Artifact a = synth_artifact_with_ngram(/*vocab*/ 20, /*ple_embed_dim*/ 320);
    const uint32_t n_rows = required_rows(a);
    const auto     path   = temp_path("match");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, n_rows);
    uint64_t          payload = 0;
    const std::string err = admit_ngram_table_from_disk(
        a, path, /*host_ram*/ (1ull << 30), /*other*/ 0, /*margin*/ 0, payload);
    ::unlink(path.c_str());
    CHECK(err.empty());
    CHECK_EQ(payload, uint64_t{n_rows} * 5 * 18);  // rows * 5 blocks/row * 18 bytes/block
}

TEST(ngram_admit_accepts_a_file_padded_above_the_hashed_row_space) {
    // The bound is >=, not ==: a table padded larger than the minimum (the
    // checkpoint may pad via split_ngram_parts) is still admitted.
    Artifact a = synth_artifact_with_ngram(/*vocab*/ 20, /*ple_embed_dim*/ 320);
    const uint32_t n_rows = required_rows(a) + 7;
    const auto     path   = temp_path("padded");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, n_rows);
    uint64_t          payload = 0;
    const std::string err = admit_ngram_table_from_disk(
        a, path, /*host_ram*/ (1ull << 30), /*other*/ 0, /*margin*/ 0, payload);
    ::unlink(path.c_str());
    CHECK(err.empty());
}

TEST(ngram_admit_refuses_a_table_too_small_for_the_hashed_row_space) {
    Artifact a = synth_artifact_with_ngram(/*vocab*/ 20, /*ple_embed_dim*/ 320);
    // One row short of the hashed row space: a lookup could reach a row this
    // file does not have.
    const uint32_t n_rows = required_rows(a) - 1;
    const auto     path   = temp_path("mismatch");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, n_rows);
    uint64_t          payload = 999;
    const std::string err = admit_ngram_table_from_disk(a, path, 0, 0, 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("n_rows") != std::string::npos);
    CHECK(err.find("below") != std::string::npos);
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
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, required_rows(a));
    uint64_t payload = 999;
    // The table's payload is many KiB; host_ram = 100 forces the refusal.
    const std::string err = admit_ngram_table_from_disk(
        a, path, /*host_ram*/ 100, /*other*/ 0, /*margin*/ 0, payload);
    ::unlink(path.c_str());
    CHECK(!err.empty());
    CHECK(err.find("host RAM budget") != std::string::npos);
    CHECK_EQ(payload, uint64_t{0});
}

TEST(ngram_admit_refuses_a_truncated_file) {
    // Header advertises a valid (large-enough) row count but the file is short
    // -- the on-disk-size check must catch this, after the row-count check.
    Artifact a = synth_artifact_with_ngram(20, 320);
    const auto path = temp_path("truncated");
    write_ngram_file(path, /*Q4_0*/ 2, /*n_cols*/ 160, required_rows(a));
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

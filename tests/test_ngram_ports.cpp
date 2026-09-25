// FEED-THE-PORTS: the device-free half of the n-gram table PORTS contract
// (exec/ngram_ports.h). What the OpenVINO half does with these -- allocate
// USM-host chunks, memcpy the GGUF's rows in, feed the split ids per forward
// -- cannot run here; what CAN is recognising the ports the emitter declares,
// refusing a bad partition by name, matching the source tensor, and splitting
// ids exactly as the emitter's Python twin (tools/q4e/ngram_ids.py
// split_by_partition) does.
#include "core/gguf.h"
#include "exec/ngram_ports.h"
#include "harness.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lgc;
using ngram::PortDims;

namespace {

// The real partition: 320,001,536 rows of 90 bytes under the A770's
// 4,294,959,104-byte cap = 47,718,400 rows a chunk (the largest multiple of
// 4096 that fits), six full chunks and a 33,691,136-row last one. These
// figures are the emitter's (q4e.serving_shape.ngram_table_chunks) and the
// contract cell's, restated here so a drift on either side shows up.
constexpr int64_t kRows      = 320001536;
constexpr int64_t kPerChunk  = 47718400;
constexpr int64_t kLastChunk = 33691136;
constexpr int64_t kRowBytes  = 90;

std::vector<PortDims> real_ports(bool with_ids = true, bool with_mask = true) {
    std::vector<PortDims> in = {
        {"inputs_embeds", {1, -1, 2560}}, {"position_ids", {1, -1}},
    };
    if (with_ids) {
        in.push_back({"ngram_chunk_ids", {1, -1, 16}});
        in.push_back({"ngram_local_ids", {1, -1, 16}});
    }
    if (with_mask) in.push_back({"conv_mask", {1, -1}});
    for (int k = 0; k < 6; ++k) in.push_back({"ngram_table." + std::to_string(k), {kPerChunk, kRowBytes}});
    in.push_back({"ngram_table.6", {kLastChunk, kRowBytes}});
    in.push_back({"past_lens", {-1}});
    return in;
}

std::string refusal(const std::vector<PortDims>& in) {
    try {
        (void)ngram::plan_ngram_ports(in);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

gguf::TensorInfo table_tensor(int32_t type = ngram::kIq4NlType, uint64_t width = 160,
                              uint64_t rows = static_cast<uint64_t>(kRows)) {
    gguf::TensorInfo t;
    t.name       = ngram::kTableTensor;
    t.dims       = {width, rows};
    t.ggml_type  = type;
    t.n_elements = static_cast<size_t>(width * rows);
    return t;
}

}  // namespace

TEST(ngram_ports_recognises_the_real_partition) {
    const auto plan = ngram::plan_ngram_ports(real_ports());
    CHECK(!plan.empty());
    CHECK_EQ(plan.chunks.size(), static_cast<size_t>(7));
    CHECK_EQ(plan.row_bytes, static_cast<size_t>(kRowBytes));
    CHECK_EQ(plan.rows_per_chunk, static_cast<size_t>(kPerChunk));
    CHECK_EQ(plan.total_rows, static_cast<size_t>(kRows));
    CHECK_EQ(plan.chunks.front().name, std::string("ngram_table.0"));
    CHECK_EQ(plan.chunks.back().name, std::string("ngram_table.6"));
    CHECK_EQ(plan.chunks.back().rows, static_cast<size_t>(kLastChunk));
    CHECK(plan.declares_ids);
    CHECK(plan.declares_conv_mask);
    // 6 x 47,718,400 + 33,691,136 is the table exactly, and every chunk is
    // under the cap it was cut for
    CHECK_EQ(6 * kPerChunk + kLastChunk, kRows);
    CHECK(kPerChunk * kRowBytes <= 4294959104LL);
}

TEST(ngram_ports_is_empty_for_an_ir_without_the_table) {
    const std::vector<PortDims> served = {
        {"inputs_embeds", {-1, -1}}, {"position_ids", {4, -1, -1}}, {"past_lens", {-1}},
        {"key_cache.0", {-1, -1, -1, -1}},
    };
    const auto plan = ngram::plan_ngram_ports(served);
    CHECK(plan.empty());
    CHECK(!plan.declares_ids);
}

TEST(ngram_ports_the_no_ple_qwen35moe_shape_declares_conv_mask_but_no_table) {
    // The `qwen3_5_moe` serving-shape IR (Qwen3.6-35B-A3B): inputs_embeds,
    // position_ids, conv_mask, beam_idx (and the paged pass's own ports), with
    // NO ngram_table.K and NO id ports. `plan.empty()` is exactly the condition
    // backend_ov.cpp's bind_ngram_ports returns INERTLY on -- this family has
    // no PLE, so the n-gram binding is not required and --ngram-gguf is not
    // needed. conv_mask is the GDN/attention padding mask, not a table port,
    // and feed_ngram_ports must feed it even when the table plan is empty.
    const std::vector<PortDims> q35 = {
        {"inputs_embeds", {1, -1, 2048}}, {"position_ids", {1, -1}},
        {"conv_mask", {1, -1}}, {"attention_mask", {1, -1}}, {"beam_idx", {-1}},
    };
    const auto plan = ngram::plan_ngram_ports(q35);
    CHECK(plan.empty());
    CHECK(!plan.declares_ids);
    CHECK(plan.declares_conv_mask);
    // a config that declares NO table is consistent with that absence: inert
    CHECK_EQ(ngram::check_declared_table(/*ngram_size=*/0, /*ple_embed_dim=*/0, plan),
             std::string(""));
}

TEST(ngram_ports_a_declared_table_with_no_port_is_refused_by_name) {
    // RED FIRST (2026-09-25): before this check the pair loaded silently -- the
    // config said "this checkpoint has a PLE", the graph declared no port to
    // carry it, and bind_ngram_ports returned inert, holding a table nothing
    // read. The absence of a table must be a FIRST-CLASS case, not a nullptr
    // that surfaces mid-decode. The control above keeps an empty plan alone
    // from passing this cell for the wrong reason.
    const std::vector<PortDims> q35 = {
        {"inputs_embeds", {1, -1, 2048}}, {"position_ids", {1, -1}}, {"conv_mask", {1, -1}},
    };
    const auto plan = ngram::plan_ngram_ports(q35);
    CHECK(plan.empty());
    const std::string why =
        ngram::check_declared_table(/*ngram_size=*/3, /*ple_embed_dim=*/2560, plan);
    CHECK(!why.empty());
    CHECK(why.find("declares an n-gram table") != std::string::npos);
    CHECK(why.find("silently absent") != std::string::npos);
    // With the table's own ports present the check is a no-op.
    CHECK_EQ(ngram::check_declared_table(3, 2560, ngram::plan_ngram_ports(real_ports())),
             std::string(""));
}

TEST(ngram_ports_refuses_a_gap_in_the_chunk_numbering) {
    auto in = real_ports();
    for (auto& p : in) {
        if (p.first == "ngram_table.2") p.first = "ngram_table.9";
    }
    const std::string why = refusal(in);
    CHECK(why.find("chunk 2 missing") != std::string::npos);
}

TEST(ngram_ports_refuses_a_dynamic_or_misshaped_chunk) {
    auto in = real_ports();
    for (auto& p : in) {
        if (p.first == "ngram_table.3") p.second = {-1, kRowBytes};
    }
    CHECK(refusal(in).find("static rank-2") != std::string::npos);
    in = real_ports();
    for (auto& p : in) {
        if (p.first == "ngram_table.3") p.second = {kPerChunk, kRowBytes, 1};
    }
    CHECK(refusal(in).find("static rank-2") != std::string::npos);
}

TEST(ngram_ports_refuses_row_bytes_that_disagree) {
    auto in = real_ports();
    for (auto& p : in) {
        if (p.first == "ngram_table.4") p.second = {kPerChunk, 80};
    }
    const std::string why = refusal(in);
    CHECK(why.find("80 bytes a row") != std::string::npos);
    CHECK(why.find("chunk 0 has 90") != std::string::npos);
}

TEST(ngram_ports_refuses_a_short_chunk_that_is_not_the_last) {
    auto in = real_ports();
    for (auto& p : in) {
        if (p.first == "ngram_table.1") p.second = {kPerChunk - 4096, kRowBytes};
    }
    CHECK(refusal(in).find("every chunk but the last") != std::string::npos);
    // ... and a last chunk LONGER than the first is just as wrong
    in = real_ports();
    for (auto& p : in) {
        if (p.first == "ngram_table.6") p.second = {kPerChunk + 1, kRowBytes};
    }
    CHECK(refusal(in).find("more than chunk 0") != std::string::npos);
}

TEST(ngram_ports_refuses_half_declared_id_ports) {
    auto in = real_ports();
    in.erase(std::remove_if(in.begin(), in.end(),
                            [](const PortDims& p) { return p.first == "ngram_local_ids"; }),
             in.end());
    const std::string why = refusal(in);
    CHECK(why.find("declared together") != std::string::npos);
}

TEST(ngram_ports_matches_the_gguf_table_tensor) {
    const auto plan = ngram::plan_ngram_ports(real_ports());
    const auto t    = table_tensor();
    const size_t bytes = static_cast<size_t>(kRows) * static_cast<size_t>(kRowBytes);
    CHECK_EQ(ngram::check_table_source(t, bytes, plan), std::string(""));
    // 28,800,138,240 bytes is the shipped tensor's own size
    CHECK_EQ(bytes, static_cast<size_t>(28800138240ULL));
}

TEST(ngram_ports_refuses_a_source_that_is_not_the_ports_format) {
    const auto plan = ngram::plan_ngram_ports(real_ports());
    const size_t bytes = static_cast<size_t>(kRows) * static_cast<size_t>(kRowBytes);
    // Q4_0 (type 2) rows would be 18 bytes a block too, but the graph decodes
    // IQ4_NL's codebook: the type is the contract, not the byte count
    CHECK(ngram::check_table_source(table_tensor(/*type=*/2), bytes, plan).find("IQ4_NL") !=
          std::string::npos);
    // a 128-wide row is 72 bytes, not the ports' 90
    CHECK(ngram::check_table_source(table_tensor(ngram::kIq4NlType, 128), bytes, plan)
              .find("the ports take 90") != std::string::npos);
    // a table of a different length than the ports partition
    CHECK(ngram::check_table_source(table_tensor(ngram::kIq4NlType, 160, kRows - 1), bytes, plan)
              .find("the ports partition") != std::string::npos);
    // a byte size that is not rows x row_bytes
    CHECK(ngram::check_table_source(table_tensor(), bytes - 1, plan).find("would be") !=
          std::string::npos);
}

TEST(ngram_ports_splits_ids_at_the_partition_like_the_python_twin) {
    const auto plan = ngram::plan_ngram_ports(real_ports());
    // both edges of chunk 0, the first row of chunk 1, the last row of the
    // table, and the min/max ids of the real-weight MEASUREMENT (window-050
    // §4.8 run 2, hash ordinal 0). Run 1's 4,023,550 / 317,350,792 were the
    // retracted ordinal-1 reading and are not cited here (REVIEW F2).
    const std::vector<int64_t> global = {0, kPerChunk - 1, kPerChunk, kRows - 1, 7226134, 316425755};
    std::vector<int32_t> chunk;
    std::vector<int64_t> local;
    ngram::split_by_partition(global, plan, chunk, local);
    const std::vector<int32_t> want_chunk = {0, 0, 1, 6, 0, 6};
    const std::vector<int64_t> want_local = {0, kPerChunk - 1, 0, kLastChunk - 1, 7226134,
                                             316425755 - 6 * kPerChunk};
    CHECK_EQ(chunk, want_chunk);
    CHECK_EQ(local, want_local);
    // and every local row is inside its chunk
    for (size_t i = 0; i < global.size(); ++i) {
        CHECK(local[i] >= 0);
        CHECK(local[i] < static_cast<int64_t>(plan.chunks[static_cast<size_t>(chunk[i])].rows));
    }
}

TEST(ngram_ports_refuses_an_id_outside_the_table) {
    const auto plan = ngram::plan_ngram_ports(real_ports());
    std::vector<int32_t> chunk;
    std::vector<int64_t> local;
    bool threw = false;
    try {
        ngram::split_by_partition({kRows}, plan, chunk, local);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("outside the table") != std::string::npos;
    }
    CHECK(threw);
}

// 0077: device-free tests for the VRAM alias check, a sampled deviation from
// Strata's full scan.
//   - the plan: blocks are 64 KiB aligned, inside the allocation, unique and
//     evenly spread; a small allocation is one block, an empty one none, and an
//     allocation larger than the sample takes GGML_CL_ALIAS_BLOCKS;
//   - the pattern: two blocks' patterns differ at byte 0, one block's equals
//     itself, so a read-back of an aliased block is not the expected one;
//   - the corruption class: a two-page model whose second write lands in the
//     first page's physical bytes reads the first back with the second's
//     pattern, so the check reports it; without the alias the check is clean;
//   - the call sites: ggml-opencl.cpp includes the header and scans the bank
//     stream ring whenever a learned order's ring is (re)allocated; a relearn
//     scans again; the USM host staging ring is not
//     scanned (host RAM, not a GPU page table); findings are named per block and
//     every scan ends with one summary line. No host-range pre-check remains
//     (the old 0077 class: different USM host ranges cannot overlap, so it could
//     never fire).
// No GPU, no OpenCL runtime.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-opencl-alias.h"

static int g_bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++g_bad; } } while (0)

static void check_plan(void) {
    // 32 MiB, the default staging ring: 512 blocks, the 16-block sample spread
    const size_t bytes = 32u << 20;
    size_t offs[GGML_CL_ALIAS_BLOCKS];
    const size_t k = ggml_cl_alias_offsets(bytes, offs, GGML_CL_ALIAS_BLOCKS);
    CHECK(k == GGML_CL_ALIAS_BLOCKS, "a 32 MiB allocation takes the full sample");
    for (size_t i = 0; i < k; ++i) {
        CHECK(offs[i] % GGML_CL_ALIAS_BLOCK == 0, "a block offset is 64 KiB aligned");
        CHECK(offs[i] + GGML_CL_ALIAS_BLOCK <= bytes, "a block lies inside the allocation");
        for (size_t j = 0; j < i; ++j) CHECK(offs[i] != offs[j], "the blocks are unique");
    }
    CHECK(offs[0] == 0, "the first block is at the start");
    // a small allocation: one block; an empty or sub-block one: none
    CHECK(ggml_cl_alias_offsets(GGML_CL_ALIAS_BLOCK, offs, GGML_CL_ALIAS_BLOCKS) == 1,
          "one block at one block's size");
    CHECK(ggml_cl_alias_offsets(GGML_CL_ALIAS_BLOCK - 1, offs, GGML_CL_ALIAS_BLOCKS) == 0,
          "no block below one block's size");
    CHECK(ggml_cl_alias_offsets(0, offs, GGML_CL_ALIAS_BLOCKS) == 0, "no block in an empty allocation");
    // a sample cap: the plan is bounded, not the allocation
    CHECK(ggml_cl_alias_offsets(1u << 30, offs, 4) == 4, "a billion bytes still takes only the cap");
}

static void check_pattern(void) {
    std::vector<uint8_t> a(GGML_CL_ALIAS_BLOCK), b(GGML_CL_ALIAS_BLOCK), c(GGML_CL_ALIAS_BLOCK);
    ggml_cl_alias_pattern(7, a.data());
    ggml_cl_alias_pattern(8, b.data());
    ggml_cl_alias_pattern(7, c.data());
    CHECK(std::memcmp(a.data(), b.data(), GGML_CL_ALIAS_BLOCK) != 0, "two ids' blocks differ");
    CHECK(std::memcmp(a.data(), c.data(), GGML_CL_ALIAS_BLOCK) == 0, "one id's block equals itself");
    CHECK(std::memcmp(a.data(), b.data(), 8) != 0, "two ids differ at the block's first word");
}

// two pages: page 0 and page 1; with aliasing page 1's bytes are page 0's, so a
// write to page 1 overwrites page 0 (the class the check exists for)
static int simulate(bool aliased) {
    std::vector<uint8_t> page0(GGML_CL_ALIAS_BLOCK), page1(GGML_CL_ALIAS_BLOCK);
    std::vector<uint8_t> * v0 = &page0, * v1 = aliased ? &page0 : &page1;
    std::vector<std::vector<uint8_t>> expected(2, std::vector<uint8_t>(GGML_CL_ALIAS_BLOCK));
    ggml_cl_alias_pattern(0, expected[0].data());
    ggml_cl_alias_pattern(1, expected[1].data());
    std::memcpy(v0->data(), expected[0].data(), GGML_CL_ALIAS_BLOCK);
    std::memcpy(v1->data(), expected[1].data(), GGML_CL_ALIAS_BLOCK);
    int bad = 0;
    if (std::memcmp(v0->data(), expected[0].data(), GGML_CL_ALIAS_BLOCK) != 0) ++bad;
    if (std::memcmp(v1->data(), expected[1].data(), GGML_CL_ALIAS_BLOCK) != 0) ++bad;
    return bad;
}

static void check_corruption_class(void) {
    CHECK(simulate(true) > 0, "an aliased block is found by the write/read-back");
    CHECK(simulate(false) == 0, "distinct blocks read back clean");
}

static void check_call_sites(const std::string & src) {
    CHECK(src.find("#include \"ggml-opencl-alias.h\"") != std::string::npos,
          "ggml-opencl.cpp includes the alias check's plan");
    CHECK(src.find("ggml_cl_alias_scan_device(ctx->queue, b.buffer, b.size, \"bank stream ring\", &alias_blocks)") != std::string::npos,
          "the bank stream ring is scanned at each learned order's ring");
    CHECK(src.find("ggml_cl_alias_scan_host") == std::string::npos,
          "the USM host staging ring is not scanned");
    CHECK(src.find("clEnqueueWriteBuffer(q, mem, CL_TRUE") != std::string::npos &&
              src.find("clEnqueueReadBuffer(q, mem, CL_TRUE") != std::string::npos,
          "the device scan writes all blocks and reads them back");
    CHECK(src.find("block %zu at %zu reads back differently") != std::string::npos,
          "a finding names the block");
    CHECK(src.find("alias check, bank stream ring: %d of %zu sampled blocks differ") != std::string::npos,
          "every scan ends with one summary line");
    // the old class: a host-range registry of distinct allocations cannot overlap
    CHECK(src.find("ggml_cl_ranges_overlap") == std::string::npos &&
              src.find("ggml_cl_note_alloc") == std::string::npos,
          "no host-range pre-check remains");
}

int main() {
    check_plan();
    check_pattern();
    check_corruption_class();
    std::ifstream f(GGML_OPENCL_SRC_PATH);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", GGML_OPENCL_SRC_PATH);
        ++g_bad;
    } else {
        const std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        check_call_sites(src);
    }
    if (g_bad) {
        std::printf("test-opencl-alias: %d failure(s)\n", g_bad);
        return 1;
    }
    std::printf("test-opencl-alias: ok\n");
    return 0;
}

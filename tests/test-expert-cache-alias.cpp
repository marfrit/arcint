// 0077: device-free tests for the slot-buffer sampled alias tripwire.
//   - the region arithmetic over every planar type the OpenCL set path converts
//     slice by slice: the offset is a multiple of the tensor's nb[2], the size is
//     exactly nb[2], inside the tensor;
//   - an empty slice or a tensor shorter than one slice has no region;
//   - the pattern: two ids differ at byte 0, one id equals itself; one differing
//     read-back is one finding;
//   - llama-expert-cache.cpp's call: the header is included, the planar types are
//     the ones sampled, all writes precede the read-backs, a finding warns and
//     the load proceeds, and the env switch is present.
// No GPU, no llama model; ggml is linked for the type sizes.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama-expert-cache-alias.h"

static int g_bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++g_bad; } } while (0)

static void check_planar_regions(void) {
    const ggml_type types[] = { GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                                GGML_TYPE_Q2_0,    GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS };
    for (ggml_type type : types) {
        // a 3-D slot tensor: ne0 = 256, ne1 = 256; nb[2] is one 2-D slice
        const size_t row  = ggml_row_size(type, 256);
        const size_t nb2  = row * 256;
        CHECK((256 * 256 / ggml_blck_size(type)) % 2 == 0, "the planar type's block count is even");
        const size_t bytes = nb2 * 8;
        for (size_t i = 0; i < 4; ++i) {
            const llama_cache_alias_span s = llama_cache_alias_region(bytes, nb2, i);
            CHECK(s.ok, "a planar tensor of eight slices has a region");
            CHECK(s.offset % nb2 == 0, "the region starts on a whole slice");
            CHECK(s.size == nb2, "the region is one whole slice");
            CHECK(s.offset + s.size <= bytes, "the region lies inside the tensor");
        }
        CHECK(llama_cache_alias_region(nb2, nb2, 0).ok, "one slice has a region");
        CHECK(llama_cache_alias_region(nb2 - 1, nb2, 0).ok == false, "less than one slice has no region");
    }
    // two consecutive tensors sample distinct, still slice-aligned starts
    const size_t nb2 = 110;
    const llama_cache_alias_span a = llama_cache_alias_region(nb2 * 64, nb2, 0);
    const llama_cache_alias_span b = llama_cache_alias_region(nb2 * 64, nb2, 1);
    CHECK(a.ok && b.ok && a.offset % nb2 == 0 && b.offset % nb2 == 0, "two tensors start on whole slices");
    CHECK(a.offset != b.offset, "consecutive tensors sample distinct regions");
    // a large slice is still one whole slice, not a capped block
    const size_t big = 40000;
    const llama_cache_alias_span c = llama_cache_alias_region(big * 8, big, 0);
    CHECK(c.ok && c.size == big, "a large slice is one whole slice");
}

static void check_pattern(void) {
    constexpr size_t BLOCK = 64u << 10;
    std::vector<uint8_t> a(BLOCK), b(BLOCK), c(BLOCK);
    llama_cache_alias_pattern(3, a.data(), a.size());
    llama_cache_alias_pattern(4, b.data(), b.size());
    llama_cache_alias_pattern(3, c.data(), c.size());
    CHECK(std::memcmp(a.data(), b.data(), BLOCK) != 0, "two ids' regions differ");
    CHECK(std::memcmp(a.data(), c.data(), BLOCK) == 0, "one id's region equals itself");
    // a slice-sized region whose tail is not a multiple of eight still fills
    std::vector<uint8_t> d(110), e(110);
    llama_cache_alias_pattern(5, d.data(), d.size());
    llama_cache_alias_pattern(6, e.data(), e.size());
    CHECK(std::memcmp(d.data(), e.data(), d.size()) != 0, "a tail-filled region differs from another id's");
    CHECK(llama_cache_alias_reads_back(3, a.data(), a.size()), "a read-back matches its own pattern");
    CHECK(!llama_cache_alias_reads_back(3, b.data(), b.size()), "one differing region is one finding");
    CHECK(llama_cache_alias_reads_back(3, c.data(), c.size()), "the same id's read-back matches");
}

static void check_call_site(const std::string & src) {
    CHECK(src.find("#include \"llama-expert-cache-alias.h\"") != std::string::npos,
          "llama-expert-cache.cpp includes the alias check");
    CHECK(src.find("llama_cache_alias_check(alias_slots)") != std::string::npos,
          "the slot buffers are checked at load");
    const size_t set = src.find("ggml_backend_tensor_set(t, region.data(), sp.offset, region.size())");
    const size_t get = src.find("ggml_backend_tensor_get(p.t, actual.data(), p.off, actual.size())");
    CHECK(set != std::string::npos && get != std::string::npos && set < get,
          "all writes precede the read-backs");
    for (const char * t : {"GGML_TYPE_IQ2_XXS", "GGML_TYPE_IQ2_XS", "GGML_TYPE_IQ2_S",
                           "GGML_TYPE_Q2_0", "GGML_TYPE_IQ3_S", "GGML_TYPE_IQ3_XXS"}) {
        CHECK(src.find(std::string("case ") + t + ":") != std::string::npos,
              "the planar types are the ones sampled");
    }
    for (const char * t : {"GGML_TYPE_Q4_0", "GGML_TYPE_Q1_0", "GGML_TYPE_Q8_0",
                           "GGML_TYPE_IQ4_NL", "GGML_TYPE_Q4_K"}) {
        CHECK(src.find(std::string("case ") + t + ":") == std::string::npos,
              "the whole-tensor SOA readers are not sampled");
    }
    CHECK(src.find("p.expected") == std::string::npos && src.find("sampled_bytes") != std::string::npos,
          "the plan holds no expected bytes and the summary reports them");
    CHECK(src.find("warn and proceed") != std::string::npos,
          "a finding warns and the load proceeds");
    CHECK(src.find("LLAMA_EXPERT_CACHE_ALIAS") != std::string::npos,
          "the record's off switch is present");
    CHECK(src.find("alias check, %d of %zu sampled slot regions differ, %zu bytes written and read back") != std::string::npos,
          "the finding is logged with its count and the sampled bytes");
}

int main() {
    check_planar_regions();
    check_pattern();
    std::ifstream f(LLAMA_EXPERT_CACHE_SRC_PATH);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", LLAMA_EXPERT_CACHE_SRC_PATH);
        ++g_bad;
    } else {
        const std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        check_call_site(src);
    }
    if (g_bad) {
        std::printf("test-expert-cache-alias: %d failure(s)\n", g_bad);
        return 1;
    }
    std::printf("test-expert-cache-alias: ok\n");
    return 0;
}

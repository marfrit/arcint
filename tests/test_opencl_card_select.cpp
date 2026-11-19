// The OpenCL card selection's pure half (src/exec/opencl_card_select.h): which
// platform and device index a --device word names, and the GGML_OPENCL_DEVICES
// value. The defect this guards: two Battlemage cards on one platform (B60 at
// 0:0, B70 at 0:1), where the old code matched only the platform's first GPU.
// One card per platform (the B60 + A770 pair) still has to resolve.
//
// Every case is RED against the stub (opencl_card_select.cpp returns an error
// for everything), and stays RED under one named one-line mutation of
// pick_cards / ggml_opencl_devices_value:
//   A the GPU ordinal compared the wrong way (`seen == ordinal` ->
//     `seen == ordinal + 1`) -> openclcard_gpu_ordinal_counts_gpus_across_platforms
//   B the name match made exact (`name.find(want) != npos` -> `name == want`)
//     -> openclcard_name_word_matches_one_gpu
//   B2 the non-GPU skip deleted -> openclcard_gpu_ordinal_counts_gpus_across_platforms
//      and openclcard_name_word_matches_one_gpu
//   C the duplicate check deleted -> openclcard_the_same_card_twice_is_refused
//   D a 0-match taken as success (`returned pick with empty error`) ->
//     openclcard_errors_name_the_word_and_candidates
//   E a 2-match resolved to its first (`if (matches.size() == 1)` -> `>= 1`) ->
//     openclcard_errors_name_the_word_and_candidates
//   F an out-of-range GPU.N clamped to the last GPU -> openclcard_gpu_ordinal_out_of_range_is_refused
//   G the GGML value joined with ";" instead of "," ->
//     openclcard_ggml_value_is_the_pick_in_asked_order
//   H the cards written/joined in enumeration order instead of the asked
//     order -> openclcard_ggml_value_is_the_pick_in_asked_order and
//     openclcard_two_cards_keep_the_asked_order

#include "exec/opencl_card_select.h"
#include "harness.h"

#include <initializer_list>
#include <string>
#include <vector>

using namespace lgc;

namespace {

ClDeviceDesc dev(int platform, int d, bool gpu, const char* name, uint64_t mem = 0) {
    ClDeviceDesc c;
    c.platform = platform;
    c.device   = d;
    c.gpu      = gpu;
    c.name     = name;
    c.mem_bytes = mem;
    return c;
}

ClDeviceDesc gpu(int platform, int d, const char* name) {
    return dev(platform, d, true, name, (uint64_t)24 << 30);
}
ClDeviceDesc cpu(int platform, int d, const char* name) {
    return dev(platform, d, false, name, (uint64_t)64 << 30);
}

// The new card host: one platform, the B60 then the B70 (device order).
std::vector<ClDeviceDesc> new_host() {
    return {gpu(0, 0, "Intel(R) Arc(TM) Pro B60 Graphics"),
            gpu(0, 1, "Intel(R) Arc(TM) Pro B70 Graphics")};
}

// The B60 + A770 host: one platform a card, as the runtime listed it.
std::vector<ClDeviceDesc> old_host() {
    return {gpu(0, 0, "Intel(R) Arc(TM) Pro B60 Graphics"),
            gpu(1, 0, "Intel(R) Arc(TM) A770 Graphics")};
}

// oneAPI lists a CPU OpenCL device first; it is not a GPU and must never be
// picked, but the GPU ordinal counts GPUs only.
std::vector<ClDeviceDesc> cpu_first_host() {
    return {cpu(0, 0, "Intel(R) Core(TM) i9-13900K CPU"),
            gpu(1, 0, "Intel(R) Arc(TM) Pro B60 Graphics"),
            gpu(1, 1, "Intel(R) Arc(TM) Pro B70 Graphics")};
}

std::vector<std::string> words(std::initializer_list<const char*> w) {
    std::vector<std::string> out;
    for (const char* s : w) out.emplace_back(s);
    return out;
}

bool has(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}

// One word resolves to exactly one card, at platform:device, with that name.
void check_one(const std::vector<ClDeviceDesc>& all, const char* want, int platform,
               int d, const char* name) {
    const ClPick r = pick_cards(all, words({want}));
    CHECK_EQ(r.error, std::string());
    if (r.cards.size() != 1) {
        CHECK_EQ(r.cards.size(), 1u);
        return;
    }
    CHECK_EQ(r.cards[0].platform, platform);
    CHECK_EQ(r.cards[0].device, d);
    CHECK_EQ(r.cards[0].name, std::string(name));
    CHECK(r.cards[0].gpu);
}

// A refusal: non-empty error, no cards, and every token named must be in the
// error text (the word, the candidates with their P:D).
void check_refused(const std::vector<ClDeviceDesc>& all,
                   const std::vector<std::string>& wanted,
                   std::initializer_list<const char*> tokens) {
    const ClPick r = pick_cards(all, wanted);
    CHECK(!r.error.empty());
    CHECK(r.cards.empty());
    for (const char* tok : tokens) CHECK(has(r.error, tok));
}

}  // namespace

TEST(openclcard_gpu_ordinal_counts_gpus_across_platforms) {
    // new host: GPU.0 and GPU both name the B60, GPU.1 the B70 on the same platform
    check_one(new_host(), "GPU.0", 0, 0, "Intel(R) Arc(TM) Pro B60 Graphics");
    check_one(new_host(), "GPU", 0, 0, "Intel(R) Arc(TM) Pro B60 Graphics");
    check_one(new_host(), "GPU.1", 0, 1, "Intel(R) Arc(TM) Pro B70 Graphics");
    // old host: the ordinal runs over platforms, one GPU each
    check_one(old_host(), "GPU.0", 0, 0, "Intel(R) Arc(TM) Pro B60 Graphics");
    check_one(old_host(), "GPU.1", 1, 0, "Intel(R) Arc(TM) A770 Graphics");
    // CPU platform first: the CPU is skipped, the B60 is GPU.0 at 1:0
    check_one(cpu_first_host(), "GPU.0", 1, 0, "Intel(R) Arc(TM) Pro B60 Graphics");
    check_one(cpu_first_host(), "GPU.1", 1, 1, "Intel(R) Arc(TM) Pro B70 Graphics");
}

TEST(openclcard_name_word_matches_one_gpu) {
    check_one(new_host(), "B70", 0, 1, "Intel(R) Arc(TM) Pro B70 Graphics");
    check_one(new_host(), "B60", 0, 0, "Intel(R) Arc(TM) Pro B60 Graphics");
    check_one(old_host(), "A770", 1, 0, "Intel(R) Arc(TM) A770 Graphics");
    // the substring crosses the CPU's name too, but only a GPU is a match
    check_one(cpu_first_host(), "B70", 1, 1, "Intel(R) Arc(TM) Pro B70 Graphics");
    check_refused(cpu_first_host(), words({"CPU"}), {"CPU"});
}

TEST(openclcard_two_cards_keep_the_asked_order) {
    const ClPick r = pick_cards(new_host(), words({"B70", "B60"}));
    CHECK_EQ(r.error, std::string());
    if (r.cards.size() != 2) {
        CHECK_EQ(r.cards.size(), 2u);
        return;
    }
    CHECK_EQ(r.cards[0].platform, 0);
    CHECK_EQ(r.cards[0].device, 1);
    CHECK_EQ(r.cards[0].name, std::string("Intel(R) Arc(TM) Pro B70 Graphics"));
    CHECK_EQ(r.cards[1].platform, 0);
    CHECK_EQ(r.cards[1].device, 0);
    CHECK_EQ(r.cards[1].name, std::string("Intel(R) Arc(TM) Pro B60 Graphics"));
}

TEST(openclcard_errors_name_the_word_and_candidates) {
    // no match: the word and the cards that were candidates, with their P:D
    check_refused(new_host(), words({"B80"}), {"B80", "0:0", "0:1"});
    // two matches: the word and both matches, with their P:D
    check_refused(new_host(), words({"Arc"}), {"Arc", "0:0", "0:1"});
}

TEST(openclcard_gpu_ordinal_out_of_range_is_refused) {
    // two GPUs, GPU.2 names a third that is not there
    check_refused(new_host(), words({"GPU.2"}), {"GPU.2"});
    // the CPU platform is not counted: GPU.1 is the last GPU there
    check_refused(cpu_first_host(), words({"GPU.2"}), {"GPU.2"});
}

TEST(openclcard_the_same_card_twice_is_refused) {
    // by name twice
    check_refused(new_host(), words({"B60", "B60"}), {"B60"});
    // two words that resolve to the same card (GPU.0 and B60 are both 0:0)
    check_refused(new_host(), words({"GPU.0", "B60"}), {"0:0"});
    // the same card by its ordinal and its name on the old host
    check_refused(old_host(), words({"GPU.1", "A770"}), {"1:0"});
}

TEST(openclcard_empty_enumeration_is_refused) {
    check_refused(std::vector<ClDeviceDesc>{}, words({"GPU.0"}), {"GPU.0"});
    check_refused(std::vector<ClDeviceDesc>{}, words({"B60"}), {"B60"});
}

TEST(openclcard_ggml_value_is_the_pick_in_asked_order) {
    // one card, and two cards in the order asked
    const std::vector<ClDeviceDesc> b70{new_host()[1]};
    CHECK_EQ(ggml_opencl_devices_value(b70), std::string("0:1"));
    CHECK_EQ(ggml_opencl_devices_value(pick_cards(new_host(), words({"B60", "B70"})).cards),
             std::string("0:0,0:1"));
    CHECK_EQ(ggml_opencl_devices_value(pick_cards(new_host(), words({"B70", "B60"})).cards),
             std::string("0:1,0:0"));
    // one card per platform: the A770 is 1:0
    CHECK_EQ(ggml_opencl_devices_value(pick_cards(old_host(), words({"A770"})).cards),
             std::string("1:0"));
    // the CPU platform is not in the value
    CHECK_EQ(ggml_opencl_devices_value(pick_cards(cpu_first_host(), words({"B70"})).cards),
             std::string("1:1"));
}

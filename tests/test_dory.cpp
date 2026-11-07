#include "core/dory.h"
#include "harness.h"

#include <string>

using namespace lgc;

// Dory1.0-4B's options and its loop arithmetic (core/dory.h): pure decisions, no model.
namespace {

// the file's own metadata (code: the GGUF's keys): 16 + 40 + 16 physical blocks, 4 loops, per_loop
DoryGguf file() {
    DoryGguf g;
    g.n_input = 16;
    g.n_recurrent = 40;
    g.n_output = 16;
    g.loops = 4;
    g.kv_mode = "per_loop";
    // the file's pattern: S-S-S-* then a dash, nine times (the dashes are MLP blocks, no KV cache)
    for (int i = 0; i < 9; ++i) g.pattern += "S-S-S-*-";
    return g;
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

}  // namespace

TEST(dory_refuses_mtp_dflash_and_layer_split) {
    const DoryOptions none;
    CHECK(has(dory_option_refusal("dory", true, false, false, none), "--llama-mtp"));
    CHECK(has(dory_option_refusal("dory", false, true, false, none), "--llama-dflash"));
    CHECK(has(dory_option_refusal("dory", false, false, true, none), "--llama-layer-split"));
    CHECK(has(dory_option_refusal("dory", true, true, true, none), "no MTP head"));
    // the combination that is fine
    CHECK(dory_option_refusal("dory", false, false, false, none).empty());
    DoryOptions o;
    o.loops = 2;
    o.kv_mode = "last_loop";
    CHECK(dory_option_refusal("dory", false, false, false, o).empty());
}

TEST(dory_flags_on_another_architecture_are_refused_and_other_flags_are_not) {
    DoryOptions loops;
    loops.loops = 2;
    DoryOptions kv;
    kv.kv_mode = "per_loop";
    CHECK(has(dory_option_refusal("qwen35moe", false, false, false, loops), "--llama-dory-loops"));
    CHECK(has(dory_option_refusal("qwen35moe", false, false, false, kv), "--llama-dory-kv"));
    CHECK(has(dory_option_refusal("qwen35moe", false, false, false, loops), "'qwen35moe'"));
    // MTP, DFlash2 and the layer split are not a Dory matter on any other model
    const DoryOptions none;
    CHECK(dory_option_refusal("qwen35moe", true, false, true, none).empty());
    CHECK(dory_option_refusal("qwen35", false, true, false, none).empty());
    CHECK(dory_option_refusal("llama", true, true, true, none).empty());
}

TEST(dory_resolve_serves_the_files_own_values_and_says_so) {
    DoryServed s;
    CHECK(dory_resolve(file(), DoryOptions{}, s).empty());
    CHECK_EQ(s.loops, 4);
    CHECK_EQ(s.kv_mode, std::string("per_loop"));
    CHECK_EQ(s.loops_source, std::string("gguf"));
    CHECK_EQ(s.kv_source, std::string("gguf"));
    CHECK_EQ(s.physical_layers, 72);
    CHECK_EQ(s.executed_layers, 192);   // 16 + 40 * 4 + 16
    CHECK_EQ(s.kv_layers, 96);          // 8 + 20 * 4 + 8 attention blocks hold a cache
}

TEST(dory_resolve_a_flag_wins_and_the_layer_count_follows) {
    DoryOptions o;
    o.loops = 2;
    o.kv_mode = "last_loop";
    DoryServed s;
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK_EQ(s.loops, 2);
    CHECK_EQ(s.executed_layers, 112);   // 16 + 40 * 2 + 16
    CHECK_EQ(s.kv_layers, 36);          // last_loop: 8 + 20 + 8, each loop overwriting one cache
    CHECK_EQ(s.physical_layers, 72);    // the weights do not change
    CHECK_EQ(s.kv_mode, std::string("last_loop"));
    CHECK_EQ(s.loops_source, std::string("flag"));
    CHECK_EQ(s.kv_source, std::string("flag"));
    o.loops = 1;
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK_EQ(s.executed_layers, 72);
}

TEST(dory_resolve_an_absent_kv_mode_is_libllamas_default_and_named_so) {
    DoryGguf g = file();
    g.kv_mode.clear();
    DoryServed s;
    CHECK(dory_resolve(g, DoryOptions{}, s).empty());
    CHECK_EQ(s.kv_mode, std::string("per_loop"));
    CHECK_EQ(s.kv_source, std::string("default"));
}

TEST(dory_resolve_refuses_what_the_port_does_not_run) {
    DoryServed s;
    DoryGguf g = file();
    g.loops = 0;   // no key, no flag: libllama would have none either
    CHECK(has(dory_resolve(g, DoryOptions{}, s), "--llama-dory-loops"));
    DoryOptions o;
    o.loops = 2;
    CHECK(dory_resolve(g, o, s).empty());   // the flag supplies it
    g = file();
    g.loops = 7;
    CHECK(has(dory_resolve(g, DoryOptions{}, s), "1 to 4"));
    g = file();
    g.kv_mode = "shared";
    CHECK(has(dory_resolve(g, DoryOptions{}, s), "per_loop or last_loop"));
    o = DoryOptions{};
    o.kv_mode = "shared";
    CHECK(has(dory_resolve(file(), o, s), "per_loop or last_loop"));
    o = DoryOptions{};
    o.loops = 5;
    CHECK(has(dory_resolve(file(), o, s), "1 to 4"));
    g = file();
    g.n_recurrent = 0;
    CHECK(!dory_resolve(g, DoryOptions{}, s).empty());
}

TEST(dory_check_loaded_catches_an_override_that_did_not_take) {
    DoryOptions o;
    o.loops = 2;
    DoryServed s;
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK(dory_check_loaded(s, 112, 72).empty());
    // libllama still built the file's 4 loops: the flag was a fallback, not a setting
    CHECK(has(dory_check_loaded(s, 192, 72), "did not take effect"));
    CHECK(has(dory_check_loaded(s, 112, 56), "physical blocks"));
}

TEST(dory_kv_layers_follow_the_mode_and_the_loops) {
    DoryServed s;
    DoryOptions o;
    o.loops = 2;
    o.kv_mode = "per_loop";
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK_EQ(s.kv_layers, 56);   // 8 + 20 * 2 + 8
    o.kv_mode = "last_loop";
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK_EQ(s.kv_layers, 36);
    o.loops = 1;
    o.kv_mode = "per_loop";
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK_EQ(s.kv_layers, 36);   // one loop: the two modes coincide
    // a pattern that does not describe the blocks is refused
    DoryGguf g = file();
    g.pattern.pop_back();
    CHECK(has(dory_resolve(g, DoryOptions{}, s), "dory.layer_pattern"));
    g = file();
    g.pattern[3] = 'x';
    CHECK(has(dory_resolve(g, DoryOptions{}, s), "dory.layer_pattern"));
}

TEST(dory_check_kv_catches_a_kv_mode_that_did_not_take) {
    DoryOptions o;
    o.kv_mode = "last_loop";
    DoryServed s;
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK(dory_check_kv(s, 36).empty());
    // libllama built a cache a loop and layer: the flag was echoed, not applied
    CHECK(has(dory_check_kv(s, 96), "did not take effect"));
    o.kv_mode = "per_loop";
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK(dory_check_kv(s, 96).empty());
    CHECK(!dory_check_kv(s, 36).empty());
}

TEST(dory_kv_size_and_the_fit_warning_at_the_default_context) {
    const std::uint64_t gib = 1ull << 30;
    // 96 layers, 4 KV heads of 256, 32768 cells: the default context at f16 is 12 GiB
    const std::uint64_t f16 = dory_kv_bytes(96, 4, 256, 32768, "f16", "f16");
    CHECK_EQ(f16, 12 * gib);
    const std::uint64_t q = dory_kv_bytes(96, 4, 256, 32768, "q8_0", "q4_0");
    CHECK(q > 4 * gib && q < 5 * gib);
    const std::uint64_t weights = 3700ull << 20;   // the Q5_K_M file, about 3.6 GiB
    // the A770 (16 GiB): f16 at the default context does not fit, q8_0:q4_0 does
    CHECK(has(dory_fit_warning(f16, weights, 16 * gib, 32768, 96, "f16", "f16"), "forecast past the card"));
    CHECK(dory_fit_warning(q, weights, 16 * gib, 32768, 96, "q8_0", "q4_0").empty());
    // the B60 (24 GiB) holds f16
    CHECK(dory_fit_warning(f16, weights, 24 * gib, 32768, 96, "f16", "f16").empty());
    // a card whose size is unknown is not warned about
    CHECK(dory_fit_warning(f16, weights, 0, 32768, 96, "f16", "f16").empty());
}

TEST(dory_boot_line_names_the_values_and_their_sources) {
    DoryServed s;
    CHECK(dory_resolve(file(), DoryOptions{}, s).empty());
    const std::string line = dory_boot_line(s);
    CHECK(has(line, "loops 4 (gguf)"));
    CHECK(has(line, "recurrent KV per_loop (gguf)"));
    CHECK(has(line, "72 blocks"));
    CHECK(has(line, "192 layers executed"));
    CHECK(has(line, "96 KV cache layers"));
}

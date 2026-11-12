#include "core/dory.h"
#include "harness.h"

#include <cmath>
#include <cstdint>
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
    const std::uint64_t weights = 3496ull << 20;   // the Q5_K_M file's tensor bytes, 3.4 GiB
    const DoryCardForecast ff = dory_card_forecast(weights, f16, 32768, 512);
    const DoryCardForecast fq = dory_card_forecast(weights, q, 32768, 512);
    // the A770 (16 GiB): f16 at the default context does not fit, q8_0:q4_0 does
    CHECK(has(dory_fit_warning(ff, 16 * gib, 32768, 96, "f16", "f16"), "forecast past the card"));
    CHECK(dory_fit_warning(fq, 16 * gib, 32768, 96, "q8_0", "q4_0").empty());
    // the B60 (24 GiB) holds f16
    CHECK(dory_fit_warning(ff, 24 * gib, 32768, 96, "f16", "f16").empty());
    // a card whose size is unknown is not warned about
    CHECK(dory_fit_warning(ff, 0, 32768, 96, "f16", "f16").empty());
    // 31,232 cells at f16: the weights and the cache alone (15,208 MiB) are past 92 % of the card (15,073), where
    // the rule before this one warned; the whole forecast (16,189 MiB) is under the card's 16,384, so no warning
    const std::uint64_t f31 = dory_kv_bytes(96, 4, 256, 31232, "f16", "f16");
    CHECK(static_cast<double>(weights + f31) > 0.92 * static_cast<double>(16 * gib));
    CHECK(dory_fit_warning(dory_card_forecast(weights, f31, 31232, 512), 16 * gib, 31232, 96, "f16", "f16").empty());
}

namespace {
const double kMiB = 1024.0 * 1024.0;
double mib(std::uint64_t b) { return static_cast<double>(b) / kMiB; }
// the served configuration of the measurement (measured-here, an A770, 2026-11-12): the Q5_K_M file, 2 loops,
// last_loop (36 cache layers), q8_0:q4_0, and the weights libllama placed on the card, from its own line
const char* const kLoadLines =
    "lgc  llama: load_tensors: offloaded 73/73 layers to GPU\n"
    "lgc  llama: load_tensors:          CPU model buffer size =   642.15 MiB\n"
    "lgc  llama: load_tensors:       OpenCL model buffer size =  2853.58 MiB\n";
DoryCardForecast served(std::int64_t cells, int ubatch) {
    return dory_card_forecast(dory_card_weight_bytes(kLoadLines), dory_kv_bytes(36, 4, 256, cells, "q8_0", "q4_0"), cells, ubatch);
}
}  // namespace

TEST(dory_card_weights_are_read_from_the_load_lines) {
    // the card's buffer counts, the host's does not: 2853.58 MiB, not the file's 3495.73
    CHECK(std::abs(mib(dory_card_weight_bytes(kLoadLines)) - 2853.58) < 0.001);
    // libllama's own text, without arcint's log prefix
    CHECK(std::abs(mib(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size =  2853.58 MiB")) - 2853.58) < 0.001);
    // host buffers under each of libllama's names
    CHECK_EQ(dory_card_weight_bytes("load_tensors:          CPU model buffer size =   642.15 MiB\n"
                                    "load_tensors:   CPU_Mapped model buffer size =   642.15 MiB\n"
                                    "load_tensors:   CPU_REPACK model buffer size =    10.00 MiB\n"),
             0u);
    // two card buffers add up
    CHECK(std::abs(mib(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size =  1000.25 MiB\n"
                                              "load_tensors:       OpenCL model buffer size =   500.25 MiB\n")) -
                   1500.5) < 0.001);
    // other text, another unit, no number: nothing is read (the caller then says so and takes the file's size)
    CHECK_EQ(dory_card_weight_bytes(""), std::uint64_t{0});
    CHECK_EQ(dory_card_weight_bytes("llama_kv_cache:     OpenCL KV buffer size =  1872.01 MiB\n"), std::uint64_t{0});
    CHECK_EQ(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size =  2853.58 GiB\n"), std::uint64_t{0});
    CHECK_EQ(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size = \n"), std::uint64_t{0});
    CHECK_EQ(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size = nan MiB\n"), std::uint64_t{0});
    CHECK_EQ(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size = inf MiB\n"), std::uint64_t{0});
    CHECK_EQ(dory_card_weight_bytes("load_tensors:       OpenCL model buffer size = -5.00 MiB\n"), std::uint64_t{0});
}

TEST(dory_compute_buffer_is_libllamas_own_figure_at_the_measured_points) {
    // libllama's "OpenCL compute buffer size" (measured-here, at context teardown, four decimals)
    struct Row { std::int64_t cells; int ubatch; double want; };
    const Row rows[] = { { 8192, 512, 78.0216 }, { 8192, 256, 39.0109 }, { 131072, 512, 318.0216 }, { 131072, 256, 159.0109 } };
    for (const Row& r : rows) CHECK(std::abs(mib(dory_compute_bytes(r.cells, r.ubatch)) - r.want) < 0.001);
    // the context term is the two f16 masks: 4 bytes a cell and micro-batch row
    CHECK_EQ(dory_compute_bytes(131072, 512) - dory_compute_bytes(8192, 512), std::uint64_t{4} * 512 * (131072 - 8192));
    CHECK_EQ(dory_compute_bytes(0, 512), std::uint64_t{0});
    CHECK_EQ(dory_compute_bytes(8192, 0), std::uint64_t{0});
    // the mask spans one sequence's cells (code: libllama's n_ctx_seq; lanes are not measured): the pool with one
    // lane or a unified cache, a lane's share with two unnamed lanes, which is 192 MiB less at 2 x 98,304
    CHECK_EQ(dory_mask_cells(196608, 1, false), std::int64_t{196608});
    CHECK_EQ(dory_mask_cells(196608, 2, false), std::int64_t{98304});
    CHECK_EQ(dory_mask_cells(163840, 2, true), std::int64_t{163840});
    CHECK_EQ(dory_compute_bytes(196608, 512) - dory_compute_bytes(dory_mask_cells(196608, 2, false), 512), std::uint64_t{192} << 20);
    // libllama pads the pool, and a lane's share, to 256 cells (code: llama_context's constructor): the forecast
    // takes the cells it allocates, not the flag's
    CHECK_EQ(dory_pool_cells(196608, 1, false), std::int64_t{196608});
    CHECK_EQ(dory_pool_cells(197888, 1, false), std::int64_t{197888});
    CHECK_EQ(dory_pool_cells(197889, 1, false), std::int64_t{198144});
    CHECK_EQ(dory_pool_cells(100000, 2, false), std::int64_t{100352});   // 100,096 / 2 = 50,048 -> 50,176 a lane
    CHECK_EQ(dory_mask_cells(100000, 2, false), std::int64_t{50176});
    CHECK_EQ(dory_pool_cells(100000, 2, true), std::int64_t{100096});
    CHECK_EQ(dory_mask_cells(197889, 1, false), std::int64_t{198144});
    // --llama-reserve-cells R keeps the buffer at R cells (code: n_ubatch_eff in llama_context::decode; not measured
    // for Dory): the mask spans the padded reserve, never more than the sequence
    CHECK_EQ(dory_mask_cells(196608, 1, false, 20480), std::int64_t{20480});
    CHECK_EQ(dory_mask_cells(196608, 1, false, 20000), std::int64_t{20224});
    CHECK_EQ(dory_mask_cells(8192, 1, false, 20480), std::int64_t{8192});
    CHECK_EQ(dory_compute_bytes(196608, 512) - dory_compute_bytes(dory_mask_cells(196608, 1, false, 20480), 512),
             std::uint64_t{344} << 20);
}

TEST(dory_card_forecast_reproduces_the_measured_peaks) {
    // the card's resident memory at the deepest fill of each context (measured-here, fdinfo, MiB; micro-batch 512):
    // the forecast is a peak figure, so it must never be below a measured peak; it stands 24.9 / 16.8 / 0.7 / 0.6
    // MiB above the four of the series the constants come from, and 0.2 MiB above the peak of the later run at
    // 196,608 cells (15,389.4 during prompts to 176,888 tokens; 15,389.1 resident after the deepest), so the three
    // deepest leave under 1 MiB. Every distance is positive and pinned to 0.1 MiB
    struct Row { std::int64_t cells; double peak; double over; };
    const Row rows[] = { { 8192, 4232.7, 24.9 }, { 32768, 5692.8, 16.8 }, { 65536, 7644.9, 0.7 }, { 131072, 11517.0, 0.6 },
                         { 196608, 15389.4, 0.2 } };
    for (const Row& r : rows) CHECK(std::abs(mib(served(r.cells, 512).total()) - r.peak - r.over) < 0.1);
    // micro-batch 256, 8,192 cells, filled to 90 %: 4,186.2 MiB (its backend term stayed 32 MiB under the peak constant)
    CHECK(std::abs(mib(served(8192, 256).total()) - 4186.2 - 32.4) < 0.1);
    // the terms at 131,072 cells: the cache is 7,488.0 MiB (libllama's two lines read 1,872.01 + 5,616.03), the
    // weights its card buffer, the compute buffer its own figure, and 858 MiB of backend memory: 11,517.6 MiB
    const DoryCardForecast f = served(131072, 512);
    CHECK(std::abs(mib(f.kv) - 7488.0) < 0.05);
    CHECK(std::abs(mib(f.weights) - 2853.58) < 0.001);
    CHECK(std::abs(mib(f.compute) - 318.0216) < 0.001);
    CHECK(std::abs(mib(f.backend) - 858.0) < 0.001);
    CHECK(std::abs(mib(f.total()) - 11517.6) < 0.05);
}

TEST(dory_fit_warning_against_the_a770s_opencl_figure) {
    // Intel's runtime reports 16,225,243,136 bytes of global memory for the A770's 16,288 MiB (measured-here:
    // clinfo's CL_DEVICE_GLOBAL_MEM_SIZE on the card host; 15,473.6 MiB, which ggml prints truncated as "global
    // mem size: 15473 MB"); the threshold moves 17 cells a MiB with it
    const std::uint64_t card = 16225243136ull;
    // 131,072 and 196,608 cells are under it (the latter by 84.0 MiB), 204,800 and 262,144 are past it
    CHECK(dory_fit_warning(served(131072, 512), card, 131072, 36, "q8_0", "q4_0").empty());
    const double t196 = mib(served(196608, 512).total());
    CHECK(t196 > 15380.0 && t196 < 15400.0);
    CHECK(dory_fit_warning(served(196608, 512), card, 196608, 36, "q8_0", "q4_0").empty());
    const double t204 = mib(served(204800, 512).total());
    CHECK(t204 > 15865.0 && t204 < 15885.0);
    const std::string w = dory_fit_warning(served(204800, 512), card, 204800, 36, "q8_0", "q4_0");
    CHECK(has(w, "forecast past the card"));
    CHECK(has(w, "15874 MiB"));
    CHECK(has(w, "36 layers x 204800 cells, q8_0:q4_0"));
    CHECK(has(dory_fit_warning(served(262144, 512), card, 262144, 36, "q8_0", "q4_0"), "forecast past the card"));
    // the first cell count that warns against that figure: 198,030 at micro-batch 512, 201,892 at 256
    CHECK(dory_fit_warning(served(198029, 512), card, 198029, 36, "q8_0", "q4_0").empty());
    CHECK(has(dory_fit_warning(served(198030, 512), card, 198030, 36, "q8_0", "q4_0"), "forecast past the card"));
    CHECK(dory_fit_warning(served(201891, 256), card, 201891, 36, "q8_0", "q4_0").empty());
    CHECK(has(dory_fit_warning(served(201892, 256), card, 201892, 36, "q8_0", "q4_0"), "forecast past the card"));
    // by the cells libllama allocates the last --n-ctx without a warning is 197,888; 197,889 allocates 198,144
    const std::int64_t c_ok = dory_pool_cells(197888, 1, false), c_past = dory_pool_cells(197889, 1, false);
    CHECK(dory_fit_warning(served(c_ok, 512), card, c_ok, 36, "q8_0", "q4_0").empty());
    CHECK(has(dory_fit_warning(served(c_past, 512), card, c_past, 36, "q8_0", "q4_0"), "36 layers x 198144 cells"));
    CHECK(std::abs(mib(card - served(196608, 512).total()) - 84.0) < 0.05);
    // no card figure: no warning
    CHECK(dory_fit_warning(served(262144, 512), 0, 262144, 36, "q8_0", "q4_0").empty());
    // a smaller micro-batch is a real lever: 204,800 cells at 256 rows forecast 231 MiB less
    CHECK(std::abs(mib(served(204800, 512).total() - served(204800, 256).total()) - 231.0) < 0.5);
}

TEST(dory_start_forecast_composes_what_the_start_logs) {
    const std::uint64_t card = 16225243136ull;
    const std::uint64_t w    = dory_card_weight_bytes(kLoadLines);
    // the served start at 196,608 cells (measured-here, 2026-11-12, an A770, micro-batch 512, one lane, no
    // reserve): the line the process logged, character for character, and no warning
    const DoryStartForecast u = dory_start_forecast(w, 36, "q8_0", "q4_0", 196608, 1, false, 0, 512);
    CHECK_EQ(u.cells, std::int64_t{196608});
    CHECK_EQ(u.mask_cells, std::int64_t{196608});
    CHECK_EQ(dory_forecast_line(u.card, card, 512, true),
             std::string("dory: card memory forecast 15390 MiB at the peak of the card's 15474 MiB (OpenCL's figure): weights "
                         "2854 (libllama's card buffers), KV 11232, compute 446 at micro-batch 512, backend 858 (measured on "
                         "an A770)"));
    CHECK(dory_fit_warning(u.card, card, u.cells, 36, "q8_0", "q4_0").empty());
    CHECK_EQ(u.card.total(), served(196608, 512).total());
    // and the served start at 204,800 cells: 15874 MiB and the warning, as logged
    const DoryStartForecast v = dory_start_forecast(w, 36, "q8_0", "q4_0", 204800, 1, false, 0, 512);
    CHECK_EQ(dory_forecast_line(v.card, card, 512, true),
             std::string("dory: card memory forecast 15874 MiB at the peak of the card's 15474 MiB (OpenCL's figure): weights "
                         "2854 (libllama's card buffers), KV 11700, compute 462 at micro-batch 512, backend 858 (measured on "
                         "an A770)"));
    CHECK_EQ(dory_fit_warning(v.card, card, v.cells, 36, "q8_0", "q4_0"),
             std::string("dory: 15874 MiB are forecast past the card's 15474 MiB (OpenCL's figure): the KV cache 11700 MiB (36 "
                         "layers x 204800 cells, q8_0:q4_0), the weights 2854, the compute buffer 462, the backend 858; lower "
                         "--n-ctx, use --llama-kv q8_0:q4_0, fewer --llama-dory-loops, --llama-dory-kv last_loop, or a smaller "
                         "--prefill-ubatch"));
    // --n-ctx 197,888 is the last without a warning; 197,889 allocates 198,144 cells, and the cache term, the mask
    // and the warning's text all take that count (code: libllama's padding; not run)
    const DoryStartForecast a = dory_start_forecast(w, 36, "q8_0", "q4_0", 197888, 1, false, 0, 512);
    CHECK(dory_fit_warning(a.card, card, a.cells, 36, "q8_0", "q4_0").empty());
    const DoryStartForecast b = dory_start_forecast(w, 36, "q8_0", "q4_0", 197889, 1, false, 0, 512);
    CHECK_EQ(b.cells, std::int64_t{198144});
    CHECK_EQ(b.mask_cells, std::int64_t{198144});
    CHECK_EQ(b.card.kv, dory_kv_bytes(36, 4, 256, 198144, "q8_0", "q4_0"));
    CHECK_EQ(b.card.compute, dory_compute_bytes(198144, 512));
    CHECK(has(dory_fit_warning(b.card, card, b.cells, 36, "q8_0", "q4_0"), "36 layers x 198144 cells"));
    // two unnamed lanes over the same pool (code, not run): the cache holds the whole pool, the mask one lane's
    // share, so the compute term is 192 MiB smaller and nothing else moves
    const DoryStartForecast l2 = dory_start_forecast(w, 36, "q8_0", "q4_0", 196608, 2, false, 0, 512);
    CHECK_EQ(l2.cells, std::int64_t{196608});
    CHECK_EQ(l2.mask_cells, std::int64_t{98304});
    CHECK_EQ(l2.card.kv, u.card.kv);
    CHECK_EQ(u.card.total() - l2.card.total(), std::uint64_t{192} << 20);
    // a pool that the lanes' padding changes: 100,000 cells over 2 lanes allocate 2 x 50,176
    const DoryStartForecast l3 = dory_start_forecast(w, 36, "q8_0", "q4_0", 100000, 2, false, 0, 512);
    CHECK_EQ(l3.cells, std::int64_t{100352});
    CHECK_EQ(l3.mask_cells, std::int64_t{50176});
    CHECK_EQ(l3.card.kv, dory_kv_bytes(36, 4, 256, 100352, "q8_0", "q4_0"));
    // a unified cache keeps the mask at the pool
    CHECK_EQ(dory_start_forecast(w, 36, "q8_0", "q4_0", 196608, 2, true, 0, 512).card.total(), u.card.total());
    // --llama-reserve-cells 20,480 (code, not run for Dory): the mask spans the reserve, the cache the pool; 344
    // MiB less compute at 196,608 cells. At 204,800 cells the forecast is 15,513.6 MiB, still 40 MiB past the
    // figure: with that reserve 204,032 cells is the largest allocation under it and 204,288 the first past it
    const DoryStartForecast r = dory_start_forecast(w, 36, "q8_0", "q4_0", 196608, 1, false, 20480, 512);
    CHECK_EQ(r.cells, std::int64_t{196608});
    CHECK_EQ(r.mask_cells, std::int64_t{20480});
    CHECK_EQ(r.card.kv, u.card.kv);
    CHECK_EQ(u.card.total() - r.card.total(), std::uint64_t{344} << 20);
    const DoryStartForecast r2 = dory_start_forecast(w, 36, "q8_0", "q4_0", 204800, 1, false, 20480, 512);
    CHECK(std::abs(mib(r2.card.total()) - 15513.6) < 0.1);
    CHECK(has(dory_fit_warning(r2.card, card, r2.cells, 36, "q8_0", "q4_0"),
              "dory: 15514 MiB are forecast past the card's 15474 MiB"));
    const DoryStartForecast r3 = dory_start_forecast(w, 36, "q8_0", "q4_0", 204032, 1, false, 20480, 512);
    CHECK(dory_fit_warning(r3.card, card, r3.cells, 36, "q8_0", "q4_0").empty());
    const DoryStartForecast r4 = dory_start_forecast(w, 36, "q8_0", "q4_0", 204288, 1, false, 20480, 512);
    CHECK(!dory_fit_warning(r4.card, card, r4.cells, 36, "q8_0", "q4_0").empty());
    // the micro-batch reaches the compute term
    CHECK_EQ(dory_start_forecast(w, 36, "q8_0", "q4_0", 196608, 1, false, 0, 256).card.compute, dory_compute_bytes(196608, 256));
}

TEST(dory_forecast_line_names_every_term) {
    const std::string line = dory_forecast_line(served(131072, 512), 16225243136ull, 512, true);
    CHECK(has(line, "card memory forecast 11518 MiB at the peak of the card's 15474 MiB (OpenCL's figure)"));
    CHECK(has(line, "weights 2854 (libllama's card buffers)"));
    CHECK(has(line, "KV 7488"));
    CHECK(has(line, "compute 318 at micro-batch 512"));
    CHECK(has(line, "backend 858"));
    // no card size: the terms still print; no load line: the line says the weight term is the whole file
    const std::string fb = dory_forecast_line(dory_card_forecast(3496ull << 20, 0, 8192, 512), 0, 512, false);
    CHECK(!has(fb, "of the card"));
    CHECK(has(fb, "weights 3496 (the whole file: no card buffer line was read)"));
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

// --llama-dory-max-loops: the effort map (contrib/llama.cpp 0118). The unit's count serves a request that
// names no effort; none/off/minimal/low run 1 loop, medium 2, high/xhigh 4, never above the unit's maximum.
TEST(dory_effort_map) {
    CHECK_EQ(dory_loops_for_effort("", 2, 4), 2);        // unset: the unit's count
    CHECK_EQ(dory_loops_for_effort("", 4, 4), 4);
    for (const char* e : {"none", "off", "minimal", "low"}) CHECK_EQ(dory_loops_for_effort(e, 2, 4), 1);
    CHECK_EQ(dory_loops_for_effort("medium", 1, 4), 2);
    CHECK_EQ(dory_loops_for_effort("high", 1, 4), 4);
    CHECK_EQ(dory_loops_for_effort("xhigh", 1, 4), 4);
    CHECK_EQ(dory_loops_for_effort("HIGH", 1, 4), 4);   // the level is not case-sensitive here
    // a level this map does not know is the unit's count
    CHECK_EQ(dory_loops_for_effort("ultra", 3, 4), 3);
}

TEST(dory_effort_map_is_capped_at_the_units_maximum) {
    CHECK_EQ(dory_loops_for_effort("high", 2, 3), 3);
    CHECK_EQ(dory_loops_for_effort("xhigh", 2, 3), 3);
    CHECK_EQ(dory_loops_for_effort("medium", 1, 1), 1);
    CHECK_EQ(dory_loops_for_effort("high", 1, 2), 2);
    CHECK_EQ(dory_loops_for_effort("medium", 1, 4), 2);
    CHECK_EQ(dory_loops_for_effort("low", 1, 4), 1);
    // the unset request is capped too (a maximum below the unit's count is refused at the start, not here)
    CHECK_EQ(dory_loops_for_effort("", 4, 3), 3);
    // the table /props prints is the function over the levels, in a fixed order
    const auto m = dory_effort_map(2, 3);
    CHECK_EQ(m.size(), static_cast<size_t>(8));
    CHECK_EQ(m[0].first, std::string("unset"));
    CHECK_EQ(m[0].second, 2);
    CHECK_EQ(m[1].first, std::string("none"));
    CHECK_EQ(m[1].second, 1);
    CHECK_EQ(m[5].first, std::string("medium"));
    CHECK_EQ(m[5].second, 2);
    CHECK_EQ(m[7].first, std::string("xhigh"));
    CHECK_EQ(m[7].second, 3);
}

// A slot's cells written at one count are not continued at another: the slot is prefilled again.
TEST(dory_reuse_decision) {
    CHECK(dory_reuse(2, 2, 100) == DoryReuse::Keep);         // same count: the prefix serves
    CHECK(dory_reuse(2, 4, 100) == DoryReuse::Reprefill);    // another count: dropped
    CHECK(dory_reuse(4, 1, 1) == DoryReuse::Reprefill);
    CHECK(dory_reuse(1, 2, 0) == DoryReuse::Keep);           // nothing cached: nothing to drop
    CHECK(dory_reuse(0, 2, 0) == DoryReuse::Keep);
}

TEST(dory_max_loops_resolution_and_refusals) {
    DoryOptions o;
    o.loops     = 2;
    o.kv_mode   = "last_loop";
    o.max_loops = 4;
    DoryServed s;
    CHECK(dory_resolve(file(), o, s).empty());
    CHECK_EQ(s.loops, 2);
    CHECK_EQ(s.max_loops, 4);
    // the tables are those of the maximum
    CHECK_EQ(s.executed_layers, 192);
    CHECK(dory_check_loaded(s, 192, 72).empty());
    CHECK(!dory_check_loaded(s, 112, 72).empty());
    CHECK(has(dory_boot_line(s), "per request up to 4"));
    // off: the tables are those of the unit's count, and the boot line is unchanged
    DoryOptions off;
    off.loops   = 2;
    off.kv_mode = "last_loop";
    DoryServed s0;
    CHECK(dory_resolve(file(), off, s0).empty());
    CHECK_EQ(s0.max_loops, 0);
    CHECK_EQ(s0.executed_layers, 112);
    CHECK(!has(dory_boot_line(s0), "per request"));
    // N below the unit's count, per_loop, a number out of range
    DoryOptions low = o;
    low.loops = 3;
    low.max_loops = 2;
    CHECK(has(dory_resolve(file(), low, s), "below the unit's"));
    CHECK(has(dory_option_refusal("dory", false, false, false, low), "below --llama-dory-loops"));
    DoryOptions pl = o;
    pl.kv_mode = "per_loop";
    CHECK(has(dory_option_refusal("dory", false, false, false, pl), "last_loop"));
    DoryOptions nokv = o;
    nokv.kv_mode.clear();   // the file says per_loop
    CHECK(!dory_option_refusal("dory", false, false, false, nokv).empty());
    DoryGguf g = file();
    g.kv_mode = "last_loop";
    DoryOptions viafile = o;
    viafile.kv_mode.clear();
    CHECK(dory_resolve(g, viafile, s).empty());   // the file's own last_loop is read back at the load
    CHECK(has(dory_resolve(file(), viafile, s), "last_loop"));
    DoryOptions big = o;
    big.max_loops = 5;
    CHECK(has(dory_option_refusal("dory", false, false, false, big), "1 to 4"));
    // another architecture, MTP
    DoryOptions only;
    only.max_loops = 4;
    CHECK(has(dory_option_refusal("qwen35", false, false, false, only), "--llama-dory-max-loops"));
    CHECK(has(dory_option_refusal("dory", true, false, false, o), "--llama-mtp"));
}

TEST(dory_max_loops_must_take_effect) {
    CHECK(dory_check_max(4, 4).empty());
    CHECK(has(dory_check_max(4, 2), "did not take effect"));
    CHECK(!dory_check_max(3, 0).empty());   // an unpatched library reports no maximum
}

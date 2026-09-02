#include "config.h"
#include "core/model_registry.h"
#include "harness.h"

#include <vector>

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

TEST(config_stub_defaults) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(cfg.stub);
    CHECK_EQ(cfg.model_id, std::string("qwen3.6-27b-a3b-coder"));
    CHECK_EQ(cfg.host, std::string("127.0.0.1"));
    CHECK_EQ(cfg.port, 8090);
    CHECK_EQ(cfg.parallel, 1);
    CHECK_EQ(cfg.kv_block_size, 32);
    CHECK_EQ(cfg.kv_dtype, std::string("fp16"));
    CHECK_EQ(cfg.mtp, std::string("auto"));
}

TEST(config_needs_something_to_serve) {
    CHECK(rejected({}));
}

TEST(config_model_and_stub_are_exclusive) {
    CHECK(rejected({"--stub", "--model", "/models/ov/x"}));
}

TEST(config_model_needs_an_openvino_build) {
    // Without the OV backend --model must refuse rather than start something
    // that cannot run; with it, the same arguments must be accepted. The build
    // that runs on the card has ARCINT_OPENVINO, and asserting only the first
    // half made this case fail there for reasons that had nothing to do with it.
#ifdef ARCINT_OPENVINO
    Config cfg;
    CHECK(run({"--model", "/models/ov/x", "--model-id", "qwen3.8-27b"}, cfg).ok);
    CHECK_EQ(cfg.model_path, std::string("/models/ov/x"));
#else
    CHECK(rejected({"--model", "/models/ov/x", "--model-id", "qwen3.8-27b"}));
    CHECK(rejected({"--model", "/models/ov/x"}));
#endif
}

TEST(config_device_and_cache_dir) {
    Config cfg;
    CHECK(run({"--stub", "--device", "GPU.1", "--cache-dir", "/tmp/blobs"}, cfg).ok);
    CHECK_EQ(cfg.device, std::string("GPU.1"));
    CHECK_EQ(cfg.cache_dir, std::string("/tmp/blobs"));

    Config def;
    CHECK(run({"--stub"}, def).ok);
    CHECK_EQ(def.device, std::string("GPU.0"));
    CHECK(def.cache_dir.empty());
    CHECK(rejected({"--stub", "--device"}));
}

TEST(config_enforces_the_allowlist) {
    CHECK(rejected({"--stub", "--model-id", "llama-3-8b"}));
    Config cfg;
    CHECK(run({"--stub", "--model-id", "qwen3.8-27b"}, cfg).ok);
}

TEST(config_long_arguments_do_not_dangle) {
    // strtol's out-pointer addresses the buffer it parsed; a temporary
    // std::string leaves it dangling. Short values survive on SSO, so the bug
    // only shows past the small-string threshold.
    CHECK(rejected({"--stub", "--port", "1234567890123456789012345678901234567890"}));
    CHECK(rejected({"--stub", "--port", "00000000000000000000000000000000000008090x"}));

    Config cfg;
    CHECK(run({"--stub", "--port", "0000000000000000000000000008090"}, cfg).ok);
    CHECK_EQ(cfg.port, 8090);
}

TEST(config_rejects_out_of_range_integers) {
    CHECK(rejected({"--stub", "--port", "99999999999999999999"}));
    CHECK(rejected({"--stub", "--n-ctx", "2147483648"}));
    CHECK(rejected({"--stub", "--parallel", "-9999999999"}));
}

TEST(config_validates_ranges) {
    CHECK(rejected({"--stub", "--port", "0"}));
    CHECK(rejected({"--stub", "--port", "70000"}));
    CHECK(rejected({"--stub", "--port", "eighty"}));
    CHECK(rejected({"--stub", "--parallel", "0"}));
    CHECK(rejected({"--stub", "--n-ctx", "-1"}));
    CHECK(rejected({"--stub", "--kv-block-size", "24"}));
    CHECK(rejected({"--stub", "--kv-dtype", "bf16"}));
    // q8 is refused on purpose: a plain int8 cast has no scales and degrades
    // the answer without failing, which is worse than not offering it.
    CHECK(rejected({"--stub", "--kv-dtype", "q8"}));
    CHECK(rejected({"--stub", "--mtp", "maybe"}));
    CHECK(rejected({"--stub", "--quant", "q2"}));
}

TEST(config_accepts_the_documented_choices) {
    for (const char* bs : {"16", "32"}) {
        Config cfg;
        CHECK(run({"--stub", "--kv-block-size", bs}, cfg).ok);
    }
    for (const char* dt : {"fp16", "fp32"}) {
        Config cfg;
        CHECK(run({"--stub", "--kv-dtype", dt}, cfg).ok);
    }
    for (const char* m : {"off", "auto"}) {
        Config cfg;
        CHECK(run({"--stub", "--model-id", "qwen3.8-27b", "--mtp", m}, cfg).ok);
    }
}

TEST(config_mtp_on_requires_an_mtp_head) {
    // --mtp on must refuse where there is nothing to switch on, rather than
    // quietly do ordinary decoding while claiming speculation. The dense export
    // carries a head now, so it is the one model that accepts it.
    Config cfg;
    CHECK(run({"--stub", "--model-id", "qwen3.8-27b", "--mtp", "on"}, cfg).ok);
    for (const std::string& id : model_ids()) {
        // Every entry that carries a head accepts the switch; every other
        // entry refuses it. Intel's 3.8 export is the second one with a head.
        if (find_model(id)->has_mtp_head) {
            Config c;
            CHECK(run({"--stub", "--model-id", id.c_str(), "--mtp", "on"}, c).ok);
            continue;
        }
        CHECK(rejected({"--stub", "--model-id", id.c_str(), "--mtp", "on"}));
    }
}

TEST(config_stub_delay_is_stub_only_and_non_negative) {
    Config cfg;
    CHECK(run({"--stub", "--stub-delay-ms", "25"}, cfg).ok);
    CHECK_EQ(cfg.stub_delay_ms, 25);

    CHECK(rejected({"--stub", "--stub-delay-ms", "-1"}));
    CHECK(rejected({"--stub", "--stub-delay-ms"}));

    Config plain;
    CHECK(run({"--stub"}, plain).ok);
    CHECK_EQ(plain.stub_delay_ms, 0);
}

TEST(config_rejects_unknown_and_dangling_options) {
    CHECK(rejected({"--stub", "--turbo"}));
    CHECK(rejected({"--stub", "--port"}));
    CHECK(rejected({"--model-id"}));
}

TEST(config_help_and_version_short_circuit_validation) {
    Config help;
    CHECK(run({"--help"}, help).ok);
    CHECK(help.show_help);

    Config version;
    CHECK(run({"--version"}, version).ok);
    CHECK(version.show_version);

    CHECK(!usage_text().empty());
}

TEST(config_verbosity_flags) {
    Config one;
    CHECK(run({"--stub", "-v"}, one).ok);
    CHECK_EQ(one.verbosity, 1);

    Config two;
    CHECK(run({"--stub", "-vv"}, two).ok);
    CHECK_EQ(two.verbosity, 2);
}


// The prefill grid and the cache grid must be one grid: a hit is where a warm
// run starts, and it has to traverse the boundaries a cold run did.
TEST(config_prefix_cache_requires_an_aligned_prefill_grid) {
    Config cfg;
    CHECK(run({"--stub", "--prefix-cache-mib", "512", "--kv-block-size", "32",
               "--prefill-chunk", "2048"}, cfg).ok);
    CHECK(rejected({"--stub", "--prefix-cache-mib", "512", "--prefill-chunk", "0"}));
    CHECK(rejected({"--stub", "--prefix-cache-mib", "512", "--kv-block-size", "48",
                    "--prefill-chunk", "2048"}));
    // Without the cache there is nothing to align to, so unchunked stays legal.
    // A fresh Config: parse_args mutates rather than resets, so reusing the one
    // above would carry --prefix-cache-mib into this case.
    Config nocache;
    CHECK(run({"--stub", "--prefill-chunk", "0"}, nocache).ok);
}

TEST(config_queue_timeout_is_a_finite_number_of_seconds) {
    Config cfg;
    CHECK(run({"--stub", "--queue-timeout", "30"}, cfg).ok);
    CHECK_NEAR(cfg.queue_timeout_s, 30.0, 1e-9);

    CHECK(run({"--stub", "--queue-timeout", "0"}, cfg).ok);
    CHECK_NEAR(cfg.queue_timeout_s, 0.0, 1e-9);

    // The default is "do not wait": a lane is a memory reservation, and a
    // client cannot tell an unbounded queue from a hang.
    Config def;
    CHECK(run({"--stub"}, def).ok);
    CHECK_NEAR(def.queue_timeout_s, 0.0, 1e-9);

    CHECK(rejected({"--stub", "--queue-timeout", "-1"}));
    CHECK(rejected({"--stub", "--queue-timeout", "nan"}));
    // Infinity would reach condition_variable::wait_for as UB and silently
    // behave as no wait at all, which is the opposite of what it asks for.
    CHECK(rejected({"--stub", "--queue-timeout", "inf"}));
    CHECK(rejected({"--stub", "--queue-timeout", "1e9"}));
    CHECK(rejected({"--stub", "--queue-timeout", "abc"}));
}

TEST(config_served_model_name_is_presentation_only) {
    Config cfg;
    CHECK(run({"--stub", "--served-model-name", "qwen3.6-coder"}, cfg).ok);
    CHECK_EQ(cfg.served_model_name, std::string("qwen3.6-coder"));
    // It does not touch which artifact is asserted: --model-id is still the
    // allowlist name, and still defaulted for the stub.
    CHECK_EQ(cfg.model_id, std::string("qwen3.6-27b-a3b-coder"));

    Config def;
    CHECK(run({"--stub"}, def).ok);
    CHECK(def.served_model_name.empty());

    // A name that is empty or blank would land on /v1/models and be discovered
    // by a roster; both are refused rather than folded into "not given".
    CHECK(rejected({"--stub", "--served-model-name", ""}));
    CHECK(rejected({"--stub", "--served-model-name", "   "}));
    CHECK(rejected({"--stub", "--served-model-name"}));

    // And it is not a way past the allowlist.
    CHECK(rejected({"--stub", "--model-id", "qwen3.6-coder",
                    "--served-model-name", "qwen3.6-coder"}));
}

TEST(config_paged_kv_is_a_documented_choice) {
    Config def;
    CHECK(run({"--stub"}, def).ok);
    // u8 by default, and the reason is the reservation rather than the
    // throughput: it is what lets two lanes reach depth and the A770 reach
    // depth at all (DESIGN §7.0.3). It costs prefill up to 22% at 115k.
    CHECK_EQ(def.paged_kv, std::string("u8"));

    Config cfg;
    CHECK(run({"--stub", "--paged-kv", "f16"}, cfg).ok);
    CHECK_EQ(cfg.paged_kv, std::string("f16"));

    CHECK(rejected({"--stub", "--paged-kv", "q8"}));
    CHECK(rejected({"--stub", "--paged-kv", "fp16"}));
    CHECK(rejected({"--stub", "--paged-kv"}));
}

// M8: --paged-kv KEY[:VALUE], asymmetric precision for the paged path
// (docs/design-m8-asymmetric-kv.md §3). Red first: before this milestone
// "u8:i4" was refused by the plain u8/f16 check above -- these cases prove
// the wider grammar without loosening what a bare value still means.
TEST(config_paged_kv_accepts_an_asymmetric_pair) {
    Config cfg;
    CHECK(run({"--stub", "--paged-kv", "u8:i4"}, cfg).ok);
    CHECK_EQ(cfg.paged_kv, std::string("u8:i4"));

    // Every documented side value, symmetric and mixed.
    for (const char* side : {"f16", "u8", "i8", "u4", "i4"}) {
        Config c;
        CHECK(run({"--stub", "--paged-kv", side}, c).ok);
    }
    Config c2;
    CHECK(run({"--stub", "--paged-kv", "i8:u4"}, c2).ok);
}

TEST(config_paged_kv_rejects_a_junk_pair) {
    CHECK(rejected({"--stub", "--paged-kv", "u8:q8"}));   // right side not in the set
    CHECK(rejected({"--stub", "--paged-kv", "q8:u8"}));   // left side not in the set
    CHECK(rejected({"--stub", "--paged-kv", "u8:i4:f16"}));  // more than one colon
    CHECK(rejected({"--stub", "--paged-kv", "u8:"}));     // empty side
    CHECK(rejected({"--stub", "--paged-kv", ":i4"}));     // empty side
}

// parse_paged_kv is the shared parser --paged-kv's validation and
// backend_ov.cpp's ARCINT_PAGED_KV override both call, so a value's meaning
// cannot drift between the two call sites (that drift was the u4/i4 bug this
// milestone found: the env override silently mapped anything unrecognised to
// u8). Exercised directly here rather than only through argv parsing.
TEST(parse_paged_kv_single_value_mirrors_to_both_sides) {
    std::string key, value;
    CHECK(parse_paged_kv("u8", key, value));
    CHECK_EQ(key, std::string("u8"));
    CHECK_EQ(value, std::string("u8"));

    CHECK(parse_paged_kv("f16", key, value));
    CHECK_EQ(key, std::string("f16"));
    CHECK_EQ(value, std::string("f16"));
}

TEST(parse_paged_kv_pair_splits_key_and_value) {
    std::string key, value;
    CHECK(parse_paged_kv("u8:i4", key, value));
    CHECK_EQ(key, std::string("u8"));
    CHECK_EQ(value, std::string("i4"));
}

TEST(parse_paged_kv_refuses_garbage) {
    std::string key, value;
    // The exact silent-u8 bug this milestone found: an unrecognised value
    // must be refused, never quietly accepted as u8.
    CHECK(!parse_paged_kv("garbage", key, value));
    CHECK(!parse_paged_kv("i4garbage", key, value));
    CHECK(!parse_paged_kv("", key, value));
    CHECK(!parse_paged_kv("u8:garbage", key, value));
    CHECK(!parse_paged_kv("garbage:u8", key, value));
    CHECK(!parse_paged_kv("u8:i4:u8", key, value));
}

TEST(parse_paged_kv_accepts_the_full_documented_set) {
    std::string key, value;
    for (const char* side : {"f16", "u8", "i8", "u4", "i4"}) {
        CHECK(parse_paged_kv(side, key, value));
        CHECK_EQ(key, std::string(side));
        CHECK_EQ(value, std::string(side));
    }
}

// M8 port audit rule, found on the card 2026-09-02: an unpatched plugin
// compiled a plain --paged-kv u8 request's KV cache ports as i8 -- same 8
// bits, its own signedness choice -- and an exact-element-type audit refused
// every symmetric u8 load over exactly that false positive. The audit must
// compare bitwidth, not exact type: same-bitwidth aliasing (u8<->i8,
// u4<->i4) passes, a genuinely different bitwidth still refuses.
TEST(kv_precision_bitwidth_matches_same_width_aliases_pass) {
    // The exact false positive measured on the card.
    CHECK(kv_precision_bitwidth_matches("u8", "i8"));
    CHECK(kv_precision_bitwidth_matches("i8", "u8"));
    CHECK(kv_precision_bitwidth_matches("u4", "i4"));
    CHECK(kv_precision_bitwidth_matches("i4", "u4"));
    // Exact match is of course still fine.
    CHECK(kv_precision_bitwidth_matches("u8", "u8"));
    CHECK(kv_precision_bitwidth_matches("f16", "f16"));
}

// Amended (on-card finding, 2026-09-02, docs/design-m8-asymmetric-kv.md
// review): this GPU plugin generation stores paged KV in 8-bit-TYPED ports
// always -- an honest 4-bit-typed port breaks the plugin's own compile, and
// the working §7.0.3 u4 deployment ran 8-bit-typed ports throughout. A
// 4-bit request (u4/i4) landing on an 8-bit port is therefore the EXPECTED
// shape on this plugin generation, not a downgrade. This is the exact red
// case the review found: before this amendment, (u4 requested, u8 actual)
// FAILED the audit and broke every legitimate 4-bit request.
TEST(kv_precision_bitwidth_matches_accepts_four_bit_packed_in_eight_bit_port) {
    CHECK(kv_precision_bitwidth_matches("u4", "u8"));  // the exact red case
    CHECK(kv_precision_bitwidth_matches("i4", "i8"));
    // Sign does not matter either, same as the 8-bit alias case above.
    CHECK(kv_precision_bitwidth_matches("i4", "u8"));
    CHECK(kv_precision_bitwidth_matches("u4", "i8"));
}

TEST(kv_precision_bitwidth_matches_refuses_a_real_width_change) {
    // The strict rule still applies outside the one known 4-in-8 packing
    // shape: an 8/16-bit request must still land on a same-width port.
    CHECK(!kv_precision_bitwidth_matches("u8", "i4"));   // wrong direction: 8 requested, 4 got
    CHECK(!kv_precision_bitwidth_matches("f16", "u8"));  // must still refuse
    CHECK(!kv_precision_bitwidth_matches("u8", "f16"));  // must still refuse
    CHECK(!kv_precision_bitwidth_matches("u4", "f16"));  // 4-bit request, but NOT the 8-bit shape
    CHECK(!kv_precision_bitwidth_matches("f16", "i4"));
}

TEST(kv_precision_bitwidth_matches_rejects_unknown_values) {
    CHECK(!kv_precision_bitwidth_matches("garbage", "u8"));
    CHECK(!kv_precision_bitwidth_matches("u8", "garbage"));
    CHECK(!kv_precision_bitwidth_matches("", ""));
}

// kv_precision_is_packed_four_bit tells the caller (backend_ov.cpp's port
// audit) which log message to print: this is true ONLY for the specific
// 4-in-8 packing shape, not for a generic same-bitwidth alias (u8<->i8) or
// for any other passing/failing combination.
TEST(kv_precision_is_packed_four_bit_identifies_the_known_shape) {
    CHECK(kv_precision_is_packed_four_bit("u4", "u8"));
    CHECK(kv_precision_is_packed_four_bit("i4", "i8"));
    CHECK(kv_precision_is_packed_four_bit("i4", "u8"));
    CHECK(kv_precision_is_packed_four_bit("u4", "i8"));
}

TEST(kv_precision_is_packed_four_bit_false_for_everything_else) {
    CHECK(!kv_precision_is_packed_four_bit("u8", "u8"));   // exact match, not this shape
    CHECK(!kv_precision_is_packed_four_bit("u8", "i8"));   // the OTHER alias shape
    CHECK(!kv_precision_is_packed_four_bit("u4", "u4"));   // an honest 4-bit port, if one existed
    CHECK(!kv_precision_is_packed_four_bit("u8", "u4"));   // wrong direction
    CHECK(!kv_precision_is_packed_four_bit("u4", "f16"));  // not 8-bit
    CHECK(!kv_precision_is_packed_four_bit("garbage", "u8"));
}

// F3 (review 2026-09-02): ARCINT_MOE_DEVICE_POOL_BYTES used to go into the
// plugin's AnyMap as a raw string, and the plugin's own stoull() has no
// full-consumption check -- "8e9" silently became 8 bytes ('e9' unconsumed
// and ignored), and a leading '-' silently wrapped through strtoull's
// two's-complement rule to a huge unsigned value. parse_u64_strict is the
// arcint-side refusal: red case first, since this milestone's whole premise
// is that a typo must refuse rather than quietly change the request.
TEST(parse_u64_strict_refuses_scientific_notation) {
    uint64_t out = 0;
    // The exact false-negative measured on the card: "8e9" must be refused,
    // never silently truncated to the "8" strtoull alone would parse.
    CHECK(!parse_u64_strict("8e9", out));
}

TEST(parse_u64_strict_accepts_a_plain_decimal) {
    uint64_t out = 0;
    CHECK(parse_u64_strict("8589934592", out));  // 8 GiB, in bytes
    CHECK_EQ(out, 8589934592ull);
}

TEST(parse_u64_strict_refuses_a_negative_wraparound) {
    uint64_t out = 0;
    // strtoull("-1", ...) alone returns ULLONG_MAX with no error -- the
    // round-trip check is what catches it (ULLONG_MAX prints back as
    // "18446744073709551615", not "-1").
    CHECK(!parse_u64_strict("-1", out));
}

TEST(parse_u64_strict_refuses_trailing_garbage_and_empty) {
    uint64_t out = 0;
    CHECK(!parse_u64_strict("123abc", out));
    CHECK(!parse_u64_strict("", out));
    // strtoull itself skips leading whitespace (fully consumes " 123"), so
    // this one is refused by the round-trip check, not the end pointer.
    CHECK(!parse_u64_strict(" 123", out));
    CHECK(!parse_u64_strict("+123", out));   // round-trip: "+123" != "123"
    CHECK(!parse_u64_strict("007", out));    // round-trip: "007" != "7"
}

TEST(parse_u64_strict_accepts_zero_and_a_large_value) {
    uint64_t out = 0;
    CHECK(parse_u64_strict("0", out));
    CHECK_EQ(out, 0ull);
    CHECK(parse_u64_strict("18446744073709551615", out));  // UINT64_MAX
    CHECK_EQ(out, 18446744073709551615ull);
}

TEST(config_mtp_layer_choice) {
    for (const char* w : {"auto", "reconstructed", "exported"}) {
        Config cfg;
        CHECK(run({"--stub", "--mtp-layer", w}, cfg).ok);
        CHECK_EQ(cfg.mtp_layer, std::string(w));
    }
    CHECK(rejected({"--stub", "--mtp-layer", "intel"}));
}

TEST(config_operator_sampler_flags) {
    Config cfg;
    CHECK(run({"--stub", "--temp", "0", "--top-p", "0.9", "--top-k", "40",
               "--repetition-penalty", "1.0", "--presence-penalty", "1.5"}, cfg).ok);
    CHECK(cfg.temp && *cfg.temp == 0.0f);
    CHECK(cfg.top_p && *cfg.top_p == 0.9f);
    CHECK(cfg.top_k && *cfg.top_k == 40);
    CHECK(cfg.repetition_penalty && *cfg.repetition_penalty == 1.0f);
    CHECK(cfg.presence_penalty && *cfg.presence_penalty == 1.5f);
}

TEST(config_operator_flags_default_unset) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(!cfg.temp);
    CHECK(!cfg.top_p);
    CHECK(!cfg.top_k);
    CHECK(!cfg.repetition_penalty);
    CHECK(!cfg.presence_penalty);
    CHECK(!cfg.think_default);
}

TEST(config_operator_flags_reject_garbage) {
    CHECK(rejected({"--stub", "--temp", "warm"}));
    CHECK(rejected({"--stub", "--top-k", "many"}));
    CHECK(rejected({"--stub", "--temp"}));
}

TEST(config_chat_template_kwarg) {
    Config cfg;
    CHECK(run({"--stub", "--chat-template-kwarg", "enable_thinking=false"}, cfg).ok);
    CHECK(cfg.think_default && *cfg.think_default == false);
    Config cfg2;
    CHECK(run({"--stub", "--chat-template-kwarg", "enable_thinking=true"}, cfg2).ok);
    CHECK(cfg2.think_default && *cfg2.think_default == true);
}

TEST(config_chat_template_kwarg_rejects_unknown_keys) {
    // Only enable_thinking exists in the render path; accepting another key
    // and ignoring it would be a lie.
    CHECK(rejected({"--stub", "--chat-template-kwarg", "add_generation_prompt=false"}));
    CHECK(rejected({"--stub", "--chat-template-kwarg", "enable_thinking=maybe"}));
    CHECK(rejected({"--stub", "--chat-template-kwarg", "enable_thinking"}));
}

TEST(config_dflash_flags) {
    Config cfg;
    CHECK(run({"--stub", "--dflash", "/models/draft", "--dflash-device", "GPU.1"}, cfg).ok);
    CHECK_EQ(cfg.dflash, std::string("/models/draft"));
    CHECK_EQ(cfg.dflash_device, std::string("GPU.1"));
}

TEST(config_one_drafter_per_server) {
    CHECK(rejected({"--stub", "--dflash", "/d", "--mtp", "on"}));
    CHECK(rejected({"--stub", "--dflash", "/d", "--draft", "4"}));
    // mtp auto yields to the requested drafter rather than conflicting
    Config cfg;
    CHECK(run({"--stub", "--dflash", "/d"}, cfg).ok);
}

TEST(config_pin_dispatch_defaults_off) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK_EQ(cfg.pin_dispatch, -1);
}

TEST(config_pin_dispatch_accepts_a_core) {
    Config cfg;
    CHECK(run({"--stub", "--pin-dispatch", "0"}, cfg).ok);
    CHECK_EQ(cfg.pin_dispatch, 0);

    Config cfg2;
    CHECK(run({"--stub", "--pin-dispatch", "7"}, cfg2).ok);
    CHECK_EQ(cfg2.pin_dispatch, 7);
}

TEST(config_pin_dispatch_rejects_garbage) {
    CHECK(rejected({"--stub", "--pin-dispatch", "-2"}));
    CHECK(rejected({"--stub", "--pin-dispatch", "1024"}));
    CHECK(rejected({"--stub", "--pin-dispatch", "not-a-core"}));
}

TEST(config_moe_cpu_tier_defaults_off) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(!cfg.moe_cpu_tier);
    CHECK_EQ(cfg.moe_cpu_tier_threads, 0);
}

TEST(config_moe_cpu_tier_accepts_with_offload_ratio) {
    Config cfg;
    CHECK(run({"--stub", "--offload-ratio", "20", "--moe-cpu-tier"}, cfg).ok);
    CHECK(cfg.moe_cpu_tier);

    Config cfg2;
    CHECK(run({"--stub", "--offload-ratio", "20", "--moe-cpu-tier",
               "--moe-cpu-tier-threads", "7"}, cfg2).ok);
    CHECK_EQ(cfg2.moe_cpu_tier_threads, 7);
}

TEST(config_moe_cpu_tier_needs_offload_ratio) {
    CHECK(rejected({"--stub", "--moe-cpu-tier"}));
}

TEST(config_moe_cpu_tier_threads_needs_tier) {
    // A thread count without the tier would be accepted and silently ignored
    // (found in review); refuse it so a mistyped invocation cannot look tuned.
    CHECK(rejected({"--stub", "--offload-ratio", "20", "--moe-cpu-tier-threads", "7"}));
}

TEST(config_moe_cpu_tier_rejects_garbage) {
    CHECK(rejected({"--stub", "--offload-ratio", "20", "--moe-cpu-tier",
                    "--moe-cpu-tier-threads", "-1"}));
    CHECK(rejected({"--stub", "--offload-ratio", "20", "--moe-cpu-tier",
                    "--moe-cpu-tier-threads", "lots"}));
}

TEST(config_fit_margin_mib_defaults_256) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK_EQ(cfg.fit_margin_mib, 256);
}

TEST(config_fit_margin_mib_accepts_a_value) {
    Config cfg;
    CHECK(run({"--stub", "--fit-margin-mib", "512"}, cfg).ok);
    CHECK_EQ(cfg.fit_margin_mib, 512);

    Config cfg2;
    CHECK(run({"--stub", "--fit-margin-mib", "0"}, cfg2).ok);
    CHECK_EQ(cfg2.fit_margin_mib, 0);
}

TEST(config_fit_margin_mib_rejects_garbage) {
    CHECK(rejected({"--stub", "--fit-margin-mib", "-1"}));
    CHECK(rejected({"--stub", "--fit-margin-mib", "not-a-number"}));
}

TEST(config_n_ctx_explicit_defaults_false) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(!cfg.n_ctx_explicit);
    CHECK_EQ(cfg.n_ctx, 0);
}

// M13 (docs/milestone-0.3.0.md): --vision is reserved, not served. The
// artifacts arcint loads are VLM exports whose vision tower and projector
// IRs are never read (src/core/artifact.cpp, src/exec/backend_ov.cpp), so
// the flag must refuse loudly -- both bare and with a value -- rather than
// silently doing nothing, which is how a mistyped invocation could look
// multimodal. No Config field backs it: see config.cpp's --vision arm.
TEST(config_vision_flag_is_reserved_and_refused) {
    Config     cfg;
    const ArgParse r = run({"--stub", "--vision"}, cfg);
    CHECK(!r.ok);
    CHECK(r.error.find("reserved") != std::string::npos);
}

TEST(config_vision_flag_with_a_value_is_still_refused) {
    // A generic "unknown option" catch-all would already reject this, so
    // assert the reserved-flag message specifically -- otherwise this case
    // would pass without --vision's own handling ever running.
    Config     cfg;
    const ArgParse r = run({"--stub", "--vision", "true"}, cfg);
    CHECK(!r.ok);
    CHECK(r.error.find("reserved") != std::string::npos);
}

TEST(config_n_ctx_explicit_true_when_passed) {
    Config cfg;
    CHECK(run({"--stub", "--n-ctx", "8192"}, cfg).ok);
    CHECK(cfg.n_ctx_explicit);
    CHECK_EQ(cfg.n_ctx, 8192);

    // Even an explicit 0 counts as explicit -- the flag was typed, so it is
    // not "nothing was asked", however unlikely a real operator is to pass
    // it.
    Config cfg2;
    CHECK(run({"--stub", "--n-ctx", "0"}, cfg2).ok);
    CHECK(cfg2.n_ctx_explicit);
    CHECK_EQ(cfg2.n_ctx, 0);
}

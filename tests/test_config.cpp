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

#include "config.h"
#include "harness.h"

#include <vector>

using namespace lgc;

namespace {

ArgParse run(std::vector<const char*> args, Config& cfg) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("ligence"));
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
    // This test binary is built without the OV backend, so --model must refuse
    // rather than start something that cannot run.
    CHECK(rejected({"--model", "/models/ov/x", "--model-id", "qwen3.8-27b"}));
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
    CHECK(rejected({"--stub", "--kv-dtype", "fp32"}));
    CHECK(rejected({"--stub", "--mtp", "maybe"}));
    CHECK(rejected({"--stub", "--quant", "q2"}));
}

TEST(config_accepts_the_documented_choices) {
    for (const char* bs : {"16", "32"}) {
        Config cfg;
        CHECK(run({"--stub", "--kv-block-size", bs}, cfg).ok);
    }
    for (const char* dt : {"fp16", "q8"}) {
        Config cfg;
        CHECK(run({"--stub", "--kv-dtype", dt}, cfg).ok);
    }
    for (const char* m : {"on", "off", "auto"}) {
        Config cfg;
        // "on" is only valid for a model that ships an MTP head.
        const bool ok = run({"--stub", "--model-id", "qwen3.8-27b", "--mtp", m}, cfg).ok;
        CHECK(ok);
    }
}

TEST(config_mtp_on_requires_an_mtp_head) {
    CHECK(rejected({"--stub", "--model-id", "qwen3.6-27b-a3b-coder", "--mtp", "on"}));
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

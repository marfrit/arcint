#include "config.h"
#include "core/sampling.h"
#include "exec/llama_layer_split.h"
#include "harness.h"

#include <cstdio>
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
    CHECK_EQ(cfg.host, std::string("127.0.0.1"));
    CHECK_EQ(cfg.port, 8090);
    CHECK_EQ(cfg.parallel, 1);
    CHECK_EQ(cfg.engine, std::string("llama"));
    CHECK(cfg.gguf_path.empty());
    CHECK_EQ(cfg.prefill_chunk, 2048);
    CHECK_EQ(cfg.device, std::string("GPU.0"));
}


TEST(config_needs_something_to_serve) {
    CHECK(rejected({}));
    Config cfg;
    const ArgParse r = run({"--port", "8090"}, cfg);
    CHECK(!r.ok);
    CHECK(r.error.find("--gguf FILE") != std::string::npos);
}

TEST(config_gguf_and_stub_are_exclusive) {
    CHECK(rejected({"--stub", "--gguf", "/m/q.gguf"}));
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
    CHECK(rejected({"--stub", "--prefill-chunk", "-1"}));
}

TEST(config_accepts_the_documented_choices) {
    for (const char* dev : {"GPU.0", "GPU.1", "B60"}) {
        Config cfg;
        CHECK(run({"--stub", "--device", dev}, cfg).ok);
        CHECK_EQ(cfg.device, std::string(dev));
    }
    Config chunk;
    CHECK(run({"--stub", "--prefill-chunk", "0"}, chunk).ok);
    CHECK(rejected({"--stub", "--device", ""}));
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
    CHECK(rejected({"--gguf"}));
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

TEST(config_served_model_name_is_a_nonblank_name) {
    Config cfg;
    CHECK(run({"--stub", "--served-model-name", "qwen3.6-coder"}, cfg).ok);
    CHECK_EQ(cfg.served_model_name, std::string("qwen3.6-coder"));

    Config def;
    CHECK(run({"--stub"}, def).ok);
    CHECK(def.served_model_name.empty());

    // A name that is empty or blank would land on /v1/models and be discovered
    // by a roster; both are refused rather than folded into "not given".
    CHECK(rejected({"--stub", "--served-model-name", ""}));
    CHECK(rejected({"--stub", "--served-model-name", "   "}));
    CHECK(rejected({"--stub", "--served-model-name"}));
}




















TEST(config_operator_sampler_flags) {
    Config cfg;
    CHECK(run({"--stub", "--temp", "0", "--top-p", "0.9", "--top-k", "40",
               "--repetition-penalty", "1.0", "--presence-penalty", "1.5", "--min-p", "0.05"}, cfg).ok);
    CHECK(cfg.temp && *cfg.temp == 0.0f);
    CHECK(cfg.top_p && *cfg.top_p == 0.9f);
    CHECK(cfg.top_k && *cfg.top_k == 40);
    CHECK(cfg.repetition_penalty && *cfg.repetition_penalty == 1.0f);
    CHECK(cfg.presence_penalty && *cfg.presence_penalty == 1.5f);
    CHECK(cfg.min_p && *cfg.min_p == 0.05f);
}

TEST(config_nothink_sampler_flags_parse_and_mark_the_row) {
    Config cfg;
    CHECK(run({"--stub", "--nothink-temp", "0.7", "--nothink-top-p", "0.8",
               "--nothink-top-k", "35", "--nothink-min-p", "0.05",
               "--nothink-presence-penalty", "0.3"}, cfg).ok);
    CHECK(cfg.nothink_temp && *cfg.nothink_temp == 0.7f);
    CHECK(cfg.nothink_top_p && *cfg.nothink_top_p == 0.8f);
    CHECK(cfg.nothink_top_k && *cfg.nothink_top_k == 35);
    CHECK(cfg.nothink_min_p && *cfg.nothink_min_p == 0.05f);
    CHECK(cfg.nothink_presence_penalty && *cfg.nothink_presence_penalty == 0.3f);

    SamplerDefaults d;
    CHECK(!apply_operator_defaults(cfg, d).has_value());
    CHECK(d.nothink_row);
    CHECK_EQ(d.provenance, std::string("operator"));
    CHECK(d.nothink_temperature == 0.7f);

    Config unset;
    CHECK(run({"--stub"}, unset).ok);
    CHECK(!unset.nothink_temp);
    CHECK(!unset.nothink_top_p);
    CHECK(!unset.nothink_top_k);
    CHECK(!unset.nothink_min_p);
    CHECK(!unset.nothink_presence_penalty);
    SamplerDefaults plain;
    CHECK(!apply_operator_defaults(unset, plain).has_value());
    CHECK(!plain.nothink_row);   // no flag: every request on the normal row
}

TEST(config_nothink_flags_reject_out_of_range_like_their_twins) {
    auto bad = [](std::vector<const char*> args) {
        Config cfg;
        if (!run(std::move(args), cfg).ok) return true;
        SamplerDefaults d;
        return apply_operator_defaults(cfg, d).has_value();
    };
    CHECK(bad({"--stub", "--nothink-temp", "2.5"}));      // [0, 2]
    CHECK(bad({"--stub", "--nothink-top-p", "0"}));       // (0, 1]
    CHECK(bad({"--stub", "--nothink-top-k", "-1"}));      // >= 0
    CHECK(bad({"--stub", "--nothink-min-p", "1.5"}));     // [0, 1]
    CHECK(bad({"--stub", "--nothink-presence-penalty", "3"}));   // [-2, 2]
    // garbage is refused at parse time, like the twins
    CHECK(rejected({"--stub", "--nothink-temp", "warm"}));
    CHECK(rejected({"--stub", "--nothink-top-k", "many"}));
    CHECK(rejected({"--stub", "--nothink-top-p"}));
}

TEST(config_nothink_flags_in_help) {
    const std::string help = usage_text();
    for (const char* flag : {"--nothink-temp", "--nothink-top-p", "--nothink-top-k",
                             "--nothink-min-p", "--nothink-presence-penalty"})
        CHECK(help.find(flag) != std::string::npos);
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



























TEST(config_n_ctx_explicit_defaults_false) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(!cfg.n_ctx_explicit);
    CHECK_EQ(cfg.n_ctx, 0);
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


















TEST(config_llama_kv) {
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf"};
        CHECK(parse_args(3, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_kv_k, std::string("f16"));
        CHECK_EQ(cfg.llama_kv_v, std::string("f16"));
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-kv", "q8_0"};
        CHECK(parse_args(5, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_kv_k, std::string("q8_0"));
        CHECK_EQ(cfg.llama_kv_v, std::string("q8_0"));   // V defaults to K
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-kv", "q8_0:q4_0"};
        CHECK(parse_args(5, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_kv_k, std::string("q8_0"));
        CHECK_EQ(cfg.llama_kv_v, std::string("q4_0"));
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-kv", "q5_0"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // not a type the kernels take
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-kv", "q4_0:q8_0"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // a pair without a kernel
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-kv", "q4_0"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // 4:4 misses the answer-level bar
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--stub", "--llama-kv", "q8_0"};
        CHECK(!parse_args(4, const_cast<char**>(argv), cfg).ok);   // an option of the libllama engine, not of the stub
    }
}

TEST(config_llama_cpu_moe_and_threads) {
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-cpu-moe", "31", "--llama-threads", "8"};
        const auto r = parse_args(7, const_cast<char**>(argv), cfg);
        CHECK(r.ok);
        CHECK_EQ(cfg.llama_cpu_moe, 31);
        CHECK_EQ(cfg.llama_threads, 8);
        CHECK_EQ(cfg.llama_mtp, 0);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp", "3"};
        CHECK(parse_args(5, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_mtp, 3);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp", "4", "--llama-mtp-vocab", "/m/ids.bin"};
        CHECK(parse_args(7, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_mtp_vocab, std::string("/m/ids.bin"));
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp-vocab", "/m/ids.bin"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // a draft vocabulary without drafts
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp", "3", "--llama-mtp-gguf", "/m/mtp.gguf"};
        CHECK(parse_args(7, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_mtp_gguf, std::string("/m/mtp.gguf"));
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp-gguf", "/m/mtp.gguf"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // an MTP file without drafts
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp", "4", "--llama-mtp-min-p", "0.5"};
        CHECK(parse_args(7, const_cast<char**>(argv), cfg).ok);
        CHECK(cfg.llama_mtp_min_p == 0.5);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp-min-p", "0.5"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // a draft threshold without drafts
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp", "4", "--llama-mtp-min-p", "1.5"};
        CHECK(!parse_args(7, const_cast<char**>(argv), cfg).ok);   // a probability
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-mtp", "8"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // a verify of 9 rows: the decode attention takes up to 8
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-cpu-moe", "-1"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // a count, not a sign
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--stub", "--llama-threads", "4"};
        CHECK(!parse_args(4, const_cast<char**>(argv), cfg).ok);   // --stub runs the model-less stub, not libllama
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--stub", "--llama-mtp", "2"};
        CHECK(!parse_args(4, const_cast<char**>(argv), cfg).ok);
    }
}

// --llama-dflash-gguf / --llama-dflash / --llama-dflash-min-p: the DFlash2
// drafter (docs/campaigns/dflash2-agent.md). One drafter a context, one card.
TEST(config_llama_dflash) {
    {
        // the drafter alone drafts its trained block less the anchor: 7
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf"}, cfg).ok);
        CHECK(cfg.llama_dflash_on());
        CHECK_EQ(cfg.llama_dflash_gguf, std::string("/m/d.gguf"));
        CHECK_EQ(cfg.llama_dflash, 7);
        CHECK(cfg.llama_dflash_min_p == 0.0);
        CHECK_EQ(cfg.llama_dflash_ubatch, 64);
        CHECK_EQ(cfg.llama_mtp, 0);
    }
    {
        // the draft context's ubatch: 8 to 512, the drafter's only
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash-ubatch", "512"}, cfg).ok);
        CHECK_EQ(cfg.llama_dflash_ubatch, 512);
    }
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash-ubatch", "4"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash-ubatch", "1024"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-ubatch", "64"}));
    CHECK(rejected({"--stub", "--llama-dflash-ubatch", "64"}));
    {
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash", "5",
                   "--llama-dflash-min-p", "0.3"}, cfg).ok);
        CHECK_EQ(cfg.llama_dflash, 5);
        CHECK(cfg.llama_dflash_min_p == 0.3);
    }
    {
        // no drafter: off, and nothing set
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf"}, cfg).ok);
        CHECK(!cfg.llama_dflash_on());
        CHECK_EQ(cfg.llama_dflash, 0);
        CHECK_EQ(cfg.llama_dflash_ubatch, 0);
    }
    // the range: 1 to 7 (a block of up to 8 positions, the drafter's trained size)
    CHECK(!rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash", "1"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash", "0"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash", "8"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash", "-1"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash", "x"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash-min-p", "1.5"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-dflash-min-p", "-0.1"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-gguf"}));   // the file is missing
    // the count and the stop need the drafter
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash", "5"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-dflash-min-p", "0.5"}));
    // one drafter a context: not with the MTP head or any of its options
    {
        Config         cfg;
        const ArgParse r = run({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-mtp", "5"}, cfg);
        CHECK(!r.ok);
        CHECK(r.error.find("exclusive") != std::string::npos);
    }
    auto exclusive = [](std::vector<const char*> args) {
        Config         cfg;
        const ArgParse r = run(std::move(args), cfg);
        return !r.ok && r.error.find("exclusive") != std::string::npos;
    };
    CHECK(exclusive({"--gguf", "/m/q.gguf", "--llama-mtp", "5", "--llama-dflash-gguf", "/m/d.gguf"}));   // either order
    CHECK(exclusive({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-mtp", "5", "--llama-mtp-vocab",
                     "/m/ids.json"}));
    CHECK(exclusive({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-mtp-vocab", "/m/ids.json"}));
    CHECK(exclusive({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-mtp-gguf", "/m/mtp.gguf"}));
    CHECK(exclusive({"--gguf", "/m/q.gguf", "--llama-dflash-gguf", "/m/d.gguf", "--llama-mtp-min-p", "0.5"}));
    // one card: refused under the layer split, with its own message
    {
        Config         cfg;
        const ArgParse r = run({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40",
                                "--llama-dflash-gguf", "/m/d.gguf"}, cfg);
        CHECK(!r.ok);
        CHECK(r.error.find("--llama-layer-split") != std::string::npos);
    }
    // an option of the libllama engine only
    CHECK(rejected({"--stub", "--llama-dflash-gguf", "/m/d.gguf"}));
    CHECK(rejected({"--stub", "--llama-dflash", "5"}));
    CHECK(rejected({"--stub", "--llama-dflash-min-p", "0.5"}));
    // the OpenVINO engine's flag stays retired, the new ones are not
    CHECK(is_retired_flag("--dflash"));
    CHECK(!is_retired_flag("--llama-dflash"));
    CHECK(!is_retired_flag("--llama-dflash-gguf"));
    // the help offers all three
    const std::string help = usage_text();
    CHECK(help.find("--llama-dflash-gguf FILE") != std::string::npos);
    CHECK(help.find("--llama-dflash N") != std::string::npos);
    CHECK(help.find("--llama-dflash-min-p P") != std::string::npos);
    CHECK(help.find("--llama-dflash-ubatch N") != std::string::npos);
}

TEST(config_llama_checkpoints) {
    {
        // the reference's defaults (llama.cpp server: 32 a slot, 8192 apart)
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf"};
        CHECK(parse_args(3, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_checkpoints, 32);
        CHECK_EQ(cfg.llama_checkpoint_step, 8192);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-checkpoints", "0", "--llama-checkpoint-step", "4096"};
        CHECK(parse_args(7, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_checkpoints, 0);   // off: a hybrid lane re-prefills when it cannot trim
        CHECK_EQ(cfg.llama_checkpoint_step, 4096);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-checkpoints", "-1"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-checkpoint-step", "0"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);   // a spacing, at least one token
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--stub", "--llama-checkpoints", "0"};
        CHECK(!parse_args(4, const_cast<char**>(argv), cfg).ok);   // an option of the libllama engine only
    }
}

TEST(config_llama_expert_cache) {
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf"};
        CHECK(parse_args(3, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_expert_cache_mib, 0);   // off by default
        CHECK(cfg.llama_expert_profile.empty());
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-cpu-moe", "48", "--llama-expert-cache", "16384",
                              "--llama-expert-profile", "/p/expert-profile.bin"};
        CHECK(parse_args(9, const_cast<char**>(argv), cfg).ok);
        CHECK_EQ(cfg.llama_expert_cache_mib, 16384);
        CHECK_EQ(cfg.llama_expert_profile, std::string("/p/expert-profile.bin"));
    }
    {
        // the cache holds experts --llama-cpu-moe keeps in host memory
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-expert-cache", "16384"};
        CHECK(!parse_args(5, const_cast<char**>(argv), cfg).ok);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-cpu-moe", "48", "--llama-expert-profile", "/p/x.bin"};
        CHECK(!parse_args(7, const_cast<char**>(argv), cfg).ok);   // a profile without a cache
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-cpu-moe", "48", "--llama-expert-cache", "-1"};
        CHECK(!parse_args(7, const_cast<char**>(argv), cfg).ok);
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--stub", "--llama-expert-cache", "64"};
        CHECK(!parse_args(4, const_cast<char**>(argv), cfg).ok);   // an option of the libllama engine only
    }
}

// --llama-mtp-card N: the card of a layer split that runs the MTP model
// (default 0, the first: it drafted in half the time of the A770 there)
TEST(config_llama_mtp_card) {
    {
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40"}, cfg).ok);
        CHECK_EQ(cfg.llama_mtp_card, 0);
    }
    {
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40", "--llama-mtp", "2", "--llama-mtp-gguf", "/m/mtp.gguf", "--llama-mtp-card", "1"}, cfg).ok);
        CHECK_EQ(cfg.llama_mtp_card, 1);
    }
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40", "--llama-mtp-card", "2"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40", "--llama-mtp-card", "-1"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40", "--llama-mtp-card", "x"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-mtp-card", "1"}));   // one card: nothing to choose
    // only the --llama-mtp-gguf model is placed by it (the GGUF's own MTP layer is the last layer's)
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "40", "--llama-mtp-card", "1"}));
    CHECK(rejected({"--stub", "--llama-mtp-card", "0"}));   // an option of the libllama engine only
}

// --device GPU.0,GPU.1 --llama-layer-split K: the libllama engine's layer split
// over two cards (docs/campaigns/layer-split-two-cards.md, 0024)
TEST(config_llama_layer_split) {
    {
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "30"}, cfg).ok);
        CHECK(cfg.llama_devices == (std::vector<std::string>{"GPU.0", "GPU.1"}));
        CHECK_EQ(cfg.llama_layer_split, 30);
    }
    {
        // one card: as before, the list holds it
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "GPU.1"}, cfg).ok);
        CHECK(cfg.llama_devices == (std::vector<std::string>{"GPU.1"}));
        CHECK_EQ(cfg.llama_layer_split, 0);
    }
    {
        // names as well as ordinals
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "B60,A770", "--llama-layer-split", "24"}, cfg).ok);
        CHECK(cfg.llama_devices == (std::vector<std::string>{"B60", "A770"}));
    }
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1"}));                                // no automatic split
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-layer-split", "24"}));                              // one card
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "0"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "-3"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "x"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1,GPU.2", "--llama-layer-split", "24"}));   // two cards
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,,GPU.1", "--llama-layer-split", "24"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,", "--llama-layer-split", "24"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.0", "--llama-layer-split", "24"}));   // one card twice
    CHECK(rejected({"--stub", "--llama-layer-split", "24"}));   // an option of the libllama engine only
    {
        // the stub takes any device string, commas included
        Config cfg;
        CHECK(run({"--stub", "--device", "HETERO:GPU.0,GPU.1"}, cfg).ok);
        CHECK_EQ(cfg.device, std::string("HETERO:GPU.0,GPU.1"));
        CHECK(cfg.llama_devices.empty());
    }
}

// --llama-expert-cache A,B: a cache per card, in --device order
TEST(config_llama_expert_cache_per_card) {
    {
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                   "--llama-expert-cache", "11000,7000"}, cfg).ok);
        CHECK(cfg.llama_expert_cache_mib_dev == (std::vector<int>{11000, 7000}));
        CHECK_EQ(cfg.llama_expert_cache_mib, 11000);
        CHECK(cfg.llama_expert_cache_on());
    }
    {
        // none on the first card: still a cache
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                   "--llama-expert-cache", "0,7000"}, cfg).ok);
        CHECK_EQ(cfg.llama_expert_cache_mib, 0);
        CHECK(cfg.llama_expert_cache_on());
    }
    {
        // one value, one card: today's meaning
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf", "--llama-cpu-moe", "48", "--llama-expert-cache", "14500"}, cfg).ok);
        CHECK_EQ(cfg.llama_expert_cache_mib, 14500);
        CHECK(cfg.llama_expert_cache_mib_dev == (std::vector<int>{14500}));
    }
    // one size per --device entry
    CHECK(rejected({"--gguf", "/m/q.gguf", "--llama-cpu-moe", "48", "--llama-expert-cache", "11000,7000"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                    "--llama-expert-cache", "11000"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                    "--llama-expert-cache", "11000,-1"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                    "--llama-expert-cache", "11000,"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                    "--llama-expert-cache", "11000,x"}));
    // a cache on either card holds experts --llama-cpu-moe keeps in host memory
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24",
                    "--llama-expert-cache", "0,7000"}));
    // no cache anywhere: no profile
    CHECK(rejected({"--gguf", "/m/q.gguf", "--device", "GPU.0,GPU.1", "--llama-layer-split", "24", "--llama-cpu-moe", "48",
                    "--llama-expert-cache", "0,0", "--llama-expert-profile", "/p/x.bin"}));
    CHECK(rejected({"--stub", "--llama-expert-cache", "64,64"}));
}

// llama.cpp's layer placement (llama-model.cpp load_tensors) with the
// fractions --llama-layer-split K sets: layers [0, K) on the first card, the
// rest and the output on the second, for every K and depth
TEST(llama_layer_split_fractions_place_k_exactly) {
    for (int n = 2; n <= 200; ++n) {
        for (int k = 1; k < n; ++k) {
            const auto ts = llama_layer_split_fractions(n, k);
            int wrong = 0;
            for (int il = 0; il < n; ++il) wrong += llama_layer_split_device(ts.data(), 2, n, il) != (il < k ? 0 : 1);
            wrong += llama_layer_split_device(ts.data(), 2, n, n) != 1;   // the output
            if (wrong != 0) {
                CHECK_EQ(n * 1000 + k, -1);   // names the failing (n, k)
                return;
            }
        }
    }
    const auto a = llama_layer_split_fractions(48, 24), b = llama_layer_split_fractions(48, 30);
    CHECK_NEAR(a[0], 23.5, 1e-6);
    CHECK_NEAR(a[1], 25.5, 1e-6);
    CHECK_NEAR(b[0], 29.5, 1e-6);
    CHECK_NEAR(b[1], 19.5, 1e-6);
    CHECK(llama_layer_split_valid(48, 1) && llama_layer_split_valid(48, 47));
    CHECK(!llama_layer_split_valid(48, 0) && !llama_layer_split_valid(48, 48) && !llama_layer_split_valid(1, 1));
    // the proportional {K, n - K} is not exact: layer 24 of 48 stays on the first card
    const float naive[2] = { 24.0f, 24.0f };
    CHECK_EQ(llama_layer_split_device(naive, 2, 48, 24), 0);
}

// Named lanes (DESIGN §4.2, amended 2026-10-07;
// docs/campaigns/lanes-agent-subagent.md): --served-model-name A,B with
// --lane-ctx CA,CB on the libllama engine, paired by index; the lane count is
// the number of names.
namespace {
// The refusal, and that it is the lanes' refusal and not some earlier one.
bool refused_for(std::vector<const char*> args, const char* needle) {
    Config         cfg;
    const ArgParse r = run(std::move(args), cfg);
    if (r.ok) return false;
    if (r.error.find(needle) == std::string::npos) {
        std::fprintf(stderr, "    (refused, but for: %s)\n", r.error.c_str());
        return false;
    }
    return true;
}
}  // namespace

TEST(config_named_lanes_pair_names_with_contexts) {
    Config cfg;
    CHECK(run({"--gguf", "/m/q.gguf", "--served-model-name", "qwen3.8-agent,qwen3.8-subagent",
               "--lane-ctx", "131072,32768"}, cfg).ok);
    CHECK(cfg.named_lanes());
    CHECK_EQ(cfg.lane_names, (std::vector<std::string>{"qwen3.8-agent", "qwen3.8-subagent"}));
    CHECK_EQ(cfg.lane_ctx, (std::vector<int>{131072, 32768}));
    CHECK_EQ(cfg.parallel, 2);   // the lane count is the number of names
    // the first name is the served one (where an unnamed request goes)
    CHECK_EQ(cfg.served_model_name, std::string("qwen3.8-agent"));

    // --parallel that agrees is accepted, one that disagrees is refused
    Config agree;
    CHECK(run({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,256",
               "--parallel", "2"}, agree).ok);
    CHECK_EQ(agree.parallel, 2);
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,256",
                       "--parallel", "1"}, "--parallel"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,256",
                       "--parallel", "3"}, "--parallel"));

    // the stub serves them too, for the device-free suites
    Config stub;
    CHECK(run({"--stub", "--served-model-name", "a,b", "--lane-ctx", "512,256"}, stub).ok);
    CHECK_EQ(stub.parallel, 2);
}

TEST(config_named_lanes_refusals) {
    // a name with whitespace ("a, b" in a unit file) would make a lane named " b"
    // that its proxy name never reaches
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a, b", "--lane-ctx", "512,256"}, "whitespace"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a ,b", "--lane-ctx", "512,256"}, "whitespace"));
    // lists of different lengths
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512"}, "--lane-ctx"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b,c", "--lane-ctx", "512,256"}, "--lane-ctx"));
    // a cap that is not a multiple of 256 (llama.cpp pads the pool to 256)
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,1000"}, "256"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "0,256"}, "256"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "-256,256"}, "256"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,x"}, "--lane-ctx"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,"}, "--lane-ctx"));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx"}));
    // --lane-ctx without names, or with one: one lane's context is --n-ctx
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--lane-ctx", "512,256"}, "--served-model-name"));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a", "--lane-ctx", "512"}, "--n-ctx"));
    // names without contexts
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b"}, "--lane-ctx"));
    // an empty or blank name, and the same name twice (a name picks one lane)
    CHECK(rejected({"--gguf", "/m/q.gguf", "--served-model-name", "a,,b", "--lane-ctx", "512,256,256"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--served-model-name", "a, ", "--lane-ctx", "512,256"}));
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,a", "--lane-ctx", "512,256"}, "twice"));
    // more than four lanes
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b,c,d,e",
                       "--lane-ctx", "256,256,256,256,256"}, "4"));
    Config four;
    CHECK(run({"--gguf", "/m/q.gguf", "--served-model-name", "a,b,c,d", "--lane-ctx", "256,256,256,256"}, four).ok);
    CHECK_EQ(four.parallel, 4);
    // --n-ctx alongside: the lanes' contexts are the context
    CHECK(refused_for({"--gguf", "/m/q.gguf", "--served-model-name", "a,b", "--lane-ctx", "512,256",
                       "--n-ctx", "768"}, "--n-ctx"));
}

TEST(config_one_name_is_unchanged) {
    Config cfg;
    CHECK(run({"--gguf", "/m/q.gguf", "--served-model-name", "qwen3.8-agent"}, cfg).ok);
    CHECK(!cfg.named_lanes());
    CHECK(cfg.lane_names.empty());
    CHECK_EQ(cfg.served_model_name, std::string("qwen3.8-agent"));
    CHECK_EQ(cfg.parallel, 1);
    // --parallel N alone: equal lanes, no names
    Config eq;
    CHECK(run({"--gguf", "/m/q.gguf", "--parallel", "2"}, eq).ok);
    CHECK(!eq.named_lanes());
    CHECK_EQ(eq.parallel, 2);
}


// The engine flag: llama is the only value; "ov" is the retired engine and is
// refused by a message that names the history tag (DESIGN 4.4).
TEST(config_engine_is_llama_only_and_ov_names_the_tag) {
    {
        Config cfg;
        CHECK(run({"--gguf", "/m/q.gguf"}, cfg).ok);
        CHECK_EQ(cfg.engine, std::string("llama"));   // a bare --gguf selects it
    }
    {
        Config cfg;
        CHECK(run({"--engine", "llama", "--gguf", "/m/q.gguf"}, cfg).ok);
        CHECK_EQ(cfg.engine, std::string("llama"));
    }
    {
        Config cfg;
        const ArgParse r = run({"--engine", "ov", "--gguf", "/m/q.gguf"}, cfg);
        CHECK(!r.ok);
        CHECK(r.error.find("openvino-history") != std::string::npos);
        CHECK(r.error.find("retired") != std::string::npos);
    }
    {
        Config cfg;
        const ArgParse r = run({"--engine", "vulkan", "--gguf", "/m/q.gguf"}, cfg);
        CHECK(!r.ok);
        CHECK(r.error.find("llama") != std::string::npos);
    }
    CHECK(rejected({"--engine"}));
    CHECK(rejected({"--engine", "llama"}));   // the engine serves a GGUF: give --gguf
}

// The retired engine's options are refused by name, with the version that
// retired them, rather than as unknown options. The list is the one the engine
// had: a flag that is neither retired nor known still reads "unknown option".
TEST(config_retired_flags_are_refused_by_name) {
    const char* const retired[] = {
        "--model", "--model-id", "--quant", "--paged-kv", "--offload-ratio", "--moe-cpu-tier",
        "--moe-cpu-tier-threads", "--moe-per-expert-dispatch", "--dflash", "--dflash-device", "--dflash-block",
        "--dflash-select", "--dflash-lambda", "--dflash-topk", "--mtp", "--mtp-layer", "--mtp-device",
        "--emb-device", "--draft", "--draft-ngram", "--gguf-mode", "--gguf-native", "--gguf-mins", "--gguf-q6k",
        "--gguf-embed", "--gguf-check", "--ngram-gguf", "--flash-next-ngram", "--prefix-cache-mib",
        "--prefix-cache-reserve", "--cache-grid", "--cache-host-mib", "--cache-dir", "--fit-margin-mib",
        "--fit-ledger-dir", "--kv-block-size", "--kv-dtype", "--kv-pool-pages", "--no-paged", "--no-logits-slice",
        "--custom-kernels", "--pin-dispatch", "--gate-pad", "--dyn-quant", "--vision", "--inspect-artifact",
        "--flash-next-offload-plan", "--gdn-checkpoint-budget", "--paged-attention-max-partitions"};
    for (const char* flag : retired) {
        CHECK(is_retired_flag(flag));
        // bare, and with a value behind it, next to a valid invocation
        for (int with_value = 0; with_value < 2; ++with_value) {
            Config cfg;
            const ArgParse r = with_value ? run({"--stub", flag, "1"}, cfg) : run({"--stub", flag}, cfg);
            CHECK(!r.ok);
            const std::string want = std::string(flag) + " was retired with the OpenVINO engine in 0.8.0 (see docs/history.md)";
            CHECK_EQ(r.error, want);
        }
    }
    // not retired, not known: the generic message
    CHECK(!is_retired_flag("--turbo"));
    CHECK(!is_retired_flag("--llama-kv"));
    CHECK(!is_retired_flag("--gguf"));
    Config cfg;
    const ArgParse r = run({"--stub", "--turbo"}, cfg);
    CHECK(!r.ok);
    CHECK(r.error.find("unknown option") != std::string::npos);
    // a retired flag is refused even where nothing else is wrong with the line
    CHECK(rejected({"--gguf", "/m/q.gguf", "--model", "/models/ov/x"}));
    // and the help does not offer any of them
    const std::string help = usage_text();
    for (const char* flag : retired) CHECK(help.find(std::string(flag) + " ") == std::string::npos &&
                                           help.find(std::string(flag) + "\n") == std::string::npos);
}

// What /props publishes as cache.kv_dtype is built from --llama-kv; this is
// the cell that fails when it ignores the configured type (the llama backend
// reported "f16" whatever the unit said until 0.8.0).
TEST(config_llama_kv_dtype_reports_the_configured_type) {
    struct Case { std::vector<const char*> args; const char* want; };
    const std::vector<Case> cases = {
        {{"--gguf", "/m/q.gguf"}, "f16"},
        {{"--gguf", "/m/q.gguf", "--llama-kv", "f16"}, "f16"},
        {{"--gguf", "/m/q.gguf", "--llama-kv", "q8_0"}, "q8_0"},
        {{"--gguf", "/m/q.gguf", "--llama-kv", "q8_0:q4_0"}, "q8_0:q4_0"},
    };
    for (const Case& c : cases) {
        Config cfg;
        CHECK(run(c.args, cfg).ok);
        CHECK_EQ(llama_kv_dtype(cfg), std::string(c.want));
    }
}

// --tts-gen-device cpu|gpu: where the audio-generation mmproj's own clip graphs
// run. "gpu" (default) keeps them on the card with the backbone; "cpu" is the
// hybrid placement the TTS real-time campaign measured as the best arm. The
// flag needs --tts, takes exactly gpu or cpu, and the stub rejects it. The
// placement itself is checked at load in backend_llama.cpp (a red if a cpu
// request builds the graphs on the card); this cell is the spec.
TEST(config_tts_gen_device_gpu_or_cpu) {
    const auto tts_args = [](std::vector<const char*> extra) {
        std::vector<const char*> a = {"--gguf", "/m/q.gguf", "--mmproj", "/m/mm.gguf",
                                      "--tts", "--tts-profiles", "/m/voices"};
        a.insert(a.end(), extra.begin(), extra.end());
        return a;
    };
    {
        Config cfg;
        CHECK(run(tts_args({}), cfg).ok);
        CHECK_EQ(cfg.tts_gen_device, std::string("gpu"));   // the card-only default
    }
    for (const char* v : {"gpu", "cpu"}) {
        Config cfg;
        CHECK(run(tts_args({"--tts-gen-device", v}), cfg).ok);
        CHECK_EQ(cfg.tts_gen_device, std::string(v));
    }
    CHECK(rejected({"--gguf", "/m/q.gguf", "--mmproj", "/m/mm.gguf", "--tts", "--tts-profiles", "/m/voices",
                    "--tts-gen-device", "card"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--mmproj", "/m/mm.gguf", "--tts", "--tts-profiles", "/m/voices",
                    "--tts-gen-device", "CPU"}));   // the values are lowercase
    CHECK(rejected({"--gguf", "/m/q.gguf", "--mmproj", "/m/mm.gguf", "--tts-gen-device", "cpu"}));   // needs --tts
    CHECK(rejected({"--gguf", "/m/q.gguf", "--tts-gen-device", "cpu"}));   // needs --tts
    CHECK(rejected({"--stub", "--tts-gen-device", "cpu"}));
    CHECK(rejected({"--gguf", "/m/q.gguf", "--mmproj", "/m/mm.gguf", "--tts", "--tts-profiles", "/m/voices",
                    "--tts-gen-device"}));   // no value
    const std::string help = usage_text();
    CHECK(help.find("--tts-gen-device") != std::string::npos);
}

TEST(config_think_eog_mask_defaults_off_and_turns_on) {
    // Off is today's behaviour; the operator turns it on by name.
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(!cfg.think_eog_mask);
    Config on;
    CHECK(run({"--stub", "--think-eog-mask"}, on).ok);
    CHECK(on.think_eog_mask);
}

TEST(config_think_eog_mask_marks_the_defaults_operator) {
    Config cfg;
    CHECK(run({"--stub", "--think-eog-mask"}, cfg).ok);
    SamplerDefaults d;
    CHECK(!apply_operator_defaults(cfg, d).has_value());
    CHECK(d.think_eog_mask);
    CHECK(d.provenance == "operator");
}

TEST(config_tool_grammar_defaults_off_and_marks_the_defaults_operator) {
    Config cfg;
    CHECK(run({"--stub"}, cfg).ok);
    CHECK(!cfg.tool_grammar);
    Config on;
    CHECK(run({"--stub", "--tool-grammar"}, on).ok);
    CHECK(on.tool_grammar);
    SamplerDefaults d;
    CHECK(!apply_operator_defaults(on, d).has_value());
    CHECK(d.tool_grammar);
    CHECK(d.provenance == "operator");
}

TEST(config_answer_penalty_flags_parse_and_travel) {
    // llava: the answer row's presence/frequency, the card's non-thinking
    // penalty after the switch. Red before the flags existed.
    Config cfg;
    CHECK(run({"--stub", "--answer-presence-penalty", "1.5",
               "--answer-frequency-penalty", "0.3"}, cfg).ok);
    CHECK(cfg.answer_presence_penalty && *cfg.answer_presence_penalty == 1.5f);
    CHECK(cfg.answer_frequency_penalty && *cfg.answer_frequency_penalty == 0.3f);

    SamplerDefaults d;
    CHECK(!apply_operator_defaults(cfg, d).has_value());
    CHECK(d.answer_presence_penalty && *d.answer_presence_penalty == 1.5f);
    CHECK(d.answer_frequency_penalty && *d.answer_frequency_penalty == 0.3f);
    CHECK(d.provenance == "operator");

    Config unset;
    CHECK(run({"--stub"}, unset).ok);
    CHECK(!unset.answer_presence_penalty);
    CHECK(!unset.answer_frequency_penalty);
}

TEST(config_answer_penalty_flags_reject_out_of_range_and_are_in_help) {
    // Same [-2, 2] as the normal row, through the same one source of truth.
    auto bad = [](std::vector<const char*> args) {
        Config cfg;
        if (!run(std::move(args), cfg).ok) return true;
        SamplerDefaults d;
        return apply_operator_defaults(cfg, d).has_value();
    };
    CHECK(bad({"--stub", "--answer-presence-penalty", "3"}));
    CHECK(bad({"--stub", "--answer-frequency-penalty", "-2.5"}));
    CHECK(rejected({"--stub", "--answer-presence-penalty", "often"}));
    CHECK(rejected({"--stub", "--answer-frequency-penalty"}));

    const std::string help = usage_text();
    for (const char* flag : {"--answer-presence-penalty", "--answer-frequency-penalty"})
        CHECK(help.find(flag) != std::string::npos);
}

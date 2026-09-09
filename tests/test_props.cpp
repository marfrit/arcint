#include "api/handlers.h"

#include "harness.h"

// /props's "cache" object (DESIGN.md §4, the served-configuration contract):
// every value in it must describe what THIS server is actually running,
// never a Config default the load path never looked at. A stub-only round
// trip (tests/roundtrip.sh) can prove the "nothing loaded" corner, but not
// the served one -- there is no card in the unit ladder to load a real model
// on. This file drives api::props() directly with a synthetic ModelStatus
// standing in for what backend_ov.cpp's load path sets, the way
// tests/test_config.cpp drives tier_prefix_cache_decision directly instead
// of loading a plugin.
namespace {

using namespace lgc;
using lgc::api::Context;
using lgc::api::SlotPool;

// A Tokenizer that is never called: props() never touches ctx.backend->
// tokenizer(), but Backend::tokenizer() is pure virtual and a FakeBackend
// still has to return *something* referenceable.
class UnusedTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(std::string_view) override { return {}; }
    std::string      decode(const std::vector<int>&) override { return {}; }
    std::string      decode_one(int) override { return {}; }
    int              eos_id() const override { return 0; }
};

class FakeBackend final : public Backend {
public:
    ModelStatus st;

    const ModelStatus& status() const override { return st; }
    Tokenizer&          tokenizer() override { return tok_; }
    std::string          render_chat(const ChatRequest&) const override { return {}; }
    // Never called: props() only reads status(), template_caps() and cache_
    // stats(), all of which have usable defaults already or are overridden
    // above.
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats&) override {
        return FinishReason::Stop;
    }

private:
    UnusedTokenizer tok_;
};

}  // namespace

TEST(props_cache_paged_reports_what_is_served) {
    // The served agent configuration this milestone's defect was measured
    // against: paged, asymmetric u8:i4, prefix cache on and holding an
    // entry, reservation measured with the u8:i4 page size (16 tokens/page).
    Config cfg;
    cfg.paged               = true;
    cfg.kv_block_size       = 32;      // the prefix-cache/stateful block, NOT the paged page size
    cfg.kv_dtype             = "fp16"; // the stateful field -- must NOT leak into a paged report
    cfg.prefix_cache_mib     = 8192;
    cfg.gdn_checkpoint_budget_mib = 512;

    FakeBackend backend;
    backend.st.id                     = "qwen3.8-27b-a3b-agent";
    backend.st.served_id              = backend.st.id;
    backend.st.loaded                 = true;
    backend.st.stub                   = false;
    backend.st.kv_precision           = "u8:i4";
    backend.st.prefix_cache_enabled   = true;
    backend.st.reservation.measured        = true;
    backend.st.reservation.kv_block_tokens = 16;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json j = lgc::api::props(ctx);
    const nlohmann::json& cache = j.at("cache");

    CHECK_EQ(cache.at("path").get<std::string>(), std::string("paged"));
    CHECK_EQ(cache.at("kv_dtype").get<std::string>(), std::string("u8:i4"));
    CHECK_EQ(cache.at("kv_block_tokens").get<int>(), 16);
    CHECK_EQ(cache.at("kv_block_size").get<int>(), 32);
    CHECK(cache.at("prefix_cache").get<bool>());
    CHECK_EQ(cache.at("prefix_cache_mib").get<int>(), 8192);
    CHECK_EQ(cache.at("gdn_checkpoint_budget_mib").get<int>(), 512);
}

TEST(props_cache_budget_asked_for_but_no_cache_serving) {
    // The exact drift ModelStatus::prefix_cache_enabled exists to catch: a
    // budget was asked for, and no cache is serving. The block must report the
    // fact (false), not the request -- reading prefix_cache_mib > 0 instead
    // would put this case back where it started.
    Config cfg;
    cfg.paged            = true;
    cfg.prefix_cache_mib = 8192;

    FakeBackend backend;
    backend.st.loaded               = true;
    backend.st.kv_precision         = "u8";
    backend.st.prefix_cache_enabled = false;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};
    const nlohmann::json  j     = lgc::api::props(ctx);
    const nlohmann::json& cache = j.at("cache");

    CHECK(!cache.at("prefix_cache").get<bool>());
    CHECK_EQ(cache.at("prefix_cache_mib").get<int>(), 8192);
}

TEST(props_cache_measured_reservation_without_a_page_size_reports_null) {
    // measured, but the page size never filled in: 0 is not a page size, so the
    // block says null rather than printing a number no pool uses.
    Config cfg;
    cfg.paged = true;

    FakeBackend backend;
    backend.st.loaded                      = true;
    backend.st.kv_precision                = "u8";
    backend.st.reservation.measured        = true;
    backend.st.reservation.kv_block_tokens = 0;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};
    const nlohmann::json  j     = lgc::api::props(ctx);
    const nlohmann::json& cache = j.at("cache");

    CHECK(cache.at("kv_block_tokens").is_null());
}

TEST(props_cache_stateful_reports_the_stateful_precision) {
    Config cfg;
    cfg.paged                     = false;
    cfg.kv_block_size             = 32;
    cfg.kv_dtype                  = "fp16";
    cfg.prefix_cache_mib          = 0;
    cfg.gdn_checkpoint_budget_mib = 512;

    FakeBackend backend;
    backend.st.id                   = "qwen3.6-27b-a3b-coder";
    backend.st.served_id            = backend.st.id;
    backend.st.loaded               = true;
    backend.st.stub                 = false;
    backend.st.kv_precision         = "fp16";
    backend.st.prefix_cache_enabled = false;
    // reservation.measured stays false: the stateful path never fills it.

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json j = lgc::api::props(ctx);
    const nlohmann::json& cache = j.at("cache");

    CHECK_EQ(cache.at("path").get<std::string>(), std::string("stateful"));
    CHECK_EQ(cache.at("kv_dtype").get<std::string>(), std::string("fp16"));
    CHECK(cache.at("kv_block_tokens").is_null());
    CHECK_EQ(cache.at("kv_block_size").get<int>(), 32);
    CHECK(!cache.at("prefix_cache").get<bool>());
    CHECK_EQ(cache.at("prefix_cache_mib").get<int>(), 0);
}

TEST(props_cache_stub_reports_null_rather_than_a_default) {
    // The stub backend loads nothing, so kv_precision stays empty and
    // prefix_cache_enabled stays false by construction (src/exec/
    // backend_stub.cpp never touches either) -- null must mean "nothing
    // served", not "fp16 was served".
    Config cfg;
    cfg.paged = true;

    FakeBackend backend;
    backend.st.id        = "qwen3.6-27b-a3b-coder";
    backend.st.served_id = backend.st.id;
    backend.st.loaded    = true;
    backend.st.stub      = true;
    // kv_precision left empty, prefix_cache_enabled left false: defaults.

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json j = lgc::api::props(ctx);
    const nlohmann::json& cache = j.at("cache");

    CHECK(cache.at("path").is_null());
    CHECK(cache.at("kv_dtype").is_null());
    CHECK(cache.at("kv_block_tokens").is_null());
    CHECK(!cache.at("prefix_cache").get<bool>());
}

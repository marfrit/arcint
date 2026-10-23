#include "api/handlers.h"

#include "harness.h"

#include <fstream>
#include <sstream>

// /props (DESIGN.md §4, the served-configuration contract): every value in it
// must describe what THIS server is actually running, never a Config default
// the load path never looked at. A stub-only round trip (tests/roundtrip.sh)
// can prove the "nothing loaded" corner, but not the served one -- there is no
// card in the unit ladder to load a real model on. This file drives
// api::props() directly with a synthetic ModelStatus standing in for what the
// llama backend's load path sets.
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
    // Never called: props() only reads status() and template_caps().
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats&) override {
        return FinishReason::Stop;
    }

private:
    UnusedTokenizer tok_;
};

}  // namespace

TEST(props_cache_kv_dtype_reports_the_configured_llama_kv) {
    // The llama backend publishes llama_kv_dtype(cfg) as the KV precision it
    // serves: whatever --llama-kv says, not a fixed "f16" (which is what it
    // reported until 0.8.0 on units that ran q8_0:q4_0).
    struct Case { const char* k; const char* v; const char* want; };
    for (const Case& c : {Case{"f16", "f16", "f16"}, Case{"q8_0", "q8_0", "q8_0"},
                          Case{"q8_0", "q4_0", "q8_0:q4_0"}}) {
        Config cfg;
        cfg.llama_kv_k = c.k;
        cfg.llama_kv_v = c.v;

        FakeBackend backend;
        backend.st.id           = "model.gguf";
        backend.st.served_id    = backend.st.id;
        backend.st.loaded       = true;
        backend.st.kv_precision = llama_kv_dtype(cfg);

        SlotPool slots(1);
        Context  ctx{&cfg, &backend, &slots};

        const nlohmann::json j = lgc::api::props(ctx);
        CHECK_EQ(j.at("cache").at("kv_dtype").get<std::string>(), std::string(c.want));
    }
}

// The llama backend needs a card to construct, so what it publishes as the KV
// precision cannot be driven here; this is the tripwire for the exact defect
// (a literal "f16" there, whatever --llama-kv said): the backend must take the
// value from llama_kv_dtype(cfg), the function the cell above pins.
TEST(llama_backend_publishes_the_configured_kv_type) {
    std::ifstream in(std::string(ARCINT_SOURCE_DIR) + "/src/exec/backend_llama.cpp");
    CHECK(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    CHECK(src.find("status_.kv_precision  = llama_kv_dtype(cfg);") != std::string::npos);
    CHECK(src.find("kv_precision  = \"") == std::string::npos);
}

TEST(props_reports_the_served_file_and_what_the_backend_knows) {
    Config cfg;
    cfg.llama_mtp = 3;

    FakeBackend backend;
    backend.st.id            = "Qwen3.8-27B-Q4_K_M.gguf";
    backend.st.served_id     = "qwen3.8-agent";
    backend.st.quant         = "Q4_K - Medium";
    backend.st.arch          = "qwen35";
    backend.st.loaded        = true;
    backend.st.n_ctx         = 245760;
    backend.st.n_ctx_train   = 262144;
    backend.st.n_layer       = 64;
    backend.st.weights_bytes = 16'000'000'000ull;
    backend.st.mtp_enabled   = true;
    backend.st.kv_precision  = "q8_0:q4_0";

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json  j = lgc::api::props(ctx);
    const nlohmann::json& m = j.at("model");
    CHECK_EQ(m.at("id").get<std::string>(), std::string("qwen3.8-agent"));
    CHECK_EQ(m.at("canonical_id").get<std::string>(), std::string("Qwen3.8-27B-Q4_K_M.gguf"));
    CHECK_EQ(m.at("quant").get<std::string>(), std::string("Q4_K - Medium"));
    CHECK_EQ(m.at("architecture").get<std::string>(), std::string("qwen35"));
    CHECK_EQ(m.at("n_ctx").get<int>(), 245760);
    CHECK_EQ(m.at("n_ctx_train").get<int>(), 262144);
    CHECK_EQ(m.at("n_layer").get<int>(), 64);
    CHECK_EQ(j.at("mtp").at("requested").get<std::string>(), std::string("on"));
    CHECK(j.at("mtp").at("enabled").get<bool>());
    CHECK_EQ(j.at("cache").at("kv_dtype").get<std::string>(), std::string("q8_0:q4_0"));
}

// The projector's own halves: /props must report libmtmd's vision/audio caps,
// so a client tells the image path from the audio-generation path without
// reading the process's flags. A VLM mmproj carries vision, a TTS mmproj
// audio; the stub is false/false.
TEST(props_reports_the_projectors_own_halves) {
    struct Case { bool vis; bool aud; };
    for (const Case& c : {Case{true, false}, Case{false, true}, Case{false, false}}) {
        FakeBackend backend;
        backend.st.id            = "backbone.gguf";
        backend.st.served_id     = backend.st.id;
        backend.st.loaded        = true;
        backend.st.mmproj_loaded = c.vis || c.aud;
        backend.st.mmproj_vision = c.vis;
        backend.st.mmproj_audio  = c.aud;

        Config   cfg;
        SlotPool slots(1);
        Context  ctx{&cfg, &backend, &slots};

        const nlohmann::json j = lgc::api::props(ctx);
        CHECK_EQ(j.at("multimodal").at("backbone_loaded").get<bool>(), true);
        CHECK_EQ(j.at("multimodal").at("mmproj_loaded").get<bool>(), c.vis || c.aud);
        CHECK_EQ(j.at("multimodal").at("vision").get<bool>(), c.vis);
        CHECK_EQ(j.at("multimodal").at("audio").get<bool>(), c.aud);
    }
}

// The values come from libmtmd's own cap query, not from --tts or an
// architecture name: the load site must call `mtmd_get_cap_from_file`.
TEST(llama_backend_publishes_the_projectors_own_caps) {
    std::ifstream in(std::string(ARCINT_SOURCE_DIR) + "/src/exec/backend_llama.cpp");
    CHECK(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    CHECK(src.find("status_.mmproj_vision = mcap.inp_vision;") != std::string::npos);
    CHECK(src.find("status_.mmproj_audio  = mcap.inp_audio;") != std::string::npos);
}

// --llama-dflash-gguf: /props says which drafter serves (one a context)
TEST(props_reports_the_dflash_drafter) {
    {
        Config cfg;
        cfg.llama_dflash_gguf = "/m/d.gguf";
        cfg.llama_dflash      = 7;
        FakeBackend backend;
        backend.st.loaded         = true;
        backend.st.dflash_enabled = true;
        backend.st.dflash_drafts  = 5;   // a drafter trained on a block of 6 caps --llama-dflash 7
        SlotPool slots(1);
        Context  ctx{&cfg, &backend, &slots};

        const nlohmann::json j = lgc::api::props(ctx);
        CHECK_EQ(j.at("dflash").at("requested").get<std::string>(), std::string("on"));
        CHECK(j.at("dflash").at("enabled").get<bool>());
        CHECK_EQ(j.at("dflash").at("drafts").get<int>(), 5);   // what is served, not the flag
        CHECK_EQ(j.at("mtp").at("requested").get<std::string>(), std::string("off"));
        CHECK(!j.at("mtp").at("enabled").get<bool>());
    }
    {
        // requested but not (yet) loaded: no count to report
        Config cfg;
        cfg.llama_dflash_gguf = "/m/d.gguf";
        cfg.llama_dflash      = 7;
        FakeBackend backend;
        SlotPool    slots(1);
        Context     ctx{&cfg, &backend, &slots};

        const nlohmann::json j = lgc::api::props(ctx);
        CHECK_EQ(j.at("dflash").at("requested").get<std::string>(), std::string("on"));
        CHECK(!j.at("dflash").at("enabled").get<bool>());
        CHECK(j.at("dflash").at("drafts").is_null());
    }
    {
        Config      cfg;
        FakeBackend backend;
        backend.st.loaded = true;
        SlotPool slots(1);
        Context  ctx{&cfg, &backend, &slots};

        const nlohmann::json j = lgc::api::props(ctx);
        CHECK_EQ(j.at("dflash").at("requested").get<std::string>(), std::string("off"));
        CHECK(!j.at("dflash").at("enabled").get<bool>());
        CHECK(j.at("dflash").at("drafts").is_null());
    }
}

TEST(props_carries_none_of_the_retired_engines_fields) {
    Config cfg;
    FakeBackend backend;
    backend.st.id     = "model.gguf";
    backend.st.loaded = true;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json j = lgc::api::props(ctx);
    CHECK(!j.contains("reservation"));
    for (const char* k : {"path", "kv_block_tokens", "kv_block_size", "prefix_cache", "prefix_cache_mib",
                          "gdn_checkpoint_budget_mib"})
        CHECK(!j.at("cache").contains(k));
    for (const char* k : {"ov_arch", "n_qsa_layer", "qsa", "arch_hash", "template_hash", "tokenizer_hash",
                          "n_gdn_layer", "n_attn_layer"})
        CHECK(!j.at("model").contains(k));

    // and /health no longer counts KV pages or a prefix cache
    const nlohmann::json h = lgc::api::health(ctx);
    CHECK(!h.contains("kv_blocks_free") && !h.contains("kv_blocks_total") && !h.contains("cache"));
}

TEST(props_stub_reports_null_rather_than_a_default) {
    // The stub backend loads nothing, so kv_precision, quant and arch stay
    // empty by construction (src/exec/backend_stub.cpp never touches them) --
    // null must mean "nothing served", not "f16 was served".
    Config cfg;
    cfg.stub = true;

    FakeBackend backend;
    backend.st.id        = "stub";
    backend.st.served_id = backend.st.id;
    backend.st.loaded    = true;
    backend.st.stub      = true;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json j = lgc::api::props(ctx);
    CHECK(j.at("cache").at("kv_dtype").is_null());
    CHECK(j.at("model").at("quant").is_null());
    CHECK(j.at("model").at("architecture").is_null());
    CHECK(j.at("model").at("n_ctx_train").is_null());
}

// The non-thinking row: /props reports which row a thinking-off request gets,
// so the operator can see it. "configured" false means no --nothink-* flag and
// every request on the normal row, exactly as before.
TEST(props_reports_the_non_thinking_row) {
    Config      cfg;
    FakeBackend backend;
    backend.st.id         = "stub";
    backend.st.served_id  = backend.st.id;
    backend.st.loaded     = true;
    backend.st.n_ctx      = 256;

    SamplerDefaults& sd = backend.st.sampler_defaults;
    sd.nothink_row              = true;
    sd.nothink_temperature      = 0.7f;
    sd.nothink_top_p            = 0.8f;
    sd.nothink_top_k            = 35;
    sd.nothink_min_p            = 0.05f;
    sd.nothink_presence_penalty = 0.3f;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};

    const nlohmann::json n = lgc::api::props(ctx).at("sampler_defaults").at("nothink");
    CHECK(n.at("configured").get<bool>());
    CHECK(n.at("temperature").get<float>() == 0.7f);
    CHECK(n.at("top_p").get<float>() == 0.8f);
    CHECK(n.at("top_k").get<int>() == 35);
    CHECK(n.at("min_p").get<float>() == 0.05f);
    CHECK(n.at("presence_penalty").get<float>() == 0.3f);

    // Unset fields are null, not a guessed value: they fall back to the
    // normal row, which is reported above them.
    sd = SamplerDefaults{};
    sd.nothink_row = true;
    sd.nothink_top_p = 0.8f;
    const nlohmann::json n2 = lgc::api::props(ctx).at("sampler_defaults").at("nothink");
    CHECK(n2.at("configured").get<bool>());
    CHECK(n2.at("temperature").is_null());
    CHECK(n2.at("top_p").get<float>() == 0.8f);
    CHECK(n2.at("top_k").is_null());

    // No --nothink-* flag: nothing configured
    sd = SamplerDefaults{};
    const nlohmann::json n3 = lgc::api::props(ctx).at("sampler_defaults").at("nothink");
    CHECK(!n3.at("configured").get<bool>());
    CHECK(n3.at("temperature").is_null());
}

TEST(props_reports_the_answer_rows_penalties) {
    // llava: /props must say which answer-row penalties are set, null for the
    // ones the operator left on the thinking row's value. Red before the keys.
    Config      cfg;
    FakeBackend backend;
    backend.st.id        = "stub";
    backend.st.served_id = backend.st.id;
    backend.st.loaded    = true;
    backend.st.n_ctx     = 256;

    SamplerDefaults& sd = backend.st.sampler_defaults;
    sd.answer_presence_penalty  = 1.5f;
    sd.answer_frequency_penalty = 0.3f;

    SlotPool slots(1);
    Context  ctx{&cfg, &backend, &slots};
    const nlohmann::json a = lgc::api::props(ctx).at("sampler_defaults").at("answer");
    CHECK(a.at("presence_penalty").get<float>() == 1.5f);
    CHECK(a.at("frequency_penalty").get<float>() == 0.3f);

    sd = SamplerDefaults{};
    sd.answer_presence_penalty = 0.0f;   // explicitly set, still reported
    const nlohmann::json a2 = lgc::api::props(ctx).at("sampler_defaults").at("answer");
    CHECK(a2.at("presence_penalty").get<float>() == 0.0f);
    CHECK(a2.at("frequency_penalty").is_null());
}

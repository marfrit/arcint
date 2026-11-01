// ops item 3 (arcint 1.4.0): POST /tokenize and POST /detokenize.
//
// Three body shapes on /tokenize:
//   - llama-server's `{content, add_special?, with_pieces?, model?}`;
//   - a chat body (`messages`, `tools`, …), counted through prepare_chat and
//     the tokens finish_prepare stored in `GenerationInput::prompt_ids_text`,
//     never a second encode;
//   - a Responses body (`input`, `instructions`, `tools`, …), the same through
//     prepare_response.
//
// The pieces are the token mode's decode pieces: `decode_one(id)`, which is
// what the renderer material's `decode.pieces_b64[id]` carries. A piece that is
// not valid UTF-8 on its own is the array of its byte values, llama-server's
// representation (a split multi-byte character is the case that produces one).
// detokenize renders specials as text (llama-server's default
// unparse_special=true), which is Tokenizer::decode.
//
// The count case is the contract that matters: for a chat body and a Responses
// body, `count` equals the `usage.prompt_tokens` a real stub generation of the
// same request reports, because both come from the same prepare and the same
// stored id vector.
//
// The accessors below never index an array the response did not carry: a red
// case must fail an assertion, not crash the runner (the review record on
// W3.5c R1).
#include "api/handlers.h"
#include "api/renderer.h"
#include "api/responses.h"
#include "api/responses_state.h"
#include "api/tokenize.h"
#include "exec/backend.h"

#include "harness.h"

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using namespace lgc;
using lgc::api::Context;
using lgc::api::PreparedChat;
using lgc::api::PreparedResponse;
using lgc::api::SlotPool;
using json = nlohmann::json;

namespace {

// One id per byte, with a BOS id the two-argument encode prepends. The
// one-byte-per-id split is what makes a multi-byte character arrive as two
// pieces that are each invalid UTF-8 -- the case the array representation is
// for. decode_one is the piece source the renderer material publishes.
class PieceTokenizer final : public Tokenizer {
public:
    static constexpr int kBos = 256;
    std::vector<int> encode(std::string_view text) override {
        std::vector<int> v;
        for (char c : text) v.push_back(static_cast<unsigned char>(c));
        return v;
    }
    std::vector<int> encode(std::string_view text, bool add_special) override {
        std::vector<int> v = encode(text);
        if (add_special) v.insert(v.begin(), kBos);
        return v;
    }
    std::string decode(const std::vector<int>& ids) override {
        std::string s;
        for (int id : ids) s += decode_one(id);
        return s;
    }
    std::string decode_one(int id) override {
        if (id == kBos) return "BOS";
        return std::string(1, static_cast<char>(id));
    }
    int eos_id() const override { return 0; }
    int vocab_size() const override { return 257; }
};

class PieceBackend final : public Backend {
public:
    PieceBackend() {
        st.id          = "stub";
        st.served_id   = "stub";
        st.loaded      = true;
        st.stub        = true;
        st.n_ctx       = 4096;
        st.n_ctx_train = kStubNCtxTrain;
    }
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok; }
    std::string        render_chat(const ChatRequest& req) const override {
        std::string out;
        for (const ChatMessage& m : req.messages) out += m.content;
        return out;
    }
    FinishReason generate(const GenerationInput& in, int, const TokenCallback&,
                          GenerationStats& stats) override {
        const std::vector<int> ids = prompt_ids_raw(in, tok);
        stats.prompt_tokens        = static_cast<int>(ids.size());
        return FinishReason::Stop;
    }

    PieceTokenizer tok;
    ModelStatus    st;
};

// The production stub backend (a real stub generation for the count cases) and
// the byte-splitting fake for the piece cases, with an optional named-lane
// context.
struct Fixture {
    Config                                 cfg;
    std::unique_ptr<Backend>               backend;
    SlotPool                               slots{1};
    std::vector<std::unique_ptr<SlotPool>> pools;
    Context                                ctx;

    explicit Fixture(int n_ctx = 4096, bool named_lanes = false) {
        cfg.stub = true;
        if (named_lanes) {
            cfg.served_model_name = "agent";
            cfg.lane_names        = {"agent", "sub"};
            cfg.lane_ctx          = {512, 256};
            cfg.parallel          = 2;
        }
        backend     = make_stub_backend(n_ctx);
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
        if (named_lanes) ctx.lanes = api::make_lanes(cfg, pools);
    }
};

// The byte-splitting fake with the same optional named-lane context.
struct PieceFixture {
    Config                                 cfg;
    PieceBackend                           backend;
    SlotPool                               slots{1};
    std::vector<std::unique_ptr<SlotPool>> pools;
    Context                                ctx;

    explicit PieceFixture(bool named_lanes = false) {
        cfg.stub = true;
        if (named_lanes) {
            cfg.served_model_name = "agent";
            cfg.lane_names        = {"agent", "sub"};
            cfg.lane_ctx          = {512, 256};
            cfg.parallel          = 2;
        }
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
        if (named_lanes) ctx.lanes = api::make_lanes(cfg, pools);
    }
};

// The production stub backend plus an in-memory ResponseStore, for the cases
// that pin what /tokenize does to a continuation's stored entry.
struct StoreFixture {
    Config                             cfg;
    std::unique_ptr<Backend>           backend;
    SlotPool                           slots{1};
    std::shared_ptr<api::ResponseStore> store;
    Context                            ctx;

    explicit StoreFixture(int n_ctx = 4096) {
        cfg.stub    = true;
        backend     = make_stub_backend(n_ctx);
        store       = std::make_shared<api::ResponseStore>(api::ResponseStore::Limits{});
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
        ctx.responses = store;
    }
};

// A stored completed turn with one user message and a valid prefix hash.
std::shared_ptr<api::StoredResponse> store_entry(StoreFixture& f, const std::string& id) {
    auto e       = std::make_shared<api::StoredResponse>();
    e->id        = id;
    e->lane      = -1;
    e->status    = "completed";
    e->chat.model = "stub";
    ChatMessage u;
    u.role    = "user";
    u.content = "find the port";
    e->chat.messages = {u};
    e->prompt        = f.backend->render_chat(e->chat);
    e->prefix_hash   = api::sha256_hex(e->prompt);
    e->prompt_tokens = 3;
    f.store->put(e);
    return e;
}

// A vocabulary that wants BOS (add_bos true): the llama backend's prompt_tokens
// prepends it when the rendered template did not write it, so the count must
// too. encode() adds no special tokens (add_special false), exactly the llama
// tokenizer's plain encode.
class BosTokenizer final : public Tokenizer {
public:
    static constexpr int kBos = 7;
    std::vector<int> encode(std::string_view text) override {
        std::vector<int> v;
        for (char c : text) v.push_back(static_cast<unsigned char>(c) + 1000);
        return v;
    }
    std::string decode(const std::vector<int>& ids) override {
        std::string s;
        for (int id : ids) s += decode_one(id);
        return s;
    }
    std::string decode_one(int id) override {
        if (id == kBos) return "BOS";
        return std::string(1, static_cast<char>(id - 1000));
    }
    int eos_id() const override { return 0; }
    int bos_id() const override { return kBos; }
    int vocab_size() const override { return 2000; }
};

class BosBackend final : public Backend {
public:
    BosBackend() {
        st.id          = "stub";
        st.served_id   = "stub";
        st.loaded      = true;
        st.stub        = true;
        st.n_ctx       = 4096;
        st.n_ctx_train = kStubNCtxTrain;
    }
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok; }
    std::string        render_chat(const ChatRequest& req) const override {
        std::string out;
        for (const ChatMessage& m : req.messages) out += m.content;
        return out;
    }
    // The vocab wants BOS; the material and /props say so.
    json template_caps() const override { return json{{"add_bos", true}}; }
    FinishReason generate(const GenerationInput& in, int, const TokenCallback&,
                          GenerationStats& stats) override {
        // The llama backend's own rule (backend_llama.cpp prompt_tokens).
        std::vector<int> ids = prompt_ids_raw(in, tok);
        if (in.prompt_ids.empty() && (ids.empty() || ids.front() != BosTokenizer::kBos))
            ids.insert(ids.begin(), BosTokenizer::kBos);
        stats.prompt_tokens = static_cast<int>(ids.size());
        return FinishReason::Stop;
    }
    BosTokenizer tok;
    ModelStatus  st;
};

struct BosFixture {
    Config     cfg;
    BosBackend backend;
    SlotPool   slots{1};
    Context    ctx;
    BosFixture() {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
};

// An mmproj unit: the projector counts image tokens that /tokenize cannot know
// without running it.
class MmBackend final : public Backend {
public:
    MmBackend() {
        st.id           = "stub";
        st.served_id    = "stub";
        st.loaded       = true;
        st.stub         = true;
        st.n_ctx        = 4096;
        st.n_ctx_train  = kStubNCtxTrain;
        st.mmproj_loaded = true;
        st.mmproj_vision = true;
    }
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok; }
    std::string        render_chat(const ChatRequest& req) const override {
        std::string out;
        for (const ChatMessage& m : req.messages) out += m.content;
        return out;
    }
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats& stats) override {
        stats.prompt_tokens = 1;
        return FinishReason::Stop;
    }
    PieceTokenizer tok;
    ModelStatus    st;
};

struct MmFixture {
    Config    cfg;
    MmBackend backend;
    SlotPool  slots{1};
    Context   ctx;
    MmFixture() {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
};

// The id list a body's `tokens` carries, or empty when it is absent or not an
// integer array.
std::vector<int> ids_of(const json& body) {
    std::vector<int> out;
    if (!body.contains("tokens") || !body["tokens"].is_array()) return out;
    for (const json& t : body["tokens"]) {
        if (!t.is_number_integer()) return {};
        out.push_back(t.get<int>());
    }
    return out;
}

// The count a body carries, or -1.
int count_of(const json& body) {
    if (!body.contains("count") || !body["count"].is_number_integer()) return -1;
    return body["count"].get<int>();
}

// One string field of the error envelope, or empty. Never calls value() on a
// body that is not an object (a red case must fail, not throw).
std::string err_field(const json& body, const char* key) {
    if (!body.contains("error") || !body["error"].is_object()) return {};
    return body["error"].value(key, std::string());
}

// The piece a response carries for an id, as bytes: a string for valid UTF-8,
// an array of byte values otherwise (llama-server's representation).
std::string piece_bytes(const json& piece) {
    if (piece.is_string()) return piece.get<std::string>();
    std::string out;
    for (const json& b : piece) out += static_cast<char>(b.get<int>());
    return out;
}

}  // namespace

// ------------------------------------------------------------ plain content

TEST(tokenize_content_returns_the_ids_the_tokenizer_makes) {
    PieceFixture f;
    const json body = json{{"model", "stub"}, {"content", "hello \xC3\xBC"}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 200);
    CHECK(ids_of(r.body) == f.backend.tok.encode("hello \xC3\xBC"));
    // No `count` on the llama-server shape.
    CHECK_EQ(count_of(r.body), -1);
}

TEST(tokenize_content_honours_add_special) {
    PieceFixture f;
    const std::vector<int> ids =
        ids_of(api::tokenize_request(f.ctx, json{{"content", "x"}, {"add_special", true}}).body);
    CHECK_EQ(ids.size(), 2u);
    if (ids.size() == 2u) CHECK_EQ(ids.front(), PieceTokenizer::kBos);
    // The default is false, llama-server's default here.
    const std::vector<int> d =
        ids_of(api::tokenize_request(f.ctx, json{{"content", "x"}}).body);
    CHECK_EQ(d.size(), 1u);
    if (d.size() == 1u) CHECK_EQ(d.front(), static_cast<int>('x'));
}

TEST(tokenize_with_pieces_carries_decode_one_bytes) {
    PieceFixture f;
    const std::string text = "a\xC3\xBC b";
    const json body = json{{"content", text}, {"with_pieces", true}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 200);
    const std::vector<int> ids = f.backend.tok.encode(text);
    CHECK(r.body.contains("tokens"));
    if (!r.body.contains("tokens") || !r.body["tokens"].is_array()) return;
    CHECK_EQ(r.body["tokens"].size(), ids.size());
    if (r.body["tokens"].size() != ids.size()) return;
    for (size_t i = 0; i < ids.size(); ++i) {
        const json& t = r.body["tokens"][i];
        CHECK_EQ(t.value("id", -1), ids[i]);
        // The token mode's decode piece: what the renderer material's
        // decode.pieces_b64[id] publishes. The piece is part of the contract,
        // so its absence is a failure, not a skip.
        CHECK(t.contains("piece"));
        if (t.contains("piece")) CHECK(piece_bytes(t["piece"]) == f.backend.tok.decode_one(ids[i]));
    }
}

TEST(tokenize_piece_of_a_split_multibyte_char_is_a_byte_array) {
    PieceFixture f;
    // "\xC3\xBC" is one two-byte code point; the byte-splitting tokenizer makes
    // two ids, so neither piece is valid UTF-8 on its own.
    const json body = json{{"content", "\xC3\xBC"}, {"with_pieces", true}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 200);
    CHECK(r.body.contains("tokens") && r.body["tokens"].is_array());
    if (!r.body.contains("tokens") || !r.body["tokens"].is_array()) return;
    CHECK_EQ(r.body["tokens"].size(), 2u);
    if (r.body["tokens"].size() != 2u) return;
    const json& p0 = r.body["tokens"][0]["piece"];
    const json& p1 = r.body["tokens"][1]["piece"];
    CHECK(p0.is_array());
    CHECK(p1.is_array());
    if (p0.is_array() && p1.is_array()) {
        CHECK_EQ(p0[0].get<int>(), 0xC3);
        CHECK_EQ(p1[0].get<int>(), 0xBC);
        // The bytes still concatenate to the character.
        CHECK_EQ(piece_bytes(p0) + piece_bytes(p1), std::string("\xC3\xBC"));
    }
}

// --------------------------------------------------------------- detokenize

TEST(detokenize_concatenates_the_decode_pieces) {
    PieceFixture f;
    const std::vector<int> ids = f.backend.tok.encode("hello \xC3\xBC");
    const api::HttpResult r = api::detokenize_request(f.ctx, json{{"tokens", ids}});
    CHECK_EQ(r.status, 200);
    CHECK_EQ(r.body.value("content", std::string()), f.backend.tok.decode(ids));
    CHECK_EQ(r.body.value("content", std::string()), std::string("hello \xC3\xBC"));
}

TEST(detokenize_renders_specials_as_text) {
    PieceFixture f;
    // The BOS id is a special token the vocabulary spells; decode_one returns
    // its text, llama-server's default unparse_special=true.
    const std::vector<int> ids = {PieceTokenizer::kBos, static_cast<int>('x')};
    const api::HttpResult r = api::detokenize_request(f.ctx, json{{"tokens", ids}});
    CHECK_EQ(r.status, 200);
    CHECK_EQ(r.body.value("content", std::string()), std::string("BOSx"));
}

TEST(detokenize_rejects_a_non_integer_token) {
    PieceFixture f;
    const api::HttpResult r =
        api::detokenize_request(f.ctx, json{{"tokens", json::array({"x"})}});
    CHECK_EQ(r.status, 400);
}

TEST(detokenize_rejects_an_out_of_range_token) {
    PieceFixture f;
    // decode_one would read past the vocabulary (the llama backend throws); the
    // route refuses the id with a 400 naming `tokens` instead.
    const api::HttpResult r =
        api::detokenize_request(f.ctx, json{{"tokens", json::array({1 << 20})}});
    CHECK_EQ(r.status, 400);
    CHECK_EQ(err_field(r.body, "param"), std::string("tokens"));
}

TEST(detokenize_rejects_a_negative_token) {
    PieceFixture f;
    const api::HttpResult r =
        api::detokenize_request(f.ctx, json{{"tokens", json::array({-1})}});
    CHECK_EQ(r.status, 400);
    CHECK_EQ(err_field(r.body, "param"), std::string("tokens"));
}

TEST(detokenize_rejects_a_float_token) {
    // get<std::vector<int>>() would truncate 1.5 to 1 and serve it.
    PieceFixture f;
    const api::HttpResult r =
        api::detokenize_request(f.ctx, json{{"tokens", json::array({1.5})}});
    CHECK_EQ(r.status, 400);
    CHECK_EQ(err_field(r.body, "param"), std::string("tokens"));
}

TEST(detokenize_rejects_a_boolean_token) {
    // get<std::vector<int>>() would convert true to 1 and serve it.
    PieceFixture f;
    const api::HttpResult r =
        api::detokenize_request(f.ctx, json{{"tokens", json::array({true})}});
    CHECK_EQ(r.status, 400);
    CHECK_EQ(err_field(r.body, "param"), std::string("tokens"));
}

TEST(detokenize_rejects_an_integer_past_int32) {
    // 2^32 + 5 wraps to 5 in get<std::vector<int>>(), which then passes the
    // vocabulary range check.
    PieceFixture f;
    const api::HttpResult r = api::detokenize_request(
        f.ctx, json{{"tokens", json::array({4294967301ll})}});
    CHECK_EQ(r.status, 400);
    CHECK_EQ(err_field(r.body, "param"), std::string("tokens"));
}

// ------------------------------------------------------------- the count

TEST(tokenize_chat_count_equals_the_generations_prompt_tokens) {
    Fixture f;
    const json body = json{{"model", "stub"},
                           {"messages", json::array({{{"role", "user"}, {"content", "hello world"}}})}};

    const api::HttpResult t = api::tokenize_request(f.ctx, body);
    CHECK_EQ(t.status, 200);

    // A real generation of the same request, through the same prepare path.
    PreparedChat prep;
    CHECK(!api::prepare_chat(f.ctx, body, prep));
    const api::HttpResult gen = api::run_chat(f.ctx, prep, 0);
    CHECK_EQ(gen.status, 200);
    const int usage_prompt =
        gen.body.contains("usage") ? gen.body["usage"].value("prompt_tokens", -1) : -1;

    CHECK_EQ(count_of(t.body), usage_prompt);
    // And the tokens are the stored id vector, not a second encode.
    CHECK(ids_of(t.body) == prep.input.prompt_ids_text);
    CHECK_EQ(count_of(t.body), static_cast<int>(prep.input.prompt_ids_text.size()));
}

TEST(tokenize_responses_count_equals_the_generations_prompt_tokens) {
    Fixture f;
    const json body = json{{"model", "stub"},
                           {"instructions", "be terse"},
                           {"input", "hello world"},
                           {"max_output_tokens", 16}};

    const api::HttpResult t = api::tokenize_request(f.ctx, body);
    CHECK_EQ(t.status, 200);

    api::ResponseRequest req;
    CHECK(!api::parse_response_request(body, req));
    PreparedResponse prep;
    CHECK(!api::prepare_response(f.ctx, std::move(req), prep));
    const api::HttpResult gen = api::run_response(f.ctx, prep, 0);
    CHECK_EQ(gen.status, 200);
    const int usage_prompt =
        gen.body.contains("usage") ? gen.body["usage"].value("input_tokens", -1) : -1;

    CHECK_EQ(count_of(t.body), usage_prompt);
    CHECK(ids_of(t.body) == prep.chat.input.prompt_ids_text);
}

// The count comes from finish_prepare's stored vector: a chat body tokenises
// once, and the /tokenize answer must not add a second pass.
class CountingTokenizer final : public Tokenizer {
public:
    mutable int encodes = 0;
    std::vector<int> encode(std::string_view text) override {
        ++encodes;
        return std::vector<int>(text.begin(), text.end());
    }
    std::vector<int> encode(std::string_view text, bool add_special) override {
        (void)add_special;
        ++encodes;
        return encode(text);
    }
    std::string decode(const std::vector<int>& ids) override {
        return std::string(ids.begin(), ids.end());
    }
    std::string decode_one(int id) override { return std::string(1, static_cast<char>(id)); }
    int         eos_id() const override { return 0; }
};

class CountingBackend final : public Backend {
public:
    CountingBackend() {
        st.id = "stub";
        st.served_id = "stub";
        st.loaded = true;
        st.stub = true;
        st.n_ctx = 4096;
        st.n_ctx_train = kStubNCtxTrain;
    }
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok; }
    std::string        render_chat(const ChatRequest& req) const override {
        std::string out;
        for (const ChatMessage& m : req.messages) out += m.content;
        return out;
    }
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats& stats) override {
        stats.prompt_tokens = 1;
        return FinishReason::Stop;
    }
    CountingTokenizer tok;
    ModelStatus       st;
};

TEST(tokenize_chat_count_does_not_encode_twice) {
    Config cfg;
    cfg.stub = true;
    CountingBackend backend;
    SlotPool        slots{1};
    Context         ctx{&cfg, &backend, &slots};

    const json body = json{{"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}};
    const api::HttpResult t = api::tokenize_request(ctx, body);
    CHECK_EQ(t.status, 200);
    CHECK(count_of(t.body) >= 0);
    CHECK_EQ(backend.tok.encodes, 1);   // prepare's one tokenisation, reused
}

TEST(tokenize_responses_truncation_auto_count_matches_the_generation) {
    // The W3.5 `auto` fit pass renders and tokenises once, then hands the
    // served ids to finish_prepare (N3). The count must be that vector's size,
    // over the fit limit but under n_ctx, where the turn is served.
    json body = json{{"model", "stub"}, {"truncation", "auto"}};
    std::string input;
    for (int i = 0; i < 2000; ++i) input += "x ";
    body["input"] = input;

    // The full prompt's count, from the same renderer the route uses.
    Fixture probe;
    api::ResponseRequest r;
    CHECK(!api::parse_response_request(body, r));
    const int full = static_cast<int>(
        probe.backend->tokenizer().encode(probe.backend->render_chat(r.chat)).size());
    CHECK(full > 0);

    Fixture f(full + 128);   // fit limit = full - 128: over it, under n_ctx
    const api::HttpResult t = api::tokenize_request(f.ctx, body);
    CHECK_EQ(t.status, 200);
    CHECK_EQ(count_of(t.body), full);

    api::ResponseRequest req;
    CHECK(!api::parse_response_request(body, req));
    PreparedResponse prep;
    CHECK(!api::prepare_response(f.ctx, std::move(req), prep));
    CHECK_EQ(prep.chat.truncation.blocks_dropped, 0);
    CHECK_EQ(prep.chat.truncation.tokens_after, full);
    const api::HttpResult gen = api::run_response(f.ctx, prep, 0);
    CHECK_EQ(gen.status, 200);
    const int usage_prompt =
        gen.body.contains("usage") ? gen.body["usage"].value("input_tokens", -1) : -1;
    CHECK_EQ(count_of(t.body), usage_prompt);
    CHECK(ids_of(t.body) == prep.chat.input.prompt_ids_text);
}

// A vocabulary that wants BOS: the generation prepends it, so the count must
// carry it too, or /tokenize would report one token fewer than the usage.
TEST(tokenize_chat_count_carries_the_bos_the_generation_adds) {
    BosFixture f;
    const json body = json{{"model", "stub"},
                           {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}};
    const api::HttpResult t = api::tokenize_request(f.ctx, body);
    CHECK_EQ(t.status, 200);

    PreparedChat prep;
    CHECK(!api::prepare_chat(f.ctx, body, prep));
    const api::HttpResult gen = api::run_chat(f.ctx, prep, 0);
    CHECK_EQ(gen.status, 200);
    const int usage_prompt =
        gen.body.contains("usage") ? gen.body["usage"].value("prompt_tokens", -1) : -1;

    CHECK_EQ(count_of(t.body), usage_prompt);
    CHECK_EQ(count_of(t.body), static_cast<int>(prep.input.prompt_ids_text.size()) + 1);
    const std::vector<int> ids = ids_of(t.body);
    CHECK(!ids.empty());
    if (!ids.empty()) CHECK_EQ(ids.front(), BosTokenizer::kBos);
}

// An mmproj unit counts the image tokens with mtmd_tokenize, which /tokenize
// does not run: the count cannot be exact, so the request is refused.
TEST(tokenize_refuses_image_parts_on_an_mmproj_unit) {
    MmFixture f;
    const json img = json{{"type", "image_url"},
                          {"image_url", {{"url", "data:image/png;base64,AAAA"}}}};
    const json body = json{{"model", "stub"},
                           {"messages", json::array({{{"role", "user"},
                               {"content", json::array({{{"type", "text"}, {"text", "hi"}}, img})}}})}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 400);
    CHECK(err_field(r.body, "message").find("image") != std::string::npos);
}

TEST(tokenize_responses_refuses_image_parts_on_an_mmproj_unit) {
    MmFixture f;
    const json body = json{{"model", "stub"},
                           {"input", json::array({{{"type", "message"}, {"role", "user"},
                               {"content", json::array({{{"type", "input_text"}, {"text", "hi"}},
                                   {{"type", "input_image"}, {"image_url", "data:image/png;base64,AAAA"}}})}}})}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 400);
    CHECK(err_field(r.body, "message").find("image") != std::string::npos);
}

// ------------------------------------------------------- the store is untouched

// A count request is not a turn: a continuation it cannot prepare must not
// evict the entry it referenced, or learning whether the next turn fits would
// destroy the chain.
TEST(tokenize_continuation_overflow_keeps_the_stored_entry) {
    StoreFixture f;
    store_entry(f, "resp-1");
    std::string input;
    for (int i = 0; i < 3000; ++i) input += "x ";
    const api::HttpResult r =
        api::tokenize_request(f.ctx, json{{"model", "stub"},
                                          {"previous_response_id", "resp-1"},
                                          {"input", input}});
    CHECK_EQ(r.status, 400);   // the lane's cap, from the same prepare path
    CHECK_EQ(err_field(r.body, "code"), std::string("context_length_exceeded"));
    CHECK(f.store->get("resp-1", -1) != nullptr);
}

TEST(tokenize_continuation_prefix_mismatch_keeps_the_stored_entry) {
    StoreFixture f;
    auto e        = store_entry(f, "resp-1");
    e->prefix_hash = "sha256:0000";
    f.store->put(e);
    const api::HttpResult r =
        api::tokenize_request(f.ctx, json{{"model", "stub"},
                                          {"previous_response_id", "resp-1"},
                                          {"input", "hi"}});
    CHECK_EQ(r.status, 400);
    CHECK_EQ(err_field(r.body, "code"), std::string("previous_response_not_found"));
    CHECK(f.store->get("resp-1", -1) != nullptr);
}

// ------------------------------------------------------------ lane routing

TEST(tokenize_content_unknown_model_is_the_same_404) {
    PieceFixture f(/*named_lanes=*/true);
    const api::HttpResult r =
        api::tokenize_request(f.ctx, json{{"model", "nope"}, {"content", "x"}});
    CHECK_EQ(r.status, 404);
    CHECK_EQ(err_field(r.body, "code"), std::string("model_not_found"));
    CHECK_EQ(err_field(r.body, "param"), std::string("model"));
    CHECK_EQ(err_field(r.body, "type"), std::string("invalid_request_error"));
}

TEST(tokenize_chat_unknown_model_is_the_same_404) {
    PieceFixture f(/*named_lanes=*/true);
    const json body = json{{"model", "nope"},
                           {"messages", json::array({{{"role", "user"}, {"content", "x"}}})}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 404);
    CHECK_EQ(err_field(r.body, "code"), std::string("model_not_found"));
}

TEST(tokenize_named_lane_serves_its_name) {
    PieceFixture f(/*named_lanes=*/true);
    const json body = json{{"model", "sub"},
                           {"messages", json::array({{{"role", "user"}, {"content", "x"}}})}};
    const api::HttpResult r = api::tokenize_request(f.ctx, body);
    CHECK_EQ(r.status, 200);
    CHECK(count_of(r.body) >= 0);
}

TEST(detokenize_unknown_model_is_a_404) {
    PieceFixture f(/*named_lanes=*/true);
    const api::HttpResult r =
        api::detokenize_request(f.ctx, json{{"model", "nope"}, {"tokens", json::array()}});
    CHECK_EQ(r.status, 404);
    CHECK_EQ(err_field(r.body, "code"), std::string("model_not_found"));
}

// ---------------------------------------------------------------- bad bodies

TEST(tokenize_without_a_known_body_is_a_400) {
    PieceFixture f;
    const api::HttpResult r = api::tokenize_request(f.ctx, json{{"nonsense", true}});
    CHECK_EQ(r.status, 400);
}

TEST(tokenize_content_must_be_a_string) {
    PieceFixture f;
    const api::HttpResult r = api::tokenize_request(f.ctx, json{{"content", 7}});
    CHECK_EQ(r.status, 400);
}

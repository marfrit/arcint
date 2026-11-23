// /v1/completions honours `arcint: {"timings": true}` as the chat route does.
// RED on the tree before the fix: parse_completion_request had no arcint block,
// so the flag was dropped silently and run_completion answered no `timings`.
//
// One-line mutation after the fix: delete the assignment `req.timings = ...` in
// read_arcint (or the `if (prep.req.timings) body["timings"] = ...` line in
// run_completion); the body loses the block and this case fails again.
#include "api/handlers.h"

#include "harness.h"

#include <string>
#include <vector>

using namespace lgc;
using nlohmann::json;

namespace {

class Stub final : public Backend {
public:
    Stub() {
        st.id          = "stub";
        st.served_id   = "stub";
        st.n_ctx_train = kStubNCtxTrain;
        st.loaded      = true;
        st.stub        = true;
        st.n_ctx       = 4096;
    }
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest&) const override { return {}; }
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats&) override {
        return FinishReason::Stop;
    }

private:
    struct Tok final : public Tokenizer {
        std::vector<int> encode(std::string_view text) override {
            std::vector<int> v;
            for (std::size_t i = 0; i < text.size(); ++i) v.push_back(static_cast<int>(i) + 100);
            return v;
        }
        std::string decode(const std::vector<int>& v) override {
            return std::string(v.size(), 'x');
        }
        std::string decode_one(int) override { return "x"; }
        int         eos_id() const override { return 0; }
    } tok_;
    ModelStatus st;
};

api::HttpResult run(const json& body) {
    Config cfg;
    Stub backend;
    api::Context ctx{&cfg, &backend, nullptr};
    api::PreparedCompletion prep;
    const auto err = api::prepare_completion(ctx, body, prep);
    if (err) return *err;
    return api::run_completion(ctx, prep, 0);
}

json completion(const json& arcint) {
    json b = {{"model", "stub"}, {"prompt", "hi"}};
    if (!arcint.is_null()) b["arcint"] = arcint;
    return b;
}

}  // namespace

TEST(completions_timings_are_absent_unless_asked) {
    const auto r = run(completion(nullptr));
    CHECK_EQ(r.status, 200);
    CHECK(!r.body.contains("timings"));
}

TEST(completions_timings_honour_the_flag) {
    const auto r = run(completion(json{{"timings", true}}));
    CHECK_EQ(r.status, 200);
    CHECK(r.body.contains("timings"));
    if (!r.body.contains("timings")) return;
    CHECK(r.body["timings"].contains("cache_n"));
    CHECK(r.body["timings"].contains("prompt_n"));
    CHECK(r.body["timings"].contains("predicted_n"));
}

TEST(completions_reject_an_unknown_arcint_field_by_name) {
    const auto r = run(completion(json{{"timings", true}, {"warp", 1}}));
    CHECK_EQ(r.status, 400);
    const std::string msg = r.body.value("error", json::object()).value("message", "");
    CHECK(msg.find("arcint.warp") != std::string::npos);
}

TEST(completions_reject_a_non_boolean_timings) {
    const auto r = run(completion(json{{"timings", "yes"}}));
    CHECK_EQ(r.status, 400);
}

#include "api/handlers.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "harness.h"

// Named lanes (DESIGN.md §4.2, amended 2026-10-07;
// docs/campaigns/lanes-agent-subagent.md): `--served-model-name A,B
// --lane-ctx CA,CB` serves two lanes of different context from one process,
// and the request's `model` field picks the lane. What is gated here is the
// API layer the libllama backend sits under, with a fake backend in its place:
// the name resolves to its lane before the request is prepared (so the §3.8
// 400 quotes that lane's cap), an unknown name is a 404, each lane has its own
// pool (a busy subagent lane never borrows the agent lane), and /v1/models,
// /props and /health say all of that. The single-name behaviour is pinned
// alongside, because it must not move.
namespace {

using namespace lgc;
using lgc::api::Context;
using lgc::api::Lane;
using lgc::api::SlotPool;
using json = nlohmann::json;

// One token per byte: a prompt's token count is its length, so a cap can be
// crossed by a known margin.
class ByteTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(std::string_view text) override {
        return std::vector<int>(text.begin(), text.end());
    }
    std::string decode(const std::vector<int>& ids) override {
        return std::string(ids.begin(), ids.end());
    }
    std::string decode_one(int id) override { return std::string(1, static_cast<char>(id)); }
    int         eos_id() const override { return 0; }
};

class FakeBackend final : public Backend {
public:
    ModelStatus      st;
    std::atomic<int> last_slot{-100};

    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest& req) const override {
        std::string out;
        for (const ChatMessage& m : req.messages) out += m.content;
        return out;
    }
    FinishReason generate(const GenerationInput&, int slot, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        last_slot = slot;
        stats.completion_tokens = 1;
        on_piece("k", 'k');
        return FinishReason::Stop;
    }

private:
    ByteTokenizer tok_;
};

// The fixture every named-lane case uses: the agent lane at 512, the
// subagent lane at 256, a 503 on the spot unless a case sets a timeout.
struct Lanes {
    Config                                 cfg;
    FakeBackend                            backend;
    SlotPool                               slots{2};
    std::vector<std::unique_ptr<SlotPool>> pools;
    Context                                ctx;

    explicit Lanes(double queue_timeout_s = 0.0) {
        cfg.stub              = true;
        cfg.served_model_name = "agent";
        cfg.lane_names        = {"agent", "sub"};
        cfg.lane_ctx          = {512, 256};
        cfg.parallel          = 2;
        cfg.queue_timeout_s   = queue_timeout_s;
        backend.st.id          = "canonical-gguf";
        backend.st.served_id   = "agent";
        backend.st.loaded      = true;
        backend.st.n_ctx       = 512;
        backend.st.n_ctx_train = 262144;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
        ctx.lanes   = lgc::api::make_lanes(cfg, pools);
    }
};

// One name, no --lane-ctx: the server as it was.
struct OneName {
    Config      cfg;
    FakeBackend backend;
    SlotPool    slots{1};
    Context     ctx;

    OneName() {
        cfg.stub              = true;
        cfg.served_model_name = "qwen3.6-coder";
        backend.st.id          = "qwen3.6-27b-a3b-coder";
        backend.st.served_id   = "qwen3.6-coder";
        backend.st.loaded      = true;
        backend.st.n_ctx       = 512;
        backend.st.n_ctx_train = 262144;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
};

json completion(const std::string& model, size_t prompt_bytes) {
    json body{{"prompt", std::string(prompt_bytes, 'x')}, {"max_tokens", 1}};
    if (!model.empty()) body["model"] = model;
    return body;
}

json chat(const std::string& model, size_t prompt_bytes) {
    json body{{"messages", json::array({{{"role", "user"},
                                         {"content", std::string(prompt_bytes, 'x')}}})},
              {"max_tokens", 1}};
    if (!model.empty()) body["model"] = model;
    return body;
}

}  // namespace

TEST(lanes_are_built_one_pool_each_in_name_order) {
    Lanes l;
    CHECK_EQ(l.ctx.lanes.size(), size_t{2});
    if (l.ctx.lanes.size() != 2) return;
    CHECK_EQ(l.ctx.lanes[0].name, std::string("agent"));
    CHECK_EQ(l.ctx.lanes[0].n_ctx, 512);
    CHECK_EQ(l.ctx.lanes[0].seq, 0);
    CHECK_EQ(l.ctx.lanes[1].name, std::string("sub"));
    CHECK_EQ(l.ctx.lanes[1].n_ctx, 256);
    CHECK_EQ(l.ctx.lanes[1].seq, 1);
    CHECK(l.ctx.lanes[0].pool != nullptr && l.ctx.lanes[1].pool != nullptr);
    CHECK(l.ctx.lanes[0].pool != l.ctx.lanes[1].pool);
    CHECK_EQ(l.ctx.lanes[0].pool->total(), 1);
    CHECK_EQ(l.ctx.lanes[1].pool->total(), 1);

    // without named lanes there are none
    OneName o;
    std::vector<std::unique_ptr<SlotPool>> pools;
    CHECK(lgc::api::make_lanes(o.cfg, pools).empty());
}

TEST(lanes_resolve_by_name_empty_to_the_first_unknown_is_404) {
    Lanes l;
    int   lane = -7;
    CHECK(!lgc::api::resolve_lane(l.ctx, "agent", lane));
    CHECK_EQ(lane, 0);
    CHECK(!lgc::api::resolve_lane(l.ctx, "sub", lane));
    CHECK_EQ(lane, 1);
    CHECK(!lgc::api::resolve_lane(l.ctx, "", lane));
    CHECK_EQ(lane, 0);
    // the artifact every lane serves: the first lane, as an empty name
    CHECK(!lgc::api::resolve_lane(l.ctx, "canonical-gguf", lane));
    CHECK_EQ(lane, 0);

    const auto err = lgc::api::resolve_lane(l.ctx, "qwen3.8-other", lane);
    CHECK(err.has_value());
    if (!err) return;
    CHECK_EQ(err->status, 404);
    const json& e = err->body.at("error");
    CHECK_EQ(e.at("type").get<std::string>(), std::string("invalid_request_error"));
    CHECK_EQ(e.at("code").get<std::string>(), std::string("model_not_found"));
    CHECK_EQ(e.at("param").get<std::string>(), std::string("model"));
    // the message names what was asked for and what is served
    const std::string msg = e.at("message").get<std::string>();
    CHECK(msg.find("qwen3.8-other") != std::string::npos);
    CHECK(msg.find("agent") != std::string::npos && msg.find("sub") != std::string::npos);
}

TEST(lanes_one_name_serves_any_model_field_as_before) {
    OneName o;
    int     lane = -7;
    CHECK(!lgc::api::resolve_lane(o.ctx, "something-else", lane));
    CHECK_EQ(lane, -1);
    lgc::api::PreparedChat prep;
    CHECK(!lgc::api::prepare_chat(o.ctx, chat("something-else", 10), prep));
    CHECK_EQ(prep.lane, -1);
    CHECK_EQ(prep.model, std::string("qwen3.6-coder"));
    // the 400 is against the process's context
    const auto err = lgc::api::prepare_chat(o.ctx, chat("", 600), prep);
    CHECK(err.has_value() && err->status == 400);
    if (err) CHECK_EQ(err->body.at("error").at("n_ctx").get<int>(), 512);
}

TEST(lanes_overflow_400_quotes_the_lanes_cap) {
    Lanes l;
    // 300 tokens: inside the agent lane (512), over the subagent's (256)
    lgc::api::PreparedCompletion prep;
    CHECK(!lgc::api::prepare_completion(l.ctx, completion("agent", 300), prep));
    CHECK_EQ(prep.lane, 0);
    CHECK_EQ(prep.model, std::string("agent"));

    const auto err = lgc::api::prepare_completion(l.ctx, completion("sub", 300), prep);
    CHECK(err.has_value());
    if (!err) return;
    CHECK_EQ(err->status, 400);
    CHECK_EQ(err->body.at("error").at("n_ctx").get<int>(), 256);
    CHECK_EQ(err->body.at("error").at("prompt_tokens").get<int>(), 300);
    CHECK_EQ(err->body.at("error").at("overflow").get<int>(), 44);

    // chat takes the same path, and an empty name is the agent lane
    lgc::api::PreparedChat c;
    CHECK(!lgc::api::prepare_chat(l.ctx, chat("", 300), c));
    CHECK_EQ(c.lane, 0);
    CHECK_EQ(c.model, std::string("agent"));
    const auto cerr = lgc::api::prepare_chat(l.ctx, chat("sub", 300), c);
    CHECK(cerr.has_value() && cerr->status == 400);
    if (cerr) CHECK_EQ(cerr->body.at("error").at("n_ctx").get<int>(), 256);
    CHECK(!lgc::api::prepare_chat(l.ctx, chat("sub", 200), c));
    CHECK_EQ(c.lane, 1);
    CHECK_EQ(c.model, std::string("sub"));
}

TEST(lanes_unknown_name_is_404_before_anything_else) {
    Lanes l;
    lgc::api::PreparedChat c;
    // an over-long prompt for an unknown name is the 404, not a 400: the name
    // is resolved first
    auto err = lgc::api::prepare_chat(l.ctx, chat("nope", 5000), c);
    CHECK(err.has_value() && err->status == 404);
    lgc::api::PreparedCompletion p;
    err = lgc::api::prepare_completion(l.ctx, completion("nope", 10), p);
    CHECK(err.has_value() && err->status == 404);
}

TEST(lanes_a_busy_lane_is_refused_on_its_own_and_never_borrows_the_other) {
    Lanes l;
    if (l.ctx.lanes.size() != 2) {
        CHECK_EQ(l.ctx.lanes.size(), size_t{2});
        return;
    }
    SlotPool::Lease sub;
    int             slot = -7;
    CHECK(!lgc::api::acquire_slot(l.ctx, 1, sub, slot));
    CHECK_EQ(slot, 1);   // the subagent lane's sequence

    // a second subagent request: the agent lane is free, and is not taken
    SlotPool::Lease second;
    int             slot2 = -7;
    const auto      err   = lgc::api::acquire_slot(l.ctx, 1, second, slot2);
    CHECK(err.has_value());
    CHECK(second.index() < 0);
    if (err) {
        CHECK_EQ(err->status, 503);
        const json& b = err->body;
        CHECK_EQ(b.at("error").at("code").get<std::string>(), std::string("no_slot_available"));
        CHECK(b.at("error").at("message").get<std::string>().find("sub") != std::string::npos);
        // the refused lane, and every lane's free count
        CHECK(b.at("slots").contains("lane") && b.contains("lanes"));
        if (!b.at("slots").contains("lane") || !b.contains("lanes")) return;
        CHECK_EQ(b.at("slots").at("lane").get<std::string>(), std::string("sub"));
        CHECK_EQ(b.at("slots").at("total").get<int>(), 1);
        CHECK_EQ(b.at("slots").at("free").get<int>(), 0);
        const json& lanes = b.at("lanes");
        CHECK_EQ(lanes.size(), size_t{2});
        if (lanes.size() == 2) {
            CHECK_EQ(lanes[0].at("name").get<std::string>(), std::string("agent"));
            CHECK_EQ(lanes[0].at("slots_free").get<int>(), 1);
            CHECK_EQ(lanes[1].at("name").get<std::string>(), std::string("sub"));
            CHECK_EQ(lanes[1].at("slots_free").get<int>(), 0);
            CHECK_EQ(lanes[1].at("n_ctx").get<int>(), 256);
        }
    }

    // the agent lane is still there for the agent
    SlotPool::Lease agent;
    int             slot3 = -7;
    CHECK(!lgc::api::acquire_slot(l.ctx, 0, agent, slot3));
    CHECK_EQ(slot3, 0);

    // and its lease frees exactly its lane
    sub = SlotPool::Lease();
    CHECK_EQ(l.ctx.lanes[1].pool->free(), 1);
    CHECK_EQ(l.ctx.lanes[0].pool->free(), 0);
}

TEST(lanes_queue_timeout_waits_for_the_same_lane) {
    Lanes l(/*queue_timeout_s=*/5.0);
    if (l.ctx.lanes.size() != 2) {
        CHECK_EQ(l.ctx.lanes.size(), size_t{2});
        return;
    }
    auto  held = std::make_unique<SlotPool::Lease>();
    int   slot = -7;
    CHECK(!lgc::api::acquire_slot(l.ctx, 1, *held, slot));

    // the subagent lane comes free after 100 ms; the agent lane was free all
    // along and must not be what the waiter gets
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        *held = SlotPool::Lease();
    });
    SlotPool::Lease waited;
    int             slot2 = -7;
    const auto      t0    = std::chrono::steady_clock::now();
    CHECK(!lgc::api::acquire_slot(l.ctx, 1, waited, slot2));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    release.join();
    CHECK_EQ(slot2, 1);
    CHECK(s >= 0.05);
    CHECK_EQ(l.ctx.lanes[0].pool->free(), 1);
}

TEST(lanes_health_reports_each_lane) {
    Lanes           l;
    SlotPool::Lease sub;
    int             slot = -7;
    CHECK(!lgc::api::acquire_slot(l.ctx, 1, sub, slot));
    const json h = lgc::api::health(l.ctx);
    // the totals a monitor reads today, over all lanes
    CHECK_EQ(h.at("slots_total").get<int>(), 2);
    CHECK_EQ(h.at("slots_free").get<int>(), 1);
    CHECK_EQ(h.at("queue_depth").get<int>(), 0);
    CHECK_EQ(h.at("model").get<std::string>(), std::string("agent"));
    CHECK(h.contains("lanes"));
    if (!h.contains("lanes")) return;
    const json& lanes = h.at("lanes");
    CHECK_EQ(lanes.size(), size_t{2});
    if (lanes.size() != 2) return;
    CHECK_EQ(lanes[0].at("name").get<std::string>(), std::string("agent"));
    CHECK_EQ(lanes[0].at("n_ctx").get<int>(), 512);
    CHECK_EQ(lanes[0].at("slots_free").get<int>(), 1);
    CHECK_EQ(lanes[0].at("slots_total").get<int>(), 1);
    CHECK_EQ(lanes[1].at("name").get<std::string>(), std::string("sub"));
    CHECK_EQ(lanes[1].at("slots_free").get<int>(), 0);
    CHECK_EQ(lanes[1].at("queue_depth").get<int>(), 0);

    // one name: no lanes array, the fields as before
    OneName    o;
    const json h1 = lgc::api::health(o.ctx);
    CHECK(!h1.contains("lanes"));
    CHECK_EQ(h1.at("slots_total").get<int>(), 1);
}

TEST(lanes_v1_models_lists_each_name_at_its_context) {
    Lanes      l;
    const json m    = lgc::api::models(l.ctx);
    const json& data = m.at("data");
    CHECK_EQ(data.size(), size_t{2});
    if (data.size() != 2) return;
    CHECK_EQ(data[0].at("id").get<std::string>(), std::string("agent"));
    CHECK_EQ(data[0].at("n_ctx").get<int>(), 512);
    CHECK_EQ(data[1].at("id").get<std::string>(), std::string("sub"));
    CHECK_EQ(data[1].at("n_ctx").get<int>(), 256);
    for (const json& e : data) {
        // every field a proxy reads today, on every entry
        CHECK_EQ(e.at("object").get<std::string>(), std::string("model"));
        CHECK_EQ(e.at("n_ctx_train").get<int>(), 262144);
        CHECK_EQ(e.at("quant").get<std::string>(), std::string("q4"));
        CHECK_EQ(e.at("lanes").get<int>(), 1);
        CHECK_EQ(e.at("canonical_id").get<std::string>(), std::string("canonical-gguf"));
    }

    OneName    o;
    const json m1 = lgc::api::models(o.ctx);
    CHECK_EQ(m1.at("data").size(), size_t{1});
    CHECK_EQ(m1.at("data")[0].at("id").get<std::string>(), std::string("qwen3.6-coder"));
    CHECK_EQ(m1.at("data")[0].at("n_ctx").get<int>(), 512);
    CHECK_EQ(m1.at("data")[0].at("lanes").get<int>(), 1);
    CHECK_EQ(m1.at("data")[0].at("canonical_id").get<std::string>(),
             std::string("qwen3.6-27b-a3b-coder"));
}

TEST(lanes_props_answers_to_every_name_and_enforces_the_field) {
    Lanes       l;
    const json  p = lgc::api::props(l.ctx);
    const json& m = p.at("model");
    CHECK(m.at("enforces_model_field").get<bool>());
    std::vector<std::string> names = m.at("answers_to").get<std::vector<std::string>>();
    CHECK_EQ(names, (std::vector<std::string>{"agent", "sub", "canonical-gguf"}));
    CHECK_EQ(m.at("n_ctx").get<int>(), 512);
    CHECK_EQ(p.at("slots").at("total").get<int>(), 2);
    CHECK(p.contains("lanes"));
    if (!p.contains("lanes")) return;
    const json& lanes = p.at("lanes");
    CHECK_EQ(lanes.size(), size_t{2});
    if (lanes.size() == 2) {
        CHECK_EQ(lanes[1].at("name").get<std::string>(), std::string("sub"));
        CHECK_EQ(lanes[1].at("n_ctx").get<int>(), 256);
    }

    OneName    o;
    const json p1 = lgc::api::props(o.ctx);
    CHECK(!p1.at("model").at("enforces_model_field").get<bool>());
    CHECK_EQ(p1.at("model").at("answers_to").get<std::vector<std::string>>(),
             (std::vector<std::string>{"qwen3.6-coder", "qwen3.6-27b-a3b-coder"}));
    CHECK(!p1.contains("lanes"));
}

TEST(lanes_the_response_names_the_lane_and_runs_on_its_sequence) {
    Lanes                  l;
    lgc::api::PreparedChat c;
    CHECK(!lgc::api::prepare_chat(l.ctx, chat("sub", 10), c));
    SlotPool::Lease lease;
    int             slot = -7;
    CHECK(!lgc::api::acquire_slot(l.ctx, c.lane, lease, slot));
    const lgc::api::HttpResult r = lgc::api::run_chat(l.ctx, c, slot);
    CHECK_EQ(r.status, 200);
    CHECK_EQ(r.body.at("model").get<std::string>(), std::string("sub"));
    CHECK_EQ(l.backend.last_slot.load(), 1);

    lgc::api::PreparedCompletion p;
    CHECK(!lgc::api::prepare_completion(l.ctx, completion("", 10), p));
    SlotPool::Lease lease2;
    int             slot2 = -7;
    CHECK(!lgc::api::acquire_slot(l.ctx, p.lane, lease2, slot2));
    const lgc::api::HttpResult r2 = lgc::api::run_completion(l.ctx, p, slot2);
    CHECK_EQ(r2.body.at("model").get<std::string>(), std::string("agent"));
    CHECK_EQ(l.backend.last_slot.load(), 0);
}

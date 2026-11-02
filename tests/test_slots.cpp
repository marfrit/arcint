#include "api/handlers.h"
#include "api/request_id.h"
#include "api/slots.h"

#include "harness.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// GET /slots (--slots): what the slots are doing, never what they are saying.
// These cases pin the shape of the body (one entry per slot, exactly the state
// fields), the binding of a busy slot to the request id the route answered with
// (1.4.2's X-Request-Id), that the binding is cleared on release, and that the
// text a request carried reaches nowhere in the body. The HTTP half - route
// presence, the 401, the flag - is tests/slots.sh.
namespace {

using namespace lgc;
using lgc::api::Context;
using lgc::api::Lane;
using lgc::api::SlotPool;
using json = nlohmann::json;

// A string no engine text would contain on its own: if it turns up in the
// body, the snapshot leaked the request.
constexpr const char* kMarker = "zqx-needle-9c4f7a2e";

Config make_cfg(bool slots = true) {
    Config cfg;
    cfg.stub     = true;
    cfg.slots    = slots;
    cfg.parallel = 2;
    return cfg;
}

struct Fixture {
    Config                   cfg;
    std::unique_ptr<Backend> backend;
    SlotPool                 slots{2};
    Context                  ctx;

    explicit Fixture(bool slots = true) : cfg(make_cfg(slots)), ctx{} {
        backend     = make_stub_backend(4096);
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
    }
};

// The idle entries, in the order the snapshot returns them.
json snapshot(const Context& ctx) { return api::slots_json(ctx); }

// The entry of one engine slot id, or null.
const json& entry(const json& body, int id) {
    static const json kNull;
    for (const json& e : body)
        if (e.value("id", -1) == id) return e;
    return kNull;
}

// Run one generation on `slot` so the slot's sequence holds cells.
GenerationStats run_on(Backend& backend, int slot, int max_tokens) {
    GenerationInput in;
    in.prompt             = std::string("recite this marker slowly: ") + kMarker;
    in.sampler.max_tokens = max_tokens;
    GenerationStats stats;
    backend.generate(in, slot, [](std::string_view, int) { return Control::Continue; }, stats);
    return stats;
}

std::vector<std::string> keys(const json& e) {
    std::vector<std::string> k;
    for (auto it = e.begin(); it != e.end(); ++it) k.push_back(it.key());
    return k;
}

}  // namespace

// The flag is off unless the operator asks for it: the default HTTP surface is
// the one every capture was taken against.
TEST(slots_flag_is_off_by_default) {
    Config cfg;
    CHECK(cfg.slots == false);
}

// Idle: one entry per slot, and per entry exactly the state fields.
TEST(slots_body_is_one_entry_per_slot_with_only_state) {
    Fixture  f;
    const json body = snapshot(f.ctx);

    CHECK(body.is_array());
    CHECK_EQ(static_cast<int>(body.size()), 2);
    const std::vector<std::string> want = {"id", "lane", "n_ctx", "is_processing", "n_cache"};
    for (const json& e : body) {
        std::vector<std::string> k = keys(e);
        std::sort(k.begin(), k.end());
        CHECK_EQ(t::show(k), t::show(want));
        CHECK_EQ(e.value("lane", ""), std::string("stub"));
        CHECK_EQ(e.value("n_ctx", 0), 4096);
        CHECK(e.value("is_processing", true) == false);
        CHECK_EQ(e.value("n_cache", -1), 0);
        CHECK(!e.contains("id_task"));
    }
    CHECK_EQ(entry(body, 0).value("id", -1), 0);
    CHECK_EQ(entry(body, 1).value("id", -1), 1);
}

// A held lease is a busy slot, and it carries the request id the route answered
// with. The id is bound where the slot is taken, so it cannot be confused with
// another request's, and it is a string: arcint's request id, not llama-server's
// integer task id.
TEST(a_busy_slot_reports_processing_and_its_request_id) {
    Fixture f;
    SlotPool::Lease lease;
    CHECK(!api::acquire_slot(f.ctx, lease, "req-slotprobe-7").has_value());
    CHECK(lease.index() >= 0);

    const json body = snapshot(f.ctx);
    CHECK_EQ(static_cast<int>(body.size()), 2);
    int busy = 0;
    for (const json& e : body) {
        if (!e.value("is_processing", false)) {
            CHECK(!e.contains("id_task"));
            continue;
        }
        ++busy;
        CHECK_EQ(e.value("id_task", ""), std::string("req-slotprobe-7"));
        CHECK_EQ(e.value("id", -1), lease.index());
    }
    CHECK_EQ(busy, 1);
}

// The id belongs to the request, not to the slot: releasing clears it, so the
// next request on the same slot cannot be reported under the old one.
TEST(releasing_clears_the_request_id) {
    Fixture f;
    {
        SlotPool::Lease lease;
        CHECK(!api::acquire_slot(f.ctx, lease, "req-before").has_value());
    }
    const json body = snapshot(f.ctx);
    for (const json& e : body) {
        CHECK(e.value("is_processing", true) == false);
        CHECK(!e.contains("id_task"));
    }
}

// The number of cells the slot's sequence holds, read from the backend: the
// prompt's cells plus what it decoded.
TEST(n_cache_counts_the_cells_the_slot_holds) {
    Fixture f;
    SlotPool::Lease lease;
    CHECK(!api::acquire_slot(f.ctx, lease, "req-cache").has_value());

    const GenerationStats stats = run_on(*f.backend, lease.index(), 4);
    CHECK(stats.prompt_tokens > 0);
    CHECK_EQ(stats.completion_tokens, 4);

    const json e = entry(snapshot(f.ctx), lease.index());
    CHECK_EQ(e.value("n_cache", -1), stats.prompt_tokens + stats.completion_tokens);
    CHECK_EQ(e.value("is_processing", false), true);
}

// The whole point of the constraint: a request's text is nowhere in the body.
TEST(no_request_text_reaches_the_body) {
    Fixture f;
    SlotPool::Lease lease;
    CHECK(!api::acquire_slot(f.ctx, lease, "req-marker").has_value());
    run_on(*f.backend, lease.index(), 3);

    const std::string dump = snapshot(f.ctx).dump();
    CHECK(dump.find(kMarker) == std::string::npos);
    // The words llama-server's /slots uses for content must not appear as
    // fields here at all.
    for (const char* forbidden : {"prompt", "generated", "next_token", "params", "content"})
        CHECK(dump.find(forbidden) == std::string::npos);
}

// Named lanes: each lane's slot says which lane it is and stops at the lane's
// own context; the engine slot id is the lane's sequence, which is what the
// backend's KV is indexed by.
TEST(named_lanes_report_their_own_names) {
    Fixture                    f;
    std::vector<std::unique_ptr<SlotPool>> pools;
    pools.push_back(std::make_unique<SlotPool>(1));
    pools.push_back(std::make_unique<SlotPool>(1));
    std::vector<Lane> lanes;
    lanes.push_back(Lane{"agent",   512, 0, pools[0].get()});
    lanes.push_back(Lane{"subagent", 256, 1, pools[1].get()});
    f.ctx.lanes = lanes;

    SlotPool::Lease lease;
    int             slot = -1;
    CHECK(!api::acquire_slot(f.ctx, 1, lease, slot, -1, "req-lane").has_value());
    CHECK_EQ(slot, 1);

    const json body = snapshot(f.ctx);
    CHECK_EQ(static_cast<int>(body.size()), 2);
    const json a = entry(body, 0);
    const json b = entry(body, 1);
    CHECK_EQ(a.value("lane", ""), std::string("agent"));
    CHECK_EQ(a.value("n_ctx", 0), 512);
    CHECK_EQ(b.value("lane", ""), std::string("subagent"));
    CHECK_EQ(b.value("n_ctx", 0), 256);
    CHECK_EQ(b.value("id_task", ""), std::string("req-lane"));
    CHECK(a.value("is_processing", true) == false);
    CHECK(b.value("is_processing", false) == true);
}

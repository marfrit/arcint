// GET /slots — see api/slots.h for what the route may and may not say.
//
// The route exists only with --slots; this function is the document itself.

#include "api/slots.h"

#include "api/handlers.h"

#include <string>
#include <vector>

namespace lgc::api {

namespace {

// Appends one pool's slots. The lane name and its context come from the lane
// table, not from the pool: the pool knows only that a slot is busy. `base_id`
// is the slot id the rest of the engine uses for this pool — the pool index
// without named lanes, the lane's own sequence with them (acquire_slot hands
// back l.seq as the slot, and the llama backend maps a slot to llama's sequence
// id 1:1, std::min(std::max(slot, 0), lanes_ - 1)).
void append_pool(nlohmann::json& out, const std::vector<SlotPool::State>& states,
                 const std::string& lane_name, int lane_n_ctx, const std::vector<int>& cells, int base_id) {
    for (const SlotPool::State& s : states) {
        const int id = base_id + s.index;
        // The KV cells this slot's sequence holds, or 0 if the backend does not
        // report positions (the stub seam, a backend without a KV memory).
        int n_cache = 0;
        if (id >= 0 && static_cast<size_t>(id) < cells.size()) n_cache = cells[static_cast<size_t>(id)];

        nlohmann::json e{{"id", id},
                         {"lane", lane_name},
                         {"n_ctx", lane_n_ctx},
                         {"is_processing", s.processing},
                         {"n_cache", n_cache}};
        // Only a busy slot names a request, and only when a route bound one (a
        // test seam takes leases without an id). The value is 1.4.2's
        // X-Request-Id — an id the caller already holds — never anything the
        // request carried.
        if (s.processing && !s.request_id.empty()) e["id_task"] = s.request_id;
        out.push_back(std::move(e));
    }
}

}  // namespace

nlohmann::json slots_json(const Context& ctx) {
    // ---- what is read, and under which lock (spec 4: read-only, lock-cheap) ----
    //   SlotPool::snapshot()      the busy flags and request ids, under the
    //                             pool's own mutex: a copy of two small
    //                             vectors, nothing else, and no nested lock.
    //   Backend::kv_positions()   the KV cell high-water per sequence, under the
    //                             backend's loan lock (mu_) — the same lock
    //                             /props already takes to read the loan state,
    //                             and the one every decode pays.
    //   the lane table            Context's own fields: fixed at startup, no lock.
    // The document is built outside all three. A GET /slots therefore never
    // holds a lock a request waits for longer than the copy of that slot's two
    // fields.
    nlohmann::json out = nlohmann::json::array();
    if (ctx.slots == nullptr) return out;

    const std::vector<int> cells =
        ctx.backend != nullptr ? ctx.backend->kv_positions() : std::vector<int>{};

    if (ctx.lanes.empty()) {
        // One name, one pool: the flat layout every single-model process runs.
        const std::string lane_name = ctx.backend != nullptr ? ctx.backend->status().served_id : std::string{};
        const int         n_ctx     = ctx.backend != nullptr ? ctx.backend->status().n_ctx : 0;
        append_pool(out, ctx.slots->snapshot(), lane_name, n_ctx, cells, 0);
        return out;
    }

    // Named lanes: each lane admits from a pool of its own (a lane is a memory
    // reservation of its own size, DESIGN §4.3), so the table is one entry per
    // lane, keyed by the lane's sequence.
    for (const Lane& l : ctx.lanes) {
        if (l.pool == nullptr) continue;
        append_pool(out, l.pool->snapshot(), l.name, l.n_ctx, cells, l.seq);
    }
    return out;
}

}  // namespace lgc::api

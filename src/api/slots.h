// GET /slots: what the engine's slots are doing, and nothing else.
//
// llama-server's /slots is the model for this route *and* the warning against
// it: its per-slot dump carries the detokenized prompt, the generated text and
// the sampling parameters of whatever is running (code, the llama.cpp tree
// arcint builds against, tools/server/server-context.cpp:733-763 - `res["prompt"]`
// at 761, `res["generated"]` at 762, `res["params"]` at 750; its README, "GET
// /slots", says the same: "per-slot metrics, such as speed, processed tokens,
// sampling parameters"). arcint serves other people's prompts, so the contract
// here is state only: id, lane, n_ctx, is_processing, n_cache, and for a busy
// slot the request id it answers (1.4.2's X-Request-Id). No prompt, no
// completion, no sampling parameters, no per-slot timings.
//
// The flag name is llama-server's - `--slots` / `--no-slots`, "expose slots
// monitoring endpoint" (code, common/arg.cpp:3626-3633 in that tree) - but two
// things are deliberately NOT copied from it. llama-server's default is ON and
// it keeps the route registered when off, answering an error (code, the same
// tree, server-context.cpp:5262-5266); arcint's default is OFF and the route
// does not exist without the flag, so it 404s like any unknown path.
#pragma once

#include <nlohmann/json.hpp>

namespace lgc::api {

struct Context;

// One entry per slot, in slot-id order. What is read under which lock is
// written out at the definition (api/slots.cpp): the slot pool's mutex is held
// only for a copy of its two small vectors, so a GET /slots cannot stall a
// request waiting for a slot.
nlohmann::json slots_json(const Context& ctx);

}  // namespace lgc::api

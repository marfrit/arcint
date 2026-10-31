#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "api/generate.h"
#include "api/handlers.h"
#include "core/chat.h"

// The Open Responses request surface (docs/campaigns/feyntalk.md W1): body →
// ResponseRequest (the mapped ChatRequest plus the spec fields the resource must
// echo), then ResponseRequest → PreparedChat through the one prepare_from_chat,
// then the non-streamed ResponseResource. The chat and completions routes are
// untouched by this module.
namespace lgc::api {

struct ResponseRequest {
    // The mapped chat request: messages (instructions as a leading system
    // message), flat tools, tool_choice, parallel_tool_calls, the sampler and
    // the thinking switch.
    ChatRequest chat;

    std::string instructions;               // echoed; "" when absent
    bool        has_instructions = false;

    nlohmann::json tools       = nlohmann::json::array();  // flat FunctionTool shape, echoed
    nlohmann::json tool_choice = "auto";                    // spec shape, echoed
    std::string    truncation  = "disabled";
    nlohmann::json text        = {{"format", {{"type", "text"}}}};
    // {effort, summary:null} when the request sent reasoning, otherwise null.
    nlohmann::json reasoning   = nullptr;
    nlohmann::json metadata    = nullptr;
    bool           store       = true;
    bool           background  = false;
    std::string    service_tier = "default";
    std::string    safety_identifier;
    bool           has_safety_identifier = false;
    std::string    prompt_cache_key;
    bool           has_prompt_cache_key = false;
    int            max_output_tokens = 0;  // 0: not sent (the resource carries null)
    int            max_tool_calls    = 0;  // 0: not sent (always null in W1)
    std::string    previous_response_id;
    bool           has_previous_response_id = false;
    int            arcint_v = 1;
    // Reasoning items of the input: accepted, kept for the store (W3), never
    // rendered into the prompt — parity with /v1/chat/completions, which has no
    // reasoning replay.
    std::vector<nlohmann::json> reasoning_items;
};

// Everything a response request can get wrong becomes here, before a byte is
// committed, in the spec's error envelope: type `invalid_request`, code
// `unsupported_parameter` with the field path for a refused field, code
// `invalid_request` for a malformed body.
std::optional<HttpResult> parse_response_request(const nlohmann::json& body, ResponseRequest& out);

struct PreparedResponse {
    ResponseRequest req;
    PreparedChat    chat;   // the one prepare_from_chat result; id = resp-N
    // A continuation's slot affinity (docs/campaigns/feyntalk.md State): the
    // slot the referenced response ran on, tried non-blockingly before the
    // lane's ordinary acquire. -1: no affinity.
    int affinity_slot = -1;
};

// Maps a parsed request onto prepare_from_chat, then numbers the response.
// previous_response_id is refused here in W1: no store exists yet, and the
// spec's answer for an unknown id is 400 previous_response_not_found (W3
// replaces the refusal with the lookup).
std::optional<HttpResult> prepare_response(const Context& ctx, ResponseRequest req,
                                           PreparedResponse& out);

// The complete ResponseResource for a finished turn: every required key of the
// 2026-04-24 OpenAPI document, output items (message, reasoning,
// function_call) and usage. `gen` carries the shared reasoning/tool-call split.
nlohmann::json response_resource(const ResponseRequest& req, const PreparedChat& prep,
                                 const ChatGeneration& gen);

// The resource's variable half, shared by the non-streamed builder above and the
// SSE writer's snapshots: every other key is the request/preparation echo and is
// filled once, here.
struct ResponseParts {
    std::string    status            = "completed";
    nlohmann::json incomplete_details = nullptr;
    nlohmann::json output            = nlohmann::json::array();
    nlohmann::json usage             = nullptr;   // null while in_progress
    nlohmann::json error             = nullptr;
    bool           completed         = true;      // completed_at: set for completed only
};
nlohmann::json response_resource(const ResponseRequest& req, const PreparedChat& prep,
                                 const ResponseParts& parts);

// One non-streamed turn: drive the prepared input exactly as run_chat does,
// then build the resource. A stop signal is a 503 envelope, a gone client no
// body (HttpResult::gone). Generation exceptions are the spec's 500/400.
HttpResult run_response(const Context& ctx, const PreparedResponse& prep, int slot);

// The admission refusal (review F1): the spec maps a busy service to 429
// too_many_requests/all_lanes_busy, while the chat route keeps its 503. The
// chat body's diagnostics — `slots` and, under named lanes, `lanes` — are
// copied into the 429 body, so a client can still tell why and retry.
HttpResult responses_busy(const HttpResult& chat_refusal);

}  // namespace lgc::api

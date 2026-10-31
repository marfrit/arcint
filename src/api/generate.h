#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "api/handlers.h"
#include "core/reasoning.h"
#include "core/stop.h"
#include "core/toolcall.h"
#include "exec/backend.h"

// The generation half shared by the chat, completions and Responses routes
// (docs/campaigns/feyntalk.md W1). Stop scanning, the reasoning split, tool-call
// parsing, the slot lines and the stats-to-JSON helpers live here once, so the
// Responses route cannot disagree with the chat route about any of them.
namespace lgc::api {

// Identifiers and timestamps. Deterministic rather than random: greedy output
// must be reproducible byte for byte (DESIGN §3.4), and a random id would break
// that in the response envelope even when every generated token matched.
std::string next_id(const char* prefix);
int64_t     now_seconds();

// The chat-shaped usage object (OpenAI's field names, so SDKs that model
// completion_tokens_details keep parsing).
nlohmann::json usage_json(const GenerationStats& stats);

// Whether the loop-guard ladder moved; the decode line and the response body
// carry the counters on this one condition. An eighth counter is added here,
// once.
bool loop_guard_moved(const GenerationStats& stats);

// feyntalk W3.5: the `truncation: "auto"` fit pass's numbers, once, for the
// non-streamed `arcint.truncated` object and the stream's arcint:truncated
// event: blocks dropped, tokens freed, prompt tokens before/after and the
// reserve the fit rule left for generation.
nlohmann::json truncation_json(const TruncationReport& truncation);

void log_slot_line(int slot, const char* phase, int tokens, double seconds, double rate,
                   const std::string& suffix = {});
void log_stats(int slot, const GenerationStats& stats, FinishReason reason);

// `emit` receives the new piece, its token id and the full text accumulated so
// far (the piece is already appended). Returning false means the client is gone.
// `pieces`, when given, receives the same (text, id) segments the reasoning
// split reads, so text and id-based splits cannot disagree.
using EmitFn = std::function<bool(std::string_view piece, int token_id, const std::string& accumulated)>;

FinishReason drive(Backend& backend, const GenerationInput& in, int slot, const EmitFn& emit,
                   GenerationStats& stats, std::string& raw,
                   std::vector<std::pair<std::string, int>>* pieces = nullptr);

// What one non-streamed chat turn produced, after the shared stop scan and the
// reasoning/tool-call split. run_chat and the Responses resource builder both
// read this, so their content, reasoning and calls are the same bytes.
struct ChatGeneration {
    GenerationStats             stats;
    FinishReason                reason = FinishReason::Stop;
    std::string                 reasoning;
    bool                        reasoning_closed = false;  // the think block's closer was seen
    std::string                 content;
    std::vector<ToolCall>       calls;
    // Tokens that produced reasoning text (review R7). Exact where the split is
    // by tag ids (a token id is one accepted output token); 0 where the split is
    // textual, because a piece may carry an opener, reasoning and a closer at
    // once and the piece count is then not a token count.
    int                         reasoning_tokens = 0;
};

// The split the non-streaming chat path has always used: token ids when the
// artifact spells a tag as one special token, the text split otherwise, then
// the Qwen tool-call parse when the request declared tools.
void finish_chat_generation(const PreparedChat& prep,
                            const std::vector<std::pair<std::string, int>>& pieces,
                            const std::string& raw, ChatGeneration& out);

}  // namespace lgc::api

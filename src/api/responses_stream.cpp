#include "api/responses_stream.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "api/error.h"
#include "api/generate.h"
#include "api/responses_state.h"
#include "core/reasoning.h"
#include "core/toolcall.h"
#include "util/log.h"
#include "util/text.h"
#include "util/utf8.h"

// The event mapping of docs/campaigns/feyntalk.md "Event mapping (point 3)".
// The reasoning/text split is the chat route's: the same ReasoningStreamer over
// the same (piece, id) sequence, with the same UTF-8 hold-back per stream, so a
// streaming client and a non-streaming one cannot disagree.
namespace lgc::api {
namespace {

using json = nlohmann::json;

// A response snapshot for response.created / in_progress / the terminal events.
json snapshot(const ResponseRequest& req, const PreparedChat& prep, const char* status,
              const json& output, const json& usage, const json& error, const json& incomplete,
              bool completed_at) {
    ResponseParts parts;
    parts.status            = status;
    parts.output            = output;
    parts.usage             = usage;
    parts.error             = error;
    parts.incomplete_details = incomplete;
    parts.completed         = completed_at;
    return response_resource(req, prep, parts);
}

// One streamed turn. The writer owns the sequence counter and the item state;
// the caller owns the generation.
class Stream final {
public:
    Stream(const PreparedResponse& prep, const SseWriter& write)
        : resp_(prep), req_(prep.req), prep_(prep.chat), write_(write) {}

    // true when every write succeeded; false once the client is gone.
    bool run(const Context& ctx, int slot);

private:
    bool send(json body) {
        if (gone_) return false;
        body["sequence_number"] = ++sequence_;
        const std::string type = body["type"].get<std::string>();
        std::string frame = "event: ";
        frame += type;
        frame += "\ndata: ";
        frame += dump_json(body);
        frame += "\n\n";
        if (!write_(frame)) {
            gone_ = true;
            return false;
        }
        return true;
    }
    bool done_frame() {
        if (gone_) return false;
        if (!write_("data: [DONE]\n\n")) {
            gone_ = true;
            return false;
        }
        return true;
    }

    int alloc_index() { return next_index_++; }

    // ---- reasoning item
    bool ensure_reasoning() {
        if (reasoning_open_) return true;
        json item = {{"type", "reasoning"},
                     {"id", next_id("rs")},
                     {"status", "in_progress"},
                     {"summary", json::array()},
                     {"content", json::array({{{"type", "reasoning_text"}, {"text", ""}}})}};
        reasoning_id_    = item["id"].get<std::string>();
        reasoning_index_ = alloc_index();
        output_.push_back(item);
        reasoning_open_ = true;
        if (!send(json{{"type", "response.output_item.added"},
                       {"output_index", reasoning_index_},
                       {"item", item}}))
            return false;
        return send(json{{"type", "response.content_part.added"},
                         {"item_id", reasoning_id_},
                         {"output_index", reasoning_index_},
                         {"content_index", 0},
                         {"part", json{{"type", "reasoning_text"}, {"text", ""}}}});
    }

    bool close_reasoning(const char* status) {
        if (!reasoning_open_) return true;
        json item = output_[reasoning_index_];
        item["status"]      = status;
        item["content"][0]["text"] = reasoning_text_;
        output_[reasoning_index_]  = item;
        reasoning_open_            = false;
        if (!send(json{{"type", "response.reasoning.done"},
                       {"item_id", reasoning_id_},
                       {"output_index", reasoning_index_},
                       {"content_index", 0},
                       {"text", reasoning_text_}}))
            return false;
        if (!send(json{{"type", "response.content_part.done"},
                       {"item_id", reasoning_id_},
                       {"output_index", reasoning_index_},
                       {"content_index", 0},
                       {"part", json{{"type", "reasoning_text"}, {"text", reasoning_text_}}}}))
            return false;
        return send(json{{"type", "response.output_item.done"},
                         {"output_index", reasoning_index_},
                         {"item", item}});
    }

    bool emit_reasoning(std::string_view bytes) {
        const std::string safe = reasoning_utf8_.push(bytes);
        if (safe.empty()) return true;
        if (!ensure_reasoning()) return false;
        reasoning_text_ += safe;
        return send(json{{"type", "response.reasoning.delta"},
                         {"item_id", reasoning_id_},
                         {"output_index", reasoning_index_},
                         {"content_index", 0},
                         {"delta", safe}});
    }
    bool flush_reasoning() {
        const std::string held = reasoning_utf8_.flush();
        if (held.empty()) return true;
        return emit_reasoning(held);
    }

    // ---- message item
    bool ensure_message() {
        if (message_open_) return true;
        json item = {{"type", "message"},
                     {"id", next_id("msg")},
                     {"role", "assistant"},
                     {"status", "in_progress"},
                     {"content", json::array({{{"type", "output_text"},
                                               {"annotations", json::array()},
                                               {"text", ""}}})}};
        message_id_    = item["id"].get<std::string>();
        message_index_ = alloc_index();
        output_.push_back(item);
        message_open_ = true;
        if (!send(json{{"type", "response.output_item.added"},
                       {"output_index", message_index_},
                       {"item", item}}))
            return false;
        return send(json{{"type", "response.content_part.added"},
                         {"item_id", message_id_},
                         {"output_index", message_index_},
                         {"content_index", 0},
                         {"part", json{{"type", "output_text"},
                                       {"annotations", json::array()},
                                       {"text", ""}}}});
    }

    bool close_message(const char* status) {
        if (!message_open_) return true;
        json item = output_[message_index_];
        item["status"]      = status;
        item["content"][0]["text"] = message_text_;
        output_[message_index_]    = item;
        message_open_              = false;
        if (!send(json{{"type", "response.output_text.done"},
                       {"item_id", message_id_},
                       {"output_index", message_index_},
                       {"content_index", 0},
                       {"text", message_text_}}))
            return false;
        if (!send(json{{"type", "response.content_part.done"},
                       {"item_id", message_id_},
                       {"output_index", message_index_},
                       {"content_index", 0},
                       {"part", json{{"type", "output_text"},
                                     {"annotations", json::array()},
                                     {"text", message_text_}}}}))
            return false;
        return send(json{{"type", "response.output_item.done"},
                         {"output_index", message_index_},
                         {"item", item}});
    }

    bool emit_content(std::string_view bytes) {
        const std::string safe = content_utf8_.push(bytes);
        if (safe.empty()) return true;
        if (!ensure_message()) return false;
        message_text_ += safe;
        return send(json{{"type", "response.output_text.delta"},
                         {"item_id", message_id_},
                         {"output_index", message_index_},
                         {"content_index", 0},
                         {"delta", safe}});
    }

    // ---- function_call items, emitted at the turn's end: arcint parses only
    // the finished call, so the argument stream has one delta (the whole JSON).
    bool emit_calls(const std::vector<ToolCall>& calls, const char* status) {
        for (size_t i = 0; i < calls.size(); ++i) {
            const ToolCall& c = calls[i];
            const std::string call_id = c.id.empty() ? log::format("call_%zu", i) : c.id;
            json item = {{"type", "function_call"},
                         {"id", next_id("fc")},
                         {"call_id", call_id},
                         {"name", c.name},
                         {"arguments", ""},
                         {"status", "in_progress"}};
            const int  index = alloc_index();
            const std::string item_id = item["id"].get<std::string>();
            output_.push_back(item);
            // The spec needs the name at added time, before any arguments.
            if (!send(json{{"type", "response.output_item.added"},
                           {"output_index", index},
                           {"item", item}}))
                return false;
            if (!send(json{{"type", "response.function_call_arguments.delta"},
                           {"item_id", item_id},
                           {"output_index", index},
                           {"delta", c.arguments}}))
                return false;
            if (!send(json{{"type", "response.function_call_arguments.done"},
                           {"item_id", item_id},
                           {"output_index", index},
                           {"arguments", c.arguments}}))
                return false;
            item["arguments"] = c.arguments;
            item["status"]    = status;
            output_[index]    = item;
            if (!send(json{{"type", "response.output_item.done"},
                           {"output_index", index},
                           {"item", item}}))
                return false;
        }
        return true;
    }

    json usage_of(const GenerationStats& stats) const {
        return json{{"input_tokens", stats.prompt_tokens},
                    {"output_tokens", stats.completion_tokens},
                    {"total_tokens", stats.prompt_tokens + stats.completion_tokens},
                    {"input_tokens_details", {{"cached_tokens", stats.cache_hit_tokens}}},
                    // R7: counted from the split where it is id-exact; 0 where
                    // the split is textual (a piece can carry opener, reasoning
                    // and closer at once, so its count would not be a token
                    // count). reasoning_tokens_ is filled only when the
                    // artifact's tags are ids.
                    {"output_tokens_details", {{"reasoning_tokens", reasoning_tokens_}}}};
    }

    // Mark every still-open item incomplete in a terminal snapshot (the spec's
    // "an incomplete item MUST be the last item" is served by closing them with
    // their done events, so this only catches an abort that wrote nothing more).
    void mark_open_incomplete() {
        for (json& item : output_) {
            if (item["status"].get<std::string>() == "in_progress") item["status"] = "incomplete";
        }
    }

    bool failed(const char* status, const std::string& message, const std::string& code,
                bool error_event) {
        // A continuation the server answers as failed is a broken chain: evict
        // the referenced entry (docs/campaigns/feyntalk.md State).
        if (resp_.req.has_previous_response_id)
            evict_continuation(*ctx_, resp_.req.previous_response_id);
        mark_open_incomplete();
        if (error_event && !send(json{{"type", "error"},
                                      {"error", json{{"type", "server_error"},
                                                     {"code", code},
                                                     {"message", message},
                                                     {"param", nullptr}}}}))
            return false;
        if (!send(json{{"type", "response.failed"},
                       {"response", snapshot(req_, prep_, status, output_, nullptr,
                                             json{{"code", code}, {"message", message}}, nullptr,
                                             false)}}))
            return false;
        return done_frame();
    }

    bool terminal(const GenerationStats& stats, FinishReason reason) {
        const bool completed = reason == FinishReason::Stop;
        const char* item_status_now = completed ? "completed" : "incomplete";
        const char* reasoning_status = item_status_now;
        // The open item closes and is the last item (spec §Items).
        if (!close_reasoning(reasoning_status)) return false;
        if (!close_message(item_status_now)) return false;

        // Review F2: the counters go out whenever one moved, not only on a
        // loop-guard abort — parity with the chat route's loop_guard object.
        if (loop_guard_moved(stats)) {
            json counters = loop_guard_json(stats);
            counters["type"]        = "arcint:loop_guard";
            counters["v"]           = 1;
            counters["response_id"] = prep_.id;
            if (!send(counters)) return false;
        }

        // A cancel is a named arcint event before the terminal incomplete (the
        // spec has no cancelled state; incomplete(cancelled) is the nearest).
        if (reason == FinishReason::Abort && stats.abort_cause == AbortCause::Cancelled) {
            json cancelled = {{"type", "arcint:response.cancelled"}, {"v", 1},
                              {"response_id", prep_.id}};
            if (!send(cancelled)) return false;
        }
        json incomplete = nullptr;
        if (!completed) {
            incomplete = json{{"reason", reason == FinishReason::Length ? "max_output_tokens"
                                        : reason == FinishReason::Loop ? "loop_guard_abort"
                                        : reason == FinishReason::Abort ? "cancelled"
                                                                      : "incomplete"}};
        }
        const char* type  = completed ? "response.completed" : "response.incomplete";
        const char* status = completed ? "completed" : "incomplete";
        json resource = snapshot(req_, prep_, status, output_, usage_of(stats), nullptr,
                                 incomplete, completed);
        if (!send(json{{"type", type}, {"response", resource}})) return false;
        // store=true retains the turn; store=false and an entries=0 store keep
        // nothing. A failed turn is not stored (the failed() path).
        remember(*ctx_, resp_, resource, slot_);
        return done_frame();
    }

    const PreparedResponse& resp_;
    const ResponseRequest&  req_;
    const PreparedChat&     prep_;
    const SseWriter&      write_;

    int  sequence_ = 0;
    bool gone_     = false;
    const Context* ctx_ = nullptr;
    int            slot_ = -1;

    json output_     = json::array();
    int  next_index_ = 0;

    // reasoning
    ReasoningStreamer reasoning_{false};
    utf8::Streamer    reasoning_utf8_;
    bool              reasoning_open_  = false;
    std::string       reasoning_id_;
    int               reasoning_index_ = -1;
    std::string       reasoning_text_;
    int               reasoning_tokens_ = 0;   // R7, id-split only

    // content
    utf8::Streamer content_utf8_;
    bool           message_open_  = false;
    std::string    message_id_;
    int            message_index_ = -1;
    std::string    content_;       // the answer part so far (tool-call scan)
    std::string    message_text_;  // bytes emitted as output_text
    size_t         streamed_ = 0;  // bytes of the answer already sent
};

bool Stream::run(const Context& ctx, int slot) {
    ctx_  = &ctx;
    slot_ = slot;
    // The spec's opening: the full resource in progress, then the same snapshot
    // as response.in_progress, before any item exists.
    const json created = snapshot(req_, prep_, "in_progress", json::array(), nullptr,
                                  nullptr, nullptr, false);
    if (!send(json{{"type", "response.created"}, {"response", created}})) return false;
    if (!send(json{{"type", "response.in_progress"}, {"response", created}})) return false;

    // The chat route's streamer, over the same (piece, id) sequence.
    reasoning_ = ReasoningStreamer(prep_.think_open, prep_.think_tags_extended,
                                   prep_.think_ids.open_close_ids);
    const bool by_ids = std::any_of(prep_.think_ids.open_close_ids.begin(),
                                    prep_.think_ids.open_close_ids.end(),
                                    [](const std::pair<int, int>& p) {
                                        return p.first >= 0 || p.second >= 0;
                                    });

    static const std::string kToolOpen = "<tool_call>";

    GenerationStats stats;
    std::string     raw;
    FinishReason    reason = FinishReason::Stop;
    try {
        reason = drive(*ctx.backend, prep_.input, slot,
                       [&](std::string_view piece, int token_id, const std::string&) -> bool {
                           const ReasoningStreamer::Step step = reasoning_.push(piece, token_id);
                           // R7: one reasoning token per accepted output token
                           // whose piece produced reasoning text; only the
                           // id-based split can say that exactly.
                           if (by_ids && token_id >= 0 && !step.reasoning.empty())
                               ++reasoning_tokens_;
                           if (!emit_reasoning(step.reasoning)) return false;
                           if (!step.content.empty()) {
                               // The block closed in this step: the reasoning
                               // streamer's held bytes go out before content.
                               if (!flush_reasoning()) return false;
                               if (reasoning_open_ && !close_reasoning("completed")) return false;
                           } else if (reasoning_.closed() && reasoning_open_) {
                               // A closer token alone closed the block; the
                               // answer begins in a later piece.
                               if (!flush_reasoning()) return false;
                               if (!close_reasoning("completed")) return false;
                           }
                           if (step.content.empty()) return true;
                           content_ += step.content;
                           if (!prep_.parse_tool_calls) return emit_content(step.content);

                           // Tool-call syntax must not leak into output_text.
                           // Same scan and hold-back as stream_chat, so a marker
                           // straddling two pieces never goes out.
                           const size_t scan_from =
                               streamed_ >= kToolOpen.size() ? streamed_ - kToolOpen.size() + 1 : 0;
                           const size_t marker = content_.find(kToolOpen, scan_from);
                           const size_t safe_end =
                               marker != std::string::npos
                                   ? marker
                                   : content_.size() -
                                         text::partial_stop_suffix(content_, kToolOpen);
                           if (safe_end > streamed_) {
                               const std::string_view pending =
                                   std::string_view(content_).substr(streamed_, safe_end - streamed_);
                               streamed_ = safe_end;
                               if (!emit_content(pending)) return false;
                           }
                           return true;
                       },
                       stats, raw, nullptr);
    } catch (const std::exception& e) {
        log::error("req", "responses stream failed: %s", e.what());
        // Two spec MUSTs meet here: an in-stream error is followed by failed,
        // and an open item is last and incomplete. The item is closed first so
        // the client sees both.
        if (!close_reasoning("incomplete")) return false;
        if (!close_message("incomplete")) return false;
        return failed("failed", e.what(), "internal_error", /*error_event=*/true);
    }

    log_stats(slot, stats, reason);
    if (gone_) return false;

    if (reason == FinishReason::Abort) {
        // ClientGone: nobody is there to receive a terminal event (the write
        // failed, or the probe saw the close). StopSignal: the process is
        // stopping; delivery is best-effort.
        if (stats.abort_cause == AbortCause::ClientGone) return false;
        if (stats.abort_cause == AbortCause::Cancelled) return terminal(stats, reason);
        return failed("failed", "the server is stopping; the request was cancelled at its last "
                                "token or batch boundary", "server_stopping",
                     /*error_event=*/false);
    }

    ReasoningStreamer::Step last = reasoning_.flush();
    if (!emit_reasoning(last.reasoning)) return false;
    if (!flush_reasoning()) return false;
    if (!last.content.empty()) {
        content_ += last.content;
        if (!prep_.parse_tool_calls && !emit_content(last.content)) return false;
    }

    std::vector<ToolCall> calls;
    if (prep_.parse_tool_calls) {
        ToolCallParse parsed = parse_qwen_tool_calls(content_, &prep_.schemas);
        if (parsed.truncated) {
            log::warn("tools", "%s", "unterminated <tool_call> block; returned as raw content");
        }
        calls = std::move(parsed.calls);
        if (parsed.content.size() > streamed_) {
            if (!emit_content(std::string_view(parsed.content).substr(streamed_))) return false;
        }
    }

    const std::string tail = content_utf8_.flush();
    if (!tail.empty() && !emit_content(tail)) return false;

    // The answer item closes before the call items are added (the campaign's
    // event order: content end, then the parsed call).
    const char* item_status_now =
        reason == FinishReason::Stop ? "completed" : "incomplete";
    if (!close_message(item_status_now)) return false;
    if (!emit_calls(calls, item_status_now)) return false;

    return terminal(stats, reason);
}

}  // namespace

void stream_response(const Context& ctx, const PreparedResponse& prep, int slot,
                     const SseWriter& write) {
    Stream stream(prep, write);
    stream.run(ctx, slot);
}

}  // namespace lgc::api

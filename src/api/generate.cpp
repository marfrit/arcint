#include "api/generate.h"

#include <atomic>
#include <ctime>
#include <set>

#include "api/handlers.h"
#include "core/reasoning.h"
#include "core/toolcall.h"
#include "util/log.h"
#include "util/text.h"

namespace lgc::api {
namespace {

using json = nlohmann::json;

std::atomic<uint64_t> g_request_counter{0};

}  // namespace

// Deterministic rather than random: greedy output must be reproducible byte for
// byte (§3.4), and a random id would break that in the response envelope even
// when every generated token matched. The equivalence suite compares generated
// content, not `created`.
std::string next_id(const char* prefix) {
    return log::format("%s-%llu", prefix,
                       static_cast<unsigned long long>(
                           g_request_counter.fetch_add(1, std::memory_order_relaxed) + 1));
}

int64_t now_seconds() { return static_cast<int64_t>(std::time(nullptr)); }

json usage_json(const GenerationStats& s) {
    // The draft counters say which performance regime this request actually
    // got: speculation engages only under greedy, and without these numbers
    // the only signal is a line in the server log the caller cannot see
    // (sampler-defaults order, 2026-09-01). OpenAI's own field names, so SDKs
    // that model completion_tokens_details keep parsing.
    return json{{"prompt_tokens", s.prompt_tokens},
                {"completion_tokens", s.completion_tokens},
                {"total_tokens", s.prompt_tokens + s.completion_tokens},
                {"completion_tokens_details",
                 {{"accepted_prediction_tokens", s.draft_accepted},
                  {"rejected_prediction_tokens", s.draft_proposed - s.draft_accepted}}}};
}

// One list for the two consumers: the decode line prints the counters when one
// moved, and the response carries the object on the same condition. An eighth
// counter is added here, once.
json truncation_json(const TruncationReport& t) {
    return json{{"blocks_dropped", t.blocks_dropped},
                {"tokens_freed", t.tokens_freed()},
                {"prompt_tokens_before", t.tokens_before},
                {"prompt_tokens_after", t.tokens_after},
                {"reserve", t.reserve}};
}

bool loop_guard_moved(const GenerationStats& stats) {
    return stats.loop_detections || stats.loop_nudges || stats.loop_slaps || stats.loop_aborts
           || stats.loop_self_closes || stats.loop_door_masked || stats.loop_door_eog_masked
           || stats.loop_context_closes || stats.loop_context_no_closer;
}

void log_slot_line(int slot, const char* phase, int tokens, double seconds, double rate,
                   const std::string& suffix) {
    log::info(log::format("slot %d", slot), "%-7s %5d tok in %5.2f s (%5.1f t/s)%s", phase, tokens,
              seconds, rate, suffix.c_str());
}

void log_stats(int slot, const GenerationStats& stats, FinishReason reason) {
    std::string prefill_suffix;
    if (stats.cache_hit_tokens > 0 && stats.prompt_tokens > 0) {
        prefill_suffix = log::format(" | cache hit %d tok (%.1f%%)", stats.cache_hit_tokens,
                                     100.0 * stats.cache_hit_tokens / stats.prompt_tokens);
        if (stats.cache_promote_seconds > 0.0) {
            prefill_suffix += log::format(" from host tier in %.2f s", stats.cache_promote_seconds);
        }
    }
    if (stats.snapshot_seconds > 0.0) {
        prefill_suffix += log::format(" | cache snapshot %.2f s", stats.snapshot_seconds);
    }
    // Where prefill went, on the same footing as the decode line. Printed only
    // when the backend fills it in, so the stub's line is the line it was.
    if (stats.prefill_forward_seconds > 0.0 || stats.prefill_embed_seconds > 0.0) {
        const double other = stats.prefill_seconds - stats.prefill_forward_seconds -
                             stats.prefill_embed_seconds - stats.prefill_blocks_seconds -
                             stats.prefill_restore_seconds - stats.prefill_wait_seconds -
                             stats.snapshot_seconds;
        prefill_suffix += log::format(
            " | graph %.2f s, embed %.2f s, pages %.2f s, restore %.2f s, wait %.2f s, "
            "other %.2f s",
            stats.prefill_forward_seconds, stats.prefill_embed_seconds,
            stats.prefill_blocks_seconds, stats.prefill_restore_seconds,
            stats.prefill_wait_seconds, other);
    }
    // An aborted prefill reports the tokens it actually prefilled, not the
    // whole prompt (the card showed a 37,723-token prompt as prefilled after
    // the client left 8 s in). rate() reads the same number.
    const int prefill_tokens = stats.prompt_tokens_prefilled >= 0 ? stats.prompt_tokens_prefilled
                                                                 : stats.prompt_tokens;
    log_slot_line(slot, "prefill", prefill_tokens, stats.prefill_seconds,
                  stats.prefill_rate(), prefill_suffix);

    const std::string decode_suffix = decode_suffix_for(stats);
    log_slot_line(slot, "decode", stats.completion_tokens, stats.decode_seconds,
                  stats.decode_rate(), decode_suffix);
    if (reason == FinishReason::Abort) {
        // The cause, not a guess: a SIGTERM during a prefill was logged as a
        // gone client until the backend carried which source ended it.
        const char* why = stats.abort_cause == AbortCause::StopSignal  ? "stop signal"
                          : stats.abort_cause == AbortCause::Cancelled ? "cancel requested"
                                                                       : "client gone";
        log::info(log::format("slot %d", slot), "aborted: %s", why);
    }
}

// Everything generic to both endpoints lives here: stop sequences, stop token
// ids, cancellation, and the accumulation of the raw text the caller needs for
// tool-call parsing. `emit` receives the new piece, its token id and the full
// text accumulated so far (the piece is already appended). Returning false means
// the client is gone.
FinishReason drive(Backend& backend, const GenerationInput& in, int slot, const EmitFn& emit,
                   GenerationStats& stats, std::string& raw,
                   std::vector<std::pair<std::string, int>>* pieces) {
    StopMatcher   stop(in.sampler.stop);
    std::set<int> stop_ids(in.sampler.stop_token_ids.begin(), in.sampler.stop_token_ids.end());
    bool          cancelled = false;

    // The probe rides the prepared input (the routes set in.peer_alive before
    // the generation starts), so the input is run as it is: no copy of the
    // prompt ids or the images per streamed request (design review, wave 2a).
    const FinishReason reason = backend.generate(
        in, slot,
        [&](std::string_view piece, int token_id) -> Control {
            const StopMatcher::Step step = stop.push(piece, token_id);
            // The reasoning split is decided by (piece, id), so the emitted
            // text goes out cut at the pieces it came from: a whole piece with
            // its id, a part of one with -1. A held tail released together
            // with the closer's piece no longer costs the closer its id.
            for (const StopMatcher::Segment& seg : step.segments) {
                if (pieces != nullptr) pieces->emplace_back(seg.text, seg.id);
                if (seg.text.empty()) continue;
                raw += seg.text;
                if (!emit(seg.text, seg.id, raw)) {
                    cancelled = true;
                    return Control::Cancel;
                }
            }
            if (step.hit) return Control::Stop;
            if (!stop_ids.empty() && stop_ids.count(token_id) != 0) return Control::Stop;
            return Control::Continue;
        },
        stats);

    if (!cancelled && reason != FinishReason::Abort) {
        // Text held back as a possible stop-sequence prefix that never
        // completed is ordinary output and must not be swallowed. An aborted
        // request is not an ordinary end: its peer is gone or the process is
        // stopping, so nothing more is written.
        for (const StopMatcher::Segment& seg : stop.flush_segments()) {
            if (pieces != nullptr) pieces->emplace_back(seg.text, seg.id);
            if (seg.text.empty()) continue;
            raw += seg.text;
            if (!emit(seg.text, seg.id, raw)) return FinishReason::Abort;
        }
    }
    return reason;
}

namespace {

// R7: one reasoning token per accepted output token whose piece produced
// reasoning text. Only the id-based split can say that exactly; the textual
// split reports 0 (a piece may carry opener, reasoning and closer together).
int reasoning_token_count(const PreparedChat& prep,
                          const std::vector<std::pair<std::string, int>>& pieces,
                          bool has_tag_ids) {
    if (!has_tag_ids) return 0;
    ReasoningStreamer streamer(prep.think_open, prep.think_tags_extended,
                               prep.think_ids.open_close_ids);
    int n = 0;
    for (const auto& [piece, id] : pieces) {
        const ReasoningStreamer::Step step = streamer.push(piece, id);
        if (id >= 0 && !step.reasoning.empty()) ++n;
    }
    // Text held back for a straddling closer is the tail of a token already
    // pushed; it is not counted a second time.
    (void)streamer.flush();
    return n;
}

}  // namespace

void finish_chat_generation(const PreparedChat& prep,
                            const std::vector<std::pair<std::string, int>>& pieces,
                            const std::string& raw, ChatGeneration& out) {
    // When the artifact's tags are single special tokens the split is decided
    // by those ids; the same (piece, id) sequence the streaming path feeds to
    // its streamer goes here too, so the two cannot disagree. Without any id,
    // the text split is exactly today's behaviour.
    bool has_tag_ids = false;
    for (const auto& [oid, cid] : prep.think_ids.open_close_ids)
        if (oid >= 0 || cid >= 0) has_tag_ids = true;
    const ReasoningSplit split = has_tag_ids
        ? split_reasoning_tokens(pieces, prep.think_open, prep.think_tags_extended,
                                 prep.think_ids.open_close_ids)
        : split_reasoning(raw, prep.think_open, prep.think_tags_extended);
    out.reasoning        = split.reasoning;
    out.reasoning_closed = split.closed;
    out.content          = split.content;
    out.reasoning_tokens = reasoning_token_count(prep, pieces, has_tag_ids);
    if (prep.parse_tool_calls) {
        ToolCallParse parsed = parse_qwen_tool_calls(out.content, &prep.schemas);
        if (parsed.truncated) {
            log::warn("tools", "%s", "unterminated <tool_call> block; returned as raw content");
        }
        out.content = std::move(parsed.content);
        out.calls   = std::move(parsed.calls);
    }
}

}  // namespace lgc::api

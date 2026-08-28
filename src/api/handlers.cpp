#include "api/handlers.h"

#include <algorithm>
#include <atomic>
#include <ctime>
#include <set>

#include "api/error.h"
#include "build_info.h"
#include "core/stop.h"
#include "core/toolcall.h"
#include "util/log.h"
#include "util/text.h"
#include "util/utf8.h"

namespace lgc::api {
namespace {

using json = nlohmann::json;

std::atomic<uint64_t> g_request_counter{0};

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
    return json{{"prompt_tokens", s.prompt_tokens},
                {"completion_tokens", s.completion_tokens},
                {"total_tokens", s.prompt_tokens + s.completion_tokens}};
}

// `with_index` adds the per-call index. A streaming delta requires it (the
// official OpenAI SDKs model ChoiceDeltaToolCall.index as a required int, and
// reject a chunk without it); a non-streaming message does not carry it.
json tool_calls_json(const std::vector<ToolCall>& calls, bool with_index) {
    json out = json::array();
    for (size_t i = 0; i < calls.size(); ++i) {
        const ToolCall& c = calls[i];
        json            entry{{"id", c.id},
                              {"type", "function"},
                              {"function", {{"name", c.name}, {"arguments", c.arguments}}}};
        if (with_index) entry["index"] = static_cast<int>(i);
        out.push_back(std::move(entry));
    }
    return out;
}

// Effective context: --n-ctx wins, else the model's trained length, else the
// backend's. Zero means "unknown", and an unknown context cannot be enforced —
// which is why the stub is given one explicitly.
int effective_n_ctx(const Context& ctx) { return ctx.backend->status().n_ctx; }

bool sse_send(const SseWriter& write, const json& payload) {
    std::string frame = "data: ";
    frame += dump_json(payload);
    frame += "\n\n";
    return write(frame);
}

void log_slot_line(int slot, const char* phase, int tokens, double seconds, double rate,
                   const std::string& suffix = {}) {
    log::info(log::format("slot %d", slot), "%-7s %5d tok in %5.2f s (%5.1f t/s)%s", phase, tokens,
              seconds, rate, suffix.c_str());
}

void log_stats(int slot, const GenerationStats& stats, FinishReason reason) {
    std::string prefill_suffix;
    if (stats.cache_hit_tokens > 0 && stats.prompt_tokens > 0) {
        prefill_suffix = log::format(" | cache hit %d tok (%.1f%%)", stats.cache_hit_tokens,
                                     100.0 * stats.cache_hit_tokens / stats.prompt_tokens);
    }
    if (stats.snapshot_seconds > 0.0) {
        prefill_suffix += log::format(" | cache snapshot %.2f s", stats.snapshot_seconds);
    }
    log_slot_line(slot, "prefill", stats.prompt_tokens, stats.prefill_seconds,
                  stats.prefill_rate(), prefill_suffix);

    std::string decode_suffix;
    if (stats.decode_seconds > 0.0) {
        const double other = stats.decode_seconds - stats.decode_forward_seconds -
                             stats.decode_embed_seconds - stats.decode_sample_seconds -
                             stats.decode_emit_seconds;
        decode_suffix = log::format(
            " | graph %.2f s, embed %.2f s, sample %.2f s, emit %.2f s, other %.2f s",
            stats.decode_forward_seconds, stats.decode_embed_seconds,
            stats.decode_sample_seconds, stats.decode_emit_seconds, other);
    }
    if (stats.draft_proposed > 0) {
        decode_suffix = log::format(
            " | draft accept %.1f%% (%d/%d), verify %.2f s, re-forward %.2f s, rollback %.2f s",
            100.0 * stats.draft_accepted / stats.draft_proposed, stats.draft_accepted,
            stats.draft_proposed, stats.draft_verify_seconds, stats.draft_reforward_seconds,
            stats.draft_rollback_seconds);
    }
    log_slot_line(slot, "decode", stats.completion_tokens, stats.decode_seconds,
                  stats.decode_rate(), decode_suffix);
    if (reason == FinishReason::Abort) {
        log::info(log::format("slot %d", slot), "%s", "aborted: client gone");
    }
}

// ---------------------------------------------------------------- generation
//
// Everything generic to both endpoints lives here: stop sequences, stop token
// ids, cancellation, and the accumulation of the raw text the caller needs for
// tool-call parsing.
// `emit` receives the new piece and the full text accumulated so far
// (the piece is already appended). Returning false means the client is gone.
using EmitFn = std::function<bool(std::string_view piece, const std::string& accumulated)>;

FinishReason drive(Backend& backend, const GenerationInput& in, const EmitFn& emit,
                   GenerationStats& stats, std::string& raw) {
    StopMatcher   stop(in.sampler.stop);
    std::set<int> stop_ids(in.sampler.stop_token_ids.begin(), in.sampler.stop_token_ids.end());
    bool          cancelled = false;

    const FinishReason reason = backend.generate(
        in,
        [&](std::string_view piece, int token_id) -> Control {
            const StopMatcher::Step step = stop.push(piece);
            if (!step.emit.empty()) {
                raw += step.emit;
                if (!emit(step.emit, raw)) {
                    cancelled = true;
                    return Control::Cancel;
                }
            }
            if (step.hit) return Control::Stop;
            if (!stop_ids.empty() && stop_ids.count(token_id) != 0) return Control::Stop;
            return Control::Continue;
        },
        stats);

    if (!cancelled) {
        // Text held back as a possible stop-sequence prefix that never
        // completed is ordinary output and must not be swallowed.
        const std::string tail = stop.flush();
        if (!tail.empty()) {
            raw += tail;
            if (!emit(tail, raw)) return FinishReason::Abort;
        }
    }
    return reason;
}

}  // namespace

// -------------------------------------------------------------------- slots
SlotPool::SlotPool(int count) : busy_(static_cast<size_t>(std::max(1, count)), false) {}

SlotPool::Lease& SlotPool::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        if (pool_ != nullptr && index_ >= 0) pool_->release(index_);
        pool_        = other.pool_;
        index_       = other.index_;
        other.pool_  = nullptr;
        other.index_ = -1;
    }
    return *this;
}

SlotPool::Lease::~Lease() {
    if (pool_ != nullptr && index_ >= 0) pool_->release(index_);
}

SlotPool::Lease SlotPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_;
    cv_.wait(lock, [this] {
        return std::find(busy_.begin(), busy_.end(), false) != busy_.end();
    });
    --waiting_;

    const auto it    = std::find(busy_.begin(), busy_.end(), false);
    const int  index = static_cast<int>(it - busy_.begin());
    *it              = true;
    return Lease(this, index);
}

void SlotPool::release(int index) {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (index >= 0 && static_cast<size_t>(index) < busy_.size()) {
            busy_[static_cast<size_t>(index)] = false;
        }
    }
    cv_.notify_one();
}

int SlotPool::total() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int>(busy_.size());
}

int SlotPool::free() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int>(std::count(busy_.begin(), busy_.end(), false));
}

int SlotPool::queue_depth() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return waiting_;
}

// ----------------------------------------------------------------- /health
json health(const Context& ctx) {
    const ModelStatus& st = ctx.backend->status();
    return json{{"status", st.loaded ? "ok" : "loading"},
                {"model", st.id},
                {"loaded", st.loaded},
                {"stub", st.stub},
                {"slots_free", ctx.slots->free()},
                {"slots_total", ctx.slots->total()},
                {"queue_depth", ctx.slots->queue_depth()}};
}

// ------------------------------------------------------------------ /props
json props(const Context& ctx) {
    const ModelStatus&    st  = ctx.backend->status();
    const Config&         cfg = *ctx.cfg;
    const SamplerDefaults sd  = st.sampler_defaults;

    json model = {{"id", st.id},
                  {"quant", quant_name(st.quant)},
                  {"loaded", st.loaded},
                  {"stub", st.stub},
                  {"n_ctx", st.n_ctx}};

    // Zero means "not pinned in the allowlist and not reported by the artifact"
    // — null says that; 0 would be a lie.
    auto maybe_int = [](int v) { return v > 0 ? json(v) : json(nullptr); };
    model["n_layer"]       = maybe_int(st.n_layer);
    model["n_gdn_layer"]   = maybe_int(st.n_gdn_layer);
    model["n_attn_layer"]  = maybe_int(st.n_attn_layer);
    model["weights_bytes"] = st.weights_bytes > 0 ? json(st.weights_bytes) : json(nullptr);

    const ModelEntry* entry = find_model(st.id);
    if (entry != nullptr) {
        auto maybe_hash = [](const std::string& h) { return h.empty() ? json(nullptr) : json(h); };

        model["ov_arch"]                 = entry->ov_arch;
        model["model_type"]              = entry->model_type;
        model["moe"]                     = entry->moe;
        model["n_expert"]                = entry->moe ? json(entry->n_expert) : json(nullptr);
        model["n_embd"]                  = maybe_int(entry->n_embd);
        model["full_attention_interval"] = maybe_int(entry->full_attention_interval);
        model["n_ctx_train"]             = maybe_int(entry->n_ctx_train);
        model["has_mtp_head"]            = entry->has_mtp_head;
        model["arch_hash"]               = maybe_hash(entry->arch_hash);
        model["template_hash"]           = maybe_hash(entry->template_hash);
        model["tokenizer_hash"]          = maybe_hash(entry->tokenizer_hash);
        // What the Pruefstand says about this artifact (DESIGN.md §3.1: the
        // calibration lesson is part of the contract, not folklore).
        model["status"] = entry->status;
        model["weights_bytes_expected"] =
            entry->weights_bytes > 0 ? json(entry->weights_bytes) : json(nullptr);
    }

    return json{
        {"model", std::move(model)},
        {"cache",
         {{"kv_block_size", cfg.kv_block_size},
          {"kv_dtype", cfg.kv_dtype},
          {"gdn_checkpoint_budget_mib", cfg.gdn_checkpoint_budget_mib},
          {"prefix_cache", false}}},  // M3
        {"mtp", {{"requested", cfg.mtp}, {"enabled", st.mtp_enabled}}},
        {"sampler_defaults",
         {{"temperature", sd.temperature},
          {"top_p", sd.top_p},
          {"top_k", sd.top_k},
          {"repetition_penalty", sd.repetition_penalty},
          {"presence_penalty", sd.presence_penalty},
          {"provenance", sd.provenance}}},
        {"slots", {{"total", ctx.slots->total()}}},
        {"build",
         {{"version", LIGENCE_VERSION},
          {"git", LIGENCE_GIT_SHA},
          {"type", LIGENCE_BUILD_TYPE},
          {"compiler", LIGENCE_COMPILER}}},
        {"endpoints",
         json::array({"/health", "/props", "/v1/chat/completions", "/v1/completions",
                      "/v1/models"})}};
}

// ---------------------------------------------------------------- prepare
namespace {

std::optional<HttpResult> finish_prepare(const Context& ctx, const std::string& prompt,
                                         const SamplerOverrides& overrides,
                                         GenerationInput& input, int& prompt_tokens) {
    SamplerParams sampler = sampler_from_defaults(ctx.backend->status().sampler_defaults);
    if (auto err = sampler_apply(sampler, overrides)) {
        return HttpResult{400, invalid_request(*err)};
    }

    prompt_tokens = static_cast<int>(ctx.backend->tokenizer().encode(prompt).size());

    // DESIGN.md §3.8 — reject, with the numbers. No truncation, no shift.
    const int n_ctx = effective_n_ctx(ctx);
    if (n_ctx > 0 && prompt_tokens >= n_ctx) {
        return HttpResult{400, context_overflow(prompt_tokens, n_ctx)};
    }

    input.prompt  = prompt;
    input.sampler = std::move(sampler);
    return std::nullopt;
}

}  // namespace

std::optional<HttpResult> prepare_chat(const Context& ctx, const json& body, PreparedChat& out) {
    ChatRequest req;
    if (auto err = parse_chat_request(body, req)) {
        return HttpResult{400, invalid_request(*err)};
    }

    const std::string& served = ctx.backend->status().id;
    if (!req.model.empty() && req.model != served) {
        log::verbose("req", "request names model '%s'; this process serves '%s'",
                     req.model.c_str(), served.c_str());
    }

    PreparedChat prep;
    prep.req = std::move(req);

    // §3.7: only parse tool-call syntax when the request actually declared
    // tools. Otherwise the raw text goes back untouched.
    prep.parse_tool_calls = !prep.req.tools.empty() && prep.req.tool_choice != "none";
    if (prep.parse_tool_calls) prep.schemas = tool_schemas(prep.req.tools);

    for (const ToolSpec& t : prep.req.tools) prep.input.tool_names.push_back(t.name);

    const std::string prompt = ctx.backend->render_chat(prep.req);
    // The rendered prompt is the one thing that decides what the model saw.
    // When an answer differs from a reference run this is the first question,
    // and reconstructing it after the fact is guesswork (§3.7).
    if (log::enabled(log::Level::Debug)) {
        log::debug("prompt", "%zu bytes:\n%s", prompt.size(), prompt.c_str());
    }
    if (auto err = finish_prepare(ctx, prompt, prep.req.sampler, prep.input, prep.prompt_tokens)) {
        return err;
    }

    prep.id      = next_id("chatcmpl");
    prep.created = now_seconds();
    out          = std::move(prep);
    return std::nullopt;
}

std::optional<HttpResult> prepare_completion(const Context& ctx, const json& body,
                                             PreparedCompletion& out) {
    CompletionRequest req;
    if (auto err = parse_completion_request(body, req)) {
        return HttpResult{400, invalid_request(*err)};
    }

    PreparedCompletion prep;
    prep.req = std::move(req);
    if (auto err =
            finish_prepare(ctx, prep.req.prompt, prep.req.sampler, prep.input, prep.prompt_tokens)) {
        return err;
    }

    prep.id      = next_id("cmpl");
    prep.created = now_seconds();
    out          = std::move(prep);
    return std::nullopt;
}

// ------------------------------------------------------------- non-streaming
HttpResult run_chat(const Context& ctx, const PreparedChat& prep) {
    SlotPool::Lease lease = ctx.slots->acquire();

    GenerationStats stats;
    std::string     raw;
    const FinishReason reason =
        drive(*ctx.backend, prep.input, [](std::string_view, const std::string&) { return true; },
              stats, raw);
    log_stats(lease.index(), stats, reason);

    std::string           content = raw;
    std::vector<ToolCall> calls;
    if (prep.parse_tool_calls) {
        ToolCallParse parsed = parse_qwen_tool_calls(raw, &prep.schemas);
        if (parsed.truncated) {
            log::warn("tools", "%s", "unterminated <tool_call> block; returned as raw content");
        }
        content = std::move(parsed.content);
        calls   = std::move(parsed.calls);
    }

    json message = {{"role", "assistant"}};
    if (calls.empty()) {
        message["content"] = content;
    } else {
        message["content"] =
            text::trim(content).empty() ? json(nullptr) : json(content);
        message["tool_calls"] = tool_calls_json(calls, /*with_index=*/false);
    }

    const char* finish = calls.empty() ? finish_reason_name(reason) : "tool_calls";

    json body = {{"id", prep.id},
                 {"object", "chat.completion"},
                 {"created", prep.created},
                 {"model", ctx.backend->status().id},
                 {"choices", json::array({{{"index", 0},
                                           {"message", std::move(message)},
                                           {"finish_reason", finish}}})},
                 {"usage", usage_json(stats)}};
    return HttpResult{200, std::move(body)};
}

HttpResult run_completion(const Context& ctx, const PreparedCompletion& prep) {
    SlotPool::Lease lease = ctx.slots->acquire();

    GenerationStats stats;
    std::string     raw;
    const FinishReason reason =
        drive(*ctx.backend, prep.input, [](std::string_view, const std::string&) { return true; },
              stats, raw);
    log_stats(lease.index(), stats, reason);

    const std::string text = prep.req.echo ? prep.req.prompt + raw : raw;

    json body = {{"id", prep.id},
                 {"object", "text_completion"},
                 {"created", prep.created},
                 {"model", ctx.backend->status().id},
                 {"choices", json::array({{{"index", 0},
                                           {"text", text},
                                           {"logprobs", nullptr},
                                           {"finish_reason", finish_reason_name(reason)}}})},
                 {"usage", usage_json(stats)}};
    return HttpResult{200, std::move(body)};
}

// ----------------------------------------------------------------- streaming
void stream_chat(const Context& ctx, const PreparedChat& prep, const SseWriter& write) {
    SlotPool::Lease lease = ctx.slots->acquire();

    const std::string& model = ctx.backend->status().id;
    auto               chunk = [&](json choices, json extra = json::object()) {
        json c = {{"id", prep.id},
                                {"object", "chat.completion.chunk"},
                                {"created", prep.created},
                                {"model", model},
                                {"choices", std::move(choices)}};
        for (auto& [k, v] : extra.items()) c[k] = v;
        return c;
    };

    if (!sse_send(write, chunk(json::array({{{"index", 0},
                                             {"delta", {{"role", "assistant"}}},
                                             {"finish_reason", nullptr}}})))) {
        return;
    }

    utf8::Streamer      utf8_stream;
    std::string         raw;
    size_t              streamed = 0;  // bytes of tool-free content already sent
    bool                gone     = false;
    static const std::string kToolOpen = "<tool_call>";

    auto emit_content = [&](std::string_view bytes) {
        const std::string safe = utf8_stream.push(bytes);
        if (safe.empty()) return true;
        return sse_send(write, chunk(json::array({{{"index", 0},
                                                   {"delta", {{"content", safe}}},
                                                   {"finish_reason", nullptr}}})));
    };

    GenerationStats stats;
    const FinishReason reason = drive(
        *ctx.backend, prep.input,
        [&](std::string_view piece, const std::string& accumulated) -> bool {
            if (!prep.parse_tool_calls) {
                if (!emit_content(piece)) {
                    gone = true;
                    return false;
                }
                return true;
            }

            // Tool-call syntax must not leak into content deltas. Everything up
            // to the first marker streams as usual; from the marker on the text
            // is buffered and comes back as a tool_calls delta at the end. The
            // partial-suffix hold-back is what stops "<tool" going out on its
            // own when the marker straddles two pieces.
            // Scan only the unexamined tail plus enough overlap for a marker
            // that straddles the boundary. Rescanning from zero on every token
            // is quadratic, and this path runs at 262k context.
            const size_t scan_from =
                streamed >= kToolOpen.size() ? streamed - kToolOpen.size() + 1 : 0;
            const size_t marker = accumulated.find(kToolOpen, scan_from);
            const size_t safe_end =
                marker != std::string::npos
                    ? marker
                    : accumulated.size() - text::partial_stop_suffix(accumulated, kToolOpen);
            if (safe_end > streamed) {
                const std::string_view pending =
                    std::string_view(accumulated).substr(streamed, safe_end - streamed);
                streamed = safe_end;
                if (!emit_content(pending)) {
                    gone = true;
                    return false;
                }
            }
            return true;
        },
        stats, raw);

    log_stats(lease.index(), stats, reason);
    if (gone) return;

    std::vector<ToolCall> calls;
    if (prep.parse_tool_calls) {
        ToolCallParse parsed = parse_qwen_tool_calls(raw, &prep.schemas);
        if (parsed.truncated) {
            log::warn("tools", "%s", "unterminated <tool_call> block; returned as raw content");
        }
        calls = std::move(parsed.calls);
        if (parsed.content.size() > streamed) {
            if (!emit_content(std::string_view(parsed.content).substr(streamed))) return;
        }
    }

    const std::string tail = utf8_stream.flush();
    if (!tail.empty()) {
        if (!sse_send(write, chunk(json::array({{{"index", 0},
                                                 {"delta", {{"content", tail}}},
                                                 {"finish_reason", nullptr}}})))) {
            return;
        }
    }

    if (!calls.empty()) {
        if (!sse_send(write, chunk(json::array({{{"index", 0},
                                                 {"delta",
                                                  {{"tool_calls",
                                                    tool_calls_json(calls, /*with_index=*/true)}}},
                                                 {"finish_reason", nullptr}}})))) {
            return;
        }
    }

    const char* finish = calls.empty() ? finish_reason_name(reason) : "tool_calls";
    if (!sse_send(write, chunk(json::array({{{"index", 0},
                                             {"delta", json::object()},
                                             {"finish_reason", finish}}})))) {
        return;
    }

    if (prep.req.stream_include_usage) {
        if (!sse_send(write, chunk(json::array(), json{{"usage", usage_json(stats)}}))) return;
    }
    write("data: [DONE]\n\n");
}

void stream_completion(const Context& ctx, const PreparedCompletion& prep,
                       const SseWriter& write) {
    SlotPool::Lease lease = ctx.slots->acquire();

    const std::string& model = ctx.backend->status().id;
    auto               chunk = [&](json choices, json extra = json::object()) {
        json c = {{"id", prep.id},
                                {"object", "text_completion"},
                                {"created", prep.created},
                                {"model", model},
                                {"choices", std::move(choices)}};
        for (auto& [k, v] : extra.items()) c[k] = v;
        return c;
    };

    utf8::Streamer utf8_stream;
    bool           gone = false;

    if (prep.req.echo && !prep.req.prompt.empty()) {
        if (!sse_send(write, chunk(json::array({{{"index", 0},
                                                 {"text", prep.req.prompt},
                                                 {"logprobs", nullptr},
                                                 {"finish_reason", nullptr}}})))) {
            return;
        }
    }

    GenerationStats stats;
    std::string     raw;
    const FinishReason reason = drive(
        *ctx.backend, prep.input,
        [&](std::string_view piece, const std::string&) -> bool {
            const std::string safe = utf8_stream.push(piece);
            if (safe.empty()) return true;
            if (!sse_send(write, chunk(json::array({{{"index", 0},
                                                     {"text", safe},
                                                     {"logprobs", nullptr},
                                                     {"finish_reason", nullptr}}})))) {
                gone = true;
                return false;
            }
            return true;
        },
        stats, raw);

    log_stats(lease.index(), stats, reason);
    if (gone) return;

    const std::string tail = utf8_stream.flush();
    if (!tail.empty()) {
        if (!sse_send(write, chunk(json::array({{{"index", 0},
                                                 {"text", tail},
                                                 {"logprobs", nullptr},
                                                 {"finish_reason", nullptr}}})))) {
            return;
        }
    }

    if (!sse_send(write, chunk(json::array({{{"index", 0},
                                             {"text", ""},
                                             {"logprobs", nullptr},
                                             {"finish_reason", finish_reason_name(reason)}}})))) {
        return;
    }

    if (prep.req.stream_include_usage) {
        if (!sse_send(write, chunk(json::array(), json{{"usage", usage_json(stats)}}))) return;
    }
    write("data: [DONE]\n\n");
}

}  // namespace lgc::api

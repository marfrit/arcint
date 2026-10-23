#include "api/handlers.h"

#include <stdexcept>
#include <optional>
#include <functional>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <set>

#include "api/error.h"
#include "build_info.h"
#include "core/stop.h"
#include "core/reasoning.h"
#include "core/toolcall.h"
#include "util/log.h"
#include "util/text.h"
#include "util/utf8.h"

namespace lgc::api {

double decode_other_seconds(const GenerationStats& stats) {
    return stats.decode_seconds - stats.decode_forward_seconds - stats.decode_embed_seconds -
           stats.decode_sample_seconds - stats.decode_emit_seconds - stats.decode_wait_seconds -
           stats.draft_verify_seconds - stats.draft_propose_seconds;
}

// The decode line's conditional terms, extracted from log_stats (review B4
// follow-up) so the stats wiring is unit-testable at the struct level: the
// value decode_suffix_for returns is the one that lands on the line.
std::string decode_suffix_for(const GenerationStats& stats) {
    std::string decode_suffix;
    if (stats.decode_seconds > 0.0) {
        const double other = decode_other_seconds(stats);
        decode_suffix = log::format(
            " | graph %.2f s, embed %.2f s, sample %.2f s, emit %.2f s, wait %.2f s, "
            "other %.2f s",
            stats.decode_forward_seconds, stats.decode_embed_seconds,
            stats.decode_sample_seconds, stats.decode_emit_seconds,
            stats.decode_wait_seconds, other);
    }
    // Printed whenever a subtracted term is non-zero, not only when a draft was
    // proposed: propose time accrues on every paged drafting cycle, so a request
    // that proposed nothing would otherwise subtract a term it never shows.
    if (stats.draft_proposed > 0 || stats.draft_propose_seconds > 0.0) {
        decode_suffix += log::format(
            " | draft accept %.1f%% (%d/%d), propose %.2f s, verify %.2f s, re-forward %.2f s, "
            "rollback %.2f s",
            stats.draft_proposed > 0 ? 100.0 * stats.draft_accepted / stats.draft_proposed : 0.0,
            stats.draft_accepted,
            stats.draft_proposed, stats.draft_propose_seconds, stats.draft_verify_seconds,
            stats.draft_reforward_seconds, stats.draft_rollback_seconds);
        // the acceptance length: the tokens a verify emits on average
        if (stats.draft_steps > 0)
            decode_suffix += log::format(", %d verifies (%.2f tokens a verify)", stats.draft_steps,
                                         static_cast<double>(stats.draft_accepted + stats.draft_steps) / stats.draft_steps);
    }
    // Visible even on a request that drafted nothing at all: a lane whose
    // drafter is currently off (disabled by an earlier failure, re-armed only
    // at the next request) proposes zero drafts and would otherwise vanish
    // from this line entirely.
    if (stats.dflash_lane_failures > 0) {
        decode_suffix +=
            log::format(" | dflash lane failures %d (cumulative)", stats.dflash_lane_failures);
    }
    // Tool-call grammar desyncs, review B4: the constraint lost sync with the
    // text and the rest of the request ran unconstrained (fail open, so the
    // turn can still end). Printed only when non-zero, as the drafting terms,
    // so a request without a broken constraint keeps the line it always had.
    if (stats.grammar_desyncs > 0) {
        decode_suffix +=
            log::format(" | grammar desync %d (turn unconstrained after desync)", stats.grammar_desyncs);
    }
    // What the other lane cost this one. Printed only when it happened, so a
    // single-stream run's line is the line it always was, and a p95 rather than
    // a mean because one long stall inside many short steps is exactly what a
    // mean hides (DESIGN.md §4.1).
    if (stats.stalled_steps > 0) {
        decode_suffix += log::format(" | stall p95 %.0f ms max %.0f ms (%d step%s, %.2f s total)",
                                     1000.0 * stats.stall_p95_seconds,
                                     1000.0 * stats.stall_max_seconds, stats.stalled_steps,
                                     stats.stalled_steps == 1 ? "" : "s",
                                     stats.stall_total_seconds);
    }
    return decode_suffix;
}

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
    log_slot_line(slot, "prefill", stats.prompt_tokens, stats.prefill_seconds,
                  stats.prefill_rate(), prefill_suffix);

    const std::string decode_suffix = decode_suffix_for(stats);
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
// `emit` receives the new piece, its token id and the full text accumulated so
// far (the piece is already appended). Returning false means the client is gone.
using EmitFn = std::function<bool(std::string_view piece, int token_id, const std::string& accumulated)>;

FinishReason drive(Backend& backend, const GenerationInput& in, int slot, const EmitFn& emit,
                   GenerationStats& stats, std::string& raw,
                   std::vector<std::pair<std::string, int>>* pieces = nullptr) {
    StopMatcher   stop(in.sampler.stop);
    std::set<int> stop_ids(in.sampler.stop_token_ids.begin(), in.sampler.stop_token_ids.end());
    bool          cancelled = false;

    const FinishReason reason = backend.generate(
        in, slot,
        [&](std::string_view piece, int token_id) -> Control {
            const StopMatcher::Step step = stop.push(piece);
            // The reasoning split is decided by (piece, id), so record what was
            // emitted with the id of the token that carried it; a piece the stop
            // matcher held back in part gets no id (it is not wholly emitted).
            if (pieces != nullptr)
                pieces->emplace_back(step.emit, step.emit.size() == piece.size() ? token_id : -1);
            if (!step.emit.empty()) {
                raw += step.emit;
                if (!emit(step.emit, token_id, raw)) {
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
            if (pieces != nullptr) pieces->emplace_back(tail, -1);
            if (!emit(tail, -1, raw)) return FinishReason::Abort;
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

SlotPool::Lease SlotPool::acquire_for(double timeout_seconds) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto some_free = [this] {
        return std::find(busy_.begin(), busy_.end(), false) != busy_.end();
    };
    if (!some_free()) {
        if (timeout_seconds <= 0.0) return Lease();
        ++waiting_;
        const bool got = cv_.wait_for(
            lock, std::chrono::duration<double>(timeout_seconds), some_free);
        --waiting_;
        if (!got) return Lease();
    }

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

// ------------------------------------------------------------- admission
std::optional<HttpResult> acquire_slot(const Context& ctx, SlotPool::Lease& out) {
    // under named lanes every request is admitted on its lane's own pool
    if (!ctx.lanes.empty()) throw std::logic_error("acquire_slot without a lane under named lanes");
    const double timeout = ctx.cfg != nullptr ? ctx.cfg->queue_timeout_s : 0.0;
    out = ctx.slots->acquire_for(timeout);
    if (out.index() >= 0) return std::nullopt;

    // Not "server busy": the numbers that say how many lanes there are and
    // how many are free (the body's slots object carries them).
    std::string why =
        log::format("all %d lane%s are busy", ctx.slots->total(),
                    ctx.slots->total() == 1 ? "" : "s");
    why += timeout > 0.0
               ? log::format("; waited %.1f s. Retry, or raise --parallel with the memory to "
                             "back it.", timeout)
               : "; no lane came free. Retry, raise --queue-timeout to wait, or raise "
                 "--parallel with the memory to back it.";

    json body = error_body(why, "server_error", "no_slot_available");
    body["slots"] = {{"total", ctx.slots->total()},
                     {"free", ctx.slots->free()},
                     {"queue_depth", ctx.slots->queue_depth()},
                     {"queue_timeout_s", timeout}};
    return HttpResult{503, std::move(body)};
}

// ------------------------------------------------------------- named lanes
std::vector<Lane> make_lanes(const Config& cfg, std::vector<std::unique_ptr<SlotPool>>& pools) {
    std::vector<Lane> lanes;
    if (!cfg.named_lanes()) return lanes;
    for (size_t i = 0; i < cfg.lane_names.size() && i < cfg.lane_ctx.size(); ++i) {
        pools.push_back(std::make_unique<SlotPool>(1));
        lanes.push_back(Lane{cfg.lane_names[i], cfg.lane_ctx[i], static_cast<int>(i), pools.back().get()});
    }
    return lanes;
}

std::optional<HttpResult> resolve_lane(const Context& ctx, const std::string& model, int& lane) {
    lane = -1;
    if (ctx.lanes.empty()) return std::nullopt;
    for (size_t i = 0; i < ctx.lanes.size(); ++i) {
        if (ctx.lanes[i].name == model) {
            lane = static_cast<int>(i);
            return std::nullopt;
        }
    }
    // No name, or the artifact's own: the first lane. Every lane serves that
    // artifact, and a client that took the canonical id from /v1/models
    // should land where an unnamed request does, not on a 404.
    if (model.empty() || model == ctx.backend->status().id) {
        lane = 0;
        return std::nullopt;
    }
    std::string served;
    for (const Lane& l : ctx.lanes) served += (served.empty() ? "" : ", ") + l.name;
    return HttpResult{404, error_body(log::format("The model '%s' does not exist: this server's lanes are %s "
                                                  "(the model field picks the lane)",
                                                  model.c_str(), served.c_str()),
                                      "invalid_request_error", "model_not_found", "model")};
}

int lane_n_ctx(const Context& ctx, int lane) {
    if (lane >= 0 && static_cast<size_t>(lane) < ctx.lanes.size()) return ctx.lanes[static_cast<size_t>(lane)].n_ctx;
    return effective_n_ctx(ctx);
}

namespace {

// Every lane's admission state, for /health and the 503
json lanes_json(const Context& ctx) {
    json out = json::array();
    for (const Lane& l : ctx.lanes) {
        out.push_back(json{{"name", l.name},
                           {"n_ctx", l.n_ctx},
                           {"slots_total", l.pool->total()},
                           {"slots_free", l.pool->free()},
                           {"queue_depth", l.pool->queue_depth()}});
    }
    return out;
}

int slots_total(const Context& ctx) {
    if (ctx.lanes.empty()) return ctx.slots->total();
    int n = 0;
    for (const Lane& l : ctx.lanes) n += l.pool->total();
    return n;
}

int slots_free(const Context& ctx) {
    if (ctx.lanes.empty()) return ctx.slots->free();
    int n = 0;
    for (const Lane& l : ctx.lanes) n += l.pool->free();
    return n;
}

int queue_depth(const Context& ctx) {
    if (ctx.lanes.empty()) return ctx.slots->queue_depth();
    int n = 0;
    for (const Lane& l : ctx.lanes) n += l.pool->queue_depth();
    return n;
}

}  // namespace

std::optional<HttpResult> acquire_slot(const Context& ctx, int lane, SlotPool::Lease& out, int& slot) {
    if (lane < 0 || static_cast<size_t>(lane) >= ctx.lanes.size()) {
        auto err = acquire_slot(ctx, out);
        slot     = out.index();
        return err;
    }
    // The lane's own pool: a busy lane waits (--queue-timeout) or is refused
    // by itself, and another lane is never taken instead (a lane is a memory
    // reservation of its own size, DESIGN §4.3)
    const Lane&  l       = ctx.lanes[static_cast<size_t>(lane)];
    const double timeout = ctx.cfg != nullptr ? ctx.cfg->queue_timeout_s : 0.0;
    out                  = l.pool->acquire_for(timeout);
    slot                 = out.index() >= 0 ? l.seq : -1;
    if (out.index() >= 0) return std::nullopt;

    std::string why = log::format("lane '%s' (n_ctx %d) is busy", l.name.c_str(), l.n_ctx);
    why += timeout > 0.0 ? log::format("; waited %.1f s. Retry, or use another lane's name if its "
                                       "context suits the request.", timeout)
                         : "; no wait configured. Retry, raise --queue-timeout to wait, or use another "
                           "lane's name if its context suits the request.";
    json body     = error_body(why, "server_error", "no_slot_available");
    body["slots"] = {{"lane", l.name},
                     {"total", l.pool->total()},
                     {"free", l.pool->free()},
                     {"queue_depth", l.pool->queue_depth()},
                     {"queue_timeout_s", timeout}};
    body["lanes"]          = lanes_json(ctx);   // as /health has them
    return HttpResult{503, std::move(body)};
}

// ----------------------------------------------------------------- /health
json health(const Context& ctx) {
    const ModelStatus& st = ctx.backend->status();
    json h = json{{"status", st.loaded ? "ok" : "loading"},
                {"model", st.served_id},
                {"loaded", st.loaded},
                {"stub", st.stub},
                {"slots_free", slots_free(ctx)},
                {"slots_total", slots_total(ctx)},
                {"queue_depth", queue_depth(ctx)}};
    // named lanes: each lane's own counts (the totals above sum them)
    if (!ctx.lanes.empty()) h["lanes"] = lanes_json(ctx);
    return h;
}

// ------------------------------------------------------------------ /props
json props(const Context& ctx) {
    const ModelStatus&    st  = ctx.backend->status();
    const Config&         cfg = *ctx.cfg;
    const SamplerDefaults sd  = st.sampler_defaults;

    // `id` is what this endpoint is called; `canonical_id` is which artifact is
    // behind it. They differ only under --served-model-name, and both are
    // always present so a rename never costs identity.
    json model = {{"id", st.served_id},
                  {"canonical_id", st.id},
                  {"served_model_name", cfg.served_model_name.empty()
                                            ? json(nullptr)
                                            : json(cfg.served_model_name)},
                  {"quant", st.quant.empty() ? json(nullptr) : json(st.quant)},
                  {"loaded", st.loaded},
                  {"stub", st.stub},
                  {"n_ctx", st.n_ctx}};

    // Which names a request may put in its `model` field. Both of these are
    // recognised; anything else is served anyway and noted at -v, because one
    // process serves exactly one model and there is nothing else it could mean.
    // Said out loud rather than left to be inferred from the list. Named lanes
    // (§4.2, amended 2026-10-07) make the field binding: each lane's name
    // picks it, the canonical id (and an empty name) the first lane, and any
    // other name is a 404.
    json answers_to = json::array();
    if (ctx.lanes.empty()) {
        answers_to.push_back(st.served_id);
    } else {
        for (const Lane& l : ctx.lanes) answers_to.push_back(l.name);
    }
    if (std::find(answers_to.begin(), answers_to.end(), json(st.id)) == answers_to.end()) answers_to.push_back(st.id);
    model["answers_to"]             = std::move(answers_to);
    model["enforces_model_field"]   = !ctx.lanes.empty();

    // Zero means "the backend reports nothing" (the stub): null says that; 0
    // would be a lie.
    auto maybe_int = [](int v) { return v > 0 ? json(v) : json(nullptr); };
    auto maybe_str = [](const std::string& v) { return v.empty() ? json(nullptr) : json(v); };
    model["architecture"]  = maybe_str(st.arch);
    model["n_ctx_train"]   = maybe_int(st.n_ctx_train);
    model["n_layer"]       = maybe_int(st.n_layer);
    model["weights_bytes"] = st.weights_bytes > 0 ? json(st.weights_bytes) : json(nullptr);

    json p = json{
        {"model", std::move(model)},
        {"chat_template_caps", ctx.backend->template_caps()},
        // What is actually served, not a Config default the load path may
        // never have read. `kv_dtype` comes off ModelStatus, which the llama
        // backend fills from --llama-kv (config.h's llama_kv_dtype); the stub
        // leaves it empty on purpose, which is what lets this print null
        // instead of guessing.
        {"cache", {{"kv_dtype", st.kv_precision.empty() ? json(nullptr) : json(st.kv_precision)}}},
        {"mtp", {{"requested", std::string(cfg.llama_mtp > 0 ? "on" : "off")},
                 {"enabled", st.mtp_enabled}}},
        {"dflash", {{"requested", std::string(cfg.llama_dflash_on() ? "on" : "off")},
                    {"enabled", st.dflash_enabled},
                    // the drafter's own count: --llama-dflash capped by its trained block
                    {"drafts", st.dflash_drafts > 0 ? json(st.dflash_drafts) : json(nullptr)}}},
        {"sampler_defaults",
         {{"temperature", sd.temperature},
          {"top_p", sd.top_p},
          {"top_k", sd.top_k},
          {"min_p", sd.min_p},
          {"repetition_penalty", sd.repetition_penalty},
          {"presence_penalty", sd.presence_penalty},
          // clean-turns: whether the think-block mask is on for requests that
          // send nothing. /props says what the server is running with, so a
          // turn that ends differently than yesterday's is explained here.
          {"think_eog_mask", sd.think_eog_mask},
          {"tool_grammar", sd.tool_grammar},
          {"provenance", sd.provenance}}},
        {"slots",
         {{"total", slots_total(ctx)},
          {"queue_timeout_s", cfg.queue_timeout_s}}},
        {"build",
         {{"version", ARCINT_VERSION},
          {"git", ARCINT_GIT_SHA},
          {"type", ARCINT_BUILD_TYPE},
          {"compiler", ARCINT_COMPILER}}},
        // The multimodal/audio presence: the backbone, the projector and its
        // own halves (libmtmd's cap), the served TTS flag and the enrolled-voice
        // count (names not required on /props by D5; GET /v1/audio/voices
        // carries them). All describe what this server actually loaded.
        {"multimodal",
         {{"backbone_loaded", st.loaded && !st.stub},
          {"mmproj_loaded", st.mmproj_loaded},
          {"vision", st.mmproj_vision},
          {"audio", st.mmproj_audio}}},
        {"tts", st.tts},
        {"voices", static_cast<int>(st.voices.size())},
        {"endpoints",
         json::array({"/health", "/props", "/v1/chat/completions", "/v1/completions",
                      "/v1/models", "/v1/audio/speech", "/v1/audio/voices"})}};
    if (!ctx.lanes.empty()) {
        json lanes = json::array();
        for (const Lane& l : ctx.lanes) lanes.push_back(json{{"name", l.name}, {"n_ctx", l.n_ctx}, {"seq", l.seq}});
        p["lanes"] = std::move(lanes);
    }
    return p;
}

// ------------------------------------------------------------- /v1/models
json models(const Context& ctx) {
    // One process serves one model (§2). The list has one entry (one per
    // named lane) so that OpenAI clients which probe this endpoint keep
    // working.
    //
    // It carries the context length because this is the only place a
    // discovering proxy looks: it reads n_ctx (among a few spellings) from
    // the model object here and asks /props for nothing but the template
    // capabilities. A context published on /props alone reaches no client,
    // and a client with no context falls back to its own default against a
    // server configured for 262144. `n_ctx` is what this process is
    // actually running with; `n_ctx_train` is the artifact's ceiling, so a
    // caller can see both and tell them apart.
    //
    // Named lanes (§4.2, amended 2026-10-07): one entry per lane name, with
    // that lane's context, so a proxy sees the subagent lane at its size.
    const ModelStatus& st = ctx.backend->status();
    auto maybe_int = [](int v) { return v > 0 ? json(v) : json(nullptr); };
    auto entry     = [&](const std::string& id, int n_ctx, int lanes) {
        json e{{"id", id},
               {"object", "model"},
               {"owned_by", "arcint"},
               {"n_ctx", maybe_int(n_ctx)},
               {"n_ctx_train", maybe_int(st.n_ctx_train)},
               {"quant", st.quant.empty() ? json(nullptr) : json(st.quant)},
               {"lanes", lanes}};
        // The artifact this endpoint actually serves, which --served-model-name
        // may have renamed. Always present, so a rename never hides identity.
        e["canonical_id"] = st.id;
        return e;
    };
    json data = json::array();
    if (ctx.lanes.empty()) {
        data.push_back(entry(st.served_id, st.n_ctx, ctx.slots->total()));
    } else {
        for (const Lane& l : ctx.lanes) data.push_back(entry(l.name, l.n_ctx, l.pool->total()));
    }
    return json{{"object", "list"}, {"data", std::move(data)}};
}

// -------------------------------------------------------- /v1/audio/voices
json voices(const Context& ctx) {
    // The enrolled names only, never a path or bytes (D5). Empty is normal: a
    // server without --tts, or one whose profiles directory has no enabled
    // entries yet. The list is a probe, not an admission gate.
    json names = json::array();
    for (const std::string& v : ctx.backend->status().voices) names.push_back(v);
    return json{{"voices", std::move(names)}};
}

// ---------------------------------------------------------------- prepare
namespace {

std::optional<HttpResult> finish_prepare(const Context& ctx, int lane, const std::string& prompt,
                                         const SamplerOverrides& overrides,
                                         GenerationInput& input, int& prompt_tokens,
                                         const std::vector<int>& prompt_ids = {}) {
    SamplerParams sampler = sampler_from_defaults(ctx.backend->status().sampler_defaults);
    if (auto err = sampler_apply(sampler, overrides)) {
        return HttpResult{400, invalid_request(*err)};
    }

    // A token-id prompt is counted as given; a text prompt is tokenised once
    // here for the count (the backend tokenises it again for the forward).
    prompt_tokens = prompt_ids.empty()
                        ? static_cast<int>(ctx.backend->tokenizer().encode(prompt).size())
                        : static_cast<int>(prompt_ids.size());

    // DESIGN.md §3.8 — reject, with the numbers. No truncation, no shift.
    // Against the request's lane: a subagent lane's prompt over its cap is
    // refused at that cap, even where the agent lane would hold it.
    const int n_ctx = lane_n_ctx(ctx, lane);
    if (n_ctx > 0 && prompt_tokens >= n_ctx) {
        return HttpResult{400, context_overflow(prompt_tokens, n_ctx)};
    }

    input.prompt     = prompt;
    input.prompt_ids = prompt_ids;
    input.sampler    = std::move(sampler);
    return std::nullopt;
}

}  // namespace

std::optional<HttpResult> prepare_chat(const Context& ctx, const json& body, PreparedChat& out) {
    ChatRequest req;
    if (auto err = parse_chat_request(body, req)) {
        return HttpResult{400, invalid_request(*err)};
    }

    // The lane first: its name may be a 404, and its cap is the 400's
    int lane = -1;
    if (auto err = resolve_lane(ctx, req.model, lane)) return err;
    const ModelStatus& st = ctx.backend->status();
    if (lane < 0 && !req.model.empty() && req.model != st.served_id && req.model != st.id) {
        log::verbose("req", "request names model '%s'; this process serves '%s'",
                     req.model.c_str(), st.served_id.c_str());
    }

    // An image part needs the multimodal projector. The parser accepts the
    // part shape; this process admits it only when --mmproj loaded (docs/
    // milestone-1.2.0.md A2), so a text-only server answers the same refusal
    // it always did.
    for (const ChatMessage& m : req.messages) {
        if (m.images.empty()) continue;
        if (!st.mmproj_loaded) {
            return HttpResult{400, invalid_request("image_url parts need a multimodal projector "
                                                    "(--mmproj; docs/milestone-1.2.0.md)")};
        }
    }
    // A speech unit's talker emits codec tokens, not text: a chat answer from
    // it would be noise with a 200 (docs/milestone-1.2.0.md A4).
    if (st.tts) {
        return HttpResult{400, invalid_request("this unit serves speech (--tts): POST /v1/audio/speech; "
                                               "chat completions are not served here")};
    }

    PreparedChat prep;
    prep.req   = std::move(req);
    prep.lane  = lane;
    prep.model = lane >= 0 ? ctx.lanes[static_cast<size_t>(lane)].name : st.served_id;

    // The operator's template default. Applied only when the request said
    // nothing itself (neither chat_template_kwargs.enable_thinking nor
    // reasoning_effort), so a request always wins.
    if (!prep.req.has_enable_thinking && ctx.cfg->think_default) {
        prep.req.enable_thinking     = *ctx.cfg->think_default;
        prep.req.has_enable_thinking = true;
    }

    // §3.7: only parse tool-call syntax when the request actually declared
    // tools. Otherwise the raw text goes back untouched.
    prep.parse_tool_calls = !prep.req.tools.empty() && prep.req.tool_choice != "none";
    if (prep.parse_tool_calls) prep.schemas = tool_schemas(prep.req.tools);

    for (const ToolSpec& t : prep.req.tools) prep.input.tool_names.push_back(t.name);
    // clean-turns step 2: the schemas reach the backend only when the calls
    // are parsed, so the grammar never constrains a call nobody would read.
    if (prep.parse_tool_calls) {
        prep.input.tools               = prep.req.tools;
        prep.input.parallel_tool_calls = prep.req.parallel_tool_calls;
    }

    const std::string prompt = ctx.backend->render_chat(prep.req);
    // The template decides whether the answer starts inside a think block; the
    // split of reasoning from content below follows that, not a guess.
    prep.think_open = prompt.ends_with("<think>\n");
    prep.think_tags_extended = ctx.backend->status().think_tags_extended;
    // The split, like the backend's mask, decides the block by token id where
    // the artifact's vocabulary spells a tag as one special token.
    prep.think_ids = ctx.backend->think_block_ids(prep.think_tags_extended);
    // The backend has to know the same thing: its mask engages before the first
    // sample, where the API's reasoning split has not seen a token yet.
    prep.input.think_open          = prep.think_open;
    prep.input.think_tags_extended = prep.think_tags_extended;
    // The rendered prompt is the one thing that decides what the model saw.
    // When an answer differs from a reference run this is the first question,
    // and reconstructing it after the fact is guesswork (§3.7).
    if (log::enabled(log::Level::Debug)) {
        log::debug("prompt", "%zu bytes:\n%s", prompt.size(), prompt.c_str());
    }
    if (auto err = finish_prepare(ctx, lane, prompt, prep.req.sampler, prep.input, prep.prompt_tokens)) {
        return err;
    }
    // the media, in marker order across the whole request (the renderer put one
    // marker per part into the prompt)
    for (const ChatMessage& m : prep.req.messages)
        for (const ImagePart& img : m.images) prep.input.images.push_back(img);

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

    int lane = -1;
    if (auto err = resolve_lane(ctx, req.model, lane)) return err;
    if (ctx.backend->status().tts) {
        return HttpResult{400, invalid_request("this unit serves speech (--tts): POST /v1/audio/speech; "
                                               "completions are not served here")};
    }

    PreparedCompletion prep;
    prep.req   = std::move(req);
    prep.lane  = lane;
    prep.model = lane >= 0 ? ctx.lanes[static_cast<size_t>(lane)].name : ctx.backend->status().served_id;
    if (auto err =
            finish_prepare(ctx, lane, prep.req.prompt, prep.req.sampler, prep.input, prep.prompt_tokens,
                           prep.req.prompt_ids)) {
        return err;
    }

    prep.id      = next_id("cmpl");
    prep.created = now_seconds();
    out          = std::move(prep);
    return std::nullopt;
}

std::optional<HttpResult> prepare_speech(const Context& ctx, const json& body, PreparedSpeech& out) {
    if (!body.is_object()) return HttpResult{400, invalid_request("request body must be a JSON object")};

    // D5, first: any reference-audio field is refused before the body is
    // parsed. The enrolment route is the tool, not this endpoint, and the
    // message says so rather than leaving a client to guess a field name.
    static const char* const kReferenceFields[] = {
        "reference_audio", "speaker_file", "sample_audio", "audio", "reference",
        "voice_sample", "speaker_ref", "speaker_reference"};
    for (const char* f : kReferenceFields) {
        if (!body.contains(f)) continue;
        return HttpResult{400, error_body(
            log::format("'%s' is not accepted: voice cloning takes enrolled voices only. Enrol a voice "
                        "with tools/tts_enrol.py (docs/speech-service.md D5), then ask for it by its "
                        "name with `voice`; GET /v1/audio/voices lists the names this server serves.",
                        f),
            "invalid_request_error", "invalid_request_error", f)};
    }

    SpeechRequest req;
    if (auto err = parse_speech_request(body, req)) return HttpResult{400, invalid_request(*err)};

    int lane = -1;
    if (auto err = resolve_lane(ctx, req.model, lane)) return err;

    // The availability refusals, in the order a build would fail them. The
    // process's own --tts is checked first, then the projector, so a text-only
    // server answers the same clear "no TTS" it always would.
    const ModelStatus& st = ctx.backend->status();
    if (!ctx.cfg->tts)
        return HttpResult{400, invalid_request("this build has no TTS: run with --tts and an "
                                               "audio-generation --mmproj (docs/milestone-1.2.0.md D4)")};
    if (!st.mmproj_loaded)
        return HttpResult{400, invalid_request("the speech routes need an --mmproj with the "
                                               "audio-generation half: this server loaded none")};
    if (!st.tts)
        return HttpResult{400, invalid_request("the loaded --mmproj carries no audio-generation half: "
                                               "the speaker encoder and acoustic decoder are missing "
                                               "(docs/milestone-1.2.0.md D4)")};

    // The voice must be one the server enrolled (D5); the list is names only.
    bool enrolled = false;
    for (const std::string& v : st.voices) if (v == req.voice) { enrolled = true; break; }
    if (!enrolled) {
        std::string known;
        for (const std::string& v : st.voices) known += (known.empty() ? "" : ", ") + v;
        return HttpResult{400, invalid_request(log::format(
            "The voice '%s' is not enrolled. This server serves %s (GET /v1/audio/voices). Enrol a "
            "voice with tools/tts_enrol.py --name NAME --audio FILE --text FILE --profiles DIR "
            "(docs/speech-service.md D5); the API never takes reference audio.",
            req.voice.c_str(), known.empty() ? "no voices" : known.c_str()))};
    }

    PreparedSpeech prep;
    prep.req   = std::move(req);
    prep.lane  = lane;
    prep.model = lane >= 0 ? ctx.lanes[static_cast<size_t>(lane)].name : st.served_id;

    SamplerParams sampler = sampler_from_defaults(st.sampler_defaults);
    if (auto err = sampler_apply(sampler, prep.req.sampler))
        return HttpResult{400, invalid_request(*err)};
    prep.input.text    = prep.req.text;
    prep.input.voice   = prep.req.voice;
    prep.input.sampler = std::move(sampler);

    out = std::move(prep);
    return std::nullopt;
}

// ------------------------------------------------------------- non-streaming
// A generation's exceptions as an answer: the backend's InputError (the
// request's own content: an image it cannot read, too many or too large
// images, a prompt over the lane with its image tokens) is the client's 400,
// anything else a 500. Without this an exception thrown while a streamed body
// runs would leave httplib's worker and end the process.
static std::optional<HttpResult> guarded(const std::function<void()>& run) {
    try {
        run();
    } catch (const ContextOverflowError& e) {
        log::warn("req", "refused: prompt %d tokens with its images, lane %d", e.prompt_tokens, e.n_ctx);
        return HttpResult{400, context_overflow(e.prompt_tokens, e.n_ctx)};
    } catch (const InputError& e) {
        log::warn("req", "refused: %s", e.what());
        return HttpResult{400, invalid_request(e.what(), "messages")};
    } catch (const std::exception& e) {
        log::error("req", "generation failed: %s", e.what());
        return HttpResult{500, error_body(e.what(), "internal_error")};
    }
    return std::nullopt;
}

HttpResult run_chat(const Context& ctx, const PreparedChat& prep, int slot) {
    GenerationStats stats;
    std::string     raw;
    std::vector<std::pair<std::string, int>> pieces;
    FinishReason reason = FinishReason::Stop;
    if (auto err = guarded([&] {
            reason = drive(*ctx.backend, prep.input, slot,
                           [](std::string_view, int, const std::string&) { return true; }, stats, raw, &pieces);
        }))
        return *err;
    log_stats(slot, stats, reason);

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
    std::string           content = split.content;
    std::vector<ToolCall> calls;
    if (prep.parse_tool_calls) {
        ToolCallParse parsed = parse_qwen_tool_calls(split.content, &prep.schemas);
        if (parsed.truncated) {
            log::warn("tools", "%s", "unterminated <tool_call> block; returned as raw content");
        }
        content = std::move(parsed.content);
        calls   = std::move(parsed.calls);
    }

    json message = {{"role", "assistant"}};
    if (!split.reasoning.empty()) {
        message["reasoning_content"] = split.reasoning;
        // Whether the think block's closer was seen before the turn ended.
        // The split computes it but the client could not see it, which made an
        // empty answer that closed the block indistinguishable from the defect
        // (a turn-ending token winning a step while the block was still open).
        message["reasoning_closed"] = split.closed;
    }
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
                 {"model", prep.model},
                 {"choices", json::array({{{"index", 0},
                                           {"message", std::move(message)},
                                           {"finish_reason", finish}}})},
                 {"usage", usage_json(stats)}};
    return HttpResult{200, std::move(body)};
}

HttpResult run_completion(const Context& ctx, const PreparedCompletion& prep, int slot) {
    GenerationStats stats;
    std::string     raw;
    FinishReason reason = FinishReason::Stop;
    if (auto err = guarded([&] {
            reason = drive(*ctx.backend, prep.input, slot,
                           [](std::string_view, int, const std::string&) { return true; }, stats, raw);
        }))
        return *err;
    log_stats(slot, stats, reason);

    const std::string text = prep.req.echo ? prep.req.prompt + raw : raw;

    json body = {{"id", prep.id},
                 {"object", "text_completion"},
                 {"created", prep.created},
                 {"model", prep.model},
                 {"choices", json::array({{{"index", 0},
                                           {"text", text},
                                           {"logprobs", nullptr},
                                           {"finish_reason", finish_reason_name(reason)}}})},
                 {"usage", usage_json(stats)}};
    return HttpResult{200, std::move(body)};
}

// --------------------------------------------------------- speech synthesis
SpeechResult run_speech(const Context& ctx, const PreparedSpeech& prep, int slot) {
    GenerationStats stats;
    std::string     wav;
    std::string     err;
    bool ok = false;
    try {
        ok = ctx.backend->generate_speech(prep.input, slot, wav, err, stats);
    } catch (const ContextOverflowError& e) {
        log::warn(log::format("slot %d", slot), "speech refused: the text and the reference take %d of %d cells",
                  e.prompt_tokens, e.n_ctx);
        return SpeechResult{400, {}, context_overflow(e.prompt_tokens, e.n_ctx)};
    }
    if (!ok) {
        // Everything a client can get wrong was refused in prepare_speech, so
        // a failure here is the server's: a profile file that vanished, a
        // pipeline that refused the input, a decode error.
        log::error(log::format("slot %d", slot), "speech '%s': %s", prep.input.voice.c_str(), err.c_str());
        return SpeechResult{500, {}, error_body(err, "internal_error")};
    }
    log::info(log::format("slot %d", slot), "speech '%s': %d frames, %zu bytes WAV in %.2f s",
              prep.input.voice.c_str(), stats.completion_tokens, wav.size(), stats.decode_seconds);
    return SpeechResult{200, std::move(wav), json::object()};
}

// ----------------------------------------------------------------- streaming
void stream_chat(const Context& ctx, const PreparedChat& prep, int slot,
                 const SseWriter& write) {
    const std::string& model = prep.model;
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
    // Reasoning goes out as reasoning_content deltas until the think block
    // closes; from there `content` accumulates the answer and the tool-call
    // logic below sees only that.
    ReasoningStreamer   reasoning(prep.think_open, prep.think_tags_extended, prep.think_ids.open_close_ids);
    std::string         content;       // the answer part of the output so far
    // Reasoning has its own hold-back stream: a multi-byte code point must not
    // be split across two SSE chunks (DESIGN §3.7), and reasoning and content
    // are separate streams, so each carries its own pending state. The content
    // path above already does this with `utf8_stream`; the reasoning path did
    // not, which is how a torn code point reached the client as replacement
    // characters.
    utf8::Streamer      reasoning_stream;
    bool                had_reasoning = false;
    auto send_reasoning = [&](const std::string& safe) {
        had_reasoning = true;
        return sse_send(write, chunk(json::array({{{"index", 0},
                                                   {"delta", {{"reasoning_content", safe}}},
                                                   {"finish_reason", nullptr}}})));
    };
    auto emit_reasoning = [&](std::string_view bytes) {
        const std::string safe = reasoning_stream.push(bytes);
        if (safe.empty()) return true;
        return send_reasoning(safe);
    };
    // Emits whatever the reasoning streamer is still holding. Called when the
    // think block closes (before any content of that step) and at end of
    // stream, so a held code point is flushed into the reasoning stream in
    // order rather than leaking into content or being dropped.
    auto flush_reasoning = [&]() -> bool {
        const std::string held = reasoning_stream.flush();
        if (held.empty()) return true;
        return send_reasoning(held);
    };

    auto emit_content = [&](std::string_view bytes) {
        const std::string safe = utf8_stream.push(bytes);
        if (safe.empty()) return true;
        return sse_send(write, chunk(json::array({{{"index", 0},
                                                   {"delta", {{"content", safe}}},
                                                   {"finish_reason", nullptr}}})));
    };

    GenerationStats stats;
    FinishReason reason = FinishReason::Stop;
    if (auto err = guarded([&] { reason = drive(
        *ctx.backend, prep.input, slot,
        [&](std::string_view piece, int token_id, const std::string&) -> bool {
            const ReasoningStreamer::Step step = reasoning.push(piece, token_id);
            if (!emit_reasoning(step.reasoning)) {
                gone = true;
                return false;
            }
            if (!step.content.empty()) {
                // The think block closed in this step: flush the reasoning
                // streamer's held bytes before any content goes out, keeping
                // reasoning-then-content order intact.
                if (!flush_reasoning()) {
                    gone = true;
                    return false;
                }
            }
            if (step.content.empty()) return true;
            content += step.content;
            const std::string& accumulated = content;
            if (!prep.parse_tool_calls) {
                if (!emit_content(step.content)) {
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
        stats, raw); })) {
        // the 200 and the first chunk are out: the error goes in the stream
        sse_send(write, err->body);
        write("data: [DONE]\n\n");
        return;
    }

    log_stats(slot, stats, reason);
    if (gone) return;
    {
        const ReasoningStreamer::Step last = reasoning.flush();
        if (!emit_reasoning(last.reasoning)) return;
        if (!flush_reasoning()) return;
        if (!last.content.empty()) {
            content += last.content;
            if (!prep.parse_tool_calls && !emit_content(last.content)) return;
        }
    }

    std::vector<ToolCall> calls;
    if (prep.parse_tool_calls) {
        ToolCallParse parsed = parse_qwen_tool_calls(content, &prep.schemas);
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
    // Review B6: the non-streaming message carries reasoning_closed, and the
    // production clients stream; the final chunk carries it too, on the same
    // condition (reasoning was non-empty). A stream client can then tell a
    // closed-block empty answer from a mid-block stop.
    json final_delta = json::object();
    if (had_reasoning)
        final_delta["reasoning_closed"] = !reasoning.in_reasoning();
    if (!sse_send(write, chunk(json::array({{{"index", 0},
                                             {"delta", final_delta},
                                             {"finish_reason", finish}}})))) {
        return;
    }

    if (prep.req.stream_include_usage) {
        if (!sse_send(write, chunk(json::array(), json{{"usage", usage_json(stats)}}))) return;
    }
    write("data: [DONE]\n\n");
}

void stream_completion(const Context& ctx, const PreparedCompletion& prep, int slot,
                       const SseWriter& write) {
    const std::string& model = prep.model;
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
    FinishReason reason = FinishReason::Stop;
    if (auto err = guarded([&] { reason = drive(
        *ctx.backend, prep.input, slot,
        [&](std::string_view piece, int, const std::string&) -> bool {
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
        stats, raw); })) {
        // the 200 and the first chunk are out: the error goes in the stream
        sse_send(write, err->body);
        write("data: [DONE]\n\n");
        return;
    }

    log_stats(slot, stats, reason);
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

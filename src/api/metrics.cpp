#include "api/metrics.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "api/handlers.h"

namespace lgc::api {
namespace {

// The fixed routes, by the name the counter carries. A path outside this list
// and the cancel template folds to "other", so a scan cannot grow the label
// set without bound.
const char* const kKnownRoutes[] = {
    "/health",         "/props",          "/metrics",
    "/v1/models",      "/v1/chat/completions", "/v1/completions",
    "/v1/responses",   "/v1/responses/renderer", "/v1/responses/:id/cancel",
    "/v1/audio/speech", "/v1/audio/voices",
};

// Prometheus label-value escaping: backslash, double quote, newline.
std::string escape_label(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            default:   out += c; break;
        }
    }
    return out;
}

void add_help_type(std::string& out, const char* name, const char* help, const char* type) {
    out += "# HELP ";
    out += name;
    out += ' ';
    out += help;
    out += "\n# TYPE ";
    out += name;
    out += ' ';
    out += type;
    out += '\n';
}

void add_counter(std::string& out, const char* name, const char* help, uint64_t value) {
    add_help_type(out, name, help, "counter");
    out += name;
    out += ' ';
    out += std::to_string(value);
    out += '\n';
}

void add_seconds(std::string& out, const char* name, const char* help, double value) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6f", value);
    add_help_type(out, name, help, "counter");
    out += name;
    out += ' ';
    out += buf;
    out += '\n';
}

struct LaneGauge {
    std::string name;
    int         depth = 0;
    int         busy  = 0;
    int         total = 0;
};

}  // namespace

std::string metrics_route_label(std::string_view path) {
    for (const char* route : kKnownRoutes)
        if (path == route) return std::string(route);

    // /v1/responses/<id>/cancel folds to its registered template. A path with
    // an extra segment inside the id does not match the route either.
    constexpr std::string_view kPrefix = "/v1/responses/";
    constexpr std::string_view kSuffix = "/cancel";
    if (path.size() > kPrefix.size() + kSuffix.size() && path.rfind(kPrefix, 0) == 0 &&
        path.substr(path.size() - kSuffix.size()) == kSuffix) {
        const std::string_view id =
            path.substr(kPrefix.size(), path.size() - kPrefix.size() - kSuffix.size());
        if (!id.empty() && id.find('/') == std::string_view::npos)
            return "/v1/responses/:id/cancel";
    }
    return "other";
}

void Metrics::observe_request(std::string_view route, int status) {
    std::lock_guard<std::mutex> guard(mutex_);
    ++requests_[{std::string(route), status}];
}

void Metrics::observe_generation(const GenerationStats& stats) {
    std::lock_guard<std::mutex> guard(mutex_);
    prompt_tokens_     += static_cast<uint64_t>(std::max(0, stats.prompt_tokens));
    completion_tokens_ += static_cast<uint64_t>(std::max(0, stats.completion_tokens));
    cache_hit_tokens_  += static_cast<uint64_t>(std::max(0, stats.cache_hit_tokens));
    draft_proposed_    += static_cast<uint64_t>(std::max(0, stats.draft_proposed));
    draft_accepted_    += static_cast<uint64_t>(std::max(0, stats.draft_accepted));
    prefill_seconds_   += stats.prefill_seconds;
    decode_seconds_    += stats.decode_seconds;
}

std::string Metrics::render(const Context& ctx) const {
    std::map<std::pair<std::string, int>, uint64_t> requests;
    uint64_t prompt_tokens = 0, completion_tokens = 0, cache_hit_tokens = 0;
    uint64_t draft_proposed = 0, draft_accepted = 0;
    double   prefill_seconds = 0.0, decode_seconds = 0.0;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        requests         = requests_;
        prompt_tokens    = prompt_tokens_;
        completion_tokens = completion_tokens_;
        cache_hit_tokens = cache_hit_tokens_;
        draft_proposed   = draft_proposed_;
        draft_accepted   = draft_accepted_;
        prefill_seconds  = prefill_seconds_;
        decode_seconds   = decode_seconds_;
    }

    std::vector<LaneGauge> lanes;
    if (ctx.lanes.empty()) {
        if (ctx.slots != nullptr)
            lanes.push_back({"default", ctx.slots->queue_depth(),
                             ctx.slots->total() - ctx.slots->free(), ctx.slots->total()});
    } else {
        for (const Lane& lane : ctx.lanes) {
            if (lane.pool == nullptr) continue;
            lanes.push_back({lane.name, lane.pool->queue_depth(),
                             lane.pool->total() - lane.pool->free(), lane.pool->total()});
        }
    }

    std::string out;
    out.reserve(1024);

    add_help_type(out, "arcint_requests_total", "HTTP requests by route and status.", "counter");
    for (const auto& [key, value] : requests) {
        out += "arcint_requests_total{route=\"";
        out += escape_label(key.first);
        out += "\",status=\"";
        out += std::to_string(key.second);
        out += "\"} ";
        out += std::to_string(value);
        out += '\n';
    }

    add_counter(out, "arcint_prompt_tokens_total", "Prompt tokens processed.", prompt_tokens);
    add_counter(out, "arcint_completion_tokens_total", "Completion tokens generated.",
                completion_tokens);
    add_counter(out, "arcint_prompt_cache_hit_tokens_total",
                "Prompt tokens reused from the cache.", cache_hit_tokens);
    add_seconds(out, "arcint_prefill_seconds_total", "Seconds spent in prefill.", prefill_seconds);
    add_seconds(out, "arcint_decode_seconds_total", "Seconds spent in decode.", decode_seconds);
    add_counter(out, "arcint_draft_proposed_total", "Speculative tokens drafted.", draft_proposed);
    add_counter(out, "arcint_draft_accepted_total", "Speculative tokens accepted.",
                draft_accepted);

    auto lane_family = [&](const char* name, const char* help, auto pick) {
        add_help_type(out, name, help, "gauge");
        for (const LaneGauge& lane : lanes) {
            out += name;
            out += "{lane=\"";
            out += escape_label(lane.name);
            out += "\"} ";
            out += std::to_string(pick(lane));
            out += '\n';
        }
    };
    lane_family("arcint_queue_depth", "Requests waiting for a lane.",
                [](const LaneGauge& l) { return l.depth; });
    lane_family("arcint_slots_busy", "Lanes currently serving a request.",
                [](const LaneGauge& l) { return l.busy; });
    lane_family("arcint_slots_total", "Lanes in the pool.",
                [](const LaneGauge& l) { return l.total; });

    const ModelStatus& st = ctx.backend->status();
    add_help_type(out, "arcint_model_loaded", "1 when the backend reports a loaded model.",
                  "gauge");
    out += "arcint_model_loaded ";
    out += (st.loaded ? '1' : '0');
    out += '\n';

    return out;
}

}  // namespace lgc::api

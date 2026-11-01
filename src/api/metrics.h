#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include "exec/backend.h"

namespace lgc::api {

struct Context;

// The counters and gauges GET /metrics serves (--metrics). Every value is read
// from work arcint already does: the HTTP layer records the status per route,
// the generation handlers record the stats they already log. No step-loop work
// is added. Off by default: without --metrics the route is not registered and
// nothing is recorded.
class Metrics {
public:
    void observe_request(std::string_view route, int status);
    void observe_generation(const GenerationStats& stats);

    // The Prometheus text body. The counters live here; the gauges (queue
    // depth, slots busy/total per lane, model loaded) are read live from ctx.
    std::string render(const Context& ctx) const;

private:
    mutable std::mutex                              mutex_;
    std::map<std::pair<std::string, int>, uint64_t> requests_;
    uint64_t prompt_tokens_     = 0;
    uint64_t completion_tokens_ = 0;
    uint64_t cache_hit_tokens_  = 0;
    uint64_t draft_proposed_    = 0;
    uint64_t draft_accepted_    = 0;
    double   prefill_seconds_   = 0.0;
    double   decode_seconds_    = 0.0;
};

// The route label of a request path for arcint_requests_total: the fixed routes
// by name, /v1/responses/<id>/cancel folded to its template, every other path
// "other" (so a scan or a 404 cannot grow the label set without bound).
std::string metrics_route_label(std::string_view path);

}  // namespace lgc::api

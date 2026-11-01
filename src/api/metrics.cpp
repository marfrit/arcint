#include "api/metrics.h"

#include "api/handlers.h"

namespace lgc::api {

void Metrics::observe_request(std::string_view, int) {}

void Metrics::observe_generation(const GenerationStats&) {}

std::string Metrics::render(const Context&) const { return {}; }

std::string metrics_route_label(std::string_view path) { return std::string(path); }

}  // namespace lgc::api

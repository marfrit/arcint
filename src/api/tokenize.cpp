#include "api/tokenize.h"

namespace lgc::api {

// Scaffolding (red first): the routes exist and the bodies parse, but the
// behaviour the cases pin is not built yet. The tokens come back empty, the
// count is absent and the model field does not resolve a lane.
HttpResult tokenize_request(const Context&, const nlohmann::json&) {
    return HttpResult{200, nlohmann::json{{"tokens", nlohmann::json::array()}}};
}

HttpResult detokenize_request(const Context&, const nlohmann::json&) {
    return HttpResult{200, nlohmann::json{{"content", std::string()}}};
}

}  // namespace lgc::api

#include "api/renderer.h"

namespace lgc::api {

const nlohmann::json& renderer_material(const Context&) {
    static const nlohmann::json kEmpty;
    return kEmpty;
}

bool renderer_ready(const Context&) { return false; }

std::string renderer_hash(const Context&) { return {}; }

HttpResult renderer_bundle(const Context&) {
    return HttpResult{200, nlohmann::json{{"skeleton", true}}};
}

}  // namespace lgc::api

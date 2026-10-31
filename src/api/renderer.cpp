#include "api/renderer.h"

#include "api/error.h"

namespace lgc::api {

const nlohmann::json& renderer_material(const Context& ctx) {
    return ctx.backend->renderer_material();
}

bool renderer_ready(const Context& ctx) {
    const nlohmann::json& m = ctx.backend->renderer_material();
    return m.is_object() && m.contains("hash") && m["hash"].is_string() &&
           !m["hash"].get<std::string>().empty();
}

std::string renderer_hash(const Context& ctx) {
    if (!renderer_ready(ctx)) return {};
    return ctx.backend->renderer_material()["hash"].get<std::string>();
}

HttpResult renderer_bundle(const Context& ctx) {
    // No material: the unit cannot serve token mode (the stub, --tts), and the
    // answer is a loud 404 rather than an empty bundle a gateway could cache.
    if (!renderer_ready(ctx)) {
        return HttpResult{404, error_body("this unit serves no renderer material: token mode is "
                                          "off here (GET /props says token_mode: \"off\")",
                                          "invalid_request_error", "renderer_not_served")};
    }
    return HttpResult{200, ctx.backend->renderer_material()};
}

}  // namespace lgc::api

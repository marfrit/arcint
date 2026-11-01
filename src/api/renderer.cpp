#include "api/renderer.h"

#include "api/error.h"

namespace lgc::api {

bool renderer_ready(const Context& ctx) {
    return ctx.backend->renderer_capable();
}

std::string renderer_hash(const Context& ctx) {
    return ctx.backend->renderer_hash_cached();   // no build, try_lock in the backend
}

RendererBundle renderer_bundle(const Context& ctx) {
    // No capability: the unit cannot serve token mode (the stub, --tts), and
    // the answer is a loud 404 rather than an empty bundle a gateway could
    // cache. The envelope's type is the Responses surface's `not_found`
    // (W6a review finding 5).
    if (!ctx.backend->renderer_capable()) {
        return {404, nullptr,
                dump_json(responses_error("this unit serves no renderer material: token mode is "
                                          "off here (GET /props says token_mode: \"off\")",
                                          "not_found", "renderer_not_served"))};
    }
    // First use builds the material once and serializes it; the body is cached
    // in the backend, so a second GET copies nothing (finding 3).
    const std::string& body = ctx.backend->renderer_material_json();
    if (body.empty()) {
        return {404, nullptr,
                dump_json(responses_error("this unit's renderer material could not be built "
                                          "(the load log has the reason); token mode is off here",
                                          "not_found", "renderer_not_served"))};
    }
    return {200, &body, {}};
}

}  // namespace lgc::api

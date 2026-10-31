#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "api/handlers.h"

// feyntalk W6a (docs/campaigns/feyntalk.md, D4/D1): the renderer material
// surface. The backend owns the material; this module reads it for `/props`,
// for the request's hash check and for `GET /v1/responses/renderer`. No auth,
// like `/props`.
namespace lgc::api {

// The material the unit serves, or an empty json when it serves none (the stub).
const nlohmann::json& renderer_material(const Context& ctx);

// Whether this unit can serve token mode: material present with a hash.
bool renderer_ready(const Context& ctx);

// The served material's hash, empty when no material is served.
std::string renderer_hash(const Context& ctx);

// `GET /v1/responses/renderer`: 200 with the material, or 404
// `renderer_not_served` for a unit without one.
HttpResult renderer_bundle(const Context& ctx);

}  // namespace lgc::api

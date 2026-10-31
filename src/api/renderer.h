#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "api/handlers.h"

// feyntalk W6a (docs/campaigns/feyntalk.md, D4/D1) and its review round: the
// renderer material surface. The backend owns the lazy build and the cached
// body; this module reads the capability for `/props`, the cached hash for
// `/props`, and hands the endpoint the pre-serialized body by reference. No
// auth, like `/props`.
namespace lgc::api {

// Whether this unit can serve token mode: the capability the backend reports,
// without building the material (W6a review finding 3). `/props` reads this.
bool renderer_ready(const Context& ctx);

// The hash of the already-built material, empty before the first build. Never
// builds (W6a review finding 3: `/props` must not pay the ~13 MB body).
std::string renderer_hash(const Context& ctx);

// The material JSON, built once on first use; empty when not capable or the
// build failed. The request's hash check reads the hash from it.
const nlohmann::json& renderer_material(const Context& ctx);

// `GET /v1/responses/renderer`: the status, the cached serialized body by
// reference (200; owned by the backend) or a serialized error envelope (404
// `renderer_not_served`, type `not_found`). Never copies the 13 MB body.
struct RendererBundle {
    int                status = 200;
    const std::string* body   = nullptr;   // 200: the backend's cached body
    std::string        error;              // 4xx: the serialized error body
};
RendererBundle renderer_bundle(const Context& ctx);

}  // namespace lgc::api

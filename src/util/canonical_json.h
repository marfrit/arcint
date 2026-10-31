#pragma once

#include <string>

#include <nlohmann/json.hpp>

// feyntalk W6a (docs/campaigns/feyntalk.md, D4): the canonical JSON the
// renderer material's hash is taken over. A JCS-style subset — objects sorted
// by key at every level, arrays in order, no insignificant whitespace, `,` and
// `:` separators, non-ASCII unescaped, integers only — so the engine and the
// gateway recompute the same digest from the same material.
namespace lgc {

// Throws std::runtime_error on a floating-point number: the hashed material
// carries integers only, and a float would not be reproducible byte for byte.
std::string canonical_dump(const nlohmann::json& v);

// "sha256:" + lowercase hex of canonical_dump(v).
std::string canonical_sha256(const nlohmann::json& v);

}  // namespace lgc

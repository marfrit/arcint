#pragma once

#include <string>
#include <string_view>

// sha256 as lowercase hex, one self-contained implementation (W6a moved it from
// src/api/responses_state.cpp so the renderer material's canonical hash and the
// response store's prefix hash use the same function).
namespace lgc {
std::string sha256_hex(std::string_view bytes);
}  // namespace lgc

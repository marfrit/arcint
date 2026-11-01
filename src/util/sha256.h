#pragma once

#include <string>
#include <string_view>

// sha256 as lowercase hex, one self-contained implementation (W6a moved it from
// src/api/responses_state.cpp so the renderer material's canonical hash and the
// response store's prefix hash use the same function).
namespace lgc {
std::string sha256_hex(std::string_view bytes);

// sha256_hex, but the working copy of the input is zeroed before it is freed.
// For a secret (an API key) whose plaintext must not stay in memory; the plain
// sha256_hex does not pay the extra pass for the large, non-secret inputs
// (the renderer material's canonical hash).
std::string sha256_hex_secure(std::string_view bytes);
}  // namespace lgc

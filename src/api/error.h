#pragma once

#include <string_view>

#include <nlohmann/json.hpp>

namespace lgc::api {

// OpenAI-shaped error envelope: {"error": {message, type, param, code}}.
nlohmann::json error_body(std::string_view message, std::string_view type,
                          std::string_view code = {}, std::string_view param = {});

nlohmann::json invalid_request(std::string_view message, std::string_view param = {});

// DESIGN.md §3.8: overflow is a hard 400 that carries the numbers, so the
// client can do its own history management. No truncation, no context shift —
// a GDN recurrent state cannot un-see tokens.
nlohmann::json context_overflow(int prompt_tokens, int n_ctx);

}  // namespace lgc::api

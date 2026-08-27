#pragma once

#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace lgc::api {

// Serialises a response body.
//
// Model output can end on a truncated code point — utf8::Streamer::flush()
// hands back an incomplete sequence on purpose, because at end of stream those
// bytes are what the model produced and swallowing them would lose content.
// nlohmann's default dump() throws type_error.316 on invalid UTF-8, which the
// exception handler would turn into a 500: one torn character at the end of a
// long generation would discard the whole response. Replacing the bad bytes
// with U+FFFD keeps the response and makes the damage visible instead.
std::string dump_json(const nlohmann::json& body);

// OpenAI-shaped error envelope: {"error": {message, type, param, code}}.
nlohmann::json error_body(std::string_view message, std::string_view type,
                          std::string_view code = {}, std::string_view param = {});

nlohmann::json invalid_request(std::string_view message, std::string_view param = {});

// DESIGN.md §3.8: overflow is a hard 400 that carries the numbers, so the
// client can do its own history management. No truncation, no context shift —
// a GDN recurrent state cannot un-see tokens.
nlohmann::json context_overflow(int prompt_tokens, int n_ctx);

}  // namespace lgc::api

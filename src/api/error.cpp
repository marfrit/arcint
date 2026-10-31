#include "api/error.h"

#include "util/log.h"

namespace lgc::api {

std::string dump_json(const nlohmann::json& body) {
    return body.dump(-1, ' ', /*ensure_ascii=*/false,
                     nlohmann::json::error_handler_t::replace);
}

nlohmann::json error_body(std::string_view message, std::string_view type,
                          std::string_view code, std::string_view param) {
    nlohmann::json err;
    err["message"] = std::string(message);
    err["type"]    = std::string(type);
    err["param"]   = param.empty() ? nlohmann::json(nullptr) : nlohmann::json(std::string(param));
    err["code"]    = code.empty() ? nlohmann::json(nullptr) : nlohmann::json(std::string(code));
    return nlohmann::json{{"error", std::move(err)}};
}

nlohmann::json invalid_request(std::string_view message, std::string_view param) {
    return error_body(message, "invalid_request_error", {}, param);
}

nlohmann::json context_overflow(int prompt_tokens, int n_ctx) {
    const int overflow = prompt_tokens - n_ctx;

    nlohmann::json body = error_body(
        log::format("prompt is %d tokens, context is %d: %d over. arcint does not truncate or "
                    "shift context; compact the history client-side and retry.",
                    prompt_tokens, n_ctx, overflow),
        "invalid_request_error", "context_length_exceeded", "messages");

    body["error"]["prompt_tokens"] = prompt_tokens;
    body["error"]["n_ctx"]         = n_ctx;
    body["error"]["overflow"]      = overflow;
    return body;
}

nlohmann::json context_overflow_truncated(int prompt_tokens, int n_ctx, int reserve,
                                         int blocks_dropped, int still_over) {
    nlohmann::json body = error_body(
        log::format("reasoning dropped: %d blocks, still %d over. prompt is %d tokens, context is "
                    "%d, generation reserve %d: compaction client-side is still needed and retry.",
                    blocks_dropped, still_over, prompt_tokens, n_ctx, reserve),
        "invalid_request_error", "context_length_exceeded", "messages");
    body["error"]["prompt_tokens"] = prompt_tokens;
    body["error"]["n_ctx"]         = n_ctx;
    body["error"]["overflow"]      = still_over;
    body["error"]["fit_reserve"]   = reserve;
    body["error"]["reasoning_dropped_blocks"] = blocks_dropped;
    return body;
}

nlohmann::json responses_error(std::string_view message, std::string_view type,
                               std::string_view code, std::string_view param) {
    // Same envelope as error_body, named separately so the chat path's bytes
    // cannot move when the Responses spellings change.
    return error_body(message, type, code, param);
}

}  // namespace lgc::api

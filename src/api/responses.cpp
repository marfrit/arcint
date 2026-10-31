#include "api/responses.h"

#include "api/error.h"

namespace lgc::api {

// RED skeleton: the mapping is not written yet, so parsing reports success
// with nothing mapped and the cases fail on their expectations.
std::optional<HttpResult> parse_response_request(const nlohmann::json&, ResponseRequest&) {
    return std::nullopt;
}

std::optional<HttpResult> prepare_response(const Context&, ResponseRequest, PreparedResponse&) {
    return HttpResult{400, responses_error("not implemented", "invalid_request")};
}

nlohmann::json response_resource(const ResponseRequest&, const PreparedChat&,
                                 const ChatGeneration&) {
    return nlohmann::json::object();
}

HttpResult run_response(const Context&, const PreparedResponse&, int) {
    return HttpResult{500, responses_error("not implemented", "server_error", "internal_error")};
}

}  // namespace lgc::api

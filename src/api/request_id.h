// X-Request-Id: one id per HTTP request, so a caller's log line and arcint's
// own log lines for the same request can be matched.
//
// The id is resolved once, in the server's pre-routing handler, and travels on
// the httplib::Response (a header) into the route lambda, which copies it into
// the prepared request struct; every log line of that request appends
// ` rid=<id>`. No globals: the id lives on the objects the request already
// owns.
#ifndef ARCINT_API_REQUEST_ID_H
#define ARCINT_API_REQUEST_ID_H

#include <string>
#include <string_view>

namespace lgc::api {

// The header name, spelled once. httplib lowercases inbound header names for
// lookup but keeps the case it was given for output, so the outbound spelling
// is this one.
inline constexpr const char* kRequestIdHeader = "X-Request-Id";

// A generated id: "req-" + 16 lowercase hex characters from a per-thread
// entropy stream. Random, not a counter: two requests that race must not
// collide, and a counter would leak the request rate.
std::string new_request_id();

struct RequestId {
    std::string id;
    // True when the client sent something and it was not usable, so `id` is a
    // generated replacement. The value the client sent is never kept in this
    // struct: it may carry control characters, and it must not reach a log.
    bool replaced = false;
};

// The inbound value is kept only when it is 1..128 characters, every one of
// them in [A-Za-z0-9._:-]. Anything else - absent, empty, longer, a space, a
// control character, a non-ASCII byte - is replaced by new_request_id() with
// replaced = true. A replacement is not an error: the request is served.
RequestId resolve_request_id(std::string_view inbound);

// " rid=<id>" for a log line, or "" when there is no id (a line emitted
// outside a request, or a response path that never resolved one).
std::string rid_suffix(std::string_view rid);

}  // namespace lgc::api

#endif  // ARCINT_API_REQUEST_ID_H

#include "api/request_id.h"

#include "util/log.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

namespace lgc::api {
namespace {

// The grammar an inbound id must satisfy to be taken over: 1..128 characters
// from [A-Za-z0-9._:-]. Deliberately narrower than "any header value": it
// excludes SP, HTAB, CR, LF, DEL, every non-ASCII byte and every character that
// could be read as a log-field separator, so a value a caller supplies can
// never forge a log field or split a line.
bool usable(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
           c == '_' || c == ':' || c == '-';
}

}  // namespace

std::string new_request_id() {
    // Per-thread on purpose: mt19937_64 is not thread-safe, and a shared
    // generator would need a lock on the request path. Seeded once per thread
    // from the device's entropy, not from a clock or a counter.
    thread_local std::mt19937_64 rng {std::random_device {}()};
    std::string                  id;
    id.reserve(20);
    id += "req-";
    static constexpr char kHex[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    // 16 hex characters = 64 bits, so exactly one draw fills the tail.
    const std::uint64_t v = rng();
    for (int b = 15; b >= 0; --b) id += kHex[(v >> (4 * b)) & 0xFu];
    return id;
}

RequestId resolve_request_id(std::string_view inbound) {
    if (inbound.size() >= 1 && inbound.size() <= 128) {
        bool ok = true;
        for (const char c : inbound) {
            if (!usable(c)) {
                ok = false;
                break;
            }
        }
        if (ok) return RequestId {std::string (inbound), false};
    }
    return RequestId {new_request_id(), !inbound.empty()};
}

std::string rid_suffix(std::string_view rid) {
    return rid.empty() ? std::string {} : log::format(" rid=%.*s", static_cast<int>(rid.size()),
                                                      rid.data());
}

}  // namespace lgc::api

#include "api/auth.h"

#include "util/sha256.h"

namespace lgc::api {

std::optional<std::string> read_api_key_file(const std::string&, std::string& error) {
    error = "not implemented";
    return std::nullopt;
}

bool constant_time_equal(std::string_view a, std::string_view b) {
    (void)a;
    (void)b;
    return false;
}

bool bearer_matches(std::string_view authorization_header, std::string_view key_sha256_hex) {
    (void)authorization_header;
    (void)key_sha256_hex;
    return false;
}

}  // namespace lgc::api

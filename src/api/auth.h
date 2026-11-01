#pragma once

#include <optional>
#include <string>
#include <string_view>

// The bearer-key half of the serving surface (--api-key / --api-key-file).
// Kept out of the HTTP layer so the file rule and the constant-time digest
// comparison are unit-testable without a server.
namespace lgc::api {

// The key file's bytes with exactly one trailing '\n' removed and nothing else
// (a lone LF; a CRLF keeps its '\r'). nullopt and a message in `error` when the
// file cannot be read. The file is read once, at boot.
std::optional<std::string> read_api_key_file(const std::string& path, std::string& error);

// Constant-time equality: no early exit, unequal lengths fold into the result
// instead of returning at the first difference. The callers compare fixed
// 64-char sha256 hex digests, so the length itself is not secret either.
bool constant_time_equal(std::string_view a, std::string_view b);

// Whether an Authorization header presents `Bearer <key>` (scheme
// case-insensitive) whose sha256 equals the stored digest. The presented key is
// hashed here and the digests compared in constant time; the stored key itself
// is never needed at request time.
bool bearer_matches(std::string_view authorization_header, std::string_view key_sha256_hex);

}  // namespace lgc::api

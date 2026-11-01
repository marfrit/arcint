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

// Boot hygiene for a configured key. False + `error` means the server must not
// start: an empty key (an empty or newline-only --api-key-file), one with a
// control character (below 0x20, or DEL), or one that starts or ends with a
// SPACE (cpp-httplib trims header values, so a client can never present it).
// True with a `warning` for a legal key that is still a footgun: a '%'
// (cpp-httplib URL-decodes header values, so a key with `%xx` can never match
// a client that sends it verbatim).
bool api_key_hygiene(std::string_view key, std::string& error, std::string& warning);

// A boot warning when the key file is group- or world-accessible (mode & 077),
// naming the mode; empty when the file is private or cannot be stat'd (the
// read already reported a missing file). Only the file's own mode is in it.
std::string api_key_file_warning(const std::string& path);

// Overwrite a string's storage with zeros before clearing it, so a plaintext
// key does not stay in this buffer's freed memory. volatile, so the compiler
// cannot elide the writes as dead stores. The server keeps only the sha256
// digest in its long-lived state; this scrubs the buffers it owns (the file
// read's buffer, the parsed key, the hash's working copy), not every copy C++
// made on the way (the ifstream's, the string's reallocations, a client's
// presented key).
void secure_scrub(std::string& s);

}  // namespace lgc::api

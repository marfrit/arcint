#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

// The allowlist pins artifact identity by sha256 prefix (models/allowlist-raw.json
// records the first 16 hex characters). Implemented here rather than pulled from
// OpenSSL: it is 100 lines, it keeps the build hermetic, and hashing a few files
// at load time is not a performance question.
namespace lgc {

class Sha256 {
public:
    Sha256() { reset(); }

    void        reset();
    void        update(const void* data, size_t len);
    void        update(std::string_view s) { update(s.data(), s.size()); }
    std::string hex();  // finalises; the object must be reset() to reuse

private:
    void transform(const uint8_t block[64]);

    uint64_t bit_length_ = 0;
    uint32_t state_[8]{};
    uint8_t  buffer_[64]{};
    size_t   buffered_ = 0;
};

std::string sha256_hex(std::string_view data);

// Full hex digest of a file, or an empty string if it cannot be read.
std::string sha256_file(const std::string& path);

// The first `n` characters of a digest — the allowlist's pinned form.
std::string hash_prefix(const std::string& hex, size_t n = 16);

}  // namespace lgc

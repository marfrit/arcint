#include "api/auth.h"

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>

#include "util/sha256.h"

namespace lgc::api {
namespace {

// ASCII case-insensitive prefix test for the auth scheme ("Bearer " is
// case-insensitive per RFC 9110; the token itself is not).
bool ascii_iequal_prefix(std::string_view text, std::string_view prefix) {
    if (text.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        char a = text[i];
        char b = prefix[i];
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

}  // namespace

std::optional<std::string> read_api_key_file(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open the key file";
        return std::nullopt;
    }
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) {
        error = "cannot read the key file";
        return std::nullopt;
    }
    // Exactly one trailing newline, and nothing else: a key that ends in '\n'
    // was written by `echo`; two were written by mistake and stay part of the
    // key.
    if (!bytes.empty() && bytes.back() == '\n') bytes.pop_back();
    return bytes;
}

bool constant_time_equal(std::string_view a, std::string_view b) {
    // No early exit. The length difference folds into `diff`, so the loop runs
    // over the longer input either way; the callers compare same-length sha256
    // digests, so this only makes the primitive safe rather than clever.
    const size_t  n    = a.size() > b.size() ? a.size() : b.size();
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(x ^ y);
    }
    return diff == 0;
}

bool bearer_matches(std::string_view authorization_header, std::string_view key_sha256_hex) {
    constexpr std::string_view kScheme = "Bearer ";
    if (!ascii_iequal_prefix(authorization_header, kScheme)) return false;
    const std::string_view presented = authorization_header.substr(kScheme.size());
    if (presented.empty()) return false;
    const std::string digest = sha256_hex(presented);
    return constant_time_equal(digest, key_sha256_hex);
}

bool api_key_hygiene(std::string_view key, std::string& error, std::string& warning) {
    error.clear();
    warning.clear();
    if (key.empty()) {
        // An empty --api-key-file (or one holding only a newline, whose one
        // trailing newline read_api_key_file strips) would leave the server
        // unauthenticated although the operator asked for auth.
        error = "the API key is empty: an empty --api-key-file (or one holding only a newline) "
                "would serve every request unauthenticated; fix the file or drop the key flags";
        return false;
    }
    if (key.find('\r') != std::string_view::npos) {
        error = "the API key contains a carriage return: it can never be presented in an HTTP "
                "header (a CRLF key file?); write the file with a single LF or none";
        return false;
    }
    auto is_space = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
    };
    const bool edge_space = is_space(key.front()) || is_space(key.back());
    const bool percent    = key.find('%') != std::string_view::npos;
    if (edge_space && percent) {
        warning = "the API key has leading or trailing whitespace and contains a percent sign "
                  "('%'): clients must send it verbatim, and cpp-httplib URL-decodes header "
                  "values, so neither form can be relied on";
    } else if (edge_space) {
        warning = "the API key has leading or trailing whitespace: clients must send it "
                  "verbatim, and whitespace in a bearer token is easy to corrupt";
    } else if (percent) {
        warning = "the API key contains a percent sign ('%'): cpp-httplib URL-decodes header "
                  "values, so a %xx escape can never match a client that sends it verbatim";
    }
    return true;
}

void secure_scrub(std::string& s) {
    if (!s.empty()) {
        // volatile: the compiler may not drop the stores as dead.
        volatile char* p = s.data();
        for (size_t i = 0; i < s.size(); ++i) p[i] = '\0';
    }
    s.clear();
}

}  // namespace lgc::api

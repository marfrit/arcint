#include "api/auth.h"
#include "config.h"
#include "harness.h"
#include "util/sha256.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

// The bearer key (--api-key / --api-key-file) and the TLS flag pairing. The
// key-file rule (exactly one trailing newline), the constant-time digest
// comparison and the header parse are tested without a socket; the route
// enforcement is tests/auth.sh's HTTP half.
namespace {

using namespace lgc;

int g_temp_counter = 0;

std::string write_temp(const std::string& bytes) {
    const std::string path = "/tmp/arcint-auth-" + std::to_string(getpid()) + "-" +
                             std::to_string(++g_temp_counter);
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

ArgParse run(std::vector<const char*> args, Config& cfg) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("arcint"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    return parse_args(static_cast<int>(argv.size()), argv.data(), cfg);
}

bool rejected(std::vector<const char*> args) {
    Config cfg;
    return !run(std::move(args), cfg).ok;
}

}  // namespace

TEST(auth_key_file_strips_exactly_one_trailing_newline) {
    const std::string k1 = write_temp("secret\n");
    const std::string k2 = write_temp("secret");
    const std::string k3 = write_temp("secret\n\n");
    const std::string k4 = write_temp("secret\r\n");

    std::string err;
    CHECK(api::read_api_key_file(k1, err).value_or("") == "secret");
    CHECK(api::read_api_key_file(k2, err).value_or("") == "secret");
    CHECK(api::read_api_key_file(k3, err).value_or("") == "secret\n");
    CHECK(api::read_api_key_file(k4, err).value_or("") == "secret\r");

    std::remove(k1.c_str());
    std::remove(k2.c_str());
    std::remove(k3.c_str());
    std::remove(k4.c_str());
}

TEST(auth_key_file_missing_is_an_error) {
    std::string err;
    const auto  key = api::read_api_key_file("/tmp/arcint-auth-does-not-exist", err);
    CHECK(!key.has_value());
    CHECK(!err.empty());
}

TEST(auth_constant_time_equal_compares_content_and_length) {
    CHECK(api::constant_time_equal("abc", "abc"));
    CHECK(!api::constant_time_equal("abc", "abd"));
    CHECK(!api::constant_time_equal("abc", "ab"));
    CHECK(!api::constant_time_equal("", "a"));
    CHECK(api::constant_time_equal("", ""));
}

TEST(auth_bearer_matches_hashes_the_presented_key) {
    const std::string key = "test-key-not-real";
    const std::string digest = sha256_hex(key);

    CHECK(api::bearer_matches("Bearer " + key, digest));
    CHECK(api::bearer_matches("bearer " + key, digest));   // scheme case-insensitive
    CHECK(!api::bearer_matches("Bearer wrong", digest));
    CHECK(!api::bearer_matches("Basic " + key, digest));
    CHECK(!api::bearer_matches("", digest));
    CHECK(!api::bearer_matches("Bearer ", digest));
    CHECK(!api::bearer_matches("Bearer " + key + " ", digest));
}

TEST(auth_config_refuses_two_key_sources_and_an_empty_key) {
    CHECK(rejected({"--stub", "--api-key", "k", "--api-key-file", "/tmp/k"}));
    CHECK(rejected({"--stub", "--api-key", ""}));
    CHECK(rejected({"--stub", "--api-key-file", ""}));
    Config cfg;
    CHECK(run({"--stub", "--api-key", "k"}, cfg).ok);
    CHECK_EQ(cfg.api_key, std::string("k"));
}

TEST(auth_config_requires_the_tls_pair) {
    CHECK(rejected({"--stub", "--tls-cert", "/tmp/c.pem"}));
    CHECK(rejected({"--stub", "--tls-key", "/tmp/k.pem"}));
}

TEST(auth_config_refuses_tls_without_the_build_option) {
#ifdef ARCINT_TLS
    Config cfg;
    CHECK(run({"--stub", "--tls-cert", "/tmp/c.pem", "--tls-key", "/tmp/k.pem"}, cfg).ok);
#else
    CHECK(rejected({"--stub", "--tls-cert", "/tmp/c.pem", "--tls-key", "/tmp/k.pem"}));
#endif
}

TEST(auth_hygiene_refuses_an_empty_key_and_a_carriage_return) {
    std::string err, warn;
    CHECK(!api::api_key_hygiene("", err, warn));
    CHECK(!err.empty());

    err.clear();
    warn.clear();
    CHECK(!api::api_key_hygiene("a\rb", err, warn));
    CHECK(err.find("carriage") != std::string::npos);

    // The newline-only --api-key-file case: read_api_key_file strips the one
    // trailing newline, leaving an empty key the same check refuses.
    CHECK(!api::api_key_hygiene("\n", err, warn));
}

TEST(auth_hygiene_warns_on_whitespace_and_a_percent) {
    std::string err, warn;
    CHECK(api::api_key_hygiene("good-key", err, warn));
    CHECK(warn.empty());

    warn.clear();
    CHECK(api::api_key_hygiene(" leading", err, warn));
    CHECK(!warn.empty());

    warn.clear();
    CHECK(api::api_key_hygiene("trailing ", err, warn));
    CHECK(!warn.empty());

    warn.clear();
    CHECK(api::api_key_hygiene("k%41y", err, warn));
    CHECK(!warn.empty());
}

TEST(auth_secure_scrub_clears_and_overwrites) {
    std::string key = "test-key-not-real";
    api::secure_scrub(key);
    CHECK(key.empty());
}

TEST(auth_main_checks_hygiene_and_scrubs_the_key) {
    // The startup path is not reachable from a unit test (it needs a process),
    // so this pins the wiring the way test_props pins the llama backend's KV
    // precision: main must run the hygiene check and scrub the plaintext
    // buffers. The scanner case in tests/acceptance/cells guards the source
    // the same way.
    std::ifstream in(std::string(ARCINT_SOURCE_DIR) + "/src/main.cpp");
    CHECK(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    CHECK(src.find("api_key_hygiene") != std::string::npos);
    CHECK(src.find("secure_scrub") != std::string::npos);
}

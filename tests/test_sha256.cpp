#include "harness.h"
#include "util/sha256.h"

using namespace lgc;

// NIST/RFC test vectors. Without these the hash could be self-consistently
// wrong, and the allowlist would be validating against ligence's own bug.
TEST(sha256_known_vectors) {
    CHECK_EQ(sha256_hex(""),
             std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    CHECK_EQ(sha256_hex("abc"),
             std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    CHECK_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
             std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
}

TEST(sha256_spans_block_boundaries) {
    // 55, 56, 63, 64 and 65 bytes: the padding cases that trip naive
    // implementations.
    const std::string a(55, 'a');
    const std::string b(56, 'a');
    const std::string c(64, 'a');
    for (const std::string& s : {a, b, c}) {
        Sha256 h;
        h.update(s);
        const std::string one_shot = h.hex();
        CHECK_EQ(one_shot.size(), 64u);
        CHECK_EQ(one_shot, sha256_hex(s));
    }
    CHECK_EQ(sha256_hex(std::string(1000000, 'a')),
             std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
}

TEST(sha256_incremental_matches_one_shot) {
    const std::string text = "the quick brown fox jumps over the lazy dog, repeatedly, at length";

    Sha256 h;
    for (size_t i = 0; i < text.size(); i += 7) h.update(text.substr(i, 7));
    CHECK_EQ(h.hex(), sha256_hex(text));
}

TEST(sha256_prefix_is_the_allowlist_form) {
    const std::string full = sha256_hex("abc");
    CHECK_EQ(hash_prefix(full), std::string("ba7816bf8f01cfea"));
    CHECK_EQ(hash_prefix(full).size(), 16u);
    CHECK_EQ(hash_prefix("short"), std::string("short"));
}

TEST(sha256_missing_file_is_empty_not_a_lie) {
    // An unreadable file must not produce the digest of nothing, or a missing
    // artifact would validate against the hash of an empty string.
    CHECK_EQ(sha256_file("/nonexistent/definitely/not/here"), std::string(""));
    CHECK(sha256_file("/nonexistent/x") != sha256_hex(""));
}

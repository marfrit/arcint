// The draft vocabulary file parser (src/exec/token_ids.h): both formats,
// sorting and deduplication, and what it refuses.
#include "exec/token_ids.h"
#include "harness.h"

#include <string>
#include <vector>

using namespace lgc;

namespace {

std::string int32s(const std::vector<int32_t>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(int32_t));
}

}  // namespace

TEST(token_ids_json_list_sorted_unique) {
    std::vector<int32_t> ids;
    std::string          err;
    CHECK(parse_token_ids("[5, 3,\n 9, 3]", true, 10, ids, err));
    CHECK_EQ(ids.size(), static_cast<size_t>(3));
    CHECK_EQ(ids[0], 3);
    CHECK_EQ(ids[1], 5);
    CHECK_EQ(ids[2], 9);
}

TEST(token_ids_int32_file) {
    std::vector<int32_t> ids;
    std::string          err;
    CHECK(parse_token_ids(int32s({7, 0, 248319}), false, 248320, ids, err));
    CHECK_EQ(ids.size(), static_cast<size_t>(3));
    CHECK_EQ(ids[0], 0);
    CHECK_EQ(ids[2], 248319);
}

TEST(token_ids_refuses_out_of_vocabulary_and_malformed) {
    std::vector<int32_t> ids;
    std::string          err;
    CHECK(!parse_token_ids("[1, 10]", true, 10, ids, err));         // 10 is past the vocabulary
    CHECK(!parse_token_ids(int32s({-1}), false, 10, ids, err));     // negative
    CHECK(!parse_token_ids("[1, -2]", true, 10, ids, err));         // not a non-negative integer
    CHECK(!parse_token_ids("{\"a\": 1}", true, 10, ids, err));      // not a list
    CHECK(!parse_token_ids(std::string(6, '\0'), false, 10, ids, err));   // not whole int32s
    CHECK(!parse_token_ids("[]", true, 10, ids, err));              // empty
}

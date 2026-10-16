// The drafter state a context checkpoint carries (src/exec/spec_state.h):
// the opaque bytes come back as they went in, and only to the drafter kind
// and the token count they were taken for.
#include "exec/spec_state.h"
#include "harness.h"

#include <cstring>
#include <vector>

using namespace lgc;

namespace {

// a payload that a truncation or an offset would change: every byte distinct
std::vector<uint8_t> ramp(size_t n) {
    std::vector<uint8_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = static_cast<uint8_t>((i * 131 + 7) & 0xff);
    return p;
}

}  // namespace

TEST(spec_state_round_trip_returns_the_bytes) {
    for (size_t n : {size_t{0}, size_t{1}, size_t{4 * 5120}, size_t{100003}}) {
        const std::vector<uint8_t> payload = ramp(n);
        const std::vector<uint8_t> s       = spec_state_pack(SpecStateKind::DflashWindow, 4096, payload.data(), n);
        CHECK_EQ(s.size(), kSpecStateHeader + n);
        SpecStateView v;
        CHECK(spec_state_view(s, SpecStateKind::DflashWindow, 4096, v));
        CHECK_EQ(v.size, n);
        CHECK(std::vector<uint8_t>(v.data, v.data + v.size) == payload);
    }
}

TEST(spec_state_mtp_row_round_trip) {
    // the MTP drafter's carried row: floats through the same bytes
    std::vector<float> row(5120);
    for (size_t i = 0; i < row.size(); ++i) row[i] = static_cast<float>(i) * 0.25f - 300.0f;
    const std::vector<uint8_t> s = spec_state_pack(SpecStateKind::MtpRow, 77, row.data(), row.size() * sizeof(float));
    SpecStateView v;
    CHECK(spec_state_view(s, SpecStateKind::MtpRow, 77, v));
    CHECK_EQ(v.size, row.size() * sizeof(float));
    std::vector<float> back(v.size / sizeof(float));
    std::memcpy(back.data(), v.data, v.size);
    CHECK(back == row);
}

TEST(spec_state_alloc_leaves_the_payload_to_fill) {
    std::vector<uint8_t> s = spec_state_alloc(SpecStateKind::DflashWindow, 9, 3);
    s[kSpecStateHeader + 0] = 1;
    s[kSpecStateHeader + 2] = 3;
    SpecStateView v;
    CHECK(spec_state_view(s, SpecStateKind::DflashWindow, 9, v));
    CHECK(std::vector<uint8_t>(v.data, v.data + v.size) == std::vector<uint8_t>({1, 0, 3}));
}

TEST(spec_state_refuses_another_drafter_or_position) {
    const std::vector<uint8_t> payload = ramp(64);
    const std::vector<uint8_t> s       = spec_state_pack(SpecStateKind::DflashWindow, 4096, payload.data(), 64);
    SpecStateView v;
    CHECK(!spec_state_view(s, SpecStateKind::MtpRow, 4096, v));         // another drafter's
    CHECK(v.data == nullptr && v.size == 0);
    CHECK(!spec_state_view(s, SpecStateKind::DflashWindow, 4095, v));   // another checkpoint's position
    CHECK(!spec_state_view({}, SpecStateKind::DflashWindow, 4096, v));  // none stored
    std::vector<uint8_t> cut = s;
    cut.pop_back();
    CHECK(!spec_state_view(cut, SpecStateKind::DflashWindow, 4096, v));   // truncated
    std::vector<uint8_t> pad = s;
    pad.push_back(0);
    CHECK(!spec_state_view(pad, SpecStateKind::DflashWindow, 4096, v));   // padded
    std::vector<uint8_t> bad = s;
    bad[0] ^= 1;
    CHECK(!spec_state_view(bad, SpecStateKind::DflashWindow, 4096, v));   // not a state
    CHECK(spec_state_view(s, SpecStateKind::DflashWindow, 4096, v));      // the original still reads
}

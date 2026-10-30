// Device-free cells for patch 0070's STRP save: tools/expert_profile_save_selftest.cpp
// compiles against the llama tree's src directory (EXPERT_PROFILE_SRC); nothing
// here names a host or a path.
//
// The round trip gives the same placement. A synthetic geometry (3 layers x 4
// experts) with the live residents {(0,0),(0,2)} in layer 0's two slots and
// (1,1) in layer 1's one slot is ranked by Strata's rule
// (rank_learned_profile, src/tests/core/expert_profile_save_test.cpp, commit
// 3d6f1718), written, read back, and the placement derived from the saved
// ranking (each layer's first n_slots pairs) is compared with the live
// residents. The in-memory ranking is also compared with the reference test's
// expected head/order, so the round trip cannot pass by being equally wrong.
//
// A second cell: a file cut at the last pair (header + n_ranked-1 records) is
// refused, and no partial ranking is returned.
//
// The version cell: the version field rewritten to 2 is refused, naming the
// file, version 2 and version 1, checked before the geometry (Strata's
// _version_test).
//
// RED MUTATIONS, each run and seen failing before the fix was restored:
//  (a) rank's heat comparator `>` -> `<` (both resident and counted groups):
//      the same-placement cell fails -- layer 0's two saved slots become
//      (0,2),(0,0) against the live (0,0),(0,2), and the expected ranking fails
//      with it. Observed: "FAIL: the ranking is Strata's rule" and
//      "FAIL: the saved ranking gives the same placement".
//  (b) drop the version check from llama_expert_profile_read: the version cell
//      fails (a version 2 file loads). Observed: "FAIL: a version 2 profile is
//      refused".
//  (c) accept a truncated record in the reader (skip the file's size check and
//      break out of the pair loop on a short fread): the truncation cell fails
//      (a partial ranking is returned). Observed: "FAIL: a truncated profile is
//      refused", "FAIL: a refusal is no partial ranking" and "FAIL: the
//      truncation refusal says truncated".
//  (d) drop the prior-rank tie-break (compare the index alone): the tie cell
//      fails -- two same-heat pairs come out in index order instead of the
//      prior's. Observed: "FAIL: ties take the prior's rank before the index".
//  (e) drop the pending incoming mark in llama_expert_profile_mark_pending:
//      the pending cell fails (the incoming expert is not resident).
//      Observed: "FAIL: a pending incoming expert is resident".
//  (f) clear the outgoing expert in phase 1 too: the pending cell fails (an
//      outgoing expert is kept resident only in phase 1, not cleared there).
//      Observed: "FAIL: an outgoing expert stays resident in phase 1".
//  (g) xe: pass prior_entries through as 0 (drop the prior mass): the xe cell
//      fails -- a 9-count pair outranks the prior's first pair. Observed:
//      "FAIL: xe mixes the prior in as pseudo-counts".
//  (h) heat counts a decayed weight instead of a raw routing event
//      (`+= 1.0` -> `+= 1.0 / (1.0 - 0.7)` in llama_expert_profile_add_heat):
//      the heat cell fails. Observed: "FAIL: heat counts one a routed pair, raw".
//  (i) heat narrowed to float (`heat[e] += (float) 1.0` over a float copy, or
//      the vector typed float): the precision cell fails. Observed:
//      "FAIL: heat keeps a count past float's 2^24".
//
// Usage: expert_profile_save_selftest [profile-path]
// Without a path it uses the temp directory and removes it at the end; with a
// path it leaves the version-1 file there for the Python test to read.
#include "llama-expert-profile.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}

bool set_version(const std::string & path, uint32_t v) {
    std::FILE * f = std::fopen(path.c_str(), "r+b");
    if (f == nullptr) return false;
    uint8_t le[4];
    for (int b = 0; b < 4; ++b) le[b] = (uint8_t) ((v >> (8 * b)) & 0xff);
    const bool ok = std::fseek(f, 4, SEEK_SET) == 0 && std::fwrite(le, 1, 4, f) == 4;
    return std::fclose(f) == 0 && ok;
}

bool has(const std::string & s, const std::string & what) { return s.find(what) != std::string::npos; }

// each layer's first n_slots pairs in the ranking: the placement the cache
// makes from the profile (the first pairs of each layer fill its slots)
std::vector<std::pair<int, int>> placement(const std::vector<std::pair<int, int>> & ranked,
                                           const std::vector<int> & n_slots) {
    std::vector<std::pair<int, int>> out;
    std::vector<int> taken(n_slots.size(), 0);
    for (const auto & p : ranked) {
        if (p.first < 0 || p.first >= (int) n_slots.size()) continue;
        if (taken[p.first] >= n_slots[p.first]) continue;
        ++taken[p.first];
        out.push_back(p);
    }
    // the live set is written layer-major (each layer's slots in their order);
    // the ranking is global, so group it the same way before comparing
    std::stable_sort(out.begin(), out.end(),
                     [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.first < b.first; });
    return out;
}
}  // namespace

int main(int argc, char ** argv) {
    using Pair = std::pair<int, int>;
    const int L = 3, E = 4;
    std::string path;
    bool keep = false;
    if (argc > 1) {
        path = argv[1];
        keep = true;
    } else {
        path = (std::filesystem::temp_directory_path() / "expert_profile_save_selftest.bin").string();
    }

    // (0,0) and (0,2) in layer 0's two slots, (1,1) in layer 1's one slot
    std::vector<Pair> live = {{0, 0}, {0, 2}, {1, 1}};
    const std::vector<int> n_slots = {2, 1, 0};
    std::vector<uint8_t> resident((size_t) L * E, 0);
    for (const auto & p : live) resident[(size_t) p.first * E + p.second] = 1;
    std::vector<double> heat((size_t) L * E, 0.0);
    heat[(size_t) 0 * E + 0] = 3.0;    // (0,0), resident
    heat[(size_t) 0 * E + 2] = 1.0;    // (0,2), resident
    heat[(size_t) 1 * E + 1] = 7.0;    // (1,1), resident
    heat[(size_t) 2 * E + 3] = 9.0;    // (2,3), hot but not resident
    const std::vector<Pair> prior = {{2, 0}, {1, 3}, {0, 1}};

    // Strata's save test's order transplanted to this set: resident by heat,
    // the counted non-resident by heat, the prior, then the layer-major fill
    const std::vector<Pair> exp = {
        {1, 1}, {0, 0}, {0, 2}, {2, 3}, {2, 0}, {1, 3}, {0, 1},
        {0, 3}, {1, 0}, {1, 2}, {2, 1}, {2, 2},
    };

    const std::vector<Pair> in = llama_expert_profile_rank(L, E, resident, heat, prior);
    check(in.size() == (size_t) L * E, "every pair is ranked");
    check(in == exp, "the ranking is Strata's rule");

    std::string err;
    check(llama_expert_profile_write(path.c_str(), L, E, in, err), "written");
    check(!std::filesystem::exists(path + ".tmp"), "no temporary file left");

    std::vector<Pair> back;
    check(llama_expert_profile_read(path.c_str(), L, E, back, err), "read back");
    check(back == in, "the round trip gives the same ranking");
    check(placement(back, n_slots) == live, "the saved ranking gives the same placement");

    // (d) ties take the prior's rank before the index (Strata's comparator):
    // two same-heat counted pairs, the prior names the higher index first
    {
        std::vector<uint8_t> none((size_t) L * E, 0);
        std::vector<double>   tied((size_t) L * E, 0.0);
        tied[(size_t) 0 * E + 1] = 1.0;
        tied[(size_t) 0 * E + 2] = 1.0;
        const std::vector<Pair> prior2 = {{0, 2}, {0, 1}};
        const std::vector<Pair> tie = llama_expert_profile_rank(L, E, none, tied, prior2);
        check(tie.size() >= 2 && tie[0] == Pair(0, 2) && tie[1] == Pair(0, 1),
              "ties take the prior's rank before the index");
    }

    // (e) pending swaps at save: the incoming expert is resident in both
    // phases; the outgoing one stays resident only in phase 1
    {
        std::vector<uint8_t> res((size_t) L * E, 0);
        res[(size_t) 1 * E + 0] = 1;   // phase 1's host table still names the outgoing's slot
        const std::vector<llama_expert_profile_swap> pend = {{1, 2, 0}};
        std::vector<uint8_t> p1 = res;
        llama_expert_profile_mark_pending(E, p1, pend, 1);
        check(p1[(size_t) 1 * E + 2] == 1, "a pending incoming expert is resident");
        check(p1[(size_t) 1 * E + 0] == 1, "an outgoing expert stays resident in phase 1");
        std::vector<uint8_t> p2 = res;
        llama_expert_profile_mark_pending(E, p2, pend, 2);
        check(p2[(size_t) 1 * E + 2] == 1, "a pending incoming expert is resident in phase 2");
        check(p2[(size_t) 1 * E + 0] == 0, "an outgoing expert is cleared in phase 2");
    }

    // (g) the xe rule: the prior mass by rank, default 800,000 routed entries,
    // beats a short conversation's counts -- a 9-count pair does not outrank it
    {
        std::vector<double> h((size_t) L * E, 0.0);
        h[(size_t) 2 * E + 3] = 9.0;
        const std::vector<Pair> prior3 = {{0, 1}, {1, 2}, {2, 0}};
        const std::vector<Pair> no_prior = llama_expert_profile_rank_xe(L, E, h, prior3, 0.0, 2.0);
        check(!no_prior.empty() && no_prior[0] == Pair(2, 3), "xe without prior mass ranks by the counts");
        const std::vector<Pair> with_prior = llama_expert_profile_rank_xe(L, E, h, prior3, 800000.0, 2.0);
        check(with_prior.size() == (size_t) L * E, "xe ranks every pair");
        check(with_prior.size() >= 4 && with_prior[0] == Pair(0, 1) && with_prior[1] == Pair(1, 2) &&
                  with_prior[2] == Pair(2, 0) && with_prior[3] == Pair(2, 3),
              "xe mixes the prior in as pseudo-counts");
    }

    // (h, i) the save's heat is raw routing counts (one a routed pair, uncounted
    // ids skipped) in double, not a decayed weight and not float
    {
        std::vector<double> h(4, 0.0);
        llama_expert_profile_add_heat(h, 4, std::vector<int32_t>{0, 1, 1, 3, 9, -1});
        check(h == std::vector<double>{1.0, 2.0, 0.0, 1.0}, "heat counts one a routed pair, raw");
        std::vector<double> d(4, 0.0);
        d[0] = 16777218.0;   // float's 2^24 + 2; +1 from here is not representable as float
        llama_expert_profile_add_heat(d, 4, std::vector<int32_t>{0});
        check(d[0] == 16777219.0, "heat keeps a count past float's 2^24");
    }

    // the truncated record: header plus one pair short, no table
    const std::string trunc = path + ".trunc";
    check(llama_expert_profile_write(trunc.c_str(), L, E, in, err), "truncation seed written");
    std::error_code ec;
    std::filesystem::resize_file(trunc, 24 + 4 * (in.size() - 1), ec);
    check(!ec, "the seed was cut by one record");
    std::vector<Pair> partial;
    err.clear();
    check(!llama_expert_profile_read(trunc.c_str(), L, E, partial, err), "a truncated profile is refused");
    check(partial.empty(), "a refusal is no partial ranking");
    check(has(err, "truncated"), "the truncation refusal says truncated");
    std::filesystem::remove(trunc);
    check(!std::filesystem::exists(trunc + ".tmp"), "the truncation seed left no .tmp");

    // the version field at 2, checked before the geometry
    check(set_version(path, 2), "version field rewritten to 2");
    std::vector<Pair> v2;
    err.clear();
    check(!llama_expert_profile_read(path.c_str(), L, E, v2, err), "a version 2 profile is refused");
    check(has(err, path) && has(err, "version 2") && has(err, "version 1"),
          "the version refusal names the file, the version found and the one read");
    check(set_version(path, 1), "version field rewritten back to 1");
    std::vector<Pair> again;
    check(llama_expert_profile_read(path.c_str(), L, E, again, err) && again == in,
          "version 1 loads again");

    if (!keep) {
        std::filesystem::remove(path);
    }
    if (fails == 0) std::puts("expert_profile_save_selftest: OK");
    return fails == 0 ? 0 : 1;
}

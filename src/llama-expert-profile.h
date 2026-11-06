#pragma once

// Strata's learned expert profile (STRP), its ranking rule and its file format,
// kept dependency-free so a device-free test compiles the .cpp alone (the same
// rule the route trace follows, src/llama-expert-route-trace.cpp of patch 0069).
//
// Two ranking rules, both from Strata (commit 3d6f1718):
//   xe  its SYCL/xe save (`save_profile`, sycl/src/program/generate.cpp): the
//       raw routing counts (one a routed pair, its device counter's unit, not a
//       decayed sum) mixed with the start's ranking as pseudo-counts, no
//       resident head. `STRATA_PROFILE_PRIOR_ENTRIES` (default 800,000 routed
//       entries), lambda 2, the prior pair rank's share exp(-2 * rank / n).
//       A short conversation must not wipe the prior.
//   cu  its `rank_learned_profile` (src/core/expert_cache.cpp): the resident
//       pairs by heat, the counted non-resident pairs by heat, then the rest;
//       ties by the prior's own rank, then the pair's index.
//
// The cu rule, read from Strata's source and its save test:
//   1. the resident pairs, ordered by heat (usage) descending;
//   2. the non-resident pairs WITH heat > 0, ordered by heat descending;
//   3. then the remaining pairs (the prior's pairs in the prior's order, then
//      the fill layer 0..n_layers-1 outer, expert 0..n_expert-1 inner);
//   every group ties by the prior's rank first, then the pair's index.
// Strata's `rank_learned_profile` sorts every pair at once with exactly that
// comparator; the staged form here gives the same order.
//
// STRP file: "STRP", u32 version 1, u32 n_layer, u32 n_expert, u32 slots,
// u32 n_ranked, then n_ranked x (u16 layer, u16 expert), little endian, then
// the trailing layer x expert rank table (i32 expert -> rank, -1 for a pair
// the ranking does not hold), as Strata's writer appends it. The reader stops
// after the pairs, as arcint's does; a writer produces the whole file so it is
// byte for byte Strata's. The `slots` field is a hint; the reader ignores it.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

enum class llama_expert_profile_rule {
    xe,   // Strata's SYCL/xe save: prior mixing, no resident head (the default)
    cu,   // Strata's rank_learned_profile: resident first, prior as a tie-break
};

// a swap a cache still has in flight, as its save sees it: the incoming expert
// has landed (a copy in flight counts as done, Strata does), the outgoing one
// is still resident only while the host table is in phase 1
struct llama_expert_profile_swap {
    int layer;
    int e_in;
    int e_out;
};

// The version and geometry checks; a refusal names the file, the version found
// and the one read (version before geometry, Strata's _version_test), or the
// geometry, or why the pairs stopped. On refusal `ranked` is left empty: no
// partial ranking. `slots` may be any value.
bool llama_expert_profile_read(const char * path, int n_layers, int n_expert,
                               std::vector<std::pair<int, int>> & ranked, std::string & err);

// Atomic: a `.tmp` beside `path`, then a rename over it; a refusal (a pair out
// of range, a short write, a failed rename) leaves an existing file untouched
// and removes the `.tmp`. `slots = n_ranked`, as Strata's writer uses
// len(ranked) for both; the trailing rank table is written after the pairs.
bool llama_expert_profile_write(const char * path, int n_layers, int n_expert,
                                const std::vector<std::pair<int, int>> & ranked, std::string & err);

// Count one graph's routing into the save's heat: exactly one a routed id, raw
// (Strata's per-routing-event device counter, the unit the prior is in), ids out
// of range skipped. Double: a month at ~33 t/s passes float's 2^24 on the
// hottest pairs. Call it in the cache's count(), never in its adapt(): the heat
// must accumulate with adaptation off (LLAMA_EXPERT_CACHE_ADAPT_EVERY=0).
void llama_expert_profile_add_heat(std::vector<double> & heat, int n_expert, const std::vector<int32_t> & ids);

// The cu ranking above. `resident` and `heat` are indexed `layer * n_expert +
// expert`; a short or missing vector reads as 0 (Strata's empty heat case). A
// prior pair out of range is skipped; a pair the prior does not name ties last.
std::vector<std::pair<int, int>> llama_expert_profile_rank(
    int n_layers, int n_expert, const std::vector<uint8_t> & resident,
    const std::vector<double> & heat, const std::vector<std::pair<int, int>> & prior);

// The xe ranking: `heat` (the raw routing counts) mixed with the start's
// ranking as pseudo-counts worth `prior_entries` routed entries, lambda 2, by
// the prior pair's rank, then Strata's rank_learned_profile with no residents.
std::vector<std::pair<int, int>> llama_expert_profile_rank_xe(
    int n_layers, int n_expert, const std::vector<double> & heat,
    const std::vector<std::pair<int, int>> & prior, double prior_entries, double lam);

// Mark a cache's swaps still in flight, as Strata's save does: every incoming
// expert is resident (its copy has landed). The outgoing expert stays resident
// only in phase 1 (the host table still names its slot); at phase 2 the
// table has already freed it, so the save clears it.
void llama_expert_profile_mark_pending(int n_expert, std::vector<uint8_t> & resident,
                                       const std::vector<llama_expert_profile_swap> & pending, int phase);

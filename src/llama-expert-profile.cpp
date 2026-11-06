#include "llama-expert-profile.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

// The heat is Strata's xe unit: raw routing counts, one a routed pair (its device
// counter), not a decayed sum. Strata's rank_learned_profile (src/core/expert_cache.cpp, commit 3d6f1718):
// resident first, then by heat descending, then by the prior's own rank, then
// by the pair's index. Written as the staged order the save test shows, which
// is the same sequence: resident by heat, the counted non-resident pairs by
// heat, the prior in its own order, the fill layer-major. Ties in either heat
// group take the prior's rank before the index (Strata's comparator), which is
// what puts a prior pair ahead of a same-heat non-prior pair.
std::vector<std::pair<int, int>> llama_expert_profile_rank(
    int n_layers, int n_expert, const std::vector<uint8_t> & resident,
    const std::vector<double> & heat, const std::vector<std::pair<int, int>> & prior) {
    std::vector<std::pair<int, int>> out;
    if (n_layers <= 0 || n_expert <= 0) {
        return out;
    }
    const size_t n = (size_t) n_layers * (size_t) n_expert;
    auto index = [n_expert](int l, int e) { return (size_t) l * (size_t) n_expert + (size_t) e; };
    auto is_resident = [&](const std::pair<int, int> & p) {
        const size_t i = index(p.first, p.second);
        return i < resident.size() && resident[i] != 0;
    };
    auto heat_of = [&](const std::pair<int, int> & p) {
        const size_t i = index(p.first, p.second);
        return i < heat.size() ? heat[i] : 0.0;
    };

    // the pairs in the fill's own enumeration: layer outer, expert inner
    std::vector<std::pair<int, int>> all;
    all.reserve(n);
    for (int l = 0; l < n_layers; ++l) {
        for (int e = 0; e < n_expert; ++e) {
            all.emplace_back(l, e);
        }
    }

    // prior rank: the position of a pair in the profile the run started from,
    // INT64_MAX when the prior does not name it (Strata's comparator)
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t & pr = prior_rank[index(l, e)];
            if (pr == INT64_MAX) {
                pr = (int64_t) r;
            }
        }
    }
    auto rank_of = [&](const std::pair<int, int> & p) { return prior_rank[index(p.first, p.second)]; };

    std::vector<std::pair<int, int>> hot_resident, hot_others;
    for (const auto & p : all) {
        if (is_resident(p)) {
            hot_resident.push_back(p);
        } else if (heat_of(p) > 0.0) {
            hot_others.push_back(p);
        }
    }
    // Strata's tie-break: heat descending, then the prior's rank, then the
    // index (layer-major). Red mutations: `>` -> `<` reverses both groups; a
    // tie-break on the index alone moves a prior pair behind a same-heat one.
    auto by_heat_desc = [&](const std::pair<int, int> & a, const std::pair<int, int> & b) {
        if (heat_of(a) != heat_of(b)) {
            return heat_of(a) > heat_of(b);
        }
        if (rank_of(a) != rank_of(b)) {
            return rank_of(a) < rank_of(b);
        }
        return index(a.first, a.second) < index(b.first, b.second);
    };
    std::sort(hot_resident.begin(), hot_resident.end(), by_heat_desc);
    std::sort(hot_others.begin(), hot_others.end(), by_heat_desc);

    std::vector<uint8_t> seen(n, 0);
    auto take = [&](const std::pair<int, int> & p) {
        if (p.first < 0 || p.first >= n_layers || p.second < 0 || p.second >= n_expert) {
            return;
        }
        const size_t i = index(p.first, p.second);
        if (seen[i]) {
            return;
        }
        seen[i] = 1;
        out.push_back(p);
    };
    for (const auto & p : hot_resident) take(p);
    for (const auto & p : hot_others)   take(p);
    // the profile the run started from, in that profile's order
    for (const auto & p : prior) take(p);
    // the fill: layer outer, expert inner, every pair still missing
    for (const auto & p : all) take(p);
    return out;
}

void llama_expert_profile_add_heat(std::vector<double> & heat, int n_expert, const std::vector<int32_t> & ids) {
    if (n_expert <= 0) {
        return;
    }
    for (int32_t e : ids) {
        if (e < 0 || e >= n_expert) {
            continue;
        }
        if ((size_t) e < heat.size()) {
            heat[(size_t) e] += 1.0;   // raw: no decay factor, one a routed pair
        }
    }
}

std::vector<std::pair<int, int>> llama_expert_profile_rank_xe(
    int n_layers, int n_expert, const std::vector<double> & heat,
    const std::vector<std::pair<int, int>> & prior, double prior_entries, double lam) {
    if (n_layers <= 0 || n_expert <= 0) {
        return std::vector<std::pair<int, int>>();
    }
    const size_t n = (size_t) n_layers * (size_t) n_expert;
    // Strata's SYCL/xe save (generate.cpp save_profile): the device's cumulative
    // counts, then the prior mass by rank, exp(-lam * rank / n) worth
    // `prior_entries` routed entries over the whole ranking. `lam` 2, the shipped
    // ranking holds ~70 % of the routing in its first 47 % of pairs.
    std::vector<double> mixed(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        mixed[i] = i < heat.size() ? heat[i] : 0.0;
    }
    if (prior_entries > 0.0 && lam > 0.0 && n > 0) {
        const double norm = lam / (1.0 - std::exp(-lam)) / (double) n;
        for (size_t r = 0; r < prior.size(); ++r) {
            const auto [l, e] = prior[r];
            if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
                continue;
            }
            const double x = (double) r / (double) n;
            mixed[(size_t) l * (size_t) n_expert + (size_t) e] +=
                prior_entries * norm * std::exp(-lam * x);
        }
    }
    // Strata passes no resident vector here: every pair ranks by the mixed heat
    const std::vector<uint8_t> none(n, 0);
    return llama_expert_profile_rank(n_layers, n_expert, none, mixed, prior);
}

void llama_expert_profile_mark_pending(int n_expert, std::vector<uint8_t> & resident,
                                       const std::vector<llama_expert_profile_swap> & pending, int phase) {
    if (n_expert <= 0) {
        return;
    }
    for (const auto & s : pending) {
        if (s.layer < 0) {
            continue;
        }
        const size_t base = (size_t) s.layer * (size_t) n_expert;
        auto put = [&](int e) {
            if (e >= 0 && e < n_expert) {
                const size_t i = base + (size_t) e;
                if (i < resident.size()) {
                    resident[i] = 1;
                }
            }
        };
        auto clear = [&](int e) {
            if (e >= 0 && e < n_expert) {
                const size_t i = base + (size_t) e;
                if (i < resident.size()) {
                    resident[i] = 0;
                }
            }
        };
        // the incoming expert has landed (a copy in flight counts as done)
        put(s.e_in);
        // the outgoing stays resident only in phase 1; at phase 2 the host
        // table has already freed its slot
        if (phase >= 2) {
            clear(s.e_out);
        }
    }
}

static void put_u32(std::FILE * f, uint32_t v) {
    uint8_t b[4];
    for (int i = 0; i < 4; ++i) {
        b[i] = (uint8_t) ((v >> (8 * i)) & 0xff);
    }
    std::fwrite(b, 1, 4, f);
}

static void put_u16(std::FILE * f, uint32_t v) {
    uint8_t b[2] = { (uint8_t) (v & 0xff), (uint8_t) ((v >> 8) & 0xff) };
    std::fwrite(b, 1, 2, f);
}

static uint32_t get_u32(const uint8_t * b) {
    return (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
}

static uint16_t get_u16(const uint8_t * b) {
    return (uint16_t) ((uint16_t) b[0] | ((uint16_t) b[1] << 8));
}

bool llama_expert_profile_read(const char * path, int n_layers, int n_expert,
                               std::vector<std::pair<int, int>> & ranked, std::string & err) {
    ranked.clear();
    err.clear();
    if (path == nullptr || *path == '\0') {
        err = "llama_expert_profile_read: no profile path";
        return false;
    }
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        err = std::string("llama_expert_profile_read: ") + path + ": cannot open: " + std::strerror(errno);
        return false;
    }

    uint8_t magic[4];
    uint8_t hb[20];
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hb, 1, 20, f) != 20) {
        std::fclose(f);
        err = std::string("llama_expert_profile_read: ") + path + ": shorter than a 24-byte STRP header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = std::string("llama_expert_profile_read: ") + path + ": not a Strata profile (magic is not STRP)";
        return false;
    }
    const uint32_t version  = get_u32(hb);
    const uint32_t n_layer  = get_u32(hb + 4);
    const uint32_t n_exp    = get_u32(hb + 8);
    const uint32_t slots    = get_u32(hb + 12);   // ignored: arcint ignores the hint
    const uint32_t n_ranked = get_u32(hb + 16);
    (void) slots;

    // the version BEFORE the geometry: the fields after it are a version 1
    // layout's, so a version 2 file of another model is refused by its number
    // (Strata's _version_test)
    if (version != 1) {
        std::fclose(f);
        err = std::string("llama_expert_profile_read: ") + path + ": profile version " +
              std::to_string(version) + ", this reader reads version 1";
        return false;
    }
    if ((int) n_layer != n_layers || (int) n_exp != n_expert) {
        std::fclose(f);
        err = std::string("llama_expert_profile_read: ") + path + ": " + std::to_string(n_layer) + "x" +
              std::to_string(n_exp) + ", not " + std::to_string(n_layers) + "x" + std::to_string(n_expert);
        return false;
    }
    // more pairs than the geometry has: a header claim, refused before the loop
    // (a truncated pair is caught by the loop's own short read)
    if (n_ranked > (uint32_t) ((size_t) n_layers * (size_t) n_expert)) {
        std::fclose(f);
        err = std::string("llama_expert_profile_read: ") + path + ": " + std::to_string(n_ranked) +
              " ranked pairs, more than the geometry's " + std::to_string(n_layers * n_expert);
        return false;
    }

    std::vector<std::pair<int, int>> got;
    got.reserve(n_ranked);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        uint8_t pb[4];
        if (std::fread(pb, 1, 4, f) != 4) {
            std::fclose(f);
            err = std::string("llama_expert_profile_read: ") + path + ": pair " + std::to_string(i) +
                  " of " + std::to_string(n_ranked) + " (truncated record)";
            return false;
        }
        const int l = get_u16(pb);
        const int e = get_u16(pb + 2);
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            std::fclose(f);
            err = std::string("llama_expert_profile_read: ") + path + ": pair " + std::to_string(i) + " (" +
                  std::to_string(l) + "," + std::to_string(e) + ") is out of range";
            return false;
        }
        got.emplace_back(l, e);
    }
    if (std::fclose(f) != 0) {
        err = std::string("llama_expert_profile_read: ") + path + ": close failed: " + std::strerror(errno);
        return false;
    }
    ranked = std::move(got);
    return true;
}

bool llama_expert_profile_write(const char * path, int n_layers, int n_expert,
                                const std::vector<std::pair<int, int>> & ranked, std::string & err) {
    err.clear();
    if (path == nullptr || *path == '\0') {
        err = "llama_expert_profile_write: no profile path";
        return false;
    }
    if (n_layers < 1 || n_layers > 65535 || n_expert < 1 || n_expert > 65535) {
        err = std::string("llama_expert_profile_write: ") + path + ": " + std::to_string(n_layers) + "x" +
              std::to_string(n_expert) + " does not fit the u16 pair fields";
        return false;
    }
    if (ranked.size() > 0xffffffffUL) {
        err = std::string("llama_expert_profile_write: ") + path + ": too many ranked pairs";
        return false;
    }
    // the range check BEFORE the file is touched: a refused save keeps an
    // existing file (Strata's save test)
    for (size_t i = 0; i < ranked.size(); ++i) {
        const int l = ranked[i].first, e = ranked[i].second;
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = std::string("llama_expert_profile_write: ") + path + ": pair " + std::to_string(i) + " (" +
                  std::to_string(l) + "," + std::to_string(e) + ") is out of range";
            return false;
        }
    }

    const std::string tmp = std::string(path) + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = std::string("llama_expert_profile_write: ") + tmp + ": cannot open: " + std::strerror(errno);
        return false;
    }
    auto fail = [&](const std::string & why) {
        std::fclose(f);
        std::remove(tmp.c_str());   // never leave a `.tmp`
        err = std::string("llama_expert_profile_write: ") + tmp + ": " + why;
        return false;
    };
    std::fwrite("STRP", 1, 4, f);
    put_u32(f, 1u);                              // version
    put_u32(f, (uint32_t) n_layers);
    put_u32(f, (uint32_t) n_expert);
    put_u32(f, (uint32_t) ranked.size());        // slots = n_ranked, Strata's writer
    put_u32(f, (uint32_t) ranked.size());        // n_ranked
    for (const auto & p : ranked) {
        put_u16(f, (uint32_t) p.first);
        put_u16(f, (uint32_t) p.second);
    }
    // the trailing table Strata's writer appends: a layer's experts -> their
    // rank, -1 for a pair the ranking does not hold. Its save test compares the
    // whole file byte for byte, so a saved profile is the same file the Python
    // writers produce; readers stop after the pairs.
    std::vector<int32_t> table((size_t) n_layers * (size_t) n_expert, -1);
    for (size_t i = 0; i < ranked.size(); ++i) {
        table[(size_t) ranked[i].first * (size_t) n_expert + (size_t) ranked[i].second] = (int32_t) i;
    }
    for (int32_t v : table) {
        put_u32(f, (uint32_t) v);
    }
    if (std::fflush(f) != 0) {
        return fail(std::string("write failed: ") + std::strerror(errno));
    }
    if (std::fclose(f) != 0) {
        const int e = errno;
        std::remove(tmp.c_str());
        err = std::string("llama_expert_profile_write: ") + tmp + ": close failed: " + std::strerror(e);
        return false;
    }
    if (std::rename(tmp.c_str(), path) != 0) {
        const int e = errno;
        std::remove(tmp.c_str());
        err = std::string("llama_expert_profile_write: ") + path + ": rename from the .tmp failed: " +
              std::strerror(e);
        return false;
    }
    return true;
}

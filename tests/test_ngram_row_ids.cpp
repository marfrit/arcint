// FIX D Link 3 (docs/design-qwen-flash-next.md "Links 2 and 3"): red-first
// coverage for exec/ngram_row_ids.h -- the (token_history) -> row_index hash
// and its dummy-weight constant derivation.
//
// Ground truth is tests/ngram_row_ids_vectors.h, produced by
// tools/gen_ngram_vectors.py, which invokes the FreeToken reference's own
// torch-free derive_ngram_hash_constants (Apache-2.0, pinned commit 505477ab)
// for the constants and transcribes the documented mixing for the expected row
// ids. The reference test vectors are NOT copied (license note in
// docs/research-freetoken.md "Code-side ground truth"); these are our own,
// reference-derived.
//
// The suite is a deletion test for the correction this milestone lands: the
// old design-doc claim "one row per token id" is a plain-embedding shape that
// gathers row == token id. `row_ids_are_not_the_old_one_row_per_token_shape`
// asserts the hashed ids differ from that naive indexing, so a regression back
// to the wrong convention fails here; `row_ids_match_reference_vectors` pins the
// exact hashed values.
#include "exec/ngram_row_ids.h"
#include "harness.h"
#include "ngram_row_ids_vectors.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace lgc;
using lgc::ngram::test_vectors::RowIdsVector;
using lgc::ngram::test_vectors::row_ids_vectors;

namespace {

ngram::HashParams params_of(const RowIdsVector& v) {
    ngram::HashParams p;
    p.ngram_size              = v.ngram_size;
    p.heads_per_ngram         = v.heads_per_ngram;
    p.eos_token_id            = v.eos_token_id;
    p.layer_multipliers       = v.layer_multipliers;
    p.ngram_heads_vocab_sizes = v.ngram_heads_vocab_sizes;
    p.ngram_heads_offsets     = v.ngram_heads_offsets;
    return p;
}

}  // namespace

// The core assertion: arcint's row_ids reproduces the reference-derived vectors
// exactly, across every fixture (2-gram-only, trigram, eos-mid-sequence, fresh
// all-eos context, single-token decode, and Qwen3.8-scale 16-head geometry).
TEST(row_ids_match_reference_vectors) {
    for (const auto& v : row_ids_vectors()) {
        const auto got = ngram::row_ids(params_of(v), v.context, v.tokens);
        CHECK_EQ(got.size(), v.expected_row_ids.size());
        for (size_t i = 0; i < got.size(); ++i) {
            // name in the message so a failure says which fixture and which
            // (token, head) slot diverged.
            if (got[i] != v.expected_row_ids[i]) {
                log::error("test", "row_ids mismatch in fixture %s at flat index %zu: got %lld, want %lld",
                           v.name, i, static_cast<long long>(got[i]),
                           static_cast<long long>(v.expected_row_ids[i]));
            }
            CHECK_EQ(got[i], v.expected_row_ids[i]);
        }
    }
}

// The negative guard for the correction. If Link 3 had been built against the
// old "one row per token id" spec, every head of token i would gather row
// (token_i mod n_rows). Prove the hashed ids are genuinely different from that
// naive shape on a non-degenerate fixture, so a silent regression to it is
// caught here rather than in a serving mismatch.
TEST(row_ids_are_not_the_old_one_row_per_token_shape) {
    const auto vectors = row_ids_vectors();
    bool checked_a_multihead_fixture = false;
    for (const auto& v : vectors) {
        const int H = (v.ngram_size - 1) * v.heads_per_ngram;
        if (H < 2) continue;  // need multiple heads to distinguish from a single-row shape
        checked_a_multihead_fixture = true;
        const auto got = ngram::row_ids(params_of(v), v.context, v.tokens);
        bool differs_from_naive = false;
        for (size_t i = 0; i < v.tokens.size(); ++i) {
            const int64_t naive = v.tokens[i] % v.total_rows;  // "one row per token id"
            for (int h = 0; h < H; ++h) {
                if (got[i * H + h] != naive) differs_from_naive = true;
            }
        }
        CHECK(differs_from_naive);
    }
    CHECK(checked_a_multihead_fixture);
}

// The dummy-weight derivation reproduces the reference constants the vectors
// were built with. This is the oracle a loaded checkpoint's int64 buffers will
// be checked against; here it checks arcint's C++ port against the reference's
// own derive_ngram_hash_constants output baked into the header. Recovering the
// per-fixture (vocab_size, ngram_vocab_size_base, ple_layer_index) from the
// generator's FIXTURES table keeps this honest: the header carries only the
// derived constants, so a wrong derivation cannot pass by construction.
TEST(derive_hash_constants_matches_reference_for_known_fixtures) {
    struct Case {
        const char* name;
        int64_t     vocab_size;
        int         ngram_size;
        int         heads_per_ngram;
        int64_t     ngram_vocab_size_base;
        int         ple_layer_index;
    };
    // Mirror of tools/gen_ngram_vectors.py::FIXTURES (name -> derive inputs).
    const Case cases[] = {
        {"basic_2gram_only", 257, 2, 3, 17, 1},
        {"trigram_no_eos", 257, 3, 2, 17, 1},
        {"trigram_eos_midseq", 257, 3, 2, 17, 2},
        {"trigram_fresh_context", 257, 3, 2, 17, 0},
        {"decode_single_token", 257, 3, 4, 17, 1},
        {"qwen_scale_16head", 151936, 3, 8, 8209, 3},
    };
    const auto vectors = row_ids_vectors();
    for (const auto& c : cases) {
        const RowIdsVector* v = nullptr;
        for (const auto& cand : vectors) {
            if (std::string(cand.name) == c.name) { v = &cand; break; }
        }
        CHECK(v != nullptr);
        if (v == nullptr) continue;
        const auto p = ngram::derive_hash_constants(c.vocab_size, c.ngram_size,
                                                    c.heads_per_ngram, c.ngram_vocab_size_base,
                                                    c.ple_layer_index);
        CHECK_EQ(p.layer_multipliers.size(), v->layer_multipliers.size());
        for (size_t i = 0; i < p.layer_multipliers.size(); ++i)
            CHECK_EQ(p.layer_multipliers[i], v->layer_multipliers[i]);
        CHECK_EQ(p.ngram_heads_vocab_sizes.size(), v->ngram_heads_vocab_sizes.size());
        for (size_t i = 0; i < p.ngram_heads_vocab_sizes.size(); ++i)
            CHECK_EQ(p.ngram_heads_vocab_sizes[i], v->ngram_heads_vocab_sizes[i]);
        CHECK_EQ(p.ngram_heads_offsets.size(), v->ngram_heads_offsets.size());
        for (size_t i = 0; i < p.ngram_heads_offsets.size(); ++i)
            CHECK_EQ(p.ngram_heads_offsets[i], v->ngram_heads_offsets[i]);
    }
}

// validate() refuses the shapes a mis-loaded checkpoint could hand it.
TEST(hash_params_validate_rejects_bad_shapes) {
    auto base = params_of(row_ids_vectors().at(1));  // trigram_no_eos: ngram_size 3, 4 heads
    base.validate();  // the good one does not throw

    bool threw = false;
    try {
        auto bad = base;
        bad.ngram_heads_vocab_sizes[0] = 0;  // modulo-by-zero waiting to happen
        bad.validate();
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);

    threw = false;
    try {
        auto bad = base;
        bad.layer_multipliers.pop_back();  // count no longer matches ngram_size
        bad.validate();
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

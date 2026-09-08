// Cross-tensor parallelism for the GGUF repack + deviation check
// (core/gguf_repack.h's gguf_repack_all, DESIGN 7.0.2ba/7.0.2bd): the loader
// used to repack the file's ~288 tensors one after another, each one itself
// parallel over its own rows; this factors the per-tensor work into a
// driver that takes several whole tensors off a list at once, in a bounded
// worker pool, while capping the intra-tensor thread count so the two
// dimensions do not multiply. Device-free -- no OpenVINO needed -- so it
// builds and runs in the stub configuration this file is compiled under.
//
// Written red-first: gguf_repack_all did not exist when this file's
// determinism test was written (it failed to compile), and only then was
// the driver added to core/gguf_repack.h/.cpp.
#include "core/gguf.h"
#include "core/gguf_repack.h"
#include "harness.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace lgc;

namespace {

std::string fixture_path() { return std::string(ARCINT_SOURCE_DIR) + "/tests/fixtures/qwen35-tiny.gguf"; }

std::vector<const gguf::TensorInfo*> two_dim_repackable_tensors(const gguf::GgufFile& f) {
    std::vector<const gguf::TensorInfo*> out;
    for (const auto& t : f.tensors())
        if (t.dims.size() == 2 && gguf::repack_supported(t.ggml_type)) out.push_back(&t);
    return out;
}

std::vector<gguf::RepackRequest> requests_of(const std::vector<const gguf::TensorInfo*>& tensors, gguf::RepackMins mins) {
    std::vector<gguf::RepackRequest> reqs;
    reqs.reserve(tensors.size());
    for (const auto* t : tensors) {
        const double bound = gguf::repack_bound_steps(t->ggml_type) * (mins == gguf::RepackMins::Split ? 2.0 : 1.0);
        reqs.push_back(gguf::RepackRequest{t, {}, bound});
    }
    return reqs;
}

std::string mkverdictdir() {
    char tmpl[] = "/tmp/arcint-parallel-verdicts-XXXXXX";
    const char* dir = ::mkdtemp(tmpl);
    return dir != nullptr ? std::string(dir) + "/gguf-verdicts" : "";
}

}  // namespace

// The determinism test: repacking the fixture's every repackable tensor
// through gguf_repack_all must give byte-identical weights/scales/min_matrix
// and identical deviation figures (max_steps, over_bound, checked) to the
// serial per-tensor calls, for threads = 1 and threads = 4, and whether the
// verdict cache is empty or warm. Every row's own repacked bytes and its own
// deviation contribution are independent of how many threads touched it
// (core/gguf_repack.h's comment on repack_tensor/repack_deviation); the
// across-row reduction inside repack_deviation is done in a fixed row order
// regardless of thread count, which is what makes this hold exactly, not
// approximately.
TEST(gguf_repack_all_matches_the_serial_calls_byte_for_byte_at_threads_1_and_4) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto tensors = two_dim_repackable_tensors(f);
    // The fixture carries all four repackable types (Q8_0 x1, Q4_K x2, Q5_K
    // x1, Q6_K x2 -- checked directly, not just "at least a few").
    CHECK_EQ(tensors.size(), size_t{6});

    // The serial reference: repack_tensor/repack_deviation called directly,
    // one tensor at a time, exactly as gguf_apply_to_template did before
    // this driver existed.
    std::vector<gguf::RepackedTensor> serial;
    std::vector<gguf::RepackDeviation> serial_dv;
    for (const auto* t : tensors) {
        const auto r = gguf::repack_tensor(f, *t);
        const double bound = gguf::repack_bound_steps(t->ggml_type);
        serial.push_back(r);
        serial_dv.push_back(gguf::repack_deviation(f, *t, r, bound));
    }

    for (unsigned threads : {1u, 4u}) {
        const auto requests = requests_of(tensors, gguf::RepackMins::Exact);
        const auto results = gguf::gguf_repack_all(f, requests, gguf::RepackMins::Exact, "", "", threads);
        CHECK_EQ(results.size(), tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            CHECK(results[i].packed.weights == serial[i].weights);
            CHECK(results[i].packed.scale == serial[i].scale);
            CHECK(results[i].packed.min_matrix == serial[i].min_matrix);
            CHECK_NEAR(results[i].max_steps, serial_dv[i].max_steps, 1e-12);
            CHECK_EQ(results[i].over_bound, serial_dv[i].over);
            CHECK_EQ(results[i].checked, serial_dv[i].values);
            CHECK(!results[i].verdict_cached);
        }
        std::printf("  gguf_repack_all threads=%u: %zu tensor(s) byte-identical to the serial calls\n", threads, tensors.size());
    }
}

// The verdict cache, empty then warm, through the driver: the first call
// checks every tensor and writes a verdict; the second reads every verdict
// back and re-checks nothing, but the repacked bytes and max_steps must
// still agree with a fresh serial repack (the verdict is a shortcut for the
// CHECK, never for the repack itself, which always runs).
TEST(gguf_repack_all_verdict_cache_empty_then_warm_matches_a_fresh_serial_repack) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto tensors = two_dim_repackable_tensors(f);
    const std::string vdir = mkverdictdir();
    CHECK(!vdir.empty());
    const std::string path = fixture_path();

    const auto requests = requests_of(tensors, gguf::RepackMins::Exact);
    const auto cold = gguf::gguf_repack_all(f, requests, gguf::RepackMins::Exact, vdir, path, 4);
    for (const auto& r : cold) CHECK(!r.verdict_cached);
    for (const auto& r : cold) CHECK(r.checked > 0);

    const auto warm = gguf::gguf_repack_all(f, requests, gguf::RepackMins::Exact, vdir, path, 1);
    CHECK_EQ(warm.size(), cold.size());
    for (size_t i = 0; i < warm.size(); ++i) {
        CHECK(warm[i].verdict_cached);
        CHECK_EQ(warm[i].checked, size_t{0});
        CHECK_EQ(warm[i].over_bound, size_t{0});
        CHECK_NEAR(warm[i].max_steps, cold[i].max_steps, 1e-9);  // round-tripped through the verdict file's text
        CHECK(warm[i].packed.weights == cold[i].packed.weights);
        CHECK(warm[i].packed.scale == cold[i].packed.scale);
    }
    std::filesystem::remove_all(vdir);
}

// The red case (kept as the record of the failure this driver closes): a
// bound of 0.0 makes every tensor with a nonzero deviation "over its bound",
// so gguf_repack_all must throw, and the message must name the failing
// tensor -- the same message gguf_apply_to_template used to throw inline
// per-tensor before the check moved into this driver.
TEST(gguf_repack_all_reports_a_tensor_over_its_bound_by_name) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto tensors = two_dim_repackable_tensors(f);
    CHECK(!tensors.empty());
    const gguf::TensorInfo* bad = tensors.front();

    std::vector<gguf::RepackRequest> requests;
    requests.push_back(gguf::RepackRequest{bad, {}, 0.0});  // an impossible bound: everything is "over"
    for (size_t i = 1; i < tensors.size(); ++i)
        requests.push_back(gguf::RepackRequest{tensors[i], {}, gguf::repack_bound_steps(tensors[i]->ggml_type)});

    bool threw = false;
    std::string what;
    try {
        (void)gguf::gguf_repack_all(f, requests, gguf::RepackMins::Exact, "", "");
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    CHECK(what.find(bad->name) != std::string::npos);
    CHECK(what.find("repacked outside its bound") != std::string::npos);
    std::printf("  over-bound message: %s\n", what.c_str());
}

// A refused type (repack_tensor itself throwing) is reported the same way --
// collected during the parallel phase and rethrown on the calling thread,
// naming the tensor, rather than crashing a worker silently.
TEST(gguf_repack_all_reports_a_refused_type_by_name) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* f32 = nullptr;
    for (const auto& t : f.tensors())
        if (t.ggml_type == 0) { f32 = &t; break; }
    CHECK(f32 != nullptr);
    const auto tensors = two_dim_repackable_tensors(f);
    CHECK(!tensors.empty());

    std::vector<gguf::RepackRequest> requests;
    requests.push_back(gguf::RepackRequest{f32, {}, gguf::repack_bound_steps(f32->ggml_type)});
    requests.push_back(gguf::RepackRequest{tensors.front(), {}, gguf::repack_bound_steps(tensors.front()->ggml_type)});

    bool threw = false;
    std::string what;
    try {
        (void)gguf::gguf_repack_all(f, requests, gguf::RepackMins::Exact, "", "");
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    CHECK(what.find(f32->name) != std::string::npos);
    CHECK(what.find("not a repacked type") != std::string::npos);
}

// gguf_repack_all's own bound-checking wiring, on the mins that ACCEPT a
// deviation instead of refusing it (Shared/Nibble, DESIGN 7.0.2bl): a bound
// of 0.0 must not throw there, only report the deviation.
TEST(gguf_repack_all_never_refuses_over_bound_under_shared_or_nibble_mins) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* q4k = nullptr;
    for (const auto& t : f.tensors())
        if (t.ggml_type == 12 && t.dims.size() == 2) { q4k = &t; break; }
    CHECK(q4k != nullptr);
    std::vector<gguf::RepackRequest> requests{gguf::RepackRequest{q4k, {}, 0.0}};
    const auto results = gguf::gguf_repack_all(f, requests, gguf::RepackMins::Shared, "", "");
    CHECK_EQ(results.size(), size_t{1});
    CHECK(results[0].over_bound > 0);  // the deviation is real and reported
    CHECK(!results[0].verdict_cached);
}

// The verdict key must carry the mins packing, not just the bound: gguf_
// apply_to_template's old (pre-0.4.2) verdict_key folded a per-packing offset
// into the bound it hashed ("the packing is part of the key"); gguf_repack_
// all's own verdict_key(file_path, t, req.bound) dropped that offset, so a
// Shared/Nibble load (which never refuses and always writes) could leave a
// verdict an Exact load of the SAME tensor/file/bound would then read back
// and trust, skipping the very check that is supposed to refuse it.
// RED before core/gguf_repack.cpp's verdict_key regained the mins offset:
// the Exact call below found the Shared call's verdict, reported it
// verdict_cached and never threw. GREEN after: the keys differ (the offsets
// in core/gguf_repack.cpp's verdict_mins_offset), so the Exact call checks
// afresh and refuses, as it always has for an over-bound Exact repack.
TEST(gguf_repack_all_verdict_key_is_not_shared_across_mins_packings) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* q4k = nullptr;
    for (const auto& t : f.tensors())
        if (t.ggml_type == 12 && t.dims.size() == 2) { q4k = &t; break; }
    CHECK(q4k != nullptr);
    const std::string vdir = mkverdictdir();
    CHECK(!vdir.empty());
    const std::string path = fixture_path();

    // Cold: a Shared repack at an impossible bound (0.0) -- Shared never
    // refuses, so this writes a verdict despite every value being "over".
    std::vector<gguf::RepackRequest> shared_req{gguf::RepackRequest{q4k, {}, 0.0}};
    const auto shared_res = gguf::gguf_repack_all(f, shared_req, gguf::RepackMins::Shared, vdir, path);
    CHECK_EQ(shared_res.size(), size_t{1});
    CHECK(shared_res[0].over_bound > 0);
    CHECK(!shared_res[0].verdict_cached);

    // An Exact request against the same tensor, file and bound must NOT come
    // back cached: it must check afresh and refuse (bound 0.0 is impossible
    // for Exact too, same as for Shared above).
    std::vector<gguf::RepackRequest> exact_req{gguf::RepackRequest{q4k, {}, 0.0}};
    bool threw = false;
    std::string what;
    try {
        (void)gguf::gguf_repack_all(f, exact_req, gguf::RepackMins::Exact, vdir, path);
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    CHECK(what.find(q4k->name) != std::string::npos);
    CHECK(what.find("repacked outside its bound") != std::string::npos);
    std::printf("  verdict key collision test: Exact correctly refused (%s)\n", what.c_str());
    std::filesystem::remove_all(vdir);
}

// An Exact verdict, once written by an Exact run, IS reused by a second
// Exact run of the same tensor/file/bound -- the ordinary cache-hit path
// (kept alongside the collision test above, which checks the opposite: that
// a DIFFERENT packing's verdict is NOT reused).
TEST(gguf_repack_all_exact_verdict_is_reused_by_a_second_exact_run) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* q4k = nullptr;
    for (const auto& t : f.tensors())
        if (t.ggml_type == 12 && t.dims.size() == 2) { q4k = &t; break; }
    CHECK(q4k != nullptr);
    const std::string vdir = mkverdictdir();
    CHECK(!vdir.empty());
    const std::string path = fixture_path();
    const double bound = gguf::repack_bound_steps(q4k->ggml_type);

    std::vector<gguf::RepackRequest> req{gguf::RepackRequest{q4k, {}, bound}};
    const auto cold = gguf::gguf_repack_all(f, req, gguf::RepackMins::Exact, vdir, path);
    CHECK_EQ(cold.size(), size_t{1});
    CHECK(!cold[0].verdict_cached);

    const auto warm = gguf::gguf_repack_all(f, req, gguf::RepackMins::Exact, vdir, path);
    CHECK_EQ(warm.size(), size_t{1});
    CHECK(warm[0].verdict_cached);
    CHECK_NEAR(warm[0].max_steps, cold[0].max_steps, 1e-9);
    std::filesystem::remove_all(vdir);
}

// The RMS reduction order: repack_deviation's across-row combination of the
// per-row sum of squares must be independent of the intra-tensor thread
// count, EXACTLY (not approximately) -- core/gguf_repack.cpp reduces
// row_sq[0..n-1] into the total in fixed row order regardless of how many
// threads computed the rows, rather than combining each thread's own
// running partial sum in thread-completion order (which the fix replaced:
// floating-point addition is not associative, so that older scheme made
// rms_steps move with the thread count alone). Verified red by hand while
// writing this test: temporarily combining row_sq in thread-interleaved
// order (thread 0's rows, then thread 1's, ...) instead of row order made
// d1.rms_steps and d16.rms_steps compare unequal (CHECK_EQ failed though
// both printed 0.00252729 -- the bits differ); reverted before this test
// was committed.
TEST(repack_deviation_rms_steps_is_exactly_equal_at_threads_1_4_and_16) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* q4k = nullptr;
    for (const auto& t : f.tensors())
        if (t.ggml_type == 12 && t.dims.size() == 2) { q4k = &t; break; }
    CHECK(q4k != nullptr);
    const auto packed = gguf::repack_tensor(f, *q4k);
    const double bound = gguf::repack_bound_steps(q4k->ggml_type);
    const auto d1  = gguf::repack_deviation(f, *q4k, packed, bound, 1);
    const auto d4  = gguf::repack_deviation(f, *q4k, packed, bound, 4);
    const auto d16 = gguf::repack_deviation(f, *q4k, packed, bound, 16);
    CHECK_EQ(d1.rms_steps, d4.rms_steps);
    CHECK_EQ(d1.rms_steps, d16.rms_steps);
    CHECK_EQ(d1.max_steps, d16.max_steps);
    CHECK_EQ(d1.over, d16.over);
    CHECK_EQ(d1.values, d16.values);
}

// A column-reordered (`dest_of`) request through gguf_repack_all, not just
// through repack_tensor directly: the same head-wise-swap construction
// tests/test_gguf_repack.cpp's "a_head_wise_column_order_is_applied_at_build"
// test uses (swap the two 128-column halves of the fixture's K=256 row), so
// the driver's own request-building path (repack_tensor called with
// req.column_dest_of, core/gguf_repack.cpp's process_one) is exercised, not
// only the direct API.
TEST(gguf_repack_all_carries_a_non_empty_column_dest_of_request_through) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* q4k = nullptr;
    for (const auto& t : f.tensors())
        if (t.ggml_type == 12 && t.dims.size() == 2) { q4k = &t; break; }
    CHECK(q4k != nullptr);
    std::vector<int64_t> dest_of(static_cast<size_t>(q4k->dims[0]));
    for (size_t c = 0; c < dest_of.size(); ++c) dest_of[c] = static_cast<int64_t>((c + 128) % dest_of.size());

    std::vector<gguf::RepackRequest> requests{gguf::RepackRequest{q4k, dest_of, gguf::repack_bound_steps(q4k->ggml_type)}};
    const auto results = gguf::gguf_repack_all(f, requests, gguf::RepackMins::Exact, "", "");
    CHECK_EQ(results.size(), size_t{1});
    CHECK(!results[0].over_bound);
    CHECK_EQ(results[0].packed.groups_per_aug, int64_t{4});  // the head-wise order's grouping, not the super-block default of 8

    // Must match a direct repack_tensor call with the same dest_of.
    const auto direct = gguf::repack_tensor(f, *q4k, &dest_of);
    CHECK(results[0].packed.weights == direct.weights);
    CHECK(results[0].packed.scale == direct.scale);
}

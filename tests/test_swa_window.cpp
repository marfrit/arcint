// A window-sized SWA KV cache (--llama-swa-window, core/swa_window.h): a prefix
// trim is exact only while the retained history reaches the next query's window.
//
// The mechanism: llama.cpp's seq_rm() on a window-sized SWA cache always returns
// true (it cannot refuse, unlike the recurrent memory), so a lane that cuts a
// prompt there leaves a hole below the window's start unless the cache still
// holds it. The backend reads llama_memory_attn_seq_pos_min() before the cut and
// resumes from a checkpoint when the hole would be there. Off (the library's
// full-size SWA cache), every trim is exact and the rule is not consulted.
#include "config.h"
#include "core/swa_window.h"
#include "harness.h"
#include <string>

using namespace lgc;

TEST(swa_window_a_q8_q4_sliding_window_checkpoint_is_refused_on_opencl_only) {
    // 1.8.1: OpenCL + a sliding window + q8_0 / q4_0 K or V: no checkpoint (the restore would
    // write no bytes). Red: the predicate returning false lets the first three through.
    // The last argument false: a llama.cpp tree without 0209 (1.8.1's and 1.8.2's).
    CHECK(swa_ckpt_restore_refused(1024, true, true, true, false));     // Inkling's units: q8_0 K/V
    CHECK(swa_ckpt_restore_refused(1024, true, false, true, false));    // f16 K, q4_0 V
    CHECK(swa_ckpt_restore_refused(1024, true, true, false, false));
    CHECK(!swa_ckpt_restore_refused(1024, true, false, false, false));  // f16 K/V: the generic write
    CHECK(!swa_ckpt_restore_refused(1024, false, true, true, false));   // not on OpenCL (the CPU's memcpy)
    CHECK(!swa_ckpt_restore_refused(0, true, true, true, false));       // no sliding window: the 27B agent at q8_0:q4_0
}

TEST(swa_window_the_props_checkpoints_string_names_the_refusal_only_where_it_took_them) {
    // 1.8.1 reported "off: <reason>" whenever the refusal was set, also for a model that
    // would not have kept checkpoints (Dory without --llama-swa-window: a sliding window,
    // no snapshotting memory). Red: the 1.8.2 rule (stale -> the reason) fails the third
    // and fourth cases. Arguments: --llama-checkpoints, can_snapshot, the refusal.
    const std::string reason = checkpoints_status(32, true, true);
    CHECK(reason.rfind("off: ", 0) == 0);                          // Inkling at q8_0 K/V: taken away
    CHECK(std::string(checkpoints_status(32, true, false)) == "on");   // Inkling at f16 K/V
    CHECK(std::string(checkpoints_status(32, false, true)) == "off");  // Dory at q8_0, not hybrid
    CHECK(std::string(checkpoints_status(0, true, true)) == "off");    // --llama-checkpoints 0
    CHECK(std::string(checkpoints_status(0, false, false)) == "off");
    CHECK(std::string(checkpoints_status(32, false, false)) == "off"); // the 27B agent: attention trims
    // red, on every run: 1.8.2's rule (ckpt_on ? "on" : stale ? the reason : "off", with ckpt_on
    // = ckpt_max > 0 && can_snapshot && !stale) names the reason for the Dory and the 0 cases
    const auto rule_182 = [](int max, bool snap, bool stale) -> std::string {
        const bool on = max > 0 && snap && !stale;
        return on ? "on" : stale ? "off: <reason>" : "off";
    };
    CHECK(rule_182(32, false, true) != checkpoints_status(32, false, true));
    CHECK(rule_182(0, true, true) != checkpoints_status(0, true, true));
}

TEST(swa_window_a_trim_inside_the_retained_history_is_exact) {
    // the lane holds [1,536, 3,071]; the cut is at 2,599, the next query at
    // 2,599 needs keys from 2,599 - 1,024 + 1 = 1,576: inside what it holds
    CHECK(swa_window_trim_complete(1024, 1536, 2599));
    // the boundary is exact: pos_min == need_from (2,599 - 1,024 + 1)
    CHECK(swa_window_trim_complete(1024, 1576, 2599));
    CHECK(!swa_window_trim_complete(1024, 1577, 2599));
}

TEST(swa_window_a_trim_below_the_retained_history_needs_a_checkpoint) {
    // a lane filled to 3,071 holds the ring [2,048, 3,071]; an edited message
    // that diverges at 2,599 leaves the window's [1,576, 2,047] absent: the
    // trim would resume with a hole, so the lane must restore a checkpoint
    CHECK(!swa_window_trim_complete(1024, 2048, 2599));
    // a divergence deeper than the whole cache is likewise incomplete
    CHECK(!swa_window_trim_complete(1024, 2048, 2049));
    // a divergence below the window's start: everything needed is absent
    CHECK(!swa_window_trim_complete(1024, 2048, 500));
}

TEST(swa_window_early_positions_count_from_zero) {
    // before the window fills, the next query's window starts at the sequence's
    // first position, not at a negative one
    CHECK(swa_window_trim_complete(1024, 0, 500));
    CHECK(!swa_window_trim_complete(1024, 64, 500));
}

TEST(swa_window_an_empty_cache_is_not_complete) {
    // pos_min -1: nothing retained, so the needed keys can never be there
    CHECK(!swa_window_trim_complete(1024, -1, 2599));
}

TEST(swa_window_a_model_without_swa_trims_as_before) {
    CHECK(swa_window_trim_complete(0, 2048, 2599));
    CHECK(swa_window_trim_complete(0, -1, 0));
}

TEST(swa_window_the_flag_is_off_by_default_and_opt_in) {
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf"};
        CHECK(parse_args(3, const_cast<char**>(argv), cfg).ok);
        CHECK(!cfg.llama_swa_window);   // prefix reuse is exact at the library default
    }
    {
        Config cfg;
        const char* argv[] = {"arcint", "--gguf", "/m/q.gguf", "--llama-swa-window"};
        CHECK(parse_args(4, const_cast<char**>(argv), cfg).ok);
        CHECK(cfg.llama_swa_window);
    }
    {
        // an option of the libllama engine, like --llama-kv
        Config cfg;
        const char* argv[] = {"arcint", "--stub", "--llama-swa-window"};
        CHECK(!parse_args(3, const_cast<char**>(argv), cfg).ok);
    }
}

TEST(swa_window_the_help_names_the_flag) {
    const std::string help = usage_text();
    CHECK(help.find("--llama-swa-window") != std::string::npos);
}

TEST(swa_window_with_0209_q8_q4_sliding_window_checkpoints_are_restored) {
    // 1.8.3: contrib/llama.cpp 0209 writes through a q8_0 / q4_0 view, so a binary built
    // against a tree with it restores these checkpoints. Red, on every run: 1.8.1's rule (no
    // view_writes argument, the refusal for every OpenCL q8_0 / q4_0 sliding window) refuses
    // each of the first three cases, which this test requires to be restored.
    const auto rule_181 = [](int n_swa, bool cl, bool k, bool v) { return n_swa > 0 && cl && (k || v); };
    CHECK(!swa_ckpt_restore_refused(1024, true, true, true, true));     // Inkling's units: q8_0 K/V
    CHECK(!swa_ckpt_restore_refused(1024, true, false, true, true));    // f16 K, q4_0 V
    CHECK(!swa_ckpt_restore_refused(1024, true, true, false, true));
    CHECK(rule_181(1024, true, true, true) != swa_ckpt_restore_refused(1024, true, true, true, true));
    CHECK(rule_181(1024, true, false, true) != swa_ckpt_restore_refused(1024, true, false, true, true));
    CHECK(rule_181(1024, true, true, false) != swa_ckpt_restore_refused(1024, true, true, false, true));
    // the cases 1.8.1 allowed stay allowed
    CHECK(!swa_ckpt_restore_refused(1024, true, false, false, true));
    CHECK(!swa_ckpt_restore_refused(1024, false, true, true, true));
    CHECK(!swa_ckpt_restore_refused(0, true, true, true, true));
}

// A libllama case: compiled out of a device-free stub build (ARCINT_LLAMA off), as the other
// libllama cases are; there it could only skip, and the unit target allows no unnamed skip.
#ifdef ARCINT_LLAMA
TEST(swa_window_the_build_finds_0209_in_its_llama_tree) {
    // The source's series carries 0209 (ARCINT_SERIES_HAS_0209), so a build against the
    // tree the recipe applies must find it (ARCINT_LLAMA_KV_VIEW_WRITES, CMakeLists.txt's
    // marker): a marker that stopped matching would turn the q8_0 / q4_0 checkpoints off
    // with every other test green. A build against another tree (a red or an older one)
    // fails here by design and says so.
#if defined(ARCINT_SERIES_HAS_0209)
#if !defined(ARCINT_LLAMA_KV_VIEW_WRITES)
    t::fail(__FILE__, __LINE__, "the series has 0209 but ARCINT_LLAMA_DIR does not: q8_0 / q4_0 sliding-window "
                                "checkpoints are off in this build");
#endif
#else
    t::skip("not a build against this source's series (no 0209 in contrib/llama.cpp/patches)");
#endif
}
#endif

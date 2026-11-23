// A window-sized SWA KV cache (--llama-swa-window, core/swa_window.h): a prefix
// trim is exact only while the retained history reaches the next query's window.
//
// The mechanism: llama.cpp's seq_rm() on a window-sized SWA cache always returns
// true (it cannot refuse, unlike the recurrent memory), so a lane that cuts a
// prompt there leaves a hole below the window's start unless the cache still
// holds it. The backend reads llama_memory_seq_pos_min() before the cut and
// resumes from a checkpoint when the hole would be there. Off (the library's
// full-size SWA cache), every trim is exact and the rule is not consulted.
#include "config.h"
#include "core/swa_window.h"
#include "harness.h"

using namespace lgc;

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

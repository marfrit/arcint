// The prefill-slot-loan's rules and scope, without a card. The mechanism (the
// expert cache's lend/refill/verify, the context's re-reserve and rebind) is
// patch 0068 of the llama.cpp series and is exercised on the B60 by the
// campaign's card commands; what is provable here is what decides *whether* a
// request borrows and that the slots go back on every exit path.
//
// Red cases (each is a one-line edit to src/exec/slot_loan.h; the named test
// then fails):
//   R1 `wanted()` without `remaining_prompt > r.resident_chunk`
//      -> short_prompt_never_lends
//   R2 `~Scope` without `end()`
//      -> exception_exit_returns_the_loan
//   R1 also makes sub_floor_prompt_is_not_lent fail (wanted() is the floor:
//      loan_chunk() can never be below it, so there is no second check)
//   R3 `end()` without the `ended_` guard (a second end refills again)
//      -> end_is_idempotent (the comment there said R1 before the third review)
//   R4 `request_start_ok()` ignoring `loan_out`
//      -> request_start_refuses_while_a_loan_is_out
//   R5 finding 4: the destructor without its try (or end() without its own)
//      -> a_throwing_return_does_not_terminate: the child aborts on
//         std::terminate, and the case fails on WIFEXITED
//   R6 finding 8's flag rule, arcint's half: `lend_param()` handing back the env
//      sentinel for a 0 MiB flag -> a_zero_flag_stays_off. The llama side of the
//      rule (init's resolution and llama_model_default_params' default) is
//      tests/test_loan_book.cpp's an_explicit_zero_flag_stays_off and
//      the_default_params_ask_for_the_environment

#include "exec/slot_loan.h"

#include "harness.h"

#include <sys/wait.h>
#include <unistd.h>

using lgc::slot_loan::Rule;
using lgc::slot_loan::Scope;

namespace {

// a begin/end pair that records what it was handed, so a test can say when each
// hook ran and with which chunk
struct Hooks {
    int  begins = 0;
    int  ends   = 0;
    uint32_t last_chunk = 0;
    bool begin_ok = true;   // false: the context refused the loan
    bool end_ok   = true;   // false: the refill's verify differed

    Scope make(Rule r, uint32_t chunk) {
        return Scope(r, chunk,
                     [this](uint32_t c) { ++begins; last_chunk = c; return begin_ok; },
                     [this]() { ++ends; return end_ok; });
    }
};

Rule enabled_rule(size_t lend = 8ull << 30) {
    Rule r;
    r.enabled    = true;
    r.lend_bytes = lend;
    return r;
}

}  // namespace

TEST(resident_chunk_is_the_floor_with_the_loan_on) {
    Rule on = enabled_rule();
    CHECK_EQ(lgc::slot_loan::floor_chunk(8192, on), 2048u);
    CHECK_EQ(lgc::slot_loan::floor_chunk(1024, on), 1024u);   // never above the chunk itself
    Rule off;
    CHECK_EQ(lgc::slot_loan::floor_chunk(8192, off), 8192u);   // today's shape unchanged
}

TEST(short_prompt_never_lends) {
    // finding 6: a 1,000-token follow-up must not pay two reserves and a refill
    Hooks h;
    Rule  r = enabled_rule();
    uint32_t chunk = 8192;
    {
        Scope s = h.make(r, chunk);
        CHECK(!s.begin(1000));
        CHECK(!s.begun());
        CHECK(!s.ended());
    }
    CHECK_EQ(h.begins, 0);
    CHECK_EQ(h.ends, 0);
}

TEST(prompt_above_the_floor_lends_at_its_own_length) {
    // 5,000 tokens at a chunk of 8,192: the reserve is sized for 5,000, not 8,192
    Hooks h;
    Rule  r = enabled_rule();
    Scope s = h.make(r, 8192);
    CHECK(s.begin(5000));
    CHECK(s.begun());
    CHECK_EQ(h.begins, 1);
    CHECK_EQ(h.last_chunk, 5000u);
    CHECK(s.end());
    CHECK(s.ended());
    CHECK_EQ(h.ends, 1);
}

TEST(normal_exit_returns_the_loan_once) {
    Hooks h;
    {
        Scope s = h.make(enabled_rule(), 8192);
        CHECK(s.begin(9000));
        CHECK(s.ok());
    }   // the block's end
    CHECK_EQ(h.begins, 1);
    CHECK_EQ(h.ends, 1);
}

TEST(exception_exit_returns_the_loan) {
    // prefill_locked throws (backend_llama.cpp:1764) and wave2a cancels at
    // ubatch boundaries; the destructor is the only path that always runs
    Hooks h;
    bool threw = false;
    try {
        Scope s = h.make(enabled_rule(), 8192);
        CHECK(s.begin(9000));
        throw std::runtime_error("llama_decode failed during prefill");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(h.begins, 1);
    CHECK_EQ(h.ends, 1);   // R2's red case
}

TEST(a_refused_loan_leaves_no_end_to_run) {
    // the context refused it: no reserve, no free, no refill
    Hooks h;
    h.begin_ok = false;
    {
        Scope s = h.make(enabled_rule(), 8192);
        CHECK(!s.begin(9000));
        CHECK(!s.begun());
        CHECK(!s.ended());
    }
    CHECK_EQ(h.begins, 1);
    CHECK_EQ(h.ends, 0);
}

TEST(a_failed_refill_is_reported) {
    // the refill's verify found a differing row: the request must fail, not
    // answer from wrong bytes
    Hooks h;
    h.end_ok = false;
    Scope s = h.make(enabled_rule(), 8192);
    CHECK(s.begin(9000));
    CHECK(!s.end());
    CHECK(!s.ok());
}

TEST(end_is_idempotent) {
    Hooks h;
    Scope s = h.make(enabled_rule(), 8192);
    CHECK(s.begin(9000));
    CHECK(s.end());
    CHECK(s.end());
    CHECK_EQ(h.ends, 1);   // R3's red case: a second end would refill again
}

TEST(begin_twice_is_refused) {
    Hooks h;
    Scope s = h.make(enabled_rule(), 8192);
    CHECK(s.begin(9000));
    CHECK(!s.begin(9000));
    CHECK_EQ(h.begins, 1);
}

TEST(sub_floor_prompt_is_not_lent) {
    // a prompt between a configured chunk and the resident floor: no reserve to
    // save, so no loan (the hidden cost is the two reserves for nothing)
    Hooks h;
    Rule r = enabled_rule();
    r.resident_chunk = 4096;
    Scope s = h.make(r, 8192);
    CHECK(!s.begin(3000));
    CHECK_EQ(h.begins, 0);   // R1's red case
}

TEST(no_lend_bytes_means_no_loan) {
    Hooks h;
    Rule r;
    r.enabled    = true;
    r.lend_bytes = 0;
    Scope s = h.make(r, 8192);
    CHECK(!s.begin(9000));
    CHECK_EQ(h.begins, 0);
}

TEST(request_start_refuses_while_a_loan_is_out) {
    // finding 4's belt: a previous request's scope failed; a lent slot must
    // never be read as an expert
    CHECK(lgc::slot_loan::request_start_ok(false));
    CHECK(!lgc::slot_loan::request_start_ok(true));   // R4's red case
}

TEST(loan_chunk_is_bounded_by_the_prompt_and_the_floor) {
    CHECK_EQ(lgc::slot_loan::loan_chunk(8192, 9000), 8192u);
    CHECK_EQ(lgc::slot_loan::loan_chunk(8192, 3000), 3000u);
    CHECK_EQ(lgc::slot_loan::loan_chunk(8192, 2048), 2048u);
}

// finding 4: llama_context_slot_loan_end re-reserves, and sched_reserve throws on
// a failed allocation. The destructor runs inside unwinding, where a throw is
// std::terminate and takes the server with it. The child process makes the
// red case measurable: without the catch it dies on SIGABRT, and the parent's
// WIFEXITED check fails.
TEST(a_throwing_return_does_not_terminate) {
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        int rc = 0;
        try {
            Scope s(enabled_rule(), 8192,
                    [](uint32_t) { return true; },
                    []() -> bool { throw std::runtime_error("failed to allocate compute pp buffers"); });
            if (!s.begin(9000)) rc = 3;
        } catch (...) {
            rc = 4;   // the destructor let the exception out (or unwinding did)
        }
        _exit(rc);
    }
    int st = 0;
    CHECK_EQ(waitpid(pid, &st, 0), pid);
    CHECK(WIFEXITED(st));   // R5's red case: no catch -> std::terminate, SIGABRT here
    CHECK_EQ(WEXITSTATUS(st), 0);
}

// the same for a begin hook that throws before it can report: the scope reports
// not begun, not ok, and an end was attempted (the context's own recovery runs
// in slot_loan_begin; a loan it left out is refused at the next request)
TEST(a_throwing_begin_is_reported_not_propagated) {
    int ends = 0;
    Scope s(enabled_rule(), 8192,
            [](uint32_t) -> bool { throw std::runtime_error("the loan's reserve failed"); },
            [&]() { ++ends; return true; });
    CHECK(!s.begin(9000));
    CHECK(!s.begun());
    CHECK(s.ended());
    CHECK(!s.ok());
    CHECK_EQ(ends, 1);
}

// finding 8's flag rule: --llama-slot-loan off (0 MiB) is an explicit off, so
// LLAMA_SLOT_LOAN_MIB cannot enable the loan behind the operator's flag. The
// env sentinel is llama.h's LLAMA_SLOT_LOAN_BYTES_ENV ((size_t) -1); this test
// holds arcint's half without llama.h -- it passes 0, not the sentinel. The
// llama half (init's rule lives there, not in this flag) is the rule cases in
// test_loan_book.cpp.
TEST(a_zero_flag_stays_off) {
    CHECK_EQ(lgc::slot_loan::lend_param(0), static_cast<size_t>(0));
    CHECK(lgc::slot_loan::lend_param(0) != static_cast<size_t>(-1));   // R6's red case
    CHECK_EQ(lgc::slot_loan::lend_param(4608), static_cast<size_t>(4608) << 20);
}

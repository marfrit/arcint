#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>

// DESIGN.md §4.1. The scheduler, and it is one object.
//
// Two lanes step their own InferRequests; the device runs one graph execution
// at a time whatever the host does, so the only real question is *in what
// order* and *with what guarantee*. Left to the plugin the answer is "an
// unordered lock", under which a decode step can wait behind an arbitrary run
// of the other lane's prefill chunks and no bound can be stated.
//
// This is a ticket lock: whoever asks first runs first. That turns the
// interleaving into a statement with a number in it — a decode step waits for
// at most one other execution, so the worst inter-token stall is one prefill
// chunk — and it is also where that wait is measured, because the waiting is
// what a caller feels. It is not a lock over shared state: each lane's state
// is its own (§3.3), and nothing inside the turn touches another lane's.
namespace lgc {

class Turnstile {
public:
    class Turn {
    public:
        Turn() = default;
        Turn(Turnstile* gate, double waited) : gate_(gate), waited_(waited) {}
        Turn(const Turn&)            = delete;
        Turn& operator=(const Turn&) = delete;
        Turn(Turn&& other) noexcept { *this = std::move(other); }
        Turn& operator=(Turn&& other) noexcept;
        ~Turn();

        // Seconds spent waiting for the device. Zero when the lane had it to
        // itself, which is how a contended step is told from a solo one.
        double waited_seconds() const { return waited_; }

    private:
        Turnstile* gate_   = nullptr;
        double     waited_ = 0.0;
    };

    // Blocks until it is this caller's turn.
    Turn take();

    // Turns *completed* so far (in-flight ones are not counted yet). Exposed
    // for the tests, which use it to prove mutual exclusion; nothing publishes
    // it, so it is not a metric anyone reads.
    uint64_t served() const;

    // Tickets handed out so far, the ones still waiting included, so
    // served() <= issued() at any instant. Exposed for the tests too: it is
    // how a test knows a contender has *asked* — the order is fixed the moment
    // the ticket is issued — without guessing at how long the scheduler took
    // to get it there. Nothing publishes it either.
    uint64_t issued() const;

private:
    void release();

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    uint64_t                next_ticket_ = 0;
    uint64_t                now_serving_ = 0;
};

}  // namespace lgc

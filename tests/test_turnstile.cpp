#include "core/turnstile.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "harness.h"

using lgc::Turnstile;

// None of these cases lets a wall-clock sleep decide an assertion. Where a
// case needs to know that another thread has asked for the device, it waits
// for the turnstile's own ticket count to advance (issued()); the only sleeps
// left are deliberate handicaps on the *other* thread, put there because the
// sleep-gated form of these tests assumed the contender reached take() within
// a fixed window and failed outright when it did not.

TEST(turnstile_lets_one_lane_through_at_a_time) {
    Turnstile        gate;
    std::atomic<int> inside{0};
    std::atomic<int> overlaps{0};
    std::atomic<int> done{0};

    auto lane = [&] {
        for (int i = 0; i < 200; ++i) {
            Turnstile::Turn turn = gate.take();
            if (inside.fetch_add(1) != 0) ++overlaps;
            std::this_thread::yield();
            inside.fetch_sub(1);
            ++done;
        }
    };
    std::thread a(lane), b(lane), c(lane);
    a.join();
    b.join();
    c.join();

    CHECK_EQ(overlaps.load(), 0);
    CHECK_EQ(done.load(), 600);
    CHECK_EQ(gate.served(), uint64_t{600});
    CHECK_EQ(gate.issued(), uint64_t{600});   // nothing left waiting
}

TEST(turnstile_serves_in_arrival_order) {
    // The property the stall bound rests on: a lane that asked first is not
    // overtaken, so a decode step waits for one execution rather than for an
    // unbounded run of the other lane's.
    //
    // "Asked first" is decided inside take(), where the ticket is issued under
    // the mutex, so that is what the main thread waits for before it starts
    // the next contender. The second contender dawdles before asking; a test
    // that started the third after a fixed wait would see 3 ask before 2 and
    // fail, whatever the turnstile did.
    Turnstile        gate;
    std::mutex       log_mutex;
    std::vector<int> order;

    Turnstile::Turn held = gate.take();   // the device is busy: ticket 0
    CHECK_EQ(gate.issued(), uint64_t{1});

    auto contender = [&](int id, std::chrono::milliseconds dawdle) {
        std::this_thread::sleep_for(dawdle);
        Turnstile::Turn turn = gate.take();
        std::lock_guard<std::mutex> guard(log_mutex);
        order.push_back(id);
    };
    auto asked = [&](uint64_t tickets) {
        while (gate.issued() < tickets) std::this_thread::yield();
    };

    std::thread second(contender, 2, std::chrono::milliseconds(20));
    asked(2);   // 2 holds ticket 1, whether or not it has reached the wait yet
    std::thread third(contender, 3, std::chrono::milliseconds(0));
    asked(3);   // 3 holds ticket 2

    held = Turnstile::Turn();   // release
    second.join();
    third.join();

    CHECK_EQ(order, (std::vector<int>{2, 3}));
    CHECK_EQ(gate.served(), uint64_t{3});
}

TEST(turnstile_reports_the_wait_it_imposed) {
    using clock = std::chrono::steady_clock;

    Turnstile gate;
    {
        Turnstile::Turn solo = gate.take();
        CHECK_EQ(solo.waited_seconds(), 0.0);   // nothing to wait for
    }

    // The bar is an interval this thread measures itself: from seeing the
    // other's ticket issued to just before giving up the turn. The other's
    // clock started before its ticket and stops after the release, so its
    // reported wait is at least that long by construction, whatever the
    // scheduler did with either thread in between.
    std::atomic<double> waited{-1.0};
    Turnstile::Turn     held = gate.take();
    const uint64_t      before = gate.issued();   // the solo turn above counts too
    std::thread         other([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // slow to ask
        waited = gate.take().waited_seconds();
    });
    while (gate.issued() == before) std::this_thread::yield();
    const auto seen = clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));   // lengthens the bar, never lowers it
    const auto releasing = clock::now();
    held = Turnstile::Turn();
    other.join();

    const double imposed = std::chrono::duration<double>(releasing - seen).count();
    CHECK(imposed > 0.0);                 // the bar is not vacuous
    CHECK(waited.load() >= imposed);      // it really did wait for the holder
}

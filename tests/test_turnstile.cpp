#include "core/turnstile.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "harness.h"

using lgc::Turnstile;

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
}

TEST(turnstile_serves_in_arrival_order) {
    // The property the stall bound rests on: a lane that asked first is not
    // overtaken, so a decode step waits for one execution rather than for an
    // unbounded run of the other lane's.
    Turnstile               gate;
    std::mutex              log_mutex;
    std::vector<int>        order;
    std::atomic<bool>       second_waiting{false};
    std::atomic<bool>       third_waiting{false};

    Turnstile::Turn held = gate.take();   // the device is busy

    auto contender = [&](int id, std::atomic<bool>& flag) {
        flag = true;
        Turnstile::Turn turn = gate.take();
        std::lock_guard<std::mutex> guard(log_mutex);
        order.push_back(id);
    };

    std::thread second(contender, 2, std::ref(second_waiting));
    while (!second_waiting.load()) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));   // 2 has its ticket

    std::thread third(contender, 3, std::ref(third_waiting));
    while (!third_waiting.load()) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));   // 3 queued behind it

    held = Turnstile::Turn();   // release
    second.join();
    third.join();

    CHECK_EQ(order.size(), size_t{2});
    CHECK_EQ(order[0], 2);
    CHECK_EQ(order[1], 3);
}

TEST(turnstile_reports_the_wait_it_imposed) {
    Turnstile gate;
    {
        Turnstile::Turn solo = gate.take();
        CHECK_EQ(solo.waited_seconds(), 0.0);   // nothing to wait for
    }

    std::atomic<double> waited{-1.0};
    Turnstile::Turn     held = gate.take();
    std::thread         other([&] { waited = gate.take().waited_seconds(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    held = Turnstile::Turn();
    other.join();

    CHECK(waited.load() >= 0.04);   // it really did wait for the holder
}

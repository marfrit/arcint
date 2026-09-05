#include "core/affinity.h"

#include <pthread.h>
#include <sched.h>

#include "harness.h"

using lgc::pin_current_thread;

TEST(pin_current_thread_pins_to_the_requested_core) {
    cpu_set_t original;
    CPU_ZERO(&original);
    CHECK_EQ(pthread_getaffinity_np(pthread_self(), sizeof(original), &original), 0);

    // The requested core must be one this process may use at all: a
    // container's cpuset can exclude CPU 0 (the 0.3.0 package build ran in
    // one allowing 1,5-7,9,11,14-15), and a literal 0 there tests the
    // host, not the pin. The first allowed core is as good a request as any.
    int core = -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &original)) { core = c; break; }
    }
    CHECK(core >= 0);

    // Probe with the identical mask first: a no-op with respect to which CPUs
    // are enabled, so it changes nothing about the assertion below, but a
    // sandbox that denies sched_setaffinity outright (a restrictive seccomp
    // profile, not merely a restrictive cpuset) fails it distinctly from a
    // bug in pin_current_thread itself.
    SKIP_UNLESS(pthread_setaffinity_np(pthread_self(), sizeof(original), &original) == 0,
                "sched_setaffinity is denied in this sandbox");

    CHECK(pin_current_thread(core));

    cpu_set_t after;
    CPU_ZERO(&after);
    CHECK_EQ(pthread_getaffinity_np(pthread_self(), sizeof(after), &after), 0);
    CHECK_EQ(CPU_COUNT(&after), 1);
    CHECK(CPU_ISSET(core, &after));

    // Restore, so later tests in this process are not left pinned to one CPU.
    CHECK_EQ(pthread_setaffinity_np(pthread_self(), sizeof(original), &original), 0);
}

TEST(pin_current_thread_rejects_a_negative_core) {
    CHECK(!pin_current_thread(-1));
}

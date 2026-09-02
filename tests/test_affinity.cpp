#include "core/affinity.h"

#include <pthread.h>
#include <sched.h>

#include "harness.h"

using lgc::pin_current_thread;

TEST(pin_current_thread_pins_to_the_requested_core) {
    cpu_set_t original;
    CPU_ZERO(&original);
    CHECK_EQ(pthread_getaffinity_np(pthread_self(), sizeof(original), &original), 0);

    CHECK(pin_current_thread(0));

    cpu_set_t after;
    CPU_ZERO(&after);
    CHECK_EQ(pthread_getaffinity_np(pthread_self(), sizeof(after), &after), 0);
    CHECK_EQ(CPU_COUNT(&after), 1);
    CHECK(CPU_ISSET(0, &after));

    // Restore, so later tests in this process are not left pinned to CPU 0.
    CHECK_EQ(pthread_setaffinity_np(pthread_self(), sizeof(original), &original), 0);
}

TEST(pin_current_thread_rejects_a_negative_core) {
    CHECK(!pin_current_thread(-1));
}

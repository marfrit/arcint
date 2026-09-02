#pragma once

namespace lgc {

// Pins the calling thread to a single CPU core (--pin-dispatch, M12b: testing
// whether decode's host-side enqueue is bound by contention for the
// dispatching thread's core). Linux only: pthread_setaffinity_np under the
// hood. `core` is taken modulo the number of online CPUs, so a caller does
// not have to know the machine's core count to stay in range. Returns false
// (and does nothing) on any other platform, for a negative core, or if the
// underlying call fails.
bool pin_current_thread(int core);

}  // namespace lgc

#include "core/affinity.h"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace lgc {

bool pin_current_thread(int core) {
#ifdef __linux__
    if (core < 0) return false;

    const long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc <= 0) return false;
    const int target = static_cast<int>(core % nproc);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(target, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)core;
    return false;
#endif
}

}  // namespace lgc

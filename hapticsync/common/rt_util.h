#pragma once
// rt_util.h — Real-time utility functions (Section 5.2.4, Appendix A)
// Provides CPU pinning, SCHED_FIFO, mlockall, prefaulting, malloc tuning.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <malloc.h>
#include <unistd.h>
#include <time.h>
#else
// Stubs for non-Linux development
#endif

namespace hs {

/// Pin the calling thread to a single CPU core.
inline bool set_affinity(int cpu) {
#ifdef __linux__
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    return pthread_setaffinity_np(pthread_self(), sizeof(s), &s) == 0;
#else
    (void)cpu;
    return true; // stub
#endif
}

/// Set the calling thread to SCHED_FIFO with the given priority.
inline bool set_fifo(int prio) {
#ifdef __linux__
    sched_param p{};
    p.sched_priority = prio;
    return pthread_setschedparam(pthread_self(), SCHED_FIFO, &p) == 0;
#else
    (void)prio;
    return true;
#endif
}

/// Disable mmap-based malloc and trimming (call before allocating RT buffers).
inline void rt_malloc_setup() {
#ifdef __linux__
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
#endif
}

/// Touch every page of a stack-allocated buffer to pre-fault pages.
inline void prefault_stack(size_t n = 1 << 20) {
    (void)n;
    volatile unsigned char buf[1 << 20];
    std::memset((void*)buf, 0, sizeof(buf));
}

/// Lock all current and future pages into RAM.
inline bool lock_all() {
#ifdef __linux__
    return mlockall(MCL_CURRENT | MCL_FUTURE) == 0;
#else
    return true;
#endif
}

/// Get current time in nanoseconds (CLOCK_MONOTONIC).
inline int64_t monotonic_ns() {
#ifdef __linux__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#else
    // Fallback using chrono
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
#endif
}

/// Verify that the calling thread has the expected affinity, policy, priority.
/// Returns true if all match. Prints mismatches to stderr.
inline bool verify_rt_setup(int expected_cpu, int expected_policy, int expected_prio) {
#ifdef __linux__
    bool ok = true;

    // Check affinity
    cpu_set_t s;
    CPU_ZERO(&s);
    pthread_getaffinity_np(pthread_self(), sizeof(s), &s);
    if (!CPU_ISSET(expected_cpu, &s)) {
        std::fprintf(stderr, "[rt_util] MISMATCH: CPU %d not in affinity set\n", expected_cpu);
        ok = false;
    }

    // Check policy and priority
    int pol;
    sched_param p{};
    pthread_getschedparam(pthread_self(), &pol, &p);
    if (pol != expected_policy) {
        std::fprintf(stderr, "[rt_util] MISMATCH: policy=%d expected=%d\n", pol, expected_policy);
        ok = false;
    }
    if (p.sched_priority != expected_prio) {
        std::fprintf(stderr, "[rt_util] MISMATCH: prio=%d expected=%d\n",
                     p.sched_priority, expected_prio);
        ok = false;
    }

    if (ok) {
        std::fprintf(stderr, "[rt_util] OK: cpu=%d policy=%d prio=%d\n",
                     expected_cpu, pol, p.sched_priority);
    }
    return ok;
#else
    (void)expected_cpu; (void)expected_policy; (void)expected_prio;
    return true;
#endif
}

} // namespace hs


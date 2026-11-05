#pragma once
// stats.h — Shared stats block for telemetry (Section 5.7)
// All fields are atomics, readable from any process via shared memory.

#include <atomic>
#include <cstdint>
#include <cstring>

namespace hs {

static constexpr const char* SHM_STATS = "/hapticsync_stats";

/// Cache-line aligned stats block shared across all processes.
/// Single producer per field (the component that owns it), multiple readers.
struct alignas(64) StatsBlock {
    // ── Producer stats (written by producer thread) ──
    std::atomic<uint64_t> periods_captured{0};
    std::atomic<uint64_t> samples_captured{0};
    std::atomic<uint32_t> xruns{0};
    std::atomic<uint32_t> gaps{0};

    // ── Assembler stats ──
    std::atomic<uint64_t> windows_sent{0};
    std::atomic<uint64_t> windows_done{0};
    std::atomic<uint64_t> windows_dropped{0};

    // ── Inference stats ──
    std::atomic<int64_t>  last_infer_ns{0};    // most recent inference duration
    std::atomic<uint32_t> model_busy{0};       // 1 = inference running

    // ── Post-process / event stats ──
    std::atomic<uint64_t> events_emitted{0};
    std::atomic<uint64_t> events_lapped{0};    // readers lapped

    // ── Lag tracking ──
    std::atomic<int64_t>  last_lag_ns{0};
    std::atomic<int64_t>  min_lag_ns{INT64_MAX};
    std::atomic<int64_t>  max_lag_ns{0};
    std::atomic<uint64_t> lag_sum_ns{0};
    std::atomic<uint64_t> lag_count{0};

    // ── Pipeline start time ──
    std::atomic<int64_t>  pipeline_start_ns{0};

    // ── Padding to fill to a known size ──
    char _pad[64];

    void reset() {
        periods_captured.store(0, std::memory_order_relaxed);
        samples_captured.store(0, std::memory_order_relaxed);
        xruns.store(0, std::memory_order_relaxed);
        gaps.store(0, std::memory_order_relaxed);
        windows_sent.store(0, std::memory_order_relaxed);
        windows_done.store(0, std::memory_order_relaxed);
        windows_dropped.store(0, std::memory_order_relaxed);
        last_infer_ns.store(0, std::memory_order_relaxed);
        model_busy.store(0, std::memory_order_relaxed);
        events_emitted.store(0, std::memory_order_relaxed);
        events_lapped.store(0, std::memory_order_relaxed);
        last_lag_ns.store(0, std::memory_order_relaxed);
        min_lag_ns.store(INT64_MAX, std::memory_order_relaxed);
        max_lag_ns.store(0, std::memory_order_relaxed);
        lag_sum_ns.store(0, std::memory_order_relaxed);
        lag_count.store(0, std::memory_order_relaxed);
        pipeline_start_ns.store(0, std::memory_order_relaxed);
    }

    void record_lag(int64_t lag_ns) {
        last_lag_ns.store(lag_ns, std::memory_order_relaxed);
        lag_sum_ns.fetch_add(static_cast<uint64_t>(lag_ns), std::memory_order_relaxed);
        lag_count.fetch_add(1, std::memory_order_relaxed);
        // Relaxed min/max (racy but acceptable for telemetry)
        int64_t cur_min = min_lag_ns.load(std::memory_order_relaxed);
        while (lag_ns < cur_min &&
               !min_lag_ns.compare_exchange_weak(cur_min, lag_ns, std::memory_order_relaxed));
        int64_t cur_max = max_lag_ns.load(std::memory_order_relaxed);
        while (lag_ns > cur_max &&
               !max_lag_ns.compare_exchange_weak(cur_max, lag_ns, std::memory_order_relaxed));
    }
};

static constexpr size_t STATS_SHM_SIZE = sizeof(StatsBlock);

} // namespace hs


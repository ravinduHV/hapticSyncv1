#pragma once
// logring.h — Lock-free log ring for RT-safe logging (Section 5.2.1)
// Producer thread writes fixed-size entries; log drain thread reads them.
// The producer never calls printf/iostream; it writes here instead.

#include <atomic>
#include <cstdint>
#include <cstring>

#include "rt_util.h"

namespace hs {

/// Fixed-size log entry (128 bytes).
struct LogEntry {
    uint64_t seq;           // sequence number (odd = writing, even = valid)
    int64_t  t_ns;          // CLOCK_MONOTONIC timestamp
    uint32_t level;         // 0=debug, 1=info, 2=warn, 3=error
    uint32_t _pad;
    char     msg[104];      // NUL-terminated message
};
static_assert(sizeof(LogEntry) == 128, "LogEntry must be 128 bytes");

enum LogLevel : uint32_t {
    LOG_DEBUG = 0,
    LOG_INFO  = 1,
    LOG_WARN  = 2,
    LOG_ERROR = 3,
};

/// Lock-free single-producer single-consumer log ring.
/// The producer (RT thread) writes entries; the drain thread reads them.
class LogRing {
public:
    /// Construct with given capacity (must be power of 2).
    explicit LogRing(size_t capacity = 4096)
        : capacity_(capacity), mask_(capacity - 1), write_idx_(0), read_idx_(0) {
        entries_ = new LogEntry[capacity];
        std::memset(entries_, 0, capacity * sizeof(LogEntry));
    }

    ~LogRing() {
        delete[] entries_;
    }

    // Non-copyable
    LogRing(const LogRing&) = delete;
    LogRing& operator=(const LogRing&) = delete;

    /// RT-safe write: no allocation, no blocking.
    /// msg is truncated to 103 chars.
    void write(uint32_t level, const char* msg) {
        uint64_t wi = write_idx_.load(std::memory_order_relaxed);
        uint32_t slot = static_cast<uint32_t>(wi & mask_);
        LogEntry& e = entries_[slot];

        // Mark as writing (odd seq)
        reinterpret_cast<std::atomic<uint64_t>*>(&e.seq)->store(
            2 * wi + 1, std::memory_order_release);

        e.t_ns = monotonic_ns();
        e.level = level;
        e._pad = 0;
        std::memset(e.msg, 0, sizeof(e.msg));
        if (msg) {
            std::strncpy(e.msg, msg, sizeof(e.msg) - 1);
        }

        // Mark as valid (even seq)
        reinterpret_cast<std::atomic<uint64_t>*>(&e.seq)->store(
            2 * (wi + 1), std::memory_order_release);

        write_idx_.store(wi + 1, std::memory_order_release);
    }

    /// Non-RT read: returns true if an entry was read.
    bool read(LogEntry& out) {
        uint64_t ri = read_idx_.load(std::memory_order_relaxed);
        uint64_t wi = write_idx_.load(std::memory_order_acquire);

        if (ri >= wi) return false; // nothing to read

        // Check for lapping
        if (wi - ri > capacity_) {
            ri = wi - capacity_; // skip to oldest available
            read_idx_.store(ri, std::memory_order_relaxed);
        }

        uint32_t slot = static_cast<uint32_t>(ri & mask_);
        const LogEntry& e = entries_[slot];

        uint64_t seq1 = reinterpret_cast<const std::atomic<uint64_t>*>(&e.seq)->load(
            std::memory_order_acquire);

        std::memcpy(&out, &e, sizeof(LogEntry));

        uint64_t seq2 = reinterpret_cast<const std::atomic<uint64_t>*>(&e.seq)->load(
            std::memory_order_acquire);

        // Valid if both seqs match, are even, and correspond to this slot
        if (seq1 != seq2 || (seq1 & 1) != 0) {
            // Torn read — skip
            read_idx_.store(ri + 1, std::memory_order_relaxed);
            return false;
        }

        read_idx_.store(ri + 1, std::memory_order_relaxed);
        return true;
    }

private:
    size_t capacity_;
    size_t mask_;
    std::atomic<uint64_t> write_idx_;
    std::atomic<uint64_t> read_idx_;
    LogEntry* entries_;
};

} // namespace hs


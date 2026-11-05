#pragma once
// ring.h — Lock-free shared-memory audio ring buffer (Section 5.2.5)
// Single writer (producer), multiple readers (assembler, telemetry).
// Mono float32. Capacity is power-of-two samples.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace hs {

static constexpr const char* SHM_AUDIO = "/hapticsync_audio";
static constexpr uint32_t RING_MAGIC   = 0x48415332; // "HAS2"
static constexpr uint32_t RING_VERSION = 1;

/// Period timestamp entry: end sample index and capture time.
struct PeriodEntry {
    uint64_t sample_idx_end; // absolute sample index at end of period
    int64_t  t_ns;           // CLOCK_MONOTONIC nanoseconds
};

/// Gap marker: where a gap occurred and how many frames were lost.
struct GapEntry {
    uint64_t sample_idx;     // absolute sample index where gap starts
    uint32_t lost_frames;    // estimated lost frames
    uint32_t _pad;
};

static constexpr size_t PERIOD_TABLE_SIZE = 256;
static constexpr size_t GAP_TABLE_SIZE    = 64;

/// Ring header in shared memory. Cache-line aligned.
struct alignas(64) AudioRingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t capacity;       // power-of-two sample count

    // Write index: absolute sample count since start, never resets.
    // Updated with release semantics after data is written.
    std::atomic<uint64_t> write_idx;

    // Counters
    std::atomic<uint32_t> xruns;
    std::atomic<uint32_t> overruns;
    std::atomic<uint32_t> period_write_idx; // index into period_table
    std::atomic<uint32_t> gap_write_idx;    // index into gap_table

    // Padding to push tables to known offsets
    char _hdr_pad[16];

    // Period timestamp ring
    PeriodEntry period_table[PERIOD_TABLE_SIZE];

    // Gap table
    GapEntry gap_table[GAP_TABLE_SIZE];
};

/// Calculate total shared memory size for audio ring.
inline size_t audio_ring_shm_size(uint32_t capacity) {
    return sizeof(AudioRingHeader) + capacity * sizeof(float);
}

/// Return the next power of two >= n.
inline uint32_t next_pow2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16;
    return n + 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioRingWriter — used by the producer thread (single writer)
// ─────────────────────────────────────────────────────────────────────────────

class AudioRingWriter {
public:
    AudioRingWriter() = default;

    /// Initialize from a shared memory pointer. Caller must ensure the
    /// memory is already allocated and sized for header + capacity floats.
    void init(void* shm, uint32_t capacity, uint32_t sample_rate) {
        hdr_ = reinterpret_cast<AudioRingHeader*>(shm);
        data_ = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(shm) + sizeof(AudioRingHeader));
        cap_ = capacity;
        mask_ = capacity - 1; // capacity must be power of 2

        hdr_->magic = RING_MAGIC;
        hdr_->version = RING_VERSION;
        hdr_->sample_rate = sample_rate;
        hdr_->capacity = capacity;
        hdr_->write_idx.store(0, std::memory_order_relaxed);
        hdr_->xruns.store(0, std::memory_order_relaxed);
        hdr_->overruns.store(0, std::memory_order_relaxed);
        hdr_->period_write_idx.store(0, std::memory_order_relaxed);
        hdr_->gap_write_idx.store(0, std::memory_order_relaxed);
    }

    /// Write mono float32 samples into the ring. RT-safe (no allocation).
    /// t_ns = CLOCK_MONOTONIC timestamp of the last sample in this batch.
    void write(const float* mono, uint32_t n, int64_t t_ns) {
        uint64_t wi = hdr_->write_idx.load(std::memory_order_relaxed);
        uint32_t pos = static_cast<uint32_t>(wi & mask_);

        // Handle wraparound: may need two memcpys
        uint32_t first = cap_ - pos;
        if (first >= n) {
            std::memcpy(data_ + pos, mono, n * sizeof(float));
        } else {
            std::memcpy(data_ + pos, mono, first * sizeof(float));
            std::memcpy(data_, mono + first, (n - first) * sizeof(float));
        }

        uint64_t new_wi = wi + n;

        // Record period timestamp
        uint32_t pi = hdr_->period_write_idx.load(std::memory_order_relaxed);
        uint32_t slot = pi % PERIOD_TABLE_SIZE;
        hdr_->period_table[slot].sample_idx_end = new_wi;
        hdr_->period_table[slot].t_ns = t_ns;
        hdr_->period_write_idx.store(pi + 1, std::memory_order_relaxed);

        // Publish write index with release semantics
        hdr_->write_idx.store(new_wi, std::memory_order_release);
    }

    /// Record a gap (xrun).
    void record_gap(uint64_t sample_idx, uint32_t lost_frames) {
        uint32_t gi = hdr_->gap_write_idx.load(std::memory_order_relaxed);
        uint32_t slot = gi % GAP_TABLE_SIZE;
        hdr_->gap_table[slot].sample_idx = sample_idx;
        hdr_->gap_table[slot].lost_frames = lost_frames;
        hdr_->gap_write_idx.store(gi + 1, std::memory_order_relaxed);
        hdr_->xruns.fetch_add(1, std::memory_order_relaxed);
    }

    /// Advance write_idx by lost_frames without writing data (after xrun).
    void advance_for_gap(uint32_t lost_frames) {
        uint64_t wi = hdr_->write_idx.load(std::memory_order_relaxed);
        hdr_->write_idx.store(wi + lost_frames, std::memory_order_release);
    }

    uint64_t write_idx() const {
        return hdr_->write_idx.load(std::memory_order_acquire);
    }

    AudioRingHeader* header() { return hdr_; }

private:
    AudioRingHeader* hdr_ = nullptr;
    float* data_ = nullptr;
    uint32_t cap_ = 0;
    uint32_t mask_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// AudioRingReader — used by assembler and telemetry (multiple readers)
// ─────────────────────────────────────────────────────────────────────────────

class AudioRingReader {
public:
    AudioRingReader() = default;

    /// Initialize from shared memory (can be read-only mapped).
    void init(const void* shm) {
        hdr_ = reinterpret_cast<const AudioRingHeader*>(shm);
        data_ = reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(shm) + sizeof(AudioRingHeader));
        cap_ = hdr_->capacity;
        mask_ = cap_ - 1;
    }

    /// Read n samples starting at absolute index 'start' into dst.
    /// Returns true if the read is valid (data was not overwritten during copy).
    bool read(uint64_t start, float* dst, uint32_t n) const {
        uint64_t wi = hdr_->write_idx.load(std::memory_order_acquire);

        // Check that the requested range is still in the ring
        if (start + n > wi) return false;          // not yet written
        if (wi - start > cap_) return false;        // overwritten (lapped)

        uint32_t pos = static_cast<uint32_t>(start & mask_);
        uint32_t first = cap_ - pos;
        if (first >= n) {
            std::memcpy(dst, data_ + pos, n * sizeof(float));
        } else {
            std::memcpy(dst, data_ + pos, first * sizeof(float));
            std::memcpy(dst + first, data_, (n - first) * sizeof(float));
        }

        // Validate: re-read write_idx to ensure data wasn't overwritten
        uint64_t wi2 = hdr_->write_idx.load(std::memory_order_acquire);
        if (wi2 - start > cap_) return false;  // lapped during copy

        return true;
    }

    uint64_t write_idx() const {
        return hdr_->write_idx.load(std::memory_order_acquire);
    }

    uint32_t capacity() const { return cap_; }
    uint32_t sample_rate() const { return hdr_->sample_rate; }
    const AudioRingHeader* header() const { return hdr_; }

    /// Interpolate capture timestamp for a given absolute sample index.
    /// Searches the period table for bracketing entries.
    int64_t estimate_timestamp(uint64_t sample_idx) const {
        uint32_t pi = hdr_->period_write_idx.load(std::memory_order_acquire);
        if (pi == 0) return 0;

        // Find the most recent period entry at or after sample_idx
        for (uint32_t i = 0; i < PERIOD_TABLE_SIZE && i < pi; ++i) {
            uint32_t slot = (pi - 1 - i) % PERIOD_TABLE_SIZE;
            const auto& pe = hdr_->period_table[slot];
            if (pe.sample_idx_end >= sample_idx) {
                // Linear interpolation: time of sample relative to period end
                double dt = (double)(pe.sample_idx_end - sample_idx)
                          / (double)hdr_->sample_rate;
                return pe.t_ns - (int64_t)(dt * 1e9);
            }
        }
        // Fallback: use most recent entry
        uint32_t slot = (pi - 1) % PERIOD_TABLE_SIZE;
        return hdr_->period_table[slot].t_ns;
    }

private:
    const AudioRingHeader* hdr_ = nullptr;
    const float* data_ = nullptr;
    uint32_t cap_ = 0;
    uint32_t mask_ = 0;
};

} // namespace hs


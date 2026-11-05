#pragma once
// events.h — Phoneme event ring buffer (Section 5.6)
// Single writer (post-process), multiple independent readers.
// Lock-free, writer never blocks. Readers detect lapping and torn reads.

#include <atomic>
#include <cstdint>
#include <cstring>

#include "rt_util.h"

namespace hs {

static constexpr const char* SHM_EVENTS = "/hapticsync_events";
static constexpr uint32_t EVENTS_MAGIC   = 0x48455631; // "HEV1"
static constexpr uint32_t EVENTS_VERSION = 1;

// ── PhonemeEvent: 64-byte fixed slot, little-endian ──────────────────────────

struct PhonemeEvent {
    uint64_t seq;          // odd = being written, even = valid (2*id)
    uint64_t event_id;     // monotonically increasing, starts at 1
    int64_t  t_frame_ns;   // CLOCK_MONOTONIC capture time of frame center
    int64_t  t_emit_ns;    // time written to ring
    uint64_t abs_frame;    // absolute 20ms frame index since stream start
    uint32_t window_id;
    uint16_t phoneme_id;   // model token id
    uint16_t flags;        // bit0: first event after gap
    float    confidence;   // softmax probability of the frame label
    char     phoneme[12];  // NUL-terminated label from vocab.json
};
static_assert(sizeof(PhonemeEvent) == 64, "PhonemeEvent must be 64 bytes");

// Flags
static constexpr uint16_t EVT_FLAG_FIRST_AFTER_GAP = 0x0001;

// ── Event ring header ────────────────────────────────────────────────────────

struct alignas(64) EventRingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t slot_count;   // default 1024
    uint32_t _pad0;
    std::atomic<uint64_t> head;  // next event_id to be written (starts at 1)
    int64_t  stream_start_ns;
    char     _pad1[32];
};

/// Total shared memory size for event ring.
inline size_t event_ring_shm_size(uint32_t slot_count) {
    return sizeof(EventRingHeader) + slot_count * sizeof(PhonemeEvent);
}

// ── EventRingWriter (single writer — post-process) ──────────────────────────

class EventRingWriter {
public:
    EventRingWriter() = default;

    void init(void* shm, uint32_t slot_count) {
        hdr_ = reinterpret_cast<EventRingHeader*>(shm);
        slots_ = reinterpret_cast<PhonemeEvent*>(
            reinterpret_cast<uint8_t*>(shm) + sizeof(EventRingHeader));
        slot_count_ = slot_count;

        hdr_->magic = EVENTS_MAGIC;
        hdr_->version = EVENTS_VERSION;
        hdr_->slot_count = slot_count;
        hdr_->head.store(1, std::memory_order_relaxed);
        hdr_->stream_start_ns = monotonic_ns();

        // Zero all slots
        std::memset(slots_, 0, slot_count * sizeof(PhonemeEvent));
    }

    /// Write a phoneme event. Returns the assigned event_id.
    uint64_t write(uint16_t phoneme_id, const char* phoneme_str,
                   float confidence, int64_t t_frame_ns, uint64_t abs_frame,
                   uint32_t window_id, uint16_t flags) {
        uint64_t id = hdr_->head.load(std::memory_order_relaxed);
        uint32_t slot = static_cast<uint32_t>((id - 1) % slot_count_);

        PhonemeEvent& e = slots_[slot];

        // Mark as being written (odd seq)
        reinterpret_cast<std::atomic<uint64_t>*>(&e.seq)->store(
            2 * id - 1, std::memory_order_release);

        e.event_id    = id;
        e.t_frame_ns  = t_frame_ns;
        e.t_emit_ns   = monotonic_ns();
        e.abs_frame   = abs_frame;
        e.window_id   = window_id;
        e.phoneme_id  = phoneme_id;
        e.flags       = flags;
        e.confidence  = confidence;

        // Copy phoneme string (NUL-terminated, max 11 chars + NUL)
        std::memset(e.phoneme, 0, sizeof(e.phoneme));
        if (phoneme_str) {
            std::strncpy(e.phoneme, phoneme_str, sizeof(e.phoneme) - 1);
        }

        // Mark as valid (even seq = 2*id)
        reinterpret_cast<std::atomic<uint64_t>*>(&e.seq)->store(
            2 * id, std::memory_order_release);

        // Advance head
        hdr_->head.store(id + 1, std::memory_order_release);

        return id;
    }

    uint64_t head() const {
        return hdr_->head.load(std::memory_order_acquire);
    }

private:
    EventRingHeader* hdr_ = nullptr;
    PhonemeEvent* slots_ = nullptr;
    uint32_t slot_count_ = 0;
};

// ── EventRingReader (one per consumer, each maintains its own cursor) ───────

class EventRingReader {
public:
    EventRingReader() = default;

    void init(const void* shm) {
        hdr_ = reinterpret_cast<const EventRingHeader*>(shm);
        slots_ = reinterpret_cast<const PhonemeEvent*>(
            reinterpret_cast<const uint8_t*>(shm) + sizeof(EventRingHeader));
        slot_count_ = hdr_->slot_count;
        cursor_ = 1; // start from event_id 1
        lapped_ = 0;
    }

    /// Try to read the next event. Returns true if an event was read.
    /// Sets 'was_lapped' if events were skipped.
    bool read(PhonemeEvent& out, bool& was_lapped) {
        was_lapped = false;
        uint64_t head = hdr_->head.load(std::memory_order_acquire);

        if (cursor_ >= head) return false; // no new events

        // Check for lapping
        if (head - cursor_ > slot_count_) {
            lapped_ += (head - slot_count_) - cursor_;
            cursor_ = head - slot_count_; // resync to oldest available
            was_lapped = true;
        }

        uint32_t slot = static_cast<uint32_t>((cursor_ - 1) % slot_count_);
        const PhonemeEvent& e = slots_[slot];

        // Read with torn-read detection
        uint64_t seq1 = reinterpret_cast<const std::atomic<uint64_t>*>(&e.seq)->load(
            std::memory_order_acquire);

        // Copy
        std::memcpy(&out, &e, sizeof(PhonemeEvent));

        uint64_t seq2 = reinterpret_cast<const std::atomic<uint64_t>*>(&e.seq)->load(
            std::memory_order_acquire);

        // Validate: both seqs must be equal, even, and match expected id
        if (seq1 != seq2 || (seq1 & 1) != 0 || seq1 != 2 * cursor_) {
            // Torn read or slot was overwritten — skip
            was_lapped = true;
            lapped_++;
            cursor_++;
            return false;
        }

        cursor_++;
        return true;
    }

    /// Reset cursor to the current head (skip all pending).
    void sync_to_head() {
        cursor_ = hdr_->head.load(std::memory_order_acquire);
    }

    uint64_t cursor() const { return cursor_; }
    uint64_t lapped_count() const { return lapped_; }

private:
    const EventRingHeader* hdr_ = nullptr;
    const PhonemeEvent* slots_ = nullptr;
    uint32_t slot_count_ = 0;
    uint64_t cursor_ = 1;
    uint64_t lapped_ = 0;
};

} // namespace hs


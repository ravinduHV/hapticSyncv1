#pragma once
// protocol.h — IPC message formats (Section 6)
// All fields little-endian, packed, explicit sizes.
// Messages sent over AF_UNIX SOCK_SEQPACKET (one message per datagram).

#include <cstdint>
#include <cstring>

namespace hs {

enum MsgType : uint32_t {
    MSG_WINDOW = 1,
    MSG_DONE   = 2,
    MSG_RESULT = 3,
};

static constexpr uint32_t PROTOCOL_VERSION = 1;

// Flags
static constexpr uint32_t WFLAG_FIRST_AFTER_GAP = 0x01;

// ── WindowMsg: assembler → inference ─────────────────────────────────────────

#pragma pack(push, 1)
struct WindowMsg {
    uint32_t type;          // MSG_WINDOW
    uint32_t version;       // PROTOCOL_VERSION
    uint64_t window_id;
    uint32_t slot;          // 0 or 1 (double-buffered)
    uint32_t n_samples;     // = (n_frames-1)*320 + 400
    uint64_t start_sample;  // absolute, multiple of 320
    int64_t  t_start_ns;    // capture time of start_sample
    int64_t  t_sent_ns;     // time this message was sent
    uint32_t n_frames;      // expected model output frames
    uint32_t commit_begin;  // frame index inside the window (inclusive)
    uint32_t commit_end;    // frame index inside the window (exclusive)
    uint32_t flags;         // bit0: first_after_gap
};
#pragma pack(pop)

// ── DoneMsg: inference → assembler ──────────────────────────────────────────

#pragma pack(push, 1)
struct DoneMsg {
    uint32_t type;          // MSG_DONE
    uint32_t version;       // PROTOCOL_VERSION
    uint64_t window_id;
    int64_t  t_infer_begin_ns;
    int64_t  t_infer_end_ns;
    uint32_t status;        // 0 = ok, nonzero = error (window discarded)
    uint32_t _pad;
};
#pragma pack(pop)

// ── ResultHeader: inference → post-process, followed by n_frames * FrameOut ─

#pragma pack(push, 1)
struct ResultHeader {
    uint32_t type;          // MSG_RESULT
    uint32_t version;       // PROTOCOL_VERSION
    uint64_t window_id;
    uint64_t start_sample;
    int64_t  t_start_ns;
    int64_t  t_infer_begin_ns;
    int64_t  t_infer_end_ns;
    uint32_t n_frames;
    uint32_t commit_begin;
    uint32_t commit_end;
    uint32_t flags;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct FrameOut {
    uint16_t label;
    uint16_t pad;
    float    conf;
};
#pragma pack(pop)

static_assert(sizeof(FrameOut) == 8, "FrameOut must be 8 bytes");

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Frame grid formulas (Wav2Vec2 geometry, Section 7.1)
/// hop = 320 samples (20 ms), receptive field = 400 samples (25 ms)
inline uint32_t n_frames_from_samples(uint32_t n_samples) {
    if (n_samples < 400) return 0;
    return (n_samples - 400) / 320 + 1;
}

inline uint32_t n_samples_from_frames(uint32_t n_frames) {
    if (n_frames == 0) return 0;
    return (n_frames - 1) * 320 + 400;
}

/// Absolute frame index: window starts at sample s, frame i within window.
/// s must be a multiple of 320.
inline uint64_t abs_frame_idx(uint64_t start_sample, uint32_t frame_in_window) {
    return start_sample / 320 + frame_in_window;
}

/// Capture time of a frame's center sample.
/// frame_in_window is the index within the window.
inline int64_t frame_center_time_ns(int64_t t_start_ns, uint32_t frame_in_window) {
    // center of frame i is at sample offset 320*i + 200
    double offset_s = (320.0 * frame_in_window + 200.0) / 16000.0;
    return t_start_ns + static_cast<int64_t>(offset_s * 1e9);
}

} // namespace hs


#pragma once
// ctc_merge.h — Stateful CTC collapse across window boundaries (Section 7.4)

#include <cstdint>
#include <optional>
#include "../common/events.h"
#include "../common/protocol.h"
#include "vocab.h"

namespace hs {

class CtcMerger {
public:
    explicit CtcMerger(const Vocab& vocab, float min_confidence = 0.0f)
        : vocab_(vocab), min_conf_(min_confidence) {
        reset();
    }

    /// Reset state (e.g. on stream start or first window after gap)
    void reset() {
        prev_label_ = vocab_.blank_id();
    }

    /// Process a single committed frame. If it marks a new run onset, returns a PhonemeEvent.
    std::optional<PhonemeEvent> process_frame(
        uint16_t label, float conf, uint64_t abs_frame,
        int64_t t_frame_ns, uint32_t window_id, uint16_t flags)
    {
        std::optional<PhonemeEvent> evt = std::nullopt;

        uint16_t blank = vocab_.blank_id();
        if (label != blank && label != prev_label_) {
            if (conf >= min_conf_) {
                PhonemeEvent e{};
                e.event_id = 0; // will be assigned by event ring writer
                e.t_frame_ns = t_frame_ns;
                e.t_emit_ns = 0; // assigned by ring writer
                e.abs_frame = abs_frame;
                e.window_id = window_id;
                e.phoneme_id = label;
                e.flags = flags;
                e.confidence = conf;
                const char* s = vocab_.get(label);
                if (s) {
                    std::strncpy(e.phoneme, s, sizeof(e.phoneme) - 1);
                }
                evt = e;
            }
        }
        prev_label_ = label;
        return evt;
    }

    uint16_t prev_label() const { return prev_label_; }

private:
    const Vocab& vocab_;
    float min_conf_;
    uint16_t prev_label_ = 41;
};

} // namespace hs


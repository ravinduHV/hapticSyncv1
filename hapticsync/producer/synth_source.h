#pragma once
// synth_source.h — Synthetic audio source for latency testing (Section 5.2.2)
// Generates sine waves or click patterns, paced with clock_nanosleep.

#include <cstdint>
#include <cmath>
#include <cstring>

#ifdef __linux__
#include <time.h>
#endif

#include "../common/rt_util.h"

namespace hs {

class SynthSource {
public:
    enum Mode { SINE, CLICK };

    SynthSource(int rate, int channels, Mode mode = SINE,
                float freq = 440.0f, float amplitude = 0.5f,
                float click_interval_s = 1.0f)
        : rate_(rate), channels_(channels), mode_(mode),
          freq_(freq), amplitude_(amplitude),
          click_interval_samples_(static_cast<int>(click_interval_s * rate)) {}

    ~SynthSource() = default;

    bool open() {
        phase_ = 0.0;
        sample_counter_ = 0;
        pacing_started_ = false;
        return true;
    }

    /// Generate and pace one period of interleaved float32 samples.
    /// buf must hold frames * channels * sizeof(float) bytes.
    /// Returns the number of frames generated.
    int read(float* buf, int frames) {
        pace(frames);

        for (int i = 0; i < frames; ++i) {
            float sample = 0.0f;

            if (mode_ == SINE) {
                sample = amplitude_ * std::sin(
                    2.0 * M_PI * freq_ * phase_ / rate_);
                phase_ += 1.0;
            } else if (mode_ == CLICK) {
                // Short click pulse at regular intervals
                int pos_in_interval = sample_counter_ % click_interval_samples_;
                if (pos_in_interval < 8) { // 8-sample click
                    sample = amplitude_;
                }
            }

            // Write same sample to all channels (interleaved)
            for (int ch = 0; ch < channels_; ++ch) {
                buf[i * channels_ + ch] = sample;
            }
            sample_counter_++;
        }

        return frames;
    }

    int64_t get_timestamp() {
        return monotonic_ns();
    }

    void close() {}

private:
    void pace(int frames) {
#ifdef __linux__
        int64_t period_ns = static_cast<int64_t>(frames) * 1000000000LL / rate_;

        if (!pacing_started_) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            next_wake_.tv_sec = now.tv_sec;
            next_wake_.tv_nsec = now.tv_nsec;
            pacing_started_ = true;
        }

        next_wake_.tv_nsec += period_ns;
        while (next_wake_.tv_nsec >= 1000000000L) {
            next_wake_.tv_sec += 1;
            next_wake_.tv_nsec -= 1000000000L;
        }

        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake_, nullptr);
#else
        (void)frames;
#endif
    }

    int rate_;
    int channels_;
    Mode mode_;
    float freq_;
    float amplitude_;
    int click_interval_samples_;
    double phase_ = 0.0;
    int sample_counter_ = 0;
    bool pacing_started_ = false;

#ifdef __linux__
    struct timespec next_wake_ = {0, 0};
#endif
};

} // namespace hs


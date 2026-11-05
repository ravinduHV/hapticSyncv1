// alsa_source.cpp — ALSA direct capture implementation (Section 5.2.2)

#include "alsa_source.h"

#include <cstdio>
#include <cstring>

namespace hs {

AlsaSource::AlsaSource(const std::string& device, int rate, int channels,
                       int period_frames, int buffer_periods)
    : device_(device), rate_(rate), channels_(channels),
      period_frames_(period_frames), buffer_periods_(buffer_periods) {}

AlsaSource::~AlsaSource() {
    close();
}

#ifdef __linux__

bool AlsaSource::open() {
    int err;

    err = snd_pcm_open(&pcm_, device_.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot open %s: %s\n",
                     device_.c_str(), snd_strerror(err));
        return false;
    }

    // ── Hardware params ─────────────────────────────────────────────────
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm_, hw);

    snd_pcm_hw_params_set_access(pcm_, hw, SND_PCM_ACCESS_RW_INTERLEAVED);

    // Try S32_LE first (common for I2S MEMS mics), fallback to S16_LE
    err = snd_pcm_hw_params_set_format(pcm_, hw, SND_PCM_FORMAT_S32_LE);
    if (err == 0) {
        is_s32_ = true;
        sample_bytes_ = 4;
        std::fprintf(stderr, "[alsa] Format: S32_LE\n");
    } else {
        err = snd_pcm_hw_params_set_format(pcm_, hw, SND_PCM_FORMAT_S16_LE);
        if (err < 0) {
            std::fprintf(stderr, "[alsa] Cannot set format: %s\n", snd_strerror(err));
            return false;
        }
        is_s32_ = false;
        sample_bytes_ = 2;
        std::fprintf(stderr, "[alsa] Format: S16_LE\n");
    }

    unsigned int actual_rate = static_cast<unsigned int>(rate_);
    err = snd_pcm_hw_params_set_rate_near(pcm_, hw, &actual_rate, nullptr);
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot set rate %d: %s\n", rate_, snd_strerror(err));
        return false;
    }
    if (static_cast<int>(actual_rate) != rate_) {
        std::fprintf(stderr, "[alsa] WARNING: requested rate %d, got %u\n",
                     rate_, actual_rate);
    }

    err = snd_pcm_hw_params_set_channels(pcm_, hw, static_cast<unsigned>(channels_));
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot set channels %d: %s\n",
                     channels_, snd_strerror(err));
        return false;
    }

    snd_pcm_uframes_t period = static_cast<snd_pcm_uframes_t>(period_frames_);
    err = snd_pcm_hw_params_set_period_size_near(pcm_, hw, &period, nullptr);
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot set period size: %s\n", snd_strerror(err));
        return false;
    }

    snd_pcm_uframes_t buffer = period * static_cast<unsigned>(buffer_periods_);
    err = snd_pcm_hw_params_set_buffer_size_near(pcm_, hw, &buffer);
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot set buffer size: %s\n", snd_strerror(err));
        return false;
    }

    err = snd_pcm_hw_params(pcm_, hw);
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot apply hw params: %s\n", snd_strerror(err));
        return false;
    }

    // ── Software params ─────────────────────────────────────────────────
    snd_pcm_sw_params_t* sw;
    snd_pcm_sw_params_alloca(&sw);
    snd_pcm_sw_params_current(pcm_, sw);

    snd_pcm_sw_params_set_avail_min(pcm_, sw, period);
    // Enable MONOTONIC timestamps
    snd_pcm_sw_params_set_tstamp_mode(pcm_, sw, SND_PCM_TSTAMP_ENABLE);
    snd_pcm_sw_params_set_tstamp_type(pcm_, sw, SND_PCM_TSTAMP_TYPE_MONOTONIC);

    err = snd_pcm_sw_params(pcm_, sw);
    if (err < 0) {
        std::fprintf(stderr, "[alsa] Cannot apply sw params: %s\n", snd_strerror(err));
        return false;
    }

    // Allocate status for timestamps
    snd_pcm_status_malloc(&status_);

    std::fprintf(stderr, "[alsa] Opened %s: rate=%u ch=%d period=%lu buf=%lu\n",
                 device_.c_str(), actual_rate, channels_,
                 (unsigned long)period, (unsigned long)buffer);

    return true;
}

int AlsaSource::read(void* buf, int frames) {
    snd_pcm_sframes_t n = snd_pcm_readi(pcm_, buf, static_cast<snd_pcm_uframes_t>(frames));
    if (n < 0) {
        if (handle_xrun(static_cast<int>(n))) {
            return 0; // recovered, caller should retry
        }
        return -1; // unrecoverable
    }
    return static_cast<int>(n);
}

int64_t AlsaSource::get_timestamp() {
    if (!pcm_ || !status_) return 0;

    snd_pcm_status(pcm_, status_);
    snd_htimestamp_t ts;
    snd_pcm_status_get_htstamp(status_, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL
         + static_cast<int64_t>(ts.tv_nsec);
}

void AlsaSource::close() {
    if (pcm_) {
        snd_pcm_drop(pcm_);
        snd_pcm_close(pcm_);
        pcm_ = nullptr;
    }
    if (status_) {
        snd_pcm_status_free(status_);
        status_ = nullptr;
    }
}

bool AlsaSource::handle_xrun(int err) {
    if (err == -EPIPE) {
        // Buffer overrun
        xruns_++;
        std::fprintf(stderr, "[alsa] XRUN (overrun) #%u — recovering\n", xruns_);
        int r = snd_pcm_prepare(pcm_);
        return (r >= 0);
    }
    if (err == -ESTRPIPE) {
        // Suspended
        xruns_++;
        std::fprintf(stderr, "[alsa] Suspended — resuming\n");
        int r;
        while ((r = snd_pcm_resume(pcm_)) == -EAGAIN) {
            // wait
            struct timespec ts = {0, 100000000}; // 100ms
            nanosleep(&ts, nullptr);
        }
        if (r < 0) {
            r = snd_pcm_prepare(pcm_);
        }
        return (r >= 0);
    }
    std::fprintf(stderr, "[alsa] Read error: %s\n", snd_strerror(err));
    return false;
}

#else
// ── Non-Linux stubs ─────────────────────────────────────────────────────────

bool AlsaSource::open() {
    std::fprintf(stderr, "[alsa] ALSA not available on this platform\n");
    return false;
}
int AlsaSource::read(void*, int) { return -1; }
int64_t AlsaSource::get_timestamp() { return 0; }
void AlsaSource::close() {}
bool AlsaSource::handle_xrun(int) { return false; }

#endif // __linux__

} // namespace hs


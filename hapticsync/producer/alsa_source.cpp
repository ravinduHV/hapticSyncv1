// alsa_source.cpp — ALSA direct capture implementation (Section 5.2.2)

#include "alsa_source.h"
#include "../common/rt_util.h"

#include <cstdio>
#include <cstring>
#include <vector>

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
    int err = -1;

    // List of candidate device strings to try if configured device is "loopback"
    std::vector<std::string> candidates;
    if (device_ == "loopback" || device_ == "aloop") {
        candidates.push_back("hw:Loopback,1,0");
        candidates.push_back("hw:Loopback,1");
        candidates.push_back("plughw:Loopback,1");
        candidates.push_back("hw:1,1");
        candidates.push_back("plughw:1,1");
    } else {
        candidates.push_back(device_);
        // If device has Loopback, add plughw fallback
        if (device_.find("Loopback") != std::string::npos && device_.rfind("plug", 0) != 0) {
            candidates.push_back("plug" + device_);
        }
    }

    for (const auto& dev : candidates) {
        err = snd_pcm_open(&pcm_, dev.c_str(), SND_PCM_STREAM_CAPTURE, 0);
        if (err == 0) {
            active_device_ = dev;
            std::fprintf(stderr, "[alsa] Successfully opened ALSA capture device: %s\n", dev.c_str());
            break;
        } else {
            std::fprintf(stderr, "[alsa] Could not open %s: %s\n", dev.c_str(), snd_strerror(err));
        }
    }

    if (err < 0 || !pcm_) {
        std::fprintf(stderr, "[alsa] FATAL: Failed to open any ALSA capture device.\n");
        return false;
    }

    // ── Hardware params ─────────────────────────────────────────────────
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm_, hw);

    snd_pcm_hw_params_set_access(pcm_, hw, SND_PCM_ACCESS_RW_INTERLEAVED);

    // Try S32_LE first (common for I2S MEMS mics), then S16_LE, then FLOAT_LE
    err = snd_pcm_hw_params_set_format(pcm_, hw, SND_PCM_FORMAT_S32_LE);
    if (err == 0) {
        format_ = FORMAT_S32_LE;
        sample_bytes_ = 4;
        std::fprintf(stderr, "[alsa] Format: S32_LE\n");
    } else {
        err = snd_pcm_hw_params_set_format(pcm_, hw, SND_PCM_FORMAT_S16_LE);
        if (err == 0) {
            format_ = FORMAT_S16_LE;
            sample_bytes_ = 2;
            std::fprintf(stderr, "[alsa] Format: S16_LE\n");
        } else {
            err = snd_pcm_hw_params_set_format(pcm_, hw, SND_PCM_FORMAT_FLOAT_LE);
            if (err == 0) {
                format_ = FORMAT_FLOAT_LE;
                sample_bytes_ = 4;
                std::fprintf(stderr, "[alsa] Format: FLOAT_LE\n");
            } else {
                std::fprintf(stderr, "[alsa] Cannot set format (S32/S16/FLOAT): %s\n", snd_strerror(err));
                return false;
            }
        }
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
    // Enable MONOTONIC timestamps (graceful fallback if unsupported by virtual/loopback device)
    snd_pcm_sw_params_set_tstamp_mode(pcm_, sw, SND_PCM_TSTAMP_ENABLE);
    snd_pcm_sw_params_set_tstamp_type(pcm_, sw, SND_PCM_TSTAMP_TYPE_MONOTONIC);

    err = snd_pcm_sw_params(pcm_, sw);
    if (err < 0) {
        // Fallback without explicit monotonic type
        snd_pcm_sw_params_current(pcm_, sw);
        snd_pcm_sw_params_set_avail_min(pcm_, sw, period);
        snd_pcm_sw_params_set_tstamp_mode(pcm_, sw, SND_PCM_TSTAMP_ENABLE);
        err = snd_pcm_sw_params(pcm_, sw);
        if (err < 0) {
            std::fprintf(stderr, "[alsa] WARNING: Cannot apply sw params: %s (continuing with default sw params)\n", snd_strerror(err));
        }
    }

    // Allocate status for timestamps
    snd_pcm_status_malloc(&status_);

    std::fprintf(stderr, "[alsa] Opened %s: rate=%u ch=%d period=%lu buf=%lu\n",
                 active_device_.c_str(), actual_rate, channels_,
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
    if (!pcm_ || !status_) return monotonic_ns();

    snd_pcm_status(pcm_, status_);
    snd_htimestamp_t ts;
    snd_pcm_status_get_htstamp(status_, &ts);
    int64_t ns = static_cast<int64_t>(ts.tv_sec) * 1000000000LL
               + static_cast<int64_t>(ts.tv_nsec);
    if (ns <= 0) {
        return monotonic_ns();
    }
    return ns;
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
int64_t AlsaSource::get_timestamp() { return monotonic_ns(); }
void AlsaSource::close() {}
bool AlsaSource::handle_xrun(int) { return false; }

#endif // __linux__

} // namespace hs



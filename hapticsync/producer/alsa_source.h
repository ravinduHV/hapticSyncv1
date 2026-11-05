#pragma once
// alsa_source.h — ALSA direct capture source (Section 5.2.2)

#include <cstdint>
#include <string>

#ifdef __linux__
#include <alsa/asoundlib.h>
#endif

namespace hs {

class AlsaSource {
public:
    AlsaSource(const std::string& device, int rate, int channels,
               int period_frames, int buffer_periods);
    ~AlsaSource();

    AlsaSource(const AlsaSource&) = delete;
    AlsaSource& operator=(const AlsaSource&) = delete;

    /// Open and configure the ALSA device. Returns true on success.
    bool open();

    /// Blocking read of one period. Returns number of frames read, or <0 on error.
    /// buf must hold period_frames * channels * sample_bytes bytes.
    int read(void* buf, int frames);

    /// Get CLOCK_MONOTONIC timestamp of the last read's final sample (nanoseconds).
    int64_t get_timestamp();

    /// Close the device.
    void close();

    /// Number of xruns encountered since open().
    uint32_t xrun_count() const { return xruns_; }

    /// Bytes per sample as configured by hardware.
    int sample_bytes() const { return sample_bytes_; }

    /// True if samples are S32_LE (need /2^31 conversion), false if S16_LE (/2^15).
    bool is_s32() const { return is_s32_; }

private:
    bool handle_xrun(int err);

    std::string device_;
    int rate_;
    int channels_;
    int period_frames_;
    int buffer_periods_;
    uint32_t xruns_ = 0;
    int sample_bytes_ = 2;
    bool is_s32_ = false;

#ifdef __linux__
    snd_pcm_t* pcm_ = nullptr;
    snd_pcm_status_t* status_ = nullptr;
#endif
};

} // namespace hs


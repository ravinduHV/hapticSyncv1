#pragma once
// wav_source.h — WAV file replay source, paced with clock_nanosleep (Section 5.2.2)

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>

#ifdef __linux__
#include <time.h>
#endif

#include "../common/rt_util.h"

namespace hs {

/// Simple WAV header (enough for PCM format).
struct WavHeader {
    char     riff[4];       // "RIFF"
    uint32_t file_size;
    char     wave[4];       // "WAVE"
    // fmt chunk
    char     fmt_id[4];     // "fmt "
    uint32_t fmt_size;
    uint16_t audio_format;  // 1 = PCM, 3 = IEEE float
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

class WavSource {
public:
    WavSource(const std::string& path, int target_rate, int target_channels)
        : path_(path), target_rate_(target_rate), target_channels_(target_channels) {}

    ~WavSource() { close(); }

    WavSource(const WavSource&) = delete;
    WavSource& operator=(const WavSource&) = delete;

    /// Open and parse the WAV file. Returns true on success.
    bool open() {
        file_ = std::fopen(path_.c_str(), "rb");
        if (!file_) {
            std::fprintf(stderr, "[wav] Cannot open %s\n", path_.c_str());
            return false;
        }

        WavHeader hdr;
        if (std::fread(&hdr, sizeof(hdr), 1, file_) != 1) {
            std::fprintf(stderr, "[wav] Cannot read header\n");
            return false;
        }

        if (std::memcmp(hdr.riff, "RIFF", 4) != 0 ||
            std::memcmp(hdr.wave, "WAVE", 4) != 0) {
            std::fprintf(stderr, "[wav] Not a valid WAV file\n");
            return false;
        }

        file_channels_ = hdr.channels;
        file_rate_ = static_cast<int>(hdr.sample_rate);
        file_bits_ = hdr.bits_per_sample;
        audio_format_ = hdr.audio_format;

        if (file_rate_ != target_rate_) {
            std::fprintf(stderr, "[wav] WARNING: file rate %d != target %d\n",
                         file_rate_, target_rate_);
        }

        // Find data chunk
        // Skip any extra fmt bytes
        if (hdr.fmt_size > 16) {
            std::fseek(file_, hdr.fmt_size - 16, SEEK_CUR);
        }

        // Search for "data" chunk
        char chunk_id[4];
        uint32_t chunk_size;
        while (std::fread(chunk_id, 4, 1, file_) == 1) {
            if (std::fread(&chunk_size, 4, 1, file_) != 1) break;
            if (std::memcmp(chunk_id, "data", 4) == 0) {
                data_start_ = std::ftell(file_);
                data_size_ = chunk_size;
                data_remaining_ = data_size_;

                std::fprintf(stderr, "[wav] Opened %s: rate=%d ch=%d bits=%d "
                             "data=%u bytes\n",
                             path_.c_str(), file_rate_, file_channels_,
                             file_bits_, data_size_);
                pacing_started_ = false;
                return true;
            }
            std::fseek(file_, chunk_size, SEEK_CUR);
        }

        std::fprintf(stderr, "[wav] No data chunk found\n");
        return false;
    }

    /// Read one period of interleaved samples into buf (as raw bytes).
    /// Paces the read to simulate real-time using clock_nanosleep(ABSTIME).
    /// Returns number of frames read.
    int read(void* buf, int frames) {
        if (!file_) return -1;

        // Pace: simulate real-time delivery
        pace(frames);

        int frame_bytes = file_channels_ * (file_bits_ / 8);
        int bytes_wanted = frames * frame_bytes;

        if (data_remaining_ <= 0) {
            // Loop: seek back to data start
            std::fseek(file_, data_start_, SEEK_SET);
            data_remaining_ = data_size_;
        }

        int to_read = bytes_wanted;
        if (to_read > static_cast<int>(data_remaining_)) {
            to_read = static_cast<int>(data_remaining_);
        }

        size_t got = std::fread(buf, 1, static_cast<size_t>(to_read), file_);
        data_remaining_ -= static_cast<uint32_t>(got);

        return static_cast<int>(got) / frame_bytes;
    }

    /// Returns current CLOCK_MONOTONIC timestamp.
    int64_t get_timestamp() {
        return monotonic_ns();
    }

    void close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    int file_channels() const { return file_channels_; }
    int file_rate() const { return file_rate_; }
    int file_bits() const { return file_bits_; }
    uint16_t audio_format() const { return audio_format_; }

private:
    void pace(int frames) {
#ifdef __linux__
        int64_t period_ns = static_cast<int64_t>(frames) * 1000000000LL / file_rate_;

        if (!pacing_started_) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            next_wake_.tv_sec = now.tv_sec;
            next_wake_.tv_nsec = now.tv_nsec;
            pacing_started_ = true;
        }

        // Advance the absolute wake time
        next_wake_.tv_nsec += period_ns;
        while (next_wake_.tv_nsec >= 1000000000L) {
            next_wake_.tv_sec += 1;
            next_wake_.tv_nsec -= 1000000000L;
        }

        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake_, nullptr);
#else
        (void)frames;
        // On non-Linux, use a basic sleep approximation
        auto period_ns = static_cast<int64_t>(frames) * 1000000000LL / file_rate_;
        auto period_ms = period_ns / 1000000;
        if (period_ms > 0) {
            // platform sleep
        }
#endif
    }

    std::string path_;
    int target_rate_;
    int target_channels_;
    int file_channels_ = 0;
    int file_rate_ = 0;
    int file_bits_ = 0;
    uint16_t audio_format_ = 0;
    FILE* file_ = nullptr;
    long data_start_ = 0;
    uint32_t data_size_ = 0;
    uint32_t data_remaining_ = 0;
    bool pacing_started_ = false;

#ifdef __linux__
    struct timespec next_wake_ = {0, 0};
#endif
};

} // namespace hs


// producer/main.cpp — Producer process main entry point (Sections 5.2, 5.3)

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>
#include <string>

#include "../common/config.h"
#include "../common/rt_util.h"
#include "../common/shm.h"
#include "../common/ring.h"
#include "../common/stats.h"
#include "../common/logring.h"
#include "alsa_source.h"
#include "wav_source.h"
#include "synth_source.h"
#include "assembler.h"

using namespace hs;

static std::atomic<bool> g_running{true};

static void sig_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
}

// Interleaved to float32 mono downmix
static inline float downmix_sample(const void* raw, int frame_idx, int channels,
                                   bool is_s32, const std::string& mode)
{
    float ch[8] = {0.0f};
    if (is_s32) {
        const int32_t* p = reinterpret_cast<const int32_t*>(raw) + frame_idx * channels;
        for (int c = 0; c < channels && c < 8; ++c) {
            ch[c] = static_cast<float>(p[c]) / 2147483648.0f; // 2^31
        }
    } else {
        const int16_t* p = reinterpret_cast<const int16_t*>(raw) + frame_idx * channels;
        for (int c = 0; c < channels && c < 8; ++c) {
            ch[c] = static_cast<float>(p[c]) / 32768.0f; // 2^15
        }
    }

    if (mode == "left") return ch[0];
    if (mode == "right" && channels > 1) return ch[1];
    // default: mean of available channels
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) sum += ch[c];
    return sum / static_cast<float>(channels);
}

int main(int argc, char* argv[]) {
    std::string config_path = "config/hapticsync.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    std::fprintf(stderr, "[producer] Starting with config: %s\n", config_path.c_str());
    Config cfg = load_config(config_path);

    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    // 1. Setup shared memory
    uint32_t audio_capacity = next_pow2(cfg.audio.ring_seconds * cfg.audio.rate);
    size_t audio_shm_sz = audio_ring_shm_size(audio_capacity);
    void* audio_shm = shm_create(SHM_AUDIO, audio_shm_sz);
    if (!audio_shm) {
        std::fprintf(stderr, "[producer] Failed to create audio shm\n");
        return 1;
    }

    void* stats_shm = shm_create(SHM_STATS, STATS_SHM_SIZE);
    if (!stats_shm) {
        std::fprintf(stderr, "[producer] Failed to create stats shm\n");
        shm_destroy(SHM_AUDIO, audio_shm, audio_shm_sz);
        return 1;
    }
    StatsBlock* stats = reinterpret_cast<StatsBlock*>(stats_shm);
    stats->reset();
    stats->pipeline_start_ns.store(monotonic_ns(), std::memory_order_relaxed);

    void* window_shm = shm_create(SHM_WINDOW, WINDOW_SHM_SIZE);
    if (!window_shm) {
        std::fprintf(stderr, "[producer] Failed to create window shm\n");
        return 1;
    }
    float* window_slots = reinterpret_cast<float*>(window_shm);

    // 2. Setup AudioRing
    AudioRingWriter audio_writer;
    audio_writer.init(audio_shm, audio_capacity, cfg.audio.rate);

    AudioRingReader audio_reader;
    audio_reader.init(audio_shm);

    // 3. Setup LogRing
    LogRing log_ring(4096);

    // 4. Socket for assembler <-> inference communication
    int assem_fd = -1;
    const char* fd_env = std::getenv("HAPTICSYNC_ASSEM_FD");
    if (fd_env) {
        assem_fd = std::atoi(fd_env);
        std::fprintf(stderr, "[producer] Using inherited socket fd=%d\n", assem_fd);
    } else {
        std::fprintf(stderr, "[producer] WARNING: HAPTICSYNC_ASSEM_FD not set (standalone/test mode)\n");
    }

    // 5. Start Log Drain Thread
    std::thread log_drain_thread([&]() {
        LogEntry e;
        while (g_running.load(std::memory_order_relaxed)) {
            while (log_ring.read(e)) {
                const char* lvl = (e.level == LOG_ERROR) ? "ERR" :
                                  (e.level == LOG_WARN)  ? "WRN" : "INF";
                std::fprintf(stderr, "[rt-log][%s] %s\n", lvl, e.msg);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    // 6. Assembler Thread (SCHED_OTHER on CPU 1)
    Assembler assembler(cfg, audio_reader, window_slots, assem_fd, stats);
    assembler.load_bench("config/bench.json");

    std::thread assembler_thread([&]() {
        set_affinity(cfg.cpus.producer);
        assembler.run();
    });

    // 7. RT Setup Sequence for Producer (Audio Capture Thread)
    // Section 5.2.4:
    // 1. set_affinity(cpu)
    // 2. mallopt tuning
    // 3. allocate/touch all buffers
    // 4. prefault_stack
    // 5. mlockall
    // 6. set_fifo(priority)
    // 7. verify
    std::fprintf(stderr, "[producer] Executing RT setup sequence...\n");
    if (!set_affinity(cfg.cpus.producer)) {
        std::fprintf(stderr, "[producer] WARNING: Failed setting CPU affinity to %d\n", cfg.cpus.producer);
    }
    rt_malloc_setup();
    prefault_stack(1 << 20);
    if (!lock_all()) {
        std::fprintf(stderr, "[producer] WARNING: mlockall failed (need CAP_SYS_RESOURCE or root)\n");
    }
    if (cfg.rt.require_rt) {
        if (!set_fifo(cfg.rt.producer_prio)) {
            std::fprintf(stderr, "[producer] FATAL: Failed setting SCHED_FIFO %d (need CAP_SYS_NICE or root)\n",
                         cfg.rt.producer_prio);
            if (cfg.rt.require_rt) {
                g_running.store(false, std::memory_order_relaxed);
            }
        }
    }

    // 8. Capture Loop based on configured source
    int period = cfg.audio.period_frames;
    int channels = cfg.audio.channels;
    std::vector<float> mono_buf(period, 0.0f);

    if (cfg.audio.source == "wav" && !cfg.audio.wav_path.empty()) {
        std::fprintf(stderr, "[producer] Using WAV file source: %s\n", cfg.audio.wav_path.c_str());
        WavSource wav(cfg.audio.wav_path, cfg.audio.rate, cfg.audio.channels);
        if (!wav.open()) {
            std::fprintf(stderr, "[producer] Failed opening WAV source\n");
            g_running.store(false, std::memory_order_relaxed);
        } else {
            int frame_bytes = wav.file_channels() * (wav.file_bits() / 8);
            std::vector<uint8_t> raw_buf(period * frame_bytes);
            bool is_s32 = (wav.file_bits() == 32);

            while (g_running.load(std::memory_order_relaxed)) {
                int n = wav.read(raw_buf.data(), period);
                if (n <= 0) continue;
                int64_t t_ns = wav.get_timestamp();

                for (int i = 0; i < n; ++i) {
                    mono_buf[i] = downmix_sample(raw_buf.data(), i, wav.file_channels(), is_s32, cfg.audio.channel_mode);
                }
                audio_writer.write(mono_buf.data(), n, t_ns);
                stats->periods_captured.fetch_add(1, std::memory_order_relaxed);
                stats->samples_captured.fetch_add(n, std::memory_order_relaxed);
            }
        }
    } else if (cfg.audio.source == "synth") {
        std::fprintf(stderr, "[producer] Using synthetic audio source\n");
        SynthSource synth(cfg.audio.rate, channels, SynthSource::SINE);
        synth.open();
        std::vector<float> raw_buf(period * channels);

        while (g_running.load(std::memory_order_relaxed)) {
            int n = synth.read(raw_buf.data(), period);
            int64_t t_ns = synth.get_timestamp();
            for (int i = 0; i < n; ++i) {
                mono_buf[i] = downmix_sample(raw_buf.data(), i, channels, true, cfg.audio.channel_mode);
            }
            audio_writer.write(mono_buf.data(), n, t_ns);
            stats->periods_captured.fetch_add(1, std::memory_order_relaxed);
            stats->samples_captured.fetch_add(n, std::memory_order_relaxed);
        }
    } else { // ALSA hardware capture (default)
        std::fprintf(stderr, "[producer] Using ALSA hardware capture: %s\n", cfg.audio.alsa_device.c_str());
        AlsaSource alsa(cfg.audio.alsa_device, cfg.audio.rate, channels,
                        cfg.audio.period_frames, cfg.audio.buffer_periods);
        if (!alsa.open()) {
            std::fprintf(stderr, "[producer] Failed opening ALSA device %s\n", cfg.audio.alsa_device.c_str());
            g_running.store(false, std::memory_order_relaxed);
        } else {
            std::vector<uint8_t> raw_buf(period * channels * 4); // max for S32
            bool is_s32 = alsa.is_s32();

            while (g_running.load(std::memory_order_relaxed)) {
                int n = alsa.read(raw_buf.data(), period);
                if (n < 0) {
                    log_ring.write(LOG_ERROR, "ALSA read error or unrecoverable xrun");
                    stats->xruns.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (n == 0) continue; // recovered from xrun

                int64_t t_ns = alsa.get_timestamp();
                for (int i = 0; i < n; ++i) {
                    mono_buf[i] = downmix_sample(raw_buf.data(), i, channels, is_s32, cfg.audio.channel_mode);
                }
                audio_writer.write(mono_buf.data(), n, t_ns);
                stats->periods_captured.fetch_add(1, std::memory_order_relaxed);
                stats->samples_captured.fetch_add(n, std::memory_order_relaxed);
            }
            alsa.close();
        }
    }

    // Cleanup and join
    std::fprintf(stderr, "[producer] Shutting down...\n");
    assembler.stop();
    if (assembler_thread.joinable()) assembler_thread.join();
    if (log_drain_thread.joinable()) log_drain_thread.join();

    shm_destroy(SHM_AUDIO, audio_shm, audio_shm_sz);
    shm_destroy(SHM_STATS, stats_shm, STATS_SHM_SIZE);
    shm_destroy(SHM_WINDOW, window_shm, WINDOW_SHM_SIZE);

    std::fprintf(stderr, "[producer] Terminated cleanly.\n");
    return 0;
}


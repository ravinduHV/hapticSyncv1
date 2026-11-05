// postproc/main.cpp — Post-process process main entry point (Sections 5.5, 7.4)

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>

#ifdef __linux__
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "../common/config.h"
#include "../common/rt_util.h"
#include "../common/shm.h"
#include "../common/events.h"
#include "../common/stats.h"
#include "../common/protocol.h"
#include "vocab.h"
#include "ctc_merge.h"

using namespace hs;

static std::atomic<bool> g_running{true};

static void sig_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
}

int main(int argc, char* argv[]) {
    std::string config_path = "config/hapticsync.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    std::fprintf(stderr, "[postproc] Starting with config: %s\n", config_path.c_str());
    Config cfg = load_config(config_path);

    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    // 1. Pin to post-processing CPU
    if (!set_affinity(cfg.cpus.post)) {
        std::fprintf(stderr, "[postproc] WARNING: Failed setting CPU affinity to %d\n", cfg.cpus.post);
    }

    // 2. Open shared memory
    size_t event_shm_sz = event_ring_shm_size(cfg.events.slot_count);
    void* event_shm = shm_create(SHM_EVENTS, event_shm_sz);
    if (!event_shm) {
        std::fprintf(stderr, "[postproc] Failed to create event shm\n");
        return 1;
    }
    EventRingWriter event_writer;
    event_writer.init(event_shm, cfg.events.slot_count);

    void* stats_shm = shm_open_or_create(SHM_STATS, STATS_SHM_SIZE);
    StatsBlock* stats = stats_shm ? reinterpret_cast<StatsBlock*>(stats_shm) : nullptr;

    // 3. Load Vocab
    Vocab vocab;
    std::string vocab_file = cfg.model.vocab_json;
    if (!vocab.load(vocab_file)) {
        std::fprintf(stderr, "[postproc] WARNING: Using fallback blank_id=%d\n", cfg.model.blank_id);
    }
    vocab.set_blank_id(static_cast<uint16_t>(cfg.model.blank_id));

    CtcMerger merger(vocab, cfg.events.min_confidence);

    // 4. Connect to RESULT socket
    int result_fd = -1;
    const char* fd_env = std::getenv("HAPTICSYNC_RESULT_FD");
    if (fd_env) {
        result_fd = std::atoi(fd_env);
        std::fprintf(stderr, "[postproc] Using socket fd=%d\n", result_fd);
    } else {
        std::fprintf(stderr, "[postproc] WARNING: HAPTICSYNC_RESULT_FD not set\n");
    }

    // Pre-allocated receive buffer for header + max frames
    static constexpr size_t MAX_RESULT_BUF = sizeof(ResultHeader) + 1024 * sizeof(FrameOut);
    std::vector<uint8_t> rx_buf(MAX_RESULT_BUF);

    std::fprintf(stderr, "[postproc] Entering main processing loop on CPU %d...\n", cfg.cpus.post);

    while (g_running.load(std::memory_order_relaxed)) {
        if (result_fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

#ifdef __linux__
        ssize_t n = recv(result_fd, rx_buf.data(), rx_buf.size(), 0); // blocking recv
        if (n <= 0) {
            if (!g_running.load(std::memory_order_relaxed)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        if (static_cast<size_t>(n) < sizeof(ResultHeader)) {
            std::fprintf(stderr, "[postproc] Received truncated message size=%zd\n", n);
            continue;
        }

        const ResultHeader* hdr = reinterpret_cast<const ResultHeader*>(rx_buf.data());
        if (hdr->type != MSG_RESULT || hdr->version != PROTOCOL_VERSION) {
            std::fprintf(stderr, "[postproc] Invalid header type=%u version=%u\n",
                         hdr->type, hdr->version);
            continue;
        }

        size_t expected_size = sizeof(ResultHeader) + hdr->n_frames * sizeof(FrameOut);
        if (static_cast<size_t>(n) < expected_size) {
            std::fprintf(stderr, "[postproc] Incomplete payload: got %zd expected %zu\n",
                         n, expected_size);
            continue;
        }

        const FrameOut* frames = reinterpret_cast<const FrameOut*>(rx_buf.data() + sizeof(ResultHeader));

        if (hdr->flags & WFLAG_FIRST_AFTER_GAP) {
            merger.reset();
        }

        uint32_t begin = hdr->commit_begin;
        uint32_t end = std::min(hdr->commit_end, hdr->n_frames);

        for (uint32_t i = begin; i < end; ++i) {
            uint64_t abs_frame = abs_frame_idx(hdr->start_sample, i);
            int64_t t_frame_ns = frame_center_time_ns(hdr->t_start_ns, i);

            uint16_t flags = 0;
            if (i == begin && (hdr->flags & WFLAG_FIRST_AFTER_GAP)) {
                flags |= EVT_FLAG_FIRST_AFTER_GAP;
            }

            auto evt = merger.process_frame(frames[i].label, frames[i].conf,
                                            abs_frame, t_frame_ns,
                                            static_cast<uint32_t>(hdr->window_id),
                                            flags);
            if (evt.has_value()) {
                event_writer.write(evt->phoneme_id, evt->phoneme,
                                   evt->confidence, evt->t_frame_ns,
                                   evt->abs_frame, evt->window_id, evt->flags);

                int64_t now_ns = monotonic_ns();
                int64_t lag_ns = now_ns - evt->t_frame_ns;

                if (stats) {
                    stats->events_emitted.fetch_add(1, std::memory_order_relaxed);
                    stats->record_lag(lag_ns);
                }
            }
        }
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif
    }

    std::fprintf(stderr, "[postproc] Shutting down...\n");
    shm_destroy(SHM_EVENTS, event_shm, event_shm_sz);
    if (stats_shm) shm_close(stats_shm, STATS_SHM_SIZE);

    return 0;
}


#pragma once
// assembler.h — Pre-model window assembler (Section 5.3)
// Runs on CPU 1, SCHED_OTHER. Decides when to send windows to inference.

#include <cstdint>
#include <cmath>
#include <atomic>

#include "../common/ring.h"
#include "../common/protocol.h"
#include "../common/stats.h"
#include "../common/config.h"

namespace hs {

/// Window slot in shared memory (double-buffered, 2 slots).
/// Each slot holds normalized float32 samples for one inference window.
static constexpr size_t MAX_WINDOW_SAMPLES = 4 * 16000; // 4 seconds max
static constexpr size_t WINDOW_SLOT_BYTES = MAX_WINDOW_SAMPLES * sizeof(float);
static constexpr const char* SHM_WINDOW = "/hapticsync_window";
static constexpr size_t WINDOW_SHM_SIZE = 2 * WINDOW_SLOT_BYTES;

/// Backlog policy identifier.
enum class Policy { CATCHUP, LATEST, FIFO, AUTO };

inline Policy parse_policy(const std::string& s) {
    if (s == "catchup") return Policy::CATCHUP;
    if (s == "latest")  return Policy::LATEST;
    if (s == "fifo")    return Policy::FIFO;
    return Policy::AUTO;
}

/// Bench parameters (from config/bench.json): t(L) = a + b*L
struct BenchParams {
    double a = 0.0; // fixed overhead (seconds)
    double b = 0.0; // cost per second of audio
    bool valid = false;
};

class Assembler {
public:
    Assembler(const Config& cfg, AudioRingReader& ring,
              float* window_slots, // 2 * MAX_WINDOW_SAMPLES floats
              int assem_sock_fd,   // socketpair to inference
              StatsBlock* stats);

    ~Assembler() = default;

    /// Main loop — runs until stop() is called.
    void run();

    /// Signal the assembler to stop.
    void stop() { running_.store(false, std::memory_order_relaxed); }

    /// Load bench parameters from bench.json.
    void load_bench(const std::string& path);

private:
    /// Build and send one window starting at the given sample index.
    /// Returns true if a window was sent.
    bool build_and_send(uint64_t start_sample, uint32_t n_samples,
                        uint32_t commit_begin, uint32_t commit_end,
                        bool first_after_gap);

    /// Normalize samples: y = (x - mean) / sqrt(var + 1e-7)
    void normalize(float* data, uint32_t n);

    /// Calculate the next window parameters based on policy.
    struct WindowPlan {
        uint64_t start_sample;
        uint32_t n_samples;
        uint32_t commit_begin;
        uint32_t commit_end;
        bool     first_after_gap;
        bool     valid;
    };
    WindowPlan plan_next_window();

    /// Handle DONE message from inference.
    void handle_done();

    /// Policy selection from bench params.
    void select_policy();

    Config cfg_;
    AudioRingReader& ring_;
    float* window_slots_;
    int sock_fd_;
    StatsBlock* stats_;

    std::atomic<bool> running_{true};

    // State
    Policy policy_ = Policy::AUTO;
    BenchParams bench_;
    uint64_t window_id_ = 0;
    int current_slot_ = 0;  // 0 or 1
    bool model_idle_ = true;
    bool gap_flag_ = false;

    // Tiling state
    uint64_t last_commit_end_sample_ = 0; // absolute sample of last committed region end
    uint64_t next_window_start_frame_ = 0; // in 20ms frame units
    bool first_window_ = true;

    // Trim params
    int trim_left_;
    int trim_right_;
    int min_window_samples_;
    int max_window_samples_;
    int min_new_samples_;
    int fixed_window_samples_;
};

} // namespace hs


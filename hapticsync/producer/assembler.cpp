// assembler.cpp — Pre-model window assembler implementation (Section 5.3)

#include "assembler.h"
#include <cstdio>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>

#ifdef __linux__
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace hs {

Assembler::Assembler(const Config& cfg, AudioRingReader& ring,
                     float* window_slots, int assem_sock_fd,
                     StatsBlock* stats)
    : cfg_(cfg), ring_(ring), window_slots_(window_slots),
      sock_fd_(assem_sock_fd), stats_(stats)
{
    trim_left_ = cfg_.window.trim_left_frames;
    trim_right_ = cfg_.window.trim_right_frames;

    min_window_samples_ = (cfg_.window.min_window_ms * cfg_.audio.rate) / 1000;
    max_window_samples_ = (cfg_.window.max_window_ms * cfg_.audio.rate) / 1000;
    min_new_samples_    = (cfg_.window.min_new_ms * cfg_.audio.rate) / 1000;
    fixed_window_samples_ = (cfg_.window.fixed_window_ms * cfg_.audio.rate) / 1000;

    policy_ = parse_policy(cfg_.window.policy);
}

void Assembler::load_bench(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    try {
        json j = json::parse(f);
        if (j.contains("a") && !j["a"].is_null() &&
            j.contains("b") && !j["b"].is_null()) {
            bench_.a = j["a"].get<double>();
            bench_.b = j["b"].get<double>();
            bench_.valid = true;
            std::fprintf(stderr, "[assembler] Loaded bench: a=%.4f s, b=%.4f\n",
                         bench_.a, bench_.b);
            select_policy();
        }
    } catch (...) {
        std::fprintf(stderr, "[assembler] Failed parsing bench file %s\n", path.c_str());
    }
}

void Assembler::select_policy() {
    if (policy_ != Policy::AUTO) return;

    if (!bench_.valid) {
        // Fallback default if bench measurements not yet populated
        std::fprintf(stderr, "[assembler] No bench params; default to latest policy.\n");
        policy_ = Policy::LATEST;
        return;
    }

    // Steady state check for catchup:
    // O = (trimL + trimR) * 0.02 + 0.005 s
    double O = (trim_left_ + trim_right_) * 0.020 + 0.005;
    if (bench_.b < 1.0) {
        double T_star = (bench_.a + bench_.b * O) / (1.0 - bench_.b);
        double L_star = T_star + O;
        double max_w_s = cfg_.window.max_window_ms / 1000.0;
        if (L_star <= max_w_s) {
            policy_ = Policy::CATCHUP;
            std::fprintf(stderr, "[assembler] Auto-selected CATCHUP policy: L*=%.3fs, T*=%.3fs\n",
                         L_star, T_star);
            return;
        }
    }
    policy_ = Policy::LATEST;
    std::fprintf(stderr, "[assembler] Auto-selected LATEST policy (b=%.4f >= 1.0 or L* exceeded)\n",
                 bench_.b);
}

void Assembler::normalize(float* data, uint32_t n) {
    if (n == 0) return;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    float mean = static_cast<float>(sum / n);

    double var_sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        float diff = data[i] - mean;
        var_sum += diff * diff;
    }
    float variance = static_cast<float>(var_sum / n);
    float inv_std = 1.0f / std::sqrt(variance + 1e-7f);

    for (uint32_t i = 0; i < n; ++i) {
        data[i] = (data[i] - mean) * inv_std;
    }
}

void Assembler::handle_done() {
#ifdef __linux__
    DoneMsg msg;
    ssize_t n = recv(sock_fd_, &msg, sizeof(msg), MSG_DONTWAIT);
    if (n == sizeof(DoneMsg) && msg.type == MSG_DONE && msg.version == PROTOCOL_VERSION) {
        model_idle_ = true;
        if (stats_) {
            stats_->model_busy.store(0, std::memory_order_relaxed);
            stats_->windows_done.fetch_add(1, std::memory_order_relaxed);
            stats_->last_infer_ns.store(msg.t_infer_end_ns - msg.t_infer_begin_ns,
                                        std::memory_order_relaxed);
        }
    }
#else
    // Stub
    model_idle_ = true;
#endif
}

Assembler::WindowPlan Assembler::plan_next_window() {
    WindowPlan plan{};
    plan.valid = false;

    uint64_t head_sample = ring_.write_idx();

    if (first_window_) {
        // First window starts at sample 0 (or nearest multiple of 320 available)
        uint32_t needed = std::max(min_window_samples_, (int)fixed_window_samples_);
        if (head_sample < needed) return plan; // not enough audio yet

        uint32_t n_frames = n_frames_from_samples(needed);
        uint32_t n_samples = n_samples_from_frames(n_frames);

        plan.start_sample = 0;
        plan.n_samples = n_samples;
        plan.commit_begin = 0;
        plan.commit_end = (n_frames > static_cast<uint32_t>(trim_right_)) ? (n_frames - trim_right_) : n_frames;
        plan.first_after_gap = true;
        plan.valid = true;
        return plan;
    }

    if (policy_ == Policy::LATEST) {
        // Fixed window length on latest audio
        uint32_t n_frames = n_frames_from_samples(fixed_window_samples_);
        uint32_t n_samples = n_samples_from_frames(n_frames);

        if (head_sample < n_samples) return plan;

        uint64_t target_start = (head_sample - n_samples) / 320 * 320;
        uint64_t expected_start = next_window_start_frame_ * 320;

        bool gap_detected = (target_start > expected_start);
        if (gap_detected) {
            if (stats_) stats_->windows_dropped.fetch_add(1, std::memory_order_relaxed);
            plan.first_after_gap = true;
            plan.commit_begin = 0;
            plan.commit_end = (n_frames > static_cast<uint32_t>(trim_right_)) ? (n_frames - trim_right_) : n_frames;
            plan.start_sample = target_start;
            plan.n_samples = n_samples;
            plan.valid = true;
        } else {
            // Can tile seamlessly
            if (head_sample < expected_start + n_samples) return plan; // wait for enough audio
            plan.first_after_gap = false;
            plan.commit_begin = trim_left_;
            plan.commit_end = (n_frames > static_cast<uint32_t>(trim_right_)) ? (n_frames - trim_right_) : n_frames;
            plan.start_sample = expected_start;
            plan.n_samples = n_samples;
            plan.valid = true;
        }
        return plan;
    }
    else if (policy_ == Policy::CATCHUP) {
        // Window covers all audio available since expected_start
        uint64_t expected_start = next_window_start_frame_ * 320;
        if (head_sample < expected_start + min_new_samples_) return plan;

        uint64_t available = head_sample - expected_start;
        uint32_t target_samples = static_cast<uint32_t>(std::min<uint64_t>(available, max_window_samples_));
        if (target_samples < static_cast<uint32_t>(min_window_samples_)) return plan;

        uint32_t n_frames = n_frames_from_samples(target_samples);
        uint32_t n_samples = n_samples_from_frames(n_frames);

        plan.first_after_gap = gap_flag_;
        gap_flag_ = false;
        plan.commit_begin = plan.first_after_gap ? 0 : trim_left_;
        plan.commit_end = (n_frames > static_cast<uint32_t>(trim_right_)) ? (n_frames - trim_right_) : n_frames;
        plan.start_sample = expected_start;
        plan.n_samples = n_samples;
        plan.valid = true;
        return plan;
    }
    else { // FIFO (testing)
        uint32_t n_frames = n_frames_from_samples(fixed_window_samples_);
        uint32_t n_samples = n_samples_from_frames(n_frames);
        uint64_t expected_start = next_window_start_frame_ * 320;

        if (head_sample < expected_start + n_samples) return plan;

        plan.first_after_gap = false;
        plan.commit_begin = trim_left_;
        plan.commit_end = (n_frames > static_cast<uint32_t>(trim_right_)) ? (n_frames - trim_right_) : n_frames;
        plan.start_sample = expected_start;
        plan.n_samples = n_samples;
        plan.valid = true;
        return plan;
    }
}

bool Assembler::build_and_send(uint64_t start_sample, uint32_t n_samples,
                              uint32_t commit_begin, uint32_t commit_end,
                              bool first_after_gap)
{
    float* slot_ptr = window_slots_ + current_slot_ * MAX_WINDOW_SAMPLES;

    if (!ring_.read(start_sample, slot_ptr, n_samples)) {
        // Read failed (overwritten or unavailable)
        gap_flag_ = true;
        if (stats_) stats_->gaps.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    if (cfg_.window.normalize_in == "cpp") {
        normalize(slot_ptr, n_samples);
    }

    WindowMsg msg{};
    msg.type = MSG_WINDOW;
    msg.version = PROTOCOL_VERSION;
    msg.window_id = ++window_id_;
    msg.slot = static_cast<uint32_t>(current_slot_);
    msg.n_samples = n_samples;
    msg.start_sample = start_sample;
    msg.t_start_ns = ring_.estimate_timestamp(start_sample);
    msg.t_sent_ns = monotonic_ns();
    msg.n_frames = n_frames_from_samples(n_samples);
    msg.commit_begin = commit_begin;
    msg.commit_end = commit_end;
    msg.flags = first_after_gap ? WFLAG_FIRST_AFTER_GAP : 0;

#ifdef __linux__
    ssize_t sent = send(sock_fd_, &msg, sizeof(msg), MSG_DONTWAIT);
    if (sent != sizeof(msg)) {
        std::fprintf(stderr, "[assembler] send failed or would block\n");
        return false;
    }
#endif

    // Switch slot
    current_slot_ = 1 - current_slot_;
    model_idle_ = false;

    // Update tiling state
    uint64_t start_frame = start_sample / 320;
    uint32_t n_prev = msg.n_frames;

    if (first_after_gap || first_window_) {
        next_window_start_frame_ = start_frame + commit_end - trim_left_;
    } else {
        next_window_start_frame_ = start_frame + n_prev - trim_right_ - trim_left_;
    }
    last_commit_end_sample_ = (start_frame + commit_end) * 320;
    first_window_ = false;

    if (stats_) {
        stats_->windows_sent.fetch_add(1, std::memory_order_relaxed);
        stats_->model_busy.store(1, std::memory_order_relaxed);
    }

    return true;
}

void Assembler::run() {
    select_policy();

    while (running_.load(std::memory_order_relaxed)) {
#ifdef __linux__
        struct pollfd pfd{};
        pfd.fd = sock_fd_;
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 5); // 5 ms timeout
        if (ret > 0 && (pfd.revents & POLLIN)) {
            handle_done();
        }
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        handle_done();
#endif

        if (model_idle_) {
            WindowPlan plan = plan_next_window();
            if (plan.valid) {
                build_and_send(plan.start_sample, plan.n_samples,
                               plan.commit_begin, plan.commit_end,
                               plan.first_after_gap);
            }
        }
    }
}

} // namespace hs


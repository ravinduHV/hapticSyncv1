#pragma once
// config.h — Load JSON configuration (Section 8)
// Uses nlohmann/json. Config is shared between C++ and Python components.

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdio>

// nlohmann/json single-include header
// On Raspberry Pi: install via apt install nlohmann-json3-dev
// For builds: #include <nlohmann/json.hpp>
#include <nlohmann/json.hpp>

namespace hs {

using json = nlohmann::json;

struct CpuConfig {
    std::vector<int> housekeeping = {0};
    int producer = 1;
    std::vector<int> model = {2};
    int post = 3;
};

struct RtConfig {
    bool require_rt = true;
    int irq_thread_prio = 60;
    int producer_prio = 50;
};

struct AudioConfig {
    std::string source = "alsa";
    std::string alsa_device = "hw:0,0";
    int rate = 16000;
    int channels = 2;
    int period_frames = 160;
    int buffer_periods = 8;
    std::string channel_mode = "mean";
    std::string wav_path;
    int ring_seconds = 60;
};

struct WindowConfig {
    std::string policy = "auto";
    int trim_left_frames = 3;
    int trim_right_frames = 3;
    int min_window_ms = 300;
    int max_window_ms = 4000;
    int min_new_ms = 200;
    int fixed_window_ms = 1000;
    std::string normalize_in = "cpp";
};

struct ModelConfig {
    std::string checkpoint = "./model/";
    int blank_id = 41;   // [PAD] token — HF CTC convention
    std::string vocab_json = "vocab.json";
};

struct EventsConfig {
    int slot_count = 1024;
    float min_confidence = 0.0f;
};

struct TelemetryConfig {
    int rate_hz = 10;
    bool dashboard = true;
    int port = 5000;
};

struct Config {
    CpuConfig cpus;
    RtConfig rt;
    AudioConfig audio;
    WindowConfig window;
    ModelConfig model;
    EventsConfig events;
    TelemetryConfig telemetry;
};

// ── JSON deserialization ────────────────────────────────────────────────────

inline void from_json(const json& j, CpuConfig& c) {
    if (j.contains("housekeeping")) j.at("housekeeping").get_to(c.housekeeping);
    if (j.contains("producer"))     j.at("producer").get_to(c.producer);
    if (j.contains("model"))        j.at("model").get_to(c.model);
    if (j.contains("post"))         j.at("post").get_to(c.post);
}

inline void from_json(const json& j, RtConfig& c) {
    if (j.contains("require_rt"))      j.at("require_rt").get_to(c.require_rt);
    if (j.contains("irq_thread_prio")) j.at("irq_thread_prio").get_to(c.irq_thread_prio);
    if (j.contains("producer_prio"))   j.at("producer_prio").get_to(c.producer_prio);
}

inline void from_json(const json& j, AudioConfig& c) {
    if (j.contains("source"))         j.at("source").get_to(c.source);
    if (j.contains("alsa_device"))    j.at("alsa_device").get_to(c.alsa_device);
    if (j.contains("rate"))           j.at("rate").get_to(c.rate);
    if (j.contains("channels"))       j.at("channels").get_to(c.channels);
    if (j.contains("period_frames"))  j.at("period_frames").get_to(c.period_frames);
    if (j.contains("buffer_periods")) j.at("buffer_periods").get_to(c.buffer_periods);
    if (j.contains("channel_mode"))   j.at("channel_mode").get_to(c.channel_mode);
    if (j.contains("wav_path") && !j.at("wav_path").is_null())
        j.at("wav_path").get_to(c.wav_path);
    if (j.contains("ring_seconds"))   j.at("ring_seconds").get_to(c.ring_seconds);
}

inline void from_json(const json& j, WindowConfig& c) {
    if (j.contains("policy"))            j.at("policy").get_to(c.policy);
    if (j.contains("trim_left_frames"))  j.at("trim_left_frames").get_to(c.trim_left_frames);
    if (j.contains("trim_right_frames")) j.at("trim_right_frames").get_to(c.trim_right_frames);
    if (j.contains("min_window_ms"))     j.at("min_window_ms").get_to(c.min_window_ms);
    if (j.contains("max_window_ms"))     j.at("max_window_ms").get_to(c.max_window_ms);
    if (j.contains("min_new_ms"))        j.at("min_new_ms").get_to(c.min_new_ms);
    if (j.contains("fixed_window_ms"))   j.at("fixed_window_ms").get_to(c.fixed_window_ms);
    if (j.contains("normalize_in"))      j.at("normalize_in").get_to(c.normalize_in);
}

inline void from_json(const json& j, ModelConfig& c) {
    if (j.contains("checkpoint"))  j.at("checkpoint").get_to(c.checkpoint);
    if (j.contains("blank_id") && !j.at("blank_id").is_null())
        j.at("blank_id").get_to(c.blank_id);
    if (j.contains("vocab_json"))  j.at("vocab_json").get_to(c.vocab_json);
}

inline void from_json(const json& j, EventsConfig& c) {
    if (j.contains("slot_count"))     j.at("slot_count").get_to(c.slot_count);
    if (j.contains("min_confidence")) j.at("min_confidence").get_to(c.min_confidence);
}

inline void from_json(const json& j, TelemetryConfig& c) {
    if (j.contains("rate_hz"))   j.at("rate_hz").get_to(c.rate_hz);
    if (j.contains("dashboard")) j.at("dashboard").get_to(c.dashboard);
    if (j.contains("port"))      j.at("port").get_to(c.port);
}

inline void from_json(const json& j, Config& c) {
    if (j.contains("cpus"))      j.at("cpus").get_to(c.cpus);
    if (j.contains("rt"))        j.at("rt").get_to(c.rt);
    if (j.contains("audio"))     j.at("audio").get_to(c.audio);
    if (j.contains("window"))    j.at("window").get_to(c.window);
    if (j.contains("model"))     j.at("model").get_to(c.model);
    if (j.contains("events"))    j.at("events").get_to(c.events);
    if (j.contains("telemetry")) j.at("telemetry").get_to(c.telemetry);
}

/// Load configuration from a JSON file. Falls back to defaults for missing keys.
inline Config load_config(const std::string& path) {
    Config cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[config] WARNING: cannot open %s, using defaults\n",
                     path.c_str());
        return cfg;
    }
    try {
        json j = json::parse(f);
        cfg = j.get<Config>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[config] ERROR parsing %s: %s\n", path.c_str(), e.what());
        throw;
    }
    return cfg;
}

} // namespace hs


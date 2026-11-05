// vocab.cpp — Vocab loader implementation

#include "vocab.h"
#include <fstream>
#include <cstdio>
#include <nlohmann/json.hpp>

namespace hs {

bool Vocab::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[vocab] Could not open %s\n", path.c_str());
        return false;
    }
    try {
        nlohmann::json j = nlohmann::json::parse(f);
        map_.clear();
        for (auto& el : j.items()) {
            uint16_t id = static_cast<uint16_t>(std::stoi(el.key()));
            map_[id] = el.value().get<std::string>();
        }
        std::fprintf(stderr, "[vocab] Loaded %zu tokens from %s\n", map_.size(), path.c_str());
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[vocab] Error parsing %s: %s\n", path.c_str(), e.what());
        return false;
    }
}

} // namespace hs


#pragma once
// vocab.h — Token ID to phoneme string mapping (Section 5.5)

#include <string>
#include <vector>
#include <unordered_map>

namespace hs {

class Vocab {
public:
    Vocab() = default;

    /// Load mapping from a JSON file (e.g., {"1": "aa", "2": "ae", ...})
    bool load(const std::string& path);

    /// Get phoneme string for given token id. Returns "[UNK]" if not found.
    const char* get(uint16_t id) const {
        auto it = map_.find(id);
        if (it != map_.end()) return it->second.c_str();
        return UNK_STR;
    }

    /// Blank token id (typically [PAD], 41 in default checkpoint)
    uint16_t blank_id() const { return blank_id_; }
    void set_blank_id(uint16_t id) { blank_id_ = id; }

private:
    std::unordered_map<uint16_t, std::string> map_;
    uint16_t blank_id_ = 41;
    static constexpr const char* UNK_STR = "[UNK]";
};

} // namespace hs

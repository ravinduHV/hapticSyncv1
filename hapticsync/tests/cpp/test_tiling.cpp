// test_tiling.cpp — Unit tests for Overlap-Save Tiling (Sections 7.1, 7.2, 11.1)

#include <cassert>
#include <cstdio>
#include <vector>
#include <set>
#include "../../common/protocol.h"

using namespace hs;

void test_geometry() {
    // Formulas:
    // n_frames = floor((n_samples - 400)/320) + 1
    // n_samples = (n_frames - 1)*320 + 400
    assert(n_frames_from_samples(400) == 1);
    assert(n_samples_from_frames(1) == 400);

    assert(n_frames_from_samples(720) == 2);
    assert(n_samples_from_frames(2) == 720);

    assert(n_frames_from_samples(2000) == 6);
    assert(n_samples_from_frames(6) == 2000);

    std::printf("[PASS] Frame grid geometry calculations\n");
}

void test_tiling_no_gaps_or_duplicates() {
    int trim_left = 3;
    int trim_right = 3;

    // Simulate 5 consecutive windows of 50 frames each
    uint32_t window_frames = 50;
    std::vector<uint64_t> committed_timeline;

    uint64_t start_frame = 0;
    bool first = true;

    for (int w = 0; w < 5; ++w) {
        uint32_t commit_begin = first ? 0 : trim_left;
        uint32_t commit_end = window_frames - trim_right;

        for (uint32_t f = commit_begin; f < commit_end; ++f) {
            uint64_t abs_f = start_frame + f;
            committed_timeline.push_back(abs_f);
        }

        // Tiling advance rule (Section 7.2):
        if (first) {
            start_frame = start_frame + commit_end - trim_left;
            first = false;
        } else {
            start_frame = start_frame + window_frames - trim_right - trim_left;
        }
    }

    // Verify committed frames are strictly contiguous: 0, 1, 2, 3...
    assert(!committed_timeline.empty());
    for (size_t i = 0; i < committed_timeline.size(); ++i) {
        assert(committed_timeline[i] == i);
    }

    std::printf("[PASS] Overlap-save tiling produces contiguous sequence without gaps/duplicates (%zu frames)\n",
                committed_timeline.size());
}

int main() {
    test_geometry();
    test_tiling_no_gaps_or_duplicates();
    std::printf("[ALL PASS] test_tiling completed successfully.\n");
    return 0;
}


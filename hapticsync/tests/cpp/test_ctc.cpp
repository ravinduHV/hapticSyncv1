// test_ctc.cpp — Unit tests for CTC collapse state machine (Sections 7.4, 11.1)

#include <cassert>
#include <cstdio>
#include <vector>
#include "../../postproc/vocab.h"
#include "../../postproc/ctc_merge.h"

using namespace hs;

void test_ctc_merging() {
    Vocab vocab;
    vocab.set_blank_id(41); // blank = 41

    CtcMerger merger(vocab, 0.0f);

    // Frame 0: token 1 ('aa') -> emits onset
    auto e1 = merger.process_frame(1, 0.9f, 0, 1000, 1, 0);
    assert(e1.has_value());
    assert(e1->phoneme_id == 1);

    // Frame 1: token 1 ('aa') -> repeated token, should NOT emit
    auto e2 = merger.process_frame(1, 0.85f, 1, 2000, 1, 0);
    assert(!e2.has_value());

    // Frame 2: token 1 ('aa') in next window -> still same label across boundary, should NOT emit
    auto e3 = merger.process_frame(1, 0.88f, 2, 3000, 2, 0);
    assert(!e3.has_value());

    // Frame 3: token 41 (blank) -> does NOT emit, but resets state
    auto e4 = merger.process_frame(41, 0.99f, 3, 4000, 2, 0);
    assert(!e4.has_value());

    // Frame 4: token 1 ('aa') -> after blank, should emit new onset!
    auto e5 = merger.process_frame(1, 0.92f, 4, 5000, 2, 0);
    assert(e5.has_value());
    assert(e5->phoneme_id == 1);

    // Frame 5: token 2 ('ae') -> different label, should emit
    auto e6 = merger.process_frame(2, 0.80f, 5, 6000, 2, 0);
    assert(e6.has_value());
    assert(e6->phoneme_id == 2);

    // Test gap reset
    merger.reset();
    // After gap, token 2 should emit again even without intervening blank
    auto e7 = merger.process_frame(2, 0.85f, 10, 10000, 3, 0);
    assert(e7.has_value());
    assert(e7->phoneme_id == 2);

    std::printf("[PASS] CTC collapse state machine rules\n");
}

int main() {
    test_ctc_merging();
    std::printf("[ALL PASS] test_ctc completed successfully.\n");
    return 0;
}


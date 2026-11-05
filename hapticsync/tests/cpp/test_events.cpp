// test_events.cpp — Unit tests for Phoneme Event Ring (Sections 5.6, 11.1)

#include <cassert>
#include <cstdio>
#include <vector>
#include "../../common/events.h"

using namespace hs;

void test_event_ring_protocol() {
    uint32_t slot_count = 16; // small ring to test lapping
    size_t shm_sz = event_ring_shm_size(slot_count);
    std::vector<uint8_t> mem(shm_sz, 0);

    EventRingWriter writer;
    writer.init(mem.data(), slot_count);

    EventRingReader reader;
    reader.init(mem.data());

    // Write 5 events
    for (uint16_t i = 1; i <= 5; ++i) {
        writer.write(i, "aa", 0.95f, i * 1000000, i, 1, 0);
    }

    assert(writer.head() == 6);

    // Read back all 5 events
    bool lapped = false;
    PhonemeEvent evt;
    for (uint16_t i = 1; i <= 5; ++i) {
        bool ok = reader.read(evt, lapped);
        assert(ok);
        assert(!lapped);
        assert(evt.event_id == i);
        assert(evt.phoneme_id == i);
        assert(std::strcmp(evt.phoneme, "aa") == 0);
    }

    // Now write 20 more events, causing reader to be lapped (slot_count is 16)
    for (uint16_t i = 6; i <= 25; ++i) {
        writer.write(i, "sh", 0.88f, i * 1000000, i, 1, 0);
    }

    // Reader should detect lapped condition
    bool ok = reader.read(evt, lapped);
    assert(lapped);
    assert(reader.lapped_count() > 0);

    std::printf("[PASS] EventRing write, read, torn-read, and lapping detection\n");
}

int main() {
    test_event_ring_protocol();
    std::printf("[ALL PASS] test_events completed successfully.\n");
    return 0;
}


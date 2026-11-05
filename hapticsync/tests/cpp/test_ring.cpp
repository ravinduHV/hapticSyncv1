// test_ring.cpp — Unit tests for AudioRing buffer (Section 11.1)

#include <cassert>
#include <cstdio>
#include <vector>
#include "../../common/ring.h"

using namespace hs;

void test_ring_wraparound() {
    uint32_t capacity = 1024; // 1024 samples
    size_t shm_sz = audio_ring_shm_size(capacity);
    std::vector<uint8_t> mem(shm_sz, 0);

    AudioRingWriter writer;
    writer.init(mem.data(), capacity, 16000);

    AudioRingReader reader;
    reader.init(mem.data());

    // Write 800 samples
    std::vector<float> data1(800);
    for (int i = 0; i < 800; ++i) data1[i] = static_cast<float>(i);
    writer.write(data1.data(), 800, 1000000);

    assert(writer.write_idx() == 800);
    assert(reader.write_idx() == 800);

    // Read back 400 samples from index 100
    std::vector<float> out(400);
    bool ok = reader.read(100, out.data(), 400);
    assert(ok);
    (void)ok;
    for (int i = 0; i < 400; ++i) {
        assert(out[i] == static_cast<float>(100 + i));
    }

    // Now write 600 more samples (will wrap around 1024)
    std::vector<float> data2(600);
    for (int i = 0; i < 600; ++i) data2[i] = static_cast<float>(800 + i);
    writer.write(data2.data(), 600, 2000000);

    assert(writer.write_idx() == 1400);

    // Read across the wraparound boundary (e.g. from 900 to 1200, length 300)
    std::vector<float> out_wrap(300);
    ok = reader.read(900, out_wrap.data(), 300);
    assert(ok);
    (void)ok;
    for (int i = 0; i < 300; ++i) {
        assert(out_wrap[i] == static_cast<float>(900 + i));
    }

    // Test lapping invalidation: index 100 has been overwritten because capacity is 1024
    // 1400 - 100 = 1300 > 1024
    ok = reader.read(100, out.data(), 100);
    assert(!ok); // MUST fail because it was lapped
    (void)ok;

    std::printf("[PASS] AudioRing wraparound and lapping test\n");
}

int main() {
    test_ring_wraparound();
    std::printf("[ALL PASS] test_ring completed successfully.\n");
    return 0;
}


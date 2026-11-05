# Haptic-Sync Handoff Documentation

## 1. Downstream Haptic Consumer Integration
The haptic consumer reads phonemes from the shared memory event ring `/hapticsync_events`.

### Contract & Struct
```c
struct PhonemeEvent {
  uint64_t seq;          // sequence number (even = valid)
  uint64_t event_id;     // monotonically increasing
  int64_t  t_frame_ns;   // CLOCK_MONOTONIC capture time of frame center
  int64_t  t_emit_ns;    // time committed to event ring
  uint64_t abs_frame;    // 20ms frame index since stream start
  uint32_t window_id;
  uint16_t phoneme_id;   // model token id
  uint16_t flags;        // bit0: first event after a gap
  float    confidence;   // softmax probability
  char     phoneme[12];  // NUL-terminated label
};
```

### Critical RT Design Rule (Ogness RT Slides)
An RT haptic consumer thread **must not** block waiting on a wake-up from the non-RT post-processing stage. It must poll the ring from a dedicated thread paced with `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` or use `EventRingReader` (see `common/events.h` and `tools/events_reader.py`).

## 2. Model Optimization Handoff
The inference stage is isolated behind the `SOCK_SEQPACKET` socketpair protocol (`WindowMsg` / `ResultHeader` / `DoneMsg` in `common/protocol.h`).
Any future quantized (INT8, ONNX Runtime, TFLite) model can be integrated into `inference/main.py` without modifying the C++ capture or post-processing pipelines.
The parameter curve `t(L) = a + b * L` in `config/bench.json` is the metric to beat.
If `b < 1.0`, the system can run sustainably in `catchup` mode with 100% audio coverage.


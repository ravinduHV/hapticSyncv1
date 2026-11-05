# Haptic-Sync Architecture Specification

## Overview
Haptic-Sync is a real-time hearing-aid phoneme detection pipeline targeting a Raspberry Pi 5 with a PREEMPT_RT Linux kernel.
The pipeline extracts phoneme event streams with microsecond-level timing metadata and publishes them to a lock-free, zero-blocking event ring for downstream haptic actuators.

## Core Pinning and Process Model

```
CPU 0: Housekeeping (systemd, non-RT IRQs, telemetry dashboard, launcher)
CPU 1: Capture & Pre-model (Audio IRQ, Producer RT thread FIFO 50, Assembler SCHED_OTHER)
CPU 2: Model Inference (Python 3 + PyTorch, 1 thread, SCHED_OTHER)
CPU 3: Post-processing (CTC collapse, overlap merge, event ring writer)
```

## Transport Architecture
- **Audio Stream**: Producer writes interleaved audio to lock-free single-writer shared memory ring (`/hapticsync_audio`).
- **Control Handshakes**: Double-buffered window slots (`/hapticsync_window`) and non-blocking `AF_UNIX` `SOCK_SEQPACKET` socketpair datagrams between C++ assembler and Python inference.
- **Phoneme Event Stream**: Post-processor commits non-blank CTC run onsets to a sequence-numbered event ring (`/hapticsync_events`).
- **Telemetry**: Read-only atomic stats block (`/hapticsync_stats`) read by CPU 0 dashboard without interfering with the RT threads.

## Algorithms
1. **Frame Grid Alignment**:
   Wav2Vec2 20 ms hop (320 samples) and 25 ms receptive field (400 samples). Window start samples are locked to multiples of 320 to guarantee global frame index stability.
2. **Overlap-Save Tiling**:
   Discards edge artifacts using left/right trims (`trimL`, `trimR`). Committed regions tile the audio timeline continuously without gaps or duplicate frames.
3. **Stateful CTC Collapse**:
   Maintains state across window boundaries so tokens spanning across window boundaries are collapsed properly. Run onsets emit immediately upon window commit.


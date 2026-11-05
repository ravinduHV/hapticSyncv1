# Architecture & Performance Analysis: Initial vs. Current Version

> **Note**: The full detailed report is maintained in [`hapticsync/docs/architecture_comparison_and_analysis.md`](hapticsync/docs/architecture_comparison_and_analysis.md).

Please refer to [`hapticsync/docs/architecture_comparison_and_analysis.md`](hapticsync/docs/architecture_comparison_and_analysis.md) for the complete comprehensive analysis, including:
1. Executive comparison matrix.
2. Architectural diagrams (Monolithic vs. Decoupled HPC).
3. Audio ingestion, scheduling (`SCHED_FIFO 50` on Core 1), and IPC POSIX Shared Memory design.
4. Empirical benchmark numbers on Raspberry Pi 5 across 1-Core FP32, 3-Core FP32, and 3-Core INT8.
5. Mathematical cost curve modeling ($t(L) = a + b \cdot L$) and catchup policy derivation.
6. Overlap-Save acoustic boundary tiling and stateful C++ CTC decoding.
7. Telemetry offloading via MQTT.


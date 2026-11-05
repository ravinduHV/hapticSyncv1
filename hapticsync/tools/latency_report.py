#!/usr/bin/env python3
"""
latency_report.py — Latency and timing distribution analyzer (Section 12.1)
Reads stats block and event ring to compute empirical lag percentiles (p50, p90, p95, p99, max).
"""

import sys
import numpy as np
from events_reader import EventRingReader


def main():
    print("── Haptic-Sync Latency & Lag Analysis Report ─────────────────")
    print("Reading phoneme events from /dev/shm/hapticsync_events...")

    try:
        reader = EventRingReader()
    except FileNotFoundError:
        print("Error: Event ring not active. Is the pipeline running?")
        sys.exit(1)

    lags = []
    print("Collecting events (press Ctrl-C to finish)...")

    try:
        for evt in reader.poll(timeout_s=0.01):
            lags.append(evt["lag_ms"])
            if len(lags) % 100 == 0:
                print(f"  Collected {len(lags)} events...", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()

    if not lags:
        print("\nNo events collected.")
        return

    arr = np.array(lags)
    print("\n\n── Empirical Lag Distribution ─────────────────────────────────")
    print(f"Total events analyzed: {len(arr)}")
    print(f"Mean lag             : {np.mean(arr):.2f} ms")
    print(f"Std deviation        : {np.std(arr):.2f} ms")
    print(f"Median (p50)         : {np.percentile(arr, 50):.2f} ms")
    print(f"90th percentile (p90): {np.percentile(arr, 90):.2f} ms")
    print(f"95th percentile (p95): {np.percentile(arr, 95):.2f} ms")
    print(f"99th percentile (p99): {np.percentile(arr, 99):.2f} ms")
    print(f"Max lag              : {np.max(arr):.2f} ms")
    print(f"Min lag              : {np.min(arr):.2f} ms")
    print("───────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
phoneme_tail.py — CLI tool to stream and tail live phoneme events (Section 5.6)
Connects to /hapticsync_events and prints events to stdout.
"""

import sys
import time
from events_reader import EventRingReader


def main():
    print("── Haptic-Sync Phoneme Stream Monitor ─────────────────────────")
    print("Connecting to /dev/shm/hapticsync_events...")

    try:
        reader = EventRingReader()
    except FileNotFoundError:
        print("Error: Event ring not active. Is the pipeline running?")
        sys.exit(1)

    print(f"Connected (capacity: {reader.slot_count} slots). Tailing events...\n")
    print(f"{'EVENT ID':<10} {'FRAME':<10} {'PHONEME':<10} {'CONF':<8} {'LAG (ms)':<10} {'FLAGS'}")
    print("─" * 60)

    try:
        for evt in reader.poll(timeout_s=0.01):
            flag_str = "GAP_RESET" if (evt["flags"] & 1) else ""
            print(f"#{evt['event_id']:<9} {evt['abs_frame']:<10} {evt['phoneme']:<10} "
                  f"{evt['confidence']:<8.3f} {evt['lag_ms']:<10.1f} {flag_str}")
    except KeyboardInterrupt:
        print("\nTail stopped.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()


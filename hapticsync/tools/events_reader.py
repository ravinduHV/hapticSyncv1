#!/usr/bin/env python3
"""
events_reader.py — Python reader library for /hapticsync_events (Section 5.6)
Implements torn-read detection, cursor tracking, and lapped detection.
Reference implementation for downstream haptic consumer.
"""

import os
import mmap
import struct
import time
from typing import Optional, Dict, Generator

SHM_EVENTS_PATH = "/dev/shm/hapticsync_events"
HEADER_FMT = "<IIIIQq"
HEADER_SIZE = 64

# PhonemeEvent (64 bytes):
# seq(Q), event_id(Q), t_frame(q), t_emit(q), abs_frame(Q), window_id(I),
# phoneme_id(H), flags(H), conf(f), phoneme(12s)
EVENT_FMT = "<QQqqQIHHf12s"
EVENT_SIZE = 64


class EventRingReader:
    def __init__(self, shm_path: str = SHM_EVENTS_PATH):
        self.shm_path = shm_path
        self._mmap = None
        self._fd = None
        self.slot_count = 1024
        self.cursor = 1
        self.lapped_count = 0
        self._open()

    def _open(self):
        if not os.path.exists(self.shm_path):
            raise FileNotFoundError(f"Shared memory {self.shm_path} does not exist")
        self._fd = os.open(self.shm_path, os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)

        # Read header
        hdr = struct.unpack_from(HEADER_FMT, self._mmap, 0)
        self.slot_count = hdr[2]
        self.cursor = 1

    def sync_to_head(self):
        """Skip to current head."""
        hdr = struct.unpack_from(HEADER_FMT, self._mmap, 0)
        self.cursor = hdr[4]

    def read_next(self) -> Optional[Dict]:
        """Read the next available event, or None if no new events."""
        if self._mmap is None:
            return None

        # Check head
        hdr = struct.unpack_from(HEADER_FMT, self._mmap, 0)
        head = hdr[4]

        if self.cursor >= head:
            return None

        # Check if we were lapped
        if head - self.cursor > self.slot_count:
            dropped = (head - self.slot_count) - self.cursor
            self.lapped_count += dropped
            self.cursor = head - self.slot_count

        slot = (self.cursor - 1) % self.slot_count
        offset = HEADER_SIZE + slot * EVENT_SIZE

        # Read with torn-read validation
        seq1 = struct.unpack_from("<Q", self._mmap, offset)[0]
        data = self._mmap[offset : offset + EVENT_SIZE]
        seq2 = struct.unpack_from("<Q", self._mmap, offset)[0]

        if seq1 != seq2 or (seq1 & 1) != 0 or seq1 != 2 * self.cursor:
            # Torn read or slot overwritten during read
            self.lapped_count += 1
            self.cursor += 1
            return None

        fields = struct.unpack_from(EVENT_FMT, data)
        self.cursor += 1

        phoneme_str = fields[9].split(b"\x00")[0].decode("utf-8", errors="ignore")
        return {
            "event_id": fields[1],
            "t_frame_ns": fields[2],
            "t_emit_ns": fields[3],
            "abs_frame": fields[4],
            "window_id": fields[5],
            "phoneme_id": fields[6],
            "flags": fields[7],
            "confidence": fields[8],
            "phoneme": phoneme_str,
            "lag_ms": round((fields[3] - fields[2]) / 1e6, 2),
        }

    def poll(self, timeout_s: float = 0.05) -> Generator[Dict, None, None]:
        """Yield events as they arrive."""
        while True:
            evt = self.read_next()
            if evt is not None:
                yield evt
            else:
                time.sleep(timeout_s)

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


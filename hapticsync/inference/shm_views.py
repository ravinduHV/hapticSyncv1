#!/usr/bin/env python3
"""
shm_views.py — Shared memory access helpers for Python (Section 5.4)
Wraps /dev/shm/hapticsync_window in zero-copy NumPy float32 views.
"""

import os
import mmap
import numpy as np

MAX_WINDOW_SAMPLES = 4 * 16000  # 64,000 samples
SLOT_BYTES = MAX_WINDOW_SAMPLES * 4  # 256,000 bytes (float32)
WINDOW_SHM_SIZE = 2 * SLOT_BYTES  # 512,000 bytes
SHM_NAME = "/dev/shm/hapticsync_window"


class WindowSlotsView:
    def __init__(self, shm_path: str = SHM_NAME):
        self.shm_path = shm_path
        self._mmap = None
        self._fd = None
        self._slots = [None, None]
        self._open()

    def _open(self):
        if os.path.exists(self.shm_path):
            self._fd = os.open(self.shm_path, os.O_RDWR)
            self._mmap = mmap.mmap(self._fd, WINDOW_SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
            # Create views for slot 0 and slot 1
            self._slots[0] = np.ndarray(
                (MAX_WINDOW_SAMPLES,),
                dtype=np.float32,
                buffer=self._mmap,
                offset=0,
            )
            self._slots[1] = np.ndarray(
                (MAX_WINDOW_SAMPLES,),
                dtype=np.float32,
                buffer=self._mmap,
                offset=SLOT_BYTES,
            )
        else:
            # Fallback for standalone test / dev mode
            self._fallback_buf = np.zeros(2 * MAX_WINDOW_SAMPLES, dtype=np.float32)
            self._slots[0] = self._fallback_buf[:MAX_WINDOW_SAMPLES]
            self._slots[1] = self._fallback_buf[MAX_WINDOW_SAMPLES:]

    def get_slot_view(self, slot_id: int, n_samples: int) -> np.ndarray:
        """Return zero-copy slice of length n_samples from slot 0 or 1."""
        if slot_id not in (0, 1):
            raise ValueError(f"Invalid slot id {slot_id}")
        return self._slots[slot_id][:n_samples]

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


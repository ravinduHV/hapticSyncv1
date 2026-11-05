#!/usr/bin/env python3
"""
telemetry/main.py — Real-time telemetry reader and dashboard server (Section 5.7)
Runs on CPU 0 (Housekeeping). Non-blocking read-only taps into shared memory rings.
"""

import os
import sys
import time
import json
import struct
import mmap
import subprocess
import threading
import collections
from datetime import datetime, timezone
import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# Pin to Housekeeping core (CPU 0)
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass

# Paths and constants
CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config/hapticsync.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

PORT = cfg.get("telemetry", {}).get("port", 5000)
RATE_HZ = cfg.get("telemetry", {}).get("rate_hz", 10)


def find_web_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "web"),
        os.path.join(os.path.dirname(__file__), "web"),
        os.path.abspath("web"),
        os.path.abspath("../web"),
        os.path.abspath("hapticsync/web"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "index.html")):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


WEB_DIR = find_web_dir()
print(f"[telemetry] Serving static web UI from: {WEB_DIR}")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared memory mappings
SHM_STATS_PATH = "/dev/shm/hapticsync_stats"
SHM_EVENTS_PATH = "/dev/shm/hapticsync_events"
SHM_AUDIO_PATH = "/dev/shm/hapticsync_audio"

STATS_FMT = "<QQIIQQQqIQQqqqQQq"
STATS_SIZE = struct.calcsize(STATS_FMT)

EVENT_FMT = "<QQqqQIHHf12s"
EVENT_SIZE = 64

# Rolling history for web client
MAX_CHUNKS = 240
_queue = collections.deque(maxlen=MAX_CHUNKS)
_chunk_counter = 0
_lock = threading.Lock()


def read_system_telemetry():
    info = {"temp_c": 0.0, "freq_mhz": 0, "throttled": "0x0"}
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        info["temp_c"] = float(out.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        pass

    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
        info["throttled"] = out.strip()
    except Exception:
        pass

    return info


class ShmReader:
    def __init__(self):
        self.stats_mmap = None
        self.events_mmap = None
        self.audio_mmap = None
        self._last_cursor = 1
        self._open_shm()

    def _open_shm(self):
        try:
            if os.path.exists(SHM_STATS_PATH) and self.stats_mmap is None:
                fd = os.open(SHM_STATS_PATH, os.O_RDONLY)
                self.stats_mmap = mmap.mmap(fd, 256, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
        except Exception:
            self.stats_mmap = None

        try:
            if os.path.exists(SHM_EVENTS_PATH) and self.events_mmap is None:
                fd = os.open(SHM_EVENTS_PATH, os.O_RDONLY)
                self.events_mmap = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
        except Exception:
            self.events_mmap = None

        try:
            if os.path.exists(SHM_AUDIO_PATH) and self.audio_mmap is None:
                fd = os.open(SHM_AUDIO_PATH, os.O_RDONLY)
                self.audio_mmap = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
        except Exception:
            self.audio_mmap = None

    def read_stats(self):
        if self.stats_mmap is None:
            self._open_shm()
            if self.stats_mmap is None:
                return {}

        try:
            self.stats_mmap.seek(0)
            data = self.stats_mmap.read(STATS_SIZE)
            fields = struct.unpack_from(STATS_FMT, data)
            return {
                "periods_captured": fields[0],
                "samples_captured": fields[1],
                "xruns": fields[2],
                "gaps": fields[3],
                "windows_sent": fields[4],
                "windows_done": fields[5],
                "windows_dropped": fields[6],
                "last_infer_ms": round(fields[7] / 1e6, 2),
                "model_busy": fields[8],
                "events_emitted": fields[9],
                "events_lapped": fields[10],
                "last_lag_ms": round(fields[11] / 1e6, 2),
                "min_lag_ms": round(fields[12] / 1e6, 2) if fields[12] < 1e12 else 0.0,
                "max_lag_ms": round(fields[13] / 1e6, 2),
                "avg_lag_ms": round((fields[14] / max(fields[15], 1)) / 1e6, 2),
                "start_ns": fields[16],
            }
        except Exception:
            return {}

    def read_new_events(self):
        if self.events_mmap is None:
            self._open_shm()
            if self.events_mmap is None:
                return []

        new_events = []
        try:
            self.events_mmap.seek(0)
            hdr = struct.unpack_from("<IIIIQq", self.events_mmap.read(32))
            slot_count = hdr[2]
            head = hdr[4]

            if self._last_cursor < 1:
                self._last_cursor = 1

            if head - self._last_cursor > slot_count:
                self._last_cursor = head - slot_count

            hdr_size = 64
            while self._last_cursor < head:
                slot = (self._last_cursor - 1) % slot_count
                offset = hdr_size + slot * EVENT_SIZE
                self.events_mmap.seek(offset)
                data = self.events_mmap.read(EVENT_SIZE)
                fields = struct.unpack_from(EVENT_FMT, data)

                seq = fields[0]
                if (seq & 1) == 0 and seq == 2 * self._last_cursor:
                    phoneme = fields[9].split(b"\x00")[0].decode("utf-8", errors="ignore")
                    new_events.append({
                        "event_id": fields[1],
                        "abs_frame": fields[4],
                        "phoneme": phoneme,
                        "confidence": round(fields[8], 3),
                        "t_frame_ns": fields[2],
                        "t_emit_ns": fields[3],
                        "lag_ms": round((fields[3] - fields[2]) / 1e6, 2),
                    })
                self._last_cursor += 1
        except Exception:
            pass

        return new_events

    def read_recent_waveform(self, n_points=150):
        if self.audio_mmap is None:
            self._open_shm()
            if self.audio_mmap is None:
                return [0.0] * n_points

        try:
            self.audio_mmap.seek(0)
            hdr = struct.unpack_from("<IIIIQ", self.audio_mmap.read(24))
            cap = hdr[3]
            wi = hdr[4]

            if wi == 0 or cap == 0:
                return [0.0] * n_points

            n_samples = min(1600, cap)
            start = wi - n_samples
            mask = cap - 1
            pos = start & mask

            hdr_offset = 64 + 256 * 16 + 64 * 16
            data_offset = hdr_offset + pos * 4

            self.audio_mmap.seek(data_offset)
            raw = self.audio_mmap.read(n_samples * 4)
            samples = np.frombuffer(raw, dtype=np.float32)

            if len(samples) == 0:
                return [0.0] * n_points

            idx = np.linspace(0, len(samples) - 1, n_points, dtype=int)
            return [round(float(s), 4) for s in samples[idx]]
        except Exception:
            return [0.0] * n_points


reader = ShmReader()


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


@app.route("/api/stats")
def api_stats():
    with _lock:
        q = list(_queue)

    all_ph = [p for item in q for p in item.get("phonemes", [])]
    freq = {}
    for p in all_ph:
        freq[p] = freq.get(p, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:12]

    st = reader.read_stats()
    avg_inf = st.get("last_infer_ms", 0.0)
    avg_rms = float(np.mean([i.get("rms_l", 0.0) for i in q])) if q else 0.0

    return jsonify({
        "total_chunks": len(q),
        "max_chunks": MAX_CHUNKS,
        "total_phonemes": len(all_ph),
        "empty_chunks": sum(1 for item in q if not item.get("phonemes")),
        "top_phonemes": [{"phoneme": k, "count": v} for k, v in top],
        "avg_rms_l": avg_rms,
        "avg_rms_r": avg_rms,
        "avg_inf_ms": avg_inf,
        "avg_timing": {
            "compute": avg_inf,
            "total": st.get("last_lag_ms", 0.0),
            "record": 10.0,
            "preprocess": 0.5,
            "fft": 0.5,
            "processor": 0.0,
            "forward": avg_inf,
            "decode": 0.2,
        },
    })


@app.route("/api/queue")
def api_queue():
    with _lock:
        items = list(_queue)
    return jsonify({
        "chunks": items,
        "total": len(items),
        "max": MAX_CHUNKS,
        "duration_s": 120,
    })


@socketio.on("connect")
def on_connect():
    with _lock:
        items = list(_queue)
    emit("queue_snapshot", {"chunks": items, "max": MAX_CHUNKS, "duration_s": 120})


def telemetry_broadcaster():
    """Background thread polling rings and emitting new chunks & updates."""
    global _chunk_counter
    interval = 1.0 / max(RATE_HZ, 1)

    while True:
        try:
            evts = reader.read_new_events()
            st = reader.read_stats()

            if evts:
                wf = reader.read_recent_waveform(150)
                rms_val = float(np.sqrt(np.mean(np.array(wf) ** 2)))
                peak_val = float(np.max(np.abs(wf))) if wf else 0.0
                ph_list = [e["phoneme"] for e in evts]

                _chunk_counter += 1
                chunk_id = f"chunk_{_chunk_counter:06d}"
                ts = datetime.now(timezone.utc)

                record = {
                    "id": chunk_id,
                    "index": _chunk_counter,
                    "timestamp": ts.isoformat(),
                    "timestamp_ms": int(ts.timestamp() * 1000),
                    "phonemes": ph_list,
                    "phoneme_str": " ".join(ph_list),
                    "rms_l": round(rms_val, 4),
                    "rms_r": round(rms_val, 4),
                    "peak_l": round(peak_val, 4),
                    "peak_r": round(peak_val, 4),
                    "dom_freq_l": 0.0,
                    "dom_freq_r": 0.0,
                    "inference_ms": st.get("last_infer_ms", 0.0),
                    "timing": {
                        "compute": st.get("last_infer_ms", 0.0),
                        "total": st.get("last_lag_ms", 0.0),
                        "record": 10.0,
                        "forward": st.get("last_infer_ms", 0.0),
                    },
                    "waveform_l": wf,
                    "waveform_r": wf,
                }

                with _lock:
                    _queue.append(record)

                socketio.emit("new_chunk", record)

        except Exception:
            pass

        time.sleep(interval)


def main():
    print(f"[telemetry] Starting dashboard server on port {PORT} (CPU 0)...")
    bg = threading.Thread(target=telemetry_broadcaster, daemon=True)
    bg.start()
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

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

app = Flask(__name__, static_folder="../web", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared memory mappings
SHM_STATS_PATH = "/dev/shm/hapticsync_stats"
SHM_EVENTS_PATH = "/dev/shm/hapticsync_events"
SHM_AUDIO_PATH = "/dev/shm/hapticsync_audio"

# Stats struct: 12 atomics + padding
# periods(Q), samples(Q), xruns(I), gaps(I), sent(Q), done(Q), dropped(Q),
# last_infer(q), model_busy(I), emitted(Q), lapped(Q), last_lag(q), min_lag(q), max_lag(q),
# lag_sum(Q), lag_count(Q), start_ns(q)
STATS_FMT = "<QQIIQQQqIQQqqqQQq"
STATS_SIZE = struct.calcsize(STATS_FMT)

# PhonemeEvent struct (64 bytes):
# seq(Q), event_id(Q), t_frame(q), t_emit(q), abs_frame(Q), window_id(I),
# phoneme_id(H), flags(H), conf(f), phoneme(12s)
EVENT_FMT = "<QQqqQIHHf12s"
EVENT_SIZE = 64


def read_system_telemetry():
    """Read CPU temperature, clock frequency, and throttling status."""
    info = {"temp_c": 0.0, "freq_mhz": 0, "throttled": "0x0"}
    try:
        # Raspberry Pi specific commands
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
        self._open_shm()

    def _open_shm(self):
        try:
            if os.path.exists(SHM_STATS_PATH):
                fd = os.open(SHM_STATS_PATH, os.O_RDONLY)
                self.stats_mmap = mmap.mmap(fd, 256, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
        except Exception:
            self.stats_mmap = None

        try:
            if os.path.exists(SHM_EVENTS_PATH):
                fd = os.open(SHM_EVENTS_PATH, os.O_RDONLY)
                # Read header to determine size
                self.events_mmap = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
        except Exception:
            self.events_mmap = None

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

    def read_recent_events(self, count=10):
        if self.events_mmap is None:
            self._open_shm()
            if self.events_mmap is None:
                return []

        try:
            # Header: magic(I), version(I), slot_count(I), pad0(I), head(Q), start_ns(q)
            self.events_mmap.seek(0)
            hdr = struct.unpack_from("<IIIIQq", self.events_mmap.read(32))
            slot_count = hdr[2]
            head = hdr[4]

            events = []
            hdr_size = 64
            read_start = max(1, head - count)
            for eid in range(read_start, head):
                slot = (eid - 1) % slot_count
                offset = hdr_size + slot * EVENT_SIZE
                self.events_mmap.seek(offset)
                data = self.events_mmap.read(EVENT_SIZE)
                fields = struct.unpack_from(EVENT_FMT, data)

                seq = fields[0]
                if (seq & 1) == 0 and seq == 2 * eid:
                    phoneme = fields[9].split(b"\x00")[0].decode("utf-8", errors="ignore")
                    events.append({
                        "event_id": fields[1],
                        "abs_frame": fields[4],
                        "phoneme": phoneme,
                        "confidence": round(fields[8], 3),
                        "t_frame_ns": fields[2],
                        "t_emit_ns": fields[3],
                        "lag_ms": round((fields[3] - fields[2]) / 1e6, 2),
                    })
            return events
        except Exception:
            return []


reader = ShmReader()


@app.route("/")
def index():
    return send_from_directory("../web", "index.html")


@app.route("/api/stats")
def api_stats():
    st = reader.read_stats()
    sys_info = read_system_telemetry()
    evts = reader.read_recent_events(20)
    return jsonify({
        "stats": st,
        "system": sys_info,
        "recent_events": evts,
    })


def telemetry_broadcaster():
    """Background thread emitting live status at RATE_HZ."""
    interval = 1.0 / max(RATE_HZ, 1)
    while True:
        try:
            st = reader.read_stats()
            sys_info = read_system_telemetry()
            evts = reader.read_recent_events(5)

            payload = {
                "stats": st,
                "system": sys_info,
                "events": evts,
                "timestamp_ms": int(time.time() * 1000),
            }
            socketio.emit("telemetry_update", payload)
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


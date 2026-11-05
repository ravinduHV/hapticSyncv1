#!/usr/bin/env python3
"""
telemetry/publisher.py — Ultra-lightweight telemetry streaming daemon (CPU 0)
Streams pipeline metrics and phoneme events to remote subscribers (e.g. PC dashboard)
via a low-overhead TCP socket server with ZERO external dependencies.
CPU usage on Raspberry Pi: < 0.5%.
"""

import os
import sys
import time
import json
import struct
import mmap
import socket
import select
import threading
import subprocess
from datetime import datetime, timezone

# Pin to Housekeeping core (CPU 0)
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config/hapticsync.json"
try:
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
except Exception:
    cfg = {}

STREAM_PORT = cfg.get("telemetry", {}).get("port", 5555)
# If config still has 5000 from old dashboard, use 5555 for publisher
if STREAM_PORT == 5000:
    STREAM_PORT = 5555

RATE_HZ = cfg.get("telemetry", {}).get("rate_hz", 10)

# Shared memory paths
SHM_STATS_PATH = "/dev/shm/hapticsync_stats"
SHM_EVENTS_PATH = "/dev/shm/hapticsync_events"
SHM_AUDIO_PATH = "/dev/shm/hapticsync_audio"

STATS_FMT = "<QQIIQQQqIQQqqqQQq"
STATS_SIZE = struct.calcsize(STATS_FMT)

EVENT_FMT = "<QQqqQIHHf12s"
EVENT_SIZE = 64

# Cache for low-frequency system checks (vcgencmd)
_cached_sys_info = {"temp_c": 0.0, "freq_mhz": 2400, "throttled": "0x0", "throttled_flags": []}
_last_sys_check = 0.0
_cached_sim_status = False
_last_sim_check = 0.0


def decode_throttled(val_str):
    try:
        val = int(val_str, 16) if isinstance(val_str, str) else int(val_str)
    except Exception:
        return []
    flags = []
    if val & 0x1: flags.append("Under-voltage detected (active)")
    if val & 0x2: flags.append("Arm frequency capped (active)")
    if val & 0x4: flags.append("Currently throttled (active)")
    if val & 0x8: flags.append("Soft temperature limit active")
    if val & 0x10000: flags.append("Under-voltage occurred previously")
    if val & 0x20000: flags.append("Arm frequency capping occurred previously")
    if val & 0x40000: flags.append("Throttling occurred previously")
    if val & 0x80000: flags.append("Soft temperature limit occurred previously")
    return flags


def get_fast_system_telemetry():
    """Reads hardware telemetry with near-zero CPU overhead."""
    global _cached_sys_info, _last_sys_check
    now = time.time()

    # Fast sysfs temperature read (<1 microsecond, no subprocess)
    temp_c = _cached_sys_info.get("temp_c", 0.0)
    try:
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_c = round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        pass
    _cached_sys_info["temp_c"] = temp_c

    # Slow check for vcgencmd throttled & clock (only once every 5 seconds)
    if now - _last_sys_check > 5.0:
        _last_sys_check = now
        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True)
            val = out.strip().replace("throttled=", "")
            _cached_sys_info["throttled"] = val
            _cached_sys_info["throttled_flags"] = decode_throttled(val)
        except Exception:
            pass

        try:
            out = subprocess.check_output(["vcgencmd", "measure_clock", "arm"], text=True)
            hz = int(out.strip().split("=")[1])
            _cached_sys_info["freq_mhz"] = round(hz / 1e6)
        except Exception:
            pass

    return _cached_sys_info


def get_simulator_status():
    """Checks simulator presence at most once every 5 seconds."""
    global _cached_sim_status, _last_sim_check
    now = time.time()
    if now - _last_sim_check > 5.0:
        _last_sim_check = now
        try:
            out = subprocess.check_output(["pgrep", "-f", "i2s_mic_simulator.py"], text=True)
            _cached_sim_status = len(out.strip().splitlines()) > 0
        except Exception:
            _cached_sim_status = False
    return _cached_sim_status


def analyze_bottlenecks(stats, sys_info, cfg_data):
    bottlenecks = []
    status = "healthy"

    win_ms = cfg_data.get("window", {}).get("fixed_window_ms", 1000)
    infer_ms = stats.get("last_infer_ms", 0.0)
    rtf = round(infer_ms / max(win_ms, 1), 2) if infer_ms > 0 else 0.0

    if rtf > 1.0:
        bottlenecks.append({
            "stage": "Wav2Vec2 Inference (CPU 2,3)",
            "severity": "critical" if rtf > 1.5 else "warning",
            "message": f"RTF is {rtf:.2f} (> 1.0). Inference takes {infer_ms:.0f}ms per {win_ms}ms window.",
            "recommendation": "Use 'latest' window policy or enable INT8 dynamic quantization."
        })
        if status == "healthy":
            status = "warning"

    dropped = stats.get("windows_dropped", 0)
    sent = stats.get("windows_sent", 0)
    if sent > 0 and dropped > 0:
        drop_rate = round((dropped / sent) * 100, 1)
        if drop_rate > 10:
            bottlenecks.append({
                "stage": "Window Queue",
                "severity": "warning",
                "message": f"{dropped} audio windows dropped ({drop_rate}% drop rate).",
                "recommendation": "Normal for 'latest' policy when RTF > 1.0 to preserve freshness."
            })

    xruns = stats.get("xruns", 0)
    if xruns > 0:
        bottlenecks.append({
            "stage": "Audio Producer (ALSA)",
            "severity": "warning",
            "message": f"{xruns} buffer overrun/underrun (XRUN) events detected on audio capture.",
            "recommendation": "Increase buffer_periods or check ALSA loopback pacing."
        })

    temp = sys_info.get("temp_c", 0.0)
    if temp >= 80.0:
        bottlenecks.append({
            "stage": "SoC Thermals",
            "severity": "critical",
            "message": f"CPU temperature is {temp:.1f}°C (exceeds thermal threshold). Frequency throttling active.",
            "recommendation": "Verify active cooler fan is operating."
        })
        status = "critical"

    return {
        "overall_status": status,
        "rtf": rtf,
        "bottlenecks": bottlenecks
    }


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

    def read_recent_waveform(self, n_points=80):
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

            n_samples = min(800, cap)
            start = wi - n_samples
            mask = cap - 1
            pos = start & mask

            hdr_offset = 64 + 256 * 16 + 64 * 16
            data_offset = hdr_offset + pos * 4

            self.audio_mmap.seek(data_offset)
            raw = self.audio_mmap.read(n_samples * 4)

            # Fast step downsampling without numpy
            count = len(raw) // 4
            step = max(count // n_points, 1)
            out = []
            for i in range(0, min(count, n_points * step), step):
                val = struct.unpack_from("<f", raw, i * 4)[0]
                out.append(round(val, 3))
            return out[:n_points]
        except Exception:
            return [0.0] * n_points


class TelemetryStreamServer:
    """Non-blocking TCP server streaming length-prefixed JSON packets to subscribers."""
    def __init__(self, port):
        self.port = port
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", port))
        self.server_sock.listen(5)
        self.server_sock.setblocking(False)
        self.clients = []
        self._lock = threading.Lock()
        print(f"[publisher] Streaming server listening on 0.0.0.0:{port} (TCP)")

    def accept_new_clients(self):
        while True:
            try:
                client_sock, addr = self.server_sock.accept()
                client_sock.setblocking(False)
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self.clients.append(client_sock)
                print(f"[publisher] Subscriber connected from {addr[0]}:{addr[1]} (active clients: {len(self.clients)})")
            except (BlockingIOError, socket.error):
                break

    def broadcast(self, msg_type: str, payload: dict):
        self.accept_new_clients()
        with self._lock:
            if not self.clients:
                return

        packet_bytes = json.dumps({"type": msg_type, "data": payload}).encode("utf-8")
        header = struct.pack("<I", len(packet_bytes))
        full_msg = header + packet_bytes

        with self._lock:
            dead_clients = []
            for c in self.clients:
                try:
                    c.sendall(full_msg)
                except Exception:
                    dead_clients.append(c)

            for dead in dead_clients:
                try:
                    dead.close()
                except Exception:
                    pass
                self.clients.remove(dead)
                print(f"[publisher] Subscriber disconnected (active clients: {len(self.clients)})")


def main():
    print("==========================================================")
    print(f"  Haptic-Sync Ultra-Lightweight Telemetry Publisher")
    print(f"  Port: {STREAM_PORT} (TCP) | CPU: 0")
    print("==========================================================")

    reader = ShmReader()
    server = TelemetryStreamServer(STREAM_PORT)

    interval = 1.0 / max(RATE_HZ, 1)
    tick = 0
    chunk_counter = 0

    try:
        while True:
            tick += 1
            evts = reader.read_new_events()
            st = reader.read_stats()

            # Broadcast new phoneme events chunk
            if evts:
                chunk_counter += 1
                wf = reader.read_recent_waveform(80)
                rms_val = round(sum(abs(x) for x in wf) / max(len(wf), 1), 4) if wf else 0.0
                ph_list = [e["phoneme"] for e in evts]
                ts = datetime.now(timezone.utc).isoformat()

                chunk_record = {
                    "id": f"chunk_{chunk_counter:06d}",
                    "index": chunk_counter,
                    "timestamp": ts,
                    "phonemes": ph_list,
                    "phoneme_str": " ".join(ph_list),
                    "rms_l": rms_val,
                    "rms_r": rms_val,
                    "inference_ms": st.get("last_infer_ms", 0.0),
                    "timing": {
                        "compute": st.get("last_infer_ms", 0.0),
                        "total": st.get("last_lag_ms", 0.0),
                    },
                    "waveform_l": wf,
                    "waveform_r": wf,
                }
                server.broadcast("new_chunk", chunk_record)

            # Broadcast pipeline update every 1 second
            if tick % max(RATE_HZ, 1) == 0:
                sys_info = get_fast_system_telemetry()
                sim_running = get_simulator_status()
                bneck_info = analyze_bottlenecks(st, sys_info, cfg)
                wf_preview = reader.read_recent_waveform(80)

                pipeline_payload = {
                    "stats": st,
                    "system": sys_info,
                    "source": {
                        "source_type": cfg.get("audio", {}).get("source", "loopback"),
                        "device": cfg.get("audio", {}).get("alsa_device", "hw:Loopback,1,0"),
                        "rate": cfg.get("audio", {}).get("rate", 16000),
                        "channels": cfg.get("audio", {}).get("channels", 2),
                        "simulator_running": sim_running,
                    },
                    "bottlenecks": bneck_info,
                    "waveform": wf_preview,
                }
                server.broadcast("pipeline_update", pipeline_payload)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[publisher] Shutting down.")


if __name__ == "__main__":
    main()


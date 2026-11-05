#!/usr/bin/env python3
"""
pc_dashboard.py — Remote Web Dashboard Server (Runs on your PC)
Connects to the lightweight telemetry publisher on Raspberry Pi,
receives pipeline events over TCP, and hosts the web dashboard locally.
Completely offloads web serving, WebSockets, and rendering from Raspberry Pi.
"""

import os
import sys
import time
import json
import struct
import socket
import select
import threading
import argparse
import collections
from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

parser = argparse.ArgumentParser(description="HapticSync Remote PC Dashboard")
parser.add_argument("--host", default="192.168.8.140", help="Raspberry Pi IP address (default: 192.168.8.140)")
parser.add_argument("--port", type=int, default=5555, help="Raspberry Pi publisher TCP port (default: 5555)")
parser.add_argument("--web-port", type=int, default=5000, help="Local PC dashboard HTTP port (default: 5000)")
args = parser.parse_args()

RPI_HOST = args.host
RPI_PORT = args.port
LOCAL_WEB_PORT = args.web_port


def find_web_dir():
    candidates = [
        os.path.join(os.path.dirname(__file__), "web"),
        os.path.join(os.path.dirname(__file__), "..", "web"),
        os.path.abspath("hapticsync/web"),
        os.path.abspath("web"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "index.html")):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


WEB_DIR = find_web_dir()
print(f"[pc_dashboard] Serving web UI from: {WEB_DIR}")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# In-memory storage for dashboard
MAX_CHUNKS = 240
_queue = collections.deque(maxlen=MAX_CHUNKS)
_latest_pipeline_data = {}
_lock = threading.Lock()
_connected_to_rpi = False


def recv_exact(sock, n_bytes):
    data = bytearray()
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def rpi_receiver_loop():
    """Background thread that connects to Raspberry Pi and receives telemetry packets."""
    global _connected_to_rpi, _latest_pipeline_data
    print(f"[pc_dashboard] Background receiver connecting to Raspberry Pi at {RPI_HOST}:{RPI_PORT}...")

    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4.0)
            sock.connect((RPI_HOST, RPI_PORT))
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            _connected_to_rpi = True
            print(f"[pc_dashboard] CONNECTED to Raspberry Pi ({RPI_HOST}:{RPI_PORT})! Streaming live events...")

            while True:
                # 1. Read 4-byte length prefix
                hdr = recv_exact(sock, 4)
                if not hdr:
                    print("[pc_dashboard] Connection closed by Raspberry Pi.")
                    break

                pkt_len = struct.unpack("<I", hdr)[0]
                if pkt_len > 10 * 1024 * 1024:  # sanity check (10MB)
                    print(f"[pc_dashboard] Invalid packet length: {pkt_len}, reconnecting...")
                    break

                # 2. Read full JSON packet
                raw_data = recv_exact(sock, pkt_len)
                if not raw_data:
                    break

                try:
                    msg = json.loads(raw_data.decode("utf-8"))
                    msg_type = msg.get("type")
                    data = msg.get("data", {})

                    if msg_type == "new_chunk":
                        with _lock:
                            _queue.append(data)
                        socketio.emit("new_chunk", data)

                    elif msg_type == "pipeline_update":
                        with _lock:
                            _latest_pipeline_data = data
                            # Populate recent events from queue if empty
                            if not data.get("recent_events"):
                                revts = []
                                for c in list(_queue)[-10:]:
                                    for p in c.get("phonemes", []):
                                        revts.append({
                                            "event_id": c.get("index", 0),
                                            "phoneme": p,
                                            "confidence": 0.95,
                                            "lag_ms": c.get("timing", {}).get("total", 0.0),
                                            "timestamp": c.get("timestamp", ""),
                                        })
                                data["recent_events"] = revts
                        socketio.emit("pipeline_update", data)

                except Exception as e:
                    print(f"[pc_dashboard] Error parsing packet: {e}")

        except (socket.error, ConnectionRefusedError, TimeoutError) as e:
            _connected_to_rpi = False
            time.sleep(2.0)
        except Exception as e:
            _connected_to_rpi = False
            print(f"[pc_dashboard] Receiver error: {e}")
            time.sleep(2.0)
        finally:
            _connected_to_rpi = False
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


# ── Flask Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/pipeline")
def pipeline():
    return send_from_directory(WEB_DIR, "pipeline.html")


@app.route("/api/pipeline")
def api_pipeline():
    with _lock:
        data = dict(_latest_pipeline_data)
        q = list(_queue)

    if not data:
        data = {
            "stats": {},
            "system": {"temp_c": 0, "throttled": "0x0"},
            "source": {"source_type": "loopback", "device": "hw:Loopback,1,0", "rate": 16000},
            "bottlenecks": {"overall_status": "connecting" if not _connected_to_rpi else "healthy", "bottlenecks": []},
            "waveform": [],
            "recent_events": [],
        }

    data["connected_to_rpi"] = _connected_to_rpi
    return jsonify(data)


@app.route("/api/stats")
def api_stats():
    with _lock:
        q = list(_queue)
        pl = dict(_latest_pipeline_data)

    all_ph = [p for item in q for p in item.get("phonemes", [])]
    freq = {}
    for p in all_ph:
        freq[p] = freq.get(p, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:12]

    st = pl.get("stats", {})
    avg_inf = st.get("last_infer_ms", 0.0)
    avg_rms = sum(i.get("rms_l", 0.0) for i in q) / max(len(q), 1) if q else 0.0

    return jsonify({
        "total_chunks": len(q),
        "max_chunks": MAX_CHUNKS,
        "total_phonemes": len(all_ph),
        "empty_chunks": sum(1 for item in q if not item.get("phonemes")),
        "top_phonemes": [{"phoneme": k, "count": v} for k, v in top],
        "avg_rms_l": round(avg_rms, 4),
        "avg_rms_r": round(avg_rms, 4),
        "avg_inf_ms": avg_inf,
        "connected_to_rpi": _connected_to_rpi,
        "avg_timing": {
            "compute": avg_inf,
            "total": st.get("last_lag_ms", 0.0),
            "record": 10.0,
            "forward": avg_inf,
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


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


@socketio.on("connect")
def on_connect():
    with _lock:
        items = list(_queue)
        pl = dict(_latest_pipeline_data)
    emit("queue_snapshot", {"chunks": items, "max": MAX_CHUNKS, "duration_s": 120})
    if pl:
        emit("pipeline_update", pl)


def main():
    print("==========================================================")
    print("  HapticSync Remote PC Web Dashboard")
    print(f"  Target Raspberry Pi : {RPI_HOST}:{RPI_PORT}")
    print(f"  Local Web Dashboard : http://localhost:{LOCAL_WEB_PORT}")
    print(f"  Pipeline Monitor   : http://localhost:{LOCAL_WEB_PORT}/pipeline")
    print("==========================================================")

    # Start TCP receiver thread
    t = threading.Thread(target=rpi_receiver_loop, daemon=True)
    t.start()

    # Run local web server on PC
    socketio.run(app, host="0.0.0.0", port=LOCAL_WEB_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()


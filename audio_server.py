#!/usr/bin/env python3
"""
audio_server.py
===============
Continuously records stereo audio chunks (0.5 s each), runs phoneme
recognition on every chunk, maintains a 2-minute rolling queue, and
streams results to a live browser dashboard via WebSocket.

Architecture
------------
  Recording thread  ->  deque(maxlen=240)  ->  Flask-SocketIO  ->  Browser
                              |
                        chunk_cache/   (WAV files, auto-cleaned)

Queue maths
-----------
  chunk duration : 0.5 s
  2 minutes      : 120 s / 0.5 s = 240 chunks max

Usage on Raspberry Pi 5
-----------------------
  # With ALSA loopback  (start i2s_mic_simulator.py first in another terminal)
  python3 audio_server.py --loopback

  # With a real I2S mic
  python3 audio_server.py

  # Specific device index
  python3 audio_server.py --device 2

  # Then open in browser on the same network:
  http://<pi-ip>:5000

Dependencies (add to pip install):
  pip install flask flask-socketio soundfile
"""

import argparse
import collections
import os
import sys
import threading
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
import soundfile as sf
from flask import Flask, jsonify, send_file
from flask_socketio import SocketIO, emit

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirror audio.py values)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000
DURATION         = 0.5          # seconds per chunk  (matches audio.py default)
CHANNELS         = 2
DTYPE            = "float32"
QUEUE_SECONDS    = 120          # 2-minute rolling window
MAX_CHUNKS       = int(QUEUE_SECONDS / DURATION)   # 240
CHUNK_CACHE_DIR  = "chunk_cache"
WAVEFORM_POINTS  = 150          # down-sampled waveform sent to browser

# ─────────────────────────────────────────────────────────────────────────────
# Flask + SocketIO
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="web", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────
_lock          = threading.Lock()
_queue         = collections.deque(maxlen=MAX_CHUNKS)   # ring buffer
_chunk_counter = 0
_server_running = True


# ─────────────────────────────────────────────────────────────────────────────
# Audio / signal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _downsample(signal, n):
    """Return n evenly-spaced values from signal (for waveform display)."""
    if len(signal) == 0:
        return [0.0] * n
    idx = np.linspace(0, len(signal) - 1, n, dtype=int)
    return signal[idx].tolist()


def _dominant_freq(signal):
    from scipy.fft import fft, fftfreq
    n  = len(signal)
    if n == 0:
        return 0.0
    yf = np.abs(fft(signal))[: n // 2]
    xf = fftfreq(n, 1.0 / SAMPLE_RATE)[: n // 2]
    return float(xf[np.argmax(yf)])


def _build_record(index, audio, phonemes, inference_ms):
    """Turn a raw audio array + phoneme list into the JSON-safe dict."""
    left  = audio[:, 0]
    right = audio[:, 1]

    chunk_id = f"chunk_{index:06d}"
    ts       = datetime.now(timezone.utc)

    # ── Save WAV ──────────────────────────────────────────────────────────
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    wav_path = os.path.join(CHUNK_CACHE_DIR, f"{chunk_id}.wav")
    sf.write(wav_path, audio, SAMPLE_RATE)

    return {
        "id":           chunk_id,
        "index":        index,
        "timestamp":    ts.isoformat(),
        "timestamp_ms": int(ts.timestamp() * 1000),
        "phonemes":     phonemes,
        "phoneme_str":  " ".join(phonemes),
        "rms_l":        float(np.sqrt(np.mean(left  ** 2))),
        "rms_r":        float(np.sqrt(np.mean(right ** 2))),
        "peak_l":       float(np.max(np.abs(left))),
        "peak_r":       float(np.max(np.abs(right))),
        "dom_freq_l":   _dominant_freq(left),
        "dom_freq_r":   _dominant_freq(right),
        "inference_ms": round(inference_ms, 1),
        "waveform_l":   _downsample(left,  WAVEFORM_POINTS),
        "waveform_r":   _downsample(right, WAVEFORM_POINTS),
        "audio_url":    f"/api/audio/{chunk_id}",
        "_wav_path":    wav_path,   # internal only, stripped before sending
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recording / inference loop  (runs in its own daemon thread)
# ─────────────────────────────────────────────────────────────────────────────

def _recording_loop(device, model):
    global _chunk_counter, _server_running

    print(f"[server] Recording loop started  |  device={device!r}  "
          f"|  chunk={DURATION}s  |  queue={MAX_CHUNKS} chunks ({QUEUE_SECONDS}s)")

    while _server_running:
        try:
            # ── Record one chunk ─────────────────────────────────────────
            n_frames = int(SAMPLE_RATE * DURATION)
            audio = sd.rec(
                n_frames,
                samplerate = SAMPLE_RATE,
                channels   = CHANNELS,
                dtype      = DTYPE,
                device     = device,
            )
            sd.wait()

            # ── Phoneme inference ────────────────────────────────────────
            mono = np.mean(audio, axis=1)
            t0   = time.perf_counter()
            phonemes = model.predict(mono)
            inf_ms   = (time.perf_counter() - t0) * 1000

            _chunk_counter += 1
            record = _build_record(_chunk_counter, audio, phonemes, inf_ms)

            # ── Add to rolling queue; collect evicted item ───────────────
            evicted = None
            with _lock:
                if len(_queue) == MAX_CHUNKS:
                    evicted = _queue[0]
                _queue.append(record)

            # ── Delete evicted WAV from disk ─────────────────────────────
            if evicted:
                p = evicted.get("_wav_path", "")
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            # ── Emit to all connected browser clients ────────────────────
            public = {k: v for k, v in record.items() if not k.startswith("_")}
            socketio.emit("new_chunk", public)

            print(f"[server] Chunk #{_chunk_counter:05d}  "
                  f"phonemes={phonemes}  inf={inf_ms:.0f}ms")

        except Exception as exc:
            print(f"[server] ERROR in recording loop: {exc}", file=sys.stderr)
            time.sleep(0.2)


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/queue")
def api_queue():
    with _lock:
        items = [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in _queue
        ]
    return jsonify({
        "chunks":     items,
        "total":      len(items),
        "max":        MAX_CHUNKS,
        "duration_s": QUEUE_SECONDS,
    })


@app.route("/api/audio/<chunk_id>")
def api_audio(chunk_id):
    # Basic validation
    if not (chunk_id.startswith("chunk_") and chunk_id[6:].isdigit()):
        return "Invalid ID", 400
    wav = os.path.join(CHUNK_CACHE_DIR, f"{chunk_id}.wav")
    if not os.path.exists(wav):
        return "Chunk not found (may have been evicted from queue)", 404
    return send_file(wav, mimetype="audio/wav")


@app.route("/api/stats")
def api_stats():
    with _lock:
        q = list(_queue)
    all_ph = [p for item in q for p in item["phonemes"]]
    freq   = {}
    for p in all_ph:
        freq[p] = freq.get(p, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:12]
    return jsonify({
        "total_chunks":   len(q),
        "max_chunks":     MAX_CHUNKS,
        "total_phonemes": len(all_ph),
        "empty_chunks":   sum(1 for item in q if not item["phonemes"]),
        "top_phonemes":   [{"phoneme": k, "count": v} for k, v in top],
        "avg_rms_l":      float(np.mean([i["rms_l"] for i in q])) if q else 0.0,
        "avg_rms_r":      float(np.mean([i["rms_r"] for i in q])) if q else 0.0,
        "avg_inf_ms":     float(np.mean([i["inference_ms"] for i in q])) if q else 0.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SocketIO events
# ─────────────────────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    """Send the full current queue as a snapshot to a newly connected client."""
    with _lock:
        items = [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in _queue
        ]
    emit("queue_snapshot", {"chunks": items, "max": MAX_CHUNKS, "duration_s": QUEUE_SECONDS})


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_loopback_input():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if "loopback" in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
            if ",1)" in d["name"]:
                return i
    for i, d in enumerate(devices):
        if "loopback" in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        prog="audio_server.py",
        description="BinaryBrains – live phoneme dashboard with 2-minute rolling queue",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--loopback", action="store_true",
                     help="Use ALSA loopback (i2s_mic_simulator.py must be running)")
    grp.add_argument("--device", type=int, default=None,
                     help="sounddevice input device index")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5000, help="HTTP port (default: 5000)")
    p.add_argument("--list-devices", action="store_true",
                   help="List sounddevice devices and exit")
    return p.parse_args()


def main():
    global _server_running
    args = _parse()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # ── Resolve input device ─────────────────────────────────────────────
    device = None
    if args.loopback:
        device = _find_loopback_input()
        if device is None:
            print("ERROR: ALSA Loopback not found. Run: sudo modprobe snd-aloop",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[server] Loopback input [{device}]: {sd.query_devices(device)['name']}")
    elif args.device is not None:
        device = args.device
        print(f"[server] Input device [{device}]: {sd.query_devices(device)['name']}")
    else:
        print("[server] Using system default input device.")

    # ── Load model ───────────────────────────────────────────────────────
    print("[server] Loading phoneme recognition model …")
    from model import RecognitionModel
    model = RecognitionModel()
    print("[server] Model ready.\n")

    # ── Start recording thread ───────────────────────────────────────────
    t = threading.Thread(target=_recording_loop, args=(device, model), daemon=True)
    t.start()

    print(f"[server] Dashboard  →  http://localhost:{args.port}")
    print(f"[server] Queue size : {MAX_CHUNKS} chunks × {DURATION}s = {QUEUE_SECONDS}s\n")

    try:
        socketio.run(app, host=args.host, port=args.port, debug=False,
                     use_reloader=False, log_output=False)
    except KeyboardInterrupt:
        _server_running = False
        print("\n[server] Stopped.")


if __name__ == "__main__":
    main()

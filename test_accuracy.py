#!/usr/bin/env python3
"""
test_accuracy.py
================
Accuracy / diagnostic test runner for the BinaryBrains phoneme recogniser.

For every recorded audio chunk this script prints a compact report:

  ┌─ Chunk #3  (0.125 s  |  2000 samples @ 16000 Hz)
  │  RMS L: 0.1823   RMS R: 0.1801   Peak L: 0.4991   Peak R: 0.4998
  │  Dominant freq L:  440.0 Hz  |  Dominant freq R:  880.0 Hz
  │  Phonemes  (4): [ ah  ih  n  iy ]
  └──────────────────────────────────────────────────────────────────

Optionally every chunk is saved as a WAV file in ./chunk_logs/ so you
can replay them with any audio player and check ground truth manually.

Works with EITHER:
  (a) a real I2S mic (physical device selected by --device)
  (b) the ALSA loopback fed by i2s_mic_simulator.py (--loopback flag)

Usage examples
--------------
  # Use loopback device (i2s_mic_simulator.py must be running)
  python3 test_accuracy.py --loopback

  # Use a specific device index (run --list-devices to find index)
  python3 test_accuracy.py --device 2

  # Save every chunk as a WAV file for manual review
  python3 test_accuracy.py --loopback --save-chunks

  # Capture only 20 chunks then stop
  python3 test_accuracy.py --loopback --chunks 20

  # List available audio devices
  python3 test_accuracy.py --list-devices
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.fft import fft, fftfreq

# Suppress transformer/torch verbose output for cleaner logs
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (mirror main.py / audio.py)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION    = 0.125          # seconds per chunk  (same as main.py)
CHANNELS    = 2
DTYPE       = "float32"

CHUNK_LOG_DIR = "chunk_logs"  # folder where WAV chunks are saved (--save-chunks)

# ANSI colours for terminal output
_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_CYAN   = "\033[96m"
_C_GREEN  = "\033[92m"
_C_YELLOW = "\033[93m"
_C_RED    = "\033[91m"
_C_DIM    = "\033[2m"


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def record_chunk(device=None):
    """
    Records one DURATION-length stereo chunk.
    Returns ndarray of shape (n_samples, 2).
    """
    n_frames = int(SAMPLE_RATE * DURATION)
    audio = sd.rec(
        n_frames,
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = DTYPE,
        device     = device,
    )
    sd.wait()
    return audio   # shape: (n_samples, 2)


def rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


def peak(signal):
    return float(np.max(np.abs(signal)))


def dominant_frequency(signal, fs=SAMPLE_RATE):
    """Returns the frequency (Hz) with the highest magnitude in the spectrum."""
    n  = len(signal)
    yf = np.abs(fft(signal))[:n // 2]
    xf = fftfreq(n, 1 / fs)[:n // 2]
    return float(xf[np.argmax(yf)])


def preprocess_to_mono(audio):
    """Average L+R channels to mono, matching audio.py:preprocess_audio."""
    return np.mean(audio, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Device discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_loopback_input():
    """Returns the sounddevice index of the ALSA Loopback capture side."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "loopback" in dev["name"].lower() and dev["max_input_channels"] >= CHANNELS:
            if ",1)" in dev["name"]:
                return i
    for i, dev in enumerate(devices):
        if "loopback" in dev["name"].lower() and dev["max_input_channels"] >= CHANNELS:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value, width=20, max_val=1.0):
    """Simple ASCII amplitude bar."""
    filled = int(round(min(value / max_val, 1.0) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_chunk_report(chunk_num, audio, phonemes, elapsed_ms, saved_path=None):
    """
    Prints a formatted diagnostic block for one audio chunk.

    Parameters
    ----------
    chunk_num  : int   – sequential chunk counter
    audio      : ndarray (n_samples, 2)  – raw stereo float32
    phonemes   : list[str] – recognised phoneme strings
    elapsed_ms : float – inference time in milliseconds
    saved_path : str | None – path if chunk was saved to WAV
    """
    n_samples = audio.shape[0]
    left  = audio[:, 0]
    right = audio[:, 1]

    rms_l  = rms(left);   rms_r  = rms(right)
    peak_l = peak(left);  peak_r = peak(right)
    dom_l  = dominant_frequency(left)
    dom_r  = dominant_frequency(right)

    ph_str  = "  ".join(phonemes) if phonemes else "(none)"
    ph_count = len(phonemes)

    width = 70
    bar = "─" * width

    print(f"\n{_C_CYAN}{_C_BOLD}┌─ Chunk #{chunk_num:>4d}  "
          f"({DURATION:.3f} s  |  {n_samples} samples @ {SAMPLE_RATE} Hz){_C_RESET}")
    print(f"{_C_CYAN}│{_C_RESET}  {_C_DIM}Recorded at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}{_C_RESET}")
    print(f"{_C_CYAN}│{_C_RESET}")

    # Amplitude stats
    print(f"{_C_CYAN}│{_C_RESET}  {_C_BOLD}Amplitude{_C_RESET}")
    print(f"{_C_CYAN}│{_C_RESET}    LEFT   RMS: {rms_l:7.4f}  {_bar(rms_l)}  Peak: {peak_l:.4f}")
    print(f"{_C_CYAN}│{_C_RESET}    RIGHT  RMS: {rms_r:7.4f}  {_bar(rms_r)}  Peak: {peak_r:.4f}")
    print(f"{_C_CYAN}│{_C_RESET}")

    # Frequency stats
    print(f"{_C_CYAN}│{_C_RESET}  {_C_BOLD}Dominant Frequency{_C_RESET}")
    print(f"{_C_CYAN}│{_C_RESET}    LEFT  : {dom_l:>8.1f} Hz")
    print(f"{_C_CYAN}│{_C_RESET}    RIGHT : {dom_r:>8.1f} Hz")
    print(f"{_C_CYAN}│{_C_RESET}")

    # Phoneme output
    phoneme_color = _C_GREEN if phonemes else _C_RED
    print(f"{_C_CYAN}│{_C_RESET}  {_C_BOLD}Phonemes{_C_RESET}  "
          f"({_C_YELLOW}{ph_count}{_C_RESET} recognised)  "
          f"[inference: {elapsed_ms:.1f} ms]")
    print(f"{_C_CYAN}│{_C_RESET}    {phoneme_color}[ {ph_str} ]{_C_RESET}")

    if saved_path:
        print(f"{_C_CYAN}│{_C_RESET}    {_C_DIM}Saved → {saved_path}{_C_RESET}")

    print(f"{_C_CYAN}└{'─' * width}{_C_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Main test loop
# ─────────────────────────────────────────────────────────────────────────────

def run_test(device, max_chunks, save_chunks, model):
    if save_chunks:
        os.makedirs(CHUNK_LOG_DIR, exist_ok=True)
        print(f"[test] Chunk WAV files will be saved to ./{CHUNK_LOG_DIR}/")

    print(f"[test] Recording chunks of {DURATION} s at {SAMPLE_RATE} Hz (stereo)")
    print(f"[test] Press Ctrl-C to stop.\n")

    chunk_num   = 0
    total_ph    = 0
    total_empty = 0
    session_start = time.time()

    try:
        while max_chunks is None or chunk_num < max_chunks:
            chunk_num += 1

            # ── Record ──────────────────────────────────────────────────────
            audio = record_chunk(device=device)

            # ── Inference ───────────────────────────────────────────────────
            mono = preprocess_to_mono(audio)
            t0   = time.perf_counter()
            phonemes = model.predict(mono)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── Stats ───────────────────────────────────────────────────────
            total_ph    += len(phonemes)
            total_empty += (1 if not phonemes else 0)

            # ── Optional WAV save ────────────────────────────────────────────
            saved_path = None
            if save_chunks:
                fname = os.path.join(
                    CHUNK_LOG_DIR,
                    f"chunk_{chunk_num:05d}_{'-'.join(phonemes) if phonemes else 'empty'}.wav"
                )
                sf.write(fname, audio, SAMPLE_RATE)
                saved_path = fname

            # ── Print report ─────────────────────────────────────────────────
            print_chunk_report(chunk_num, audio, phonemes, elapsed_ms, saved_path)

    except KeyboardInterrupt:
        pass

    # ── Session summary ──────────────────────────────────────────────────────
    elapsed_total = time.time() - session_start
    print(f"\n{'═'*70}")
    print(f"  SESSION SUMMARY")
    print(f"{'─'*70}")
    print(f"  Total chunks   : {chunk_num}")
    print(f"  Total phonemes : {total_ph}  "
          f"(avg {total_ph/max(chunk_num,1):.2f} per chunk)")
    print(f"  Empty chunks   : {total_empty}  "
          f"({100*total_empty/max(chunk_num,1):.1f}% silence)")
    print(f"  Elapsed time   : {elapsed_total:.1f} s")
    if save_chunks:
        print(f"  WAV chunks     : ./{CHUNK_LOG_DIR}/")
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="test_accuracy.py",
        description="Chunk-level diagnostic / accuracy tester for BinaryBrains phoneme recogniser",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--loopback", action="store_true",
        help="Auto-detect and use the ALSA Loopback capture device "
             "(requires i2s_mic_simulator.py to be running)",
    )
    grp.add_argument(
        "--device", type=int, default=None,
        help="Sounddevice input device index (use --list-devices to find it)",
    )
    p.add_argument(
        "--chunks", type=int, default=None, metavar="N",
        help="Stop after N chunks (default: run forever until Ctrl-C)",
    )
    p.add_argument(
        "--save-chunks", action="store_true",
        help=f"Save every audio chunk as a WAV file in ./{CHUNK_LOG_DIR}/",
    )
    p.add_argument(
        "--duration", type=float, default=DURATION,
        help=f"Chunk duration in seconds (default: {DURATION})",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="Print all sounddevice devices and exit",
    )
    return p.parse_args()


def main():
    global DURATION
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    DURATION = args.duration

    # ── Resolve input device ─────────────────────────────────────────────────
    device = None
    if args.loopback:
        device = find_loopback_input()
        if device is None:
            print(
                "ERROR: No ALSA Loopback INPUT device found.\n"
                "  sudo modprobe snd-aloop\n"
                "  python3 i2s_mic_simulator.py --source sine   (in another terminal)",
                file=sys.stderr,
            )
            sys.exit(1)
        dev_info = sd.query_devices(device)
        print(f"[test] Loopback capture device [{device}]: {dev_info['name']}")
    elif args.device is not None:
        device = args.device
        dev_info = sd.query_devices(device)
        print(f"[test] Using device [{device}]: {dev_info['name']}")
    else:
        # Use sounddevice system default
        print("[test] Using system default input device.")

    # ── Load model ───────────────────────────────────────────────────────────
    print("[test] Loading recognition model (this may take a moment) ...")
    from model import RecognitionModel
    model = RecognitionModel()
    print("[test] Model ready.\n")

    # ── Run ──────────────────────────────────────────────────────────────────
    run_test(
        device      = device,
        max_chunks  = args.chunks,
        save_chunks = args.save_chunks,
        model       = model,
    )


if __name__ == "__main__":
    main()

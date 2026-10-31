#!/usr/bin/env python3
"""
i2s_mic_simulator.py
====================
Simulates an I2S stereo microphone on Raspberry Pi 5 (Linux) by writing
audio data into an ALSA loopback device.  The main program reads from the
OTHER side of that loopback via sounddevice - it sees a normal stereo
16 kHz device, exactly as if a physical I2S mic (e.g. INMP441 / SPH0645)
were connected.

How it works
------------
1.  The Linux kernel module ``snd-aloop`` creates a pair of virtual ALSA
    PCM devices (hw:Loopback,0 and hw:Loopback,1).  Whatever is written
    to hw:Loopback,0 immediately appears on hw:Loopback,1 (and vice-versa).
2.  This script writes audio to hw:Loopback,0  (the "mic" side).
3.  main.py / audio.py opens hw:Loopback,1  (the "capture" side) via
    sounddevice.

Audio source options (choose with --source flag)
-------------------------------------------------
  sine   - a pair of sine tones (different frequency per channel),
           useful for verifying FFT analysis in signal_processing.py
  noise  - white Gaussian noise (both channels), mimics ambient mic noise
  wav    - read a WAV file and loop it (pass path with --wav)

Setup (run once on the Raspberry Pi)
--------------------------------------
  sudo modprobe snd-aloop     # load the loopback module
  # To persist across reboots:
  echo snd-aloop | sudo tee -a /etc/modules

  # Install Python deps (already in requirement.txt except soundfile):
  pip install sounddevice soundfile numpy

Usage
------
  # Terminal 1 - start the simulator FIRST
  python3 i2s_mic_simulator.py --source sine
  python3 i2s_mic_simulator.py --source noise
  python3 i2s_mic_simulator.py --source wav --wav my_speech.wav

  # Terminal 2 - run the main program with the loopback as input device
  python3 main_loopback.py
  # OR: set SOUNDDEVICE_DEVICE env var before running main.py
"""

import argparse
import sys
import time
import signal
import threading

import numpy as np
import sounddevice as sd

# ------------------------------------------------------------------------------
# Constants - must match main.py / audio.py exactly
# ------------------------------------------------------------------------------
SAMPLE_RATE = 16_000        # Hz - matches audio.py
CHANNELS    = 2             # stereo (Left + Right), matches sd.rec(channels=2)
DTYPE       = "float32"     # matches sd.rec(dtype="float32")
BLOCKSIZE   = 512           # frames per callback; smaller = lower latency


# ------------------------------------------------------------------------------
# I2S-style signal generation helpers
# ------------------------------------------------------------------------------

def _make_sine_generator(freq_left=440.0, freq_right=880.0, amplitude=0.5):
    """
    Returns a stateful function that yields stereo float32 frames.
    Left channel : freq_left  Hz sine - simulates one I2S capsule.
    Right channel: freq_right Hz sine - simulates the other I2S capsule.
    The different frequencies let signal_processing.py confirm both
    FFT channels are working independently.
    """
    phase_left  = 0.0
    phase_right = 0.0
    step_left   = 2.0 * np.pi * freq_left  / SAMPLE_RATE
    step_right  = 2.0 * np.pi * freq_right / SAMPLE_RATE

    def generate(frames):
        nonlocal phase_left, phase_right
        t = np.arange(frames, dtype=np.float64)
        left  = amplitude * np.sin(phase_left  + t * step_left)
        right = amplitude * np.sin(phase_right + t * step_right)
        phase_left  = (phase_left  + frames * step_left)  % (2.0 * np.pi)
        phase_right = (phase_right + frames * step_right) % (2.0 * np.pi)
        return np.stack([left, right], axis=1).astype(np.float32)

    return generate


def _make_noise_generator(amplitude=0.05):
    """
    Returns a stateful function yielding stereo white Gaussian noise,
    independently on each channel (Left != Right), mimicking two I2S
    capsules in different spatial positions picking up slightly different
    ambient sound fields.
    """
    rng = np.random.default_rng()

    def generate(frames):
        data = rng.standard_normal((frames, CHANNELS)).astype(np.float32)
        return data * amplitude

    return generate


def _make_wav_generator(wav_path):
    """
    Loads a WAV file and returns a generator that loops through it
    indefinitely in stereo float32 chunks at SAMPLE_RATE.
    - Resamples if the file sample rate != SAMPLE_RATE (requires soxr).
    - Mono files are duplicated to both channels.
    """
    import soundfile as sf

    data, file_sr = sf.read(wav_path, dtype="float32", always_2d=True)

    # Resample if necessary
    if file_sr != SAMPLE_RATE:
        try:
            import soxr
            data = soxr.resample(data, file_sr, SAMPLE_RATE, quality="HQ")
            print(f"[i2s_sim] Resampled WAV from {file_sr} Hz -> {SAMPLE_RATE} Hz")
        except ImportError:
            print(
                f"[i2s_sim] WARNING: soxr not installed; WAV is {file_sr} Hz but "
                f"simulator expects {SAMPLE_RATE} Hz. Install soxr or supply a "
                "16 kHz file.",
                file=sys.stderr,
            )

    # Ensure stereo
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
        print("[i2s_sim] Mono WAV duplicated to both I2S channels.")
    elif data.shape[1] > 2:
        data = data[:, :2]
        print("[i2s_sim] WAV trimmed to first two channels.")

    total_frames = data.shape[0]
    pos = [0]

    def generate(frames):
        out = np.empty((frames, CHANNELS), dtype=np.float32)
        written = 0
        while written < frames:
            available = total_frames - pos[0]
            need = frames - written
            take = min(available, need)
            out[written : written + take] = data[pos[0] : pos[0] + take]
            written += take
            pos[0]  += take
            if pos[0] >= total_frames:
                pos[0] = 0  # loop back to start
        return out

    return generate


# ------------------------------------------------------------------------------
# ALSA Loopback device discovery
# ------------------------------------------------------------------------------

def _find_loopback_output_device():
    """
    Scans sounddevice device list for the ALSA snd-aloop PLAYBACK side
    (hw:Loopback,0 - the side we WRITE to).
    Returns device index, or raises RuntimeError if not found.
    """
    devices = sd.query_devices()
    candidates = []
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if "loopback" in name and dev["max_output_channels"] >= CHANNELS:
            candidates.append((i, dev["name"]))

    if not candidates:
        raise RuntimeError(
            "No ALSA Loopback OUTPUT device found.\n"
            "  Load the kernel module:  sudo modprobe snd-aloop\n"
            "  Then list devices:       python3 i2s_mic_simulator.py --list-devices"
        )

    # Prefer subdevice 0 (playback/write side)
    for idx, name in candidates:
        if ",0)" in name:
            print(f"[i2s_sim] Auto-selected output device [{idx}]: {name}")
            return idx

    idx, name = candidates[0]
    print(f"[i2s_sim] Auto-selected output device [{idx}]: {name}")
    return idx


def _find_loopback_input_device():
    """Returns capture side (hw:Loopback,1) index for user info only."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "loopback" in dev["name"].lower() and dev["max_input_channels"] >= CHANNELS:
            if ",1)" in dev["name"]:
                return i
    # fallback: first loopback input
    for i, dev in enumerate(devices):
        if "loopback" in dev["name"].lower() and dev["max_input_channels"] >= CHANNELS:
            return i
    return -1


# ------------------------------------------------------------------------------
# Simulator engine
# ------------------------------------------------------------------------------

class I2SMicSimulator:
    """
    Streams synthesized / recorded audio into the ALSA Loopback write side
    so that any program reading from the Loopback read side via sounddevice
    perceives it as a real I2S stereo microphone.

    The sounddevice OutputStream callback fires at BLOCKSIZE-frame intervals,
    mimicking the DMA-driven bursts of a hardware I2S peripheral.
    """

    def __init__(self, generate_fn, device_index):
        self._generate  = generate_fn
        self._device    = device_index
        self._stop_evt  = threading.Event()
        self._stream    = None

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"[i2s_sim] Stream status: {status}", file=sys.stderr)
        outdata[:] = self._generate(frames)

    def start(self):
        print("[i2s_sim] Starting I2S mic simulator ...")
        print(f"          Sample rate : {SAMPLE_RATE} Hz")
        print(f"          Channels    : {CHANNELS}  (L + R stereo)")
        print(f"          Block size  : {BLOCKSIZE} frames")
        print(f"          Output dev  : {self._device}")
        print()
        self._stream = sd.OutputStream(
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            dtype      = DTYPE,
            device     = self._device,
            blocksize  = BLOCKSIZE,
            callback   = self._callback,
        )
        self._stream.start()
        print("[i2s_sim] Streaming ... press Ctrl-C to stop.")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        print("\n[i2s_sim] Stopped.")

    def wait(self):
        """Block until SIGINT/SIGTERM sets the stop event."""
        try:
            while not self._stop_evt.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        prog="i2s_mic_simulator.py",
        description="I2S stereo microphone simulator via ALSA snd-aloop (Raspberry Pi 5)",
    )
    p.add_argument(
        "--source", choices=["sine", "noise", "wav"], default="sine",
        help="Audio source: sine (default), noise, or wav",
    )
    p.add_argument(
        "--wav", metavar="FILE", default=None,
        help="WAV file path (required with --source=wav)",
    )
    p.add_argument(
        "--freq-left", type=float, default=440.0,
        help="Left-channel sine frequency Hz (default: 440)",
    )
    p.add_argument(
        "--freq-right", type=float, default=880.0,
        help="Right-channel sine frequency Hz (default: 880)",
    )
    p.add_argument(
        "--amplitude", type=float, default=0.4,
        help="Signal amplitude 0.0-1.0 (default: 0.4)",
    )
    p.add_argument(
        "--device", type=int, default=None,
        help="Force sounddevice output device index (auto-detect by default)",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List all sounddevice devices and exit",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    # Choose audio source
    if args.source == "sine":
        generate_fn = _make_sine_generator(
            freq_left  = args.freq_left,
            freq_right = args.freq_right,
            amplitude  = args.amplitude,
        )
        print(f"[i2s_sim] Source: SINE  L={args.freq_left} Hz  R={args.freq_right} Hz  "
              f"amp={args.amplitude}")

    elif args.source == "noise":
        generate_fn = _make_noise_generator(amplitude=args.amplitude)
        print(f"[i2s_sim] Source: WHITE NOISE  amp={args.amplitude}")

    elif args.source == "wav":
        if not args.wav:
            print("ERROR: --wav FILE is required when --source=wav", file=sys.stderr)
            sys.exit(1)
        generate_fn = _make_wav_generator(args.wav)
        print(f"[i2s_sim] Source: WAV  file={args.wav}")

    # Find ALSA loopback output device
    try:
        out_idx = args.device if args.device is not None else _find_loopback_output_device()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Print capture-side info for user reference
    cap_idx = _find_loopback_input_device()
    if cap_idx >= 0:
        cap_dev = sd.query_devices(cap_idx)
        print(f"[i2s_sim] Capture device (what main.py should use): "
              f"[{cap_idx}] {cap_dev['name']}")
        print(f"[i2s_sim] To configure main.py, see main_loopback.py")
        print()

    # Install signal handlers and run
    sim = I2SMicSimulator(generate_fn, out_idx)

    def _shutdown(signum, frame):
        sim._stop_evt.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    sim.start()
    sim.wait()


if __name__ == "__main__":
    main()

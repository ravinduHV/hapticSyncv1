#!/usr/bin/env python3
"""
main_loopback.py
================
Drop-in replacement for main.py that redirects sounddevice to the ALSA
Loopback capture device instead of a physical microphone.

Run i2s_mic_simulator.py first (in a separate terminal), then run this.

Usage on Raspberry Pi 5:
  # Terminal 1
  python3 i2s_mic_simulator.py --source sine

  # Terminal 2
  python3 main_loopback.py
"""

import sys
import sounddevice as sd


def _find_loopback_input_index():
    """
    Returns the sounddevice index of the ALSA Loopback CAPTURE side
    (hw:Loopback,1 - the side the main program should READ from).
    """
    devices = sd.query_devices()
    # Prefer the explicit subdevice-1 entry
    for i, dev in enumerate(devices):
        if "loopback" in dev["name"].lower() and dev["max_input_channels"] >= 2:
            if ",1)" in dev["name"]:
                return i
    # Fallback: first loopback input device
    for i, dev in enumerate(devices):
        if "loopback" in dev["name"].lower() and dev["max_input_channels"] >= 2:
            return i
    return None


cap_idx = _find_loopback_input_index()
if cap_idx is None:
    print(
        "[main_loopback] ERROR: No ALSA Loopback INPUT device found.\n"
        "  Make sure snd-aloop is loaded:  sudo modprobe snd-aloop\n"
        "  And that i2s_mic_simulator.py is running in another terminal.",
        file=sys.stderr,
    )
    sys.exit(1)

cap_dev = sd.query_devices(cap_idx)
print(f"[main_loopback] Using loopback capture device [{cap_idx}]: {cap_dev['name']}")
print("[main_loopback] Redirecting sounddevice default input to loopback ...\n")

# Set BOTH input and output defaults so sd.rec() picks up the loopback
sd.default.device   = (cap_idx, cap_idx)   # (input, output)
sd.default.samplerate = 16_000
sd.default.channels   = 2
sd.default.dtype      = "float32"

# Now import and run main.py logic unchanged
# (All imports happen AFTER sd.default is set, so sd.rec() in audio.py
#  will automatically use the loopback device.)
from model import RecognitionModel
from audio import preprocess_audio, record_audio
from signal_processing import get_frequencies

SAMPLE_RATE = 16000
DURATION    = 0.125

model = RecognitionModel()

while True:
    audio = record_audio()

    freqs_left,  mags_left  = get_frequencies(audio[:, 0], fs=16000)
    freqs_right, mags_right = get_frequencies(audio[:, 1], fs=16000)

    phonemes = model.predict(preprocess_audio(audio))
    print("Sound Frequencies (Left Channel):",  freqs_left)
    print("Sound Magnitudes  (Left Channel):",  mags_left)
    print("Sound Frequencies (Right Channel):", freqs_right)
    print("Sound Magnitudes  (Right Channel):", mags_right)
    print("Recognized phonemes:", phonemes)

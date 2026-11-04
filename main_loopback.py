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

# Now import and run main.py logic with per-step timing
# (All imports happen AFTER sd.default is set, so sd.rec() in audio.py
#  will automatically use the loopback device.)
import time
import numpy as np
import torch
from model import RecognitionModel
from audio import preprocess_audio, record_audio
from signal_processing import get_frequencies

SAMPLE_RATE = 16000
DURATION    = 0.125

print("[main_loopback] Loading recognition model ...")
model_wrapper = RecognitionModel()
model = model_wrapper.model
processor = model_wrapper.processor
id2phoneme = model_wrapper.id2phoneme
device = next(model.parameters()).device
print("[main_loopback] Model ready. Starting measurement loop ...\n")

def _sync():
    """Ensure GPU synchronization if CUDA is being used."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Rolling timing history for moving averages (last 20 iterations)
HISTORY_LEN = 20
timings = {
    "record": [],
    "preprocess": [],
    "fft": [],
    "processor": [],
    "forward": [],
    "ctc_decode": [],
    "total": [],
    "pure_compute": []
}

iteration = 0
while True:
    iteration += 1
    t_start = time.perf_counter()

    # ── Step 1: Audio Recording ──────────────────────────────────────────────
    t0 = time.perf_counter()
    audio = record_audio(sample_rate=SAMPLE_RATE, duration=DURATION)
    t1 = time.perf_counter()
    time_record = (t1 - t0) * 1000

    # ── Step 2: Preprocess Audio (Stereo -> Mono) ────────────────────────────
    t0 = time.perf_counter()
    mono_audio = preprocess_audio(audio)
    t1 = time.perf_counter()
    time_preprocess = (t1 - t0) * 1000

    # ── Step 3: Spectral Analysis (FFT on Left & Right) ──────────────────────
    t0 = time.perf_counter()
    freqs_left,  mags_left  = get_frequencies(audio[:, 0], fs=SAMPLE_RATE)
    freqs_right, mags_right = get_frequencies(audio[:, 1], fs=SAMPLE_RATE)
    t1 = time.perf_counter()
    time_fft = (t1 - t0) * 1000

    # ── Step 4: Wav2Vec2 Feature Extraction (Processor) ──────────────────────
    t0 = time.perf_counter()
    inputs = processor(mono_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    _sync()
    t1 = time.perf_counter()
    time_processor = (t1 - t0) * 1000

    # ── Step 5: Model Forward Pass (Inference) ───────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_values).logits
    _sync()
    t1 = time.perf_counter()
    time_forward = (t1 - t0) * 1000

    # ── Step 6: CTC Decoding / Phoneme Mapping ───────────────────────────────
    t0 = time.perf_counter()
    pred_ids = torch.argmax(logits, dim=-1)[0]
    phonemes = [id2phoneme.get(int(i), "[UNK]") for i in pred_ids if int(i) not in [0, 41]]
    t1 = time.perf_counter()
    time_decode = (t1 - t0) * 1000

    t_end = time.perf_counter()
    time_total = (t_end - t_start) * 1000
    time_compute = time_total - time_record

    # Update rolling history
    timings["record"].append(time_record)
    timings["preprocess"].append(time_preprocess)
    timings["fft"].append(time_fft)
    timings["processor"].append(time_processor)
    timings["forward"].append(time_forward)
    timings["ctc_decode"].append(time_decode)
    timings["total"].append(time_total)
    timings["pure_compute"].append(time_compute)

    for k in timings:
        if len(timings[k]) > HISTORY_LEN:
            timings[k].pop(0)

    # ── Formatted Output ─────────────────────────────────────────────────────
    print(f"┌─ Iteration #{iteration:04d} (Chunk: {DURATION}s) ──────────────────────────────────")
    print(f"│ 1. Audio Record (wait)  : {time_record:7.2f} ms  (Avg: {np.mean(timings['record']):6.2f} ms)")
    print(f"│ 2. Stereo -> Mono       : {time_preprocess:7.2f} ms  (Avg: {np.mean(timings['preprocess']):6.2f} ms)")
    print(f"│ 3. Dual FFT Analysis    : {time_fft:7.2f} ms  (Avg: {np.mean(timings['fft']):6.2f} ms)")
    print(f"│ 4. Feature Processor    : {time_processor:7.2f} ms  (Avg: {np.mean(timings['processor']):6.2f} ms)")
    print(f"│ 5. Model Forward Pass   : {time_forward:7.2f} ms  (Avg: {np.mean(timings['forward']):6.2f} ms)")
    print(f"│ 6. CTC Greedy Decode    : {time_decode:7.2f} ms  (Avg: {np.mean(timings['ctc_decode']):6.2f} ms)")
    print(f"├───────────────────────────────────────────────────────────────────")
    print(f"│ Compute Latency (2 to 6): {time_compute:7.2f} ms  (Avg: {np.mean(timings['pure_compute']):6.2f} ms)")
    print(f"│ Total Latency (1 to 6)  : {time_total:7.2f} ms  (Avg: {np.mean(timings['total']):6.2f} ms)")
    print(f"│ Phonemes                : {phonemes if phonemes else '[]'}")
    print(f"└───────────────────────────────────────────────────────────────────\n")

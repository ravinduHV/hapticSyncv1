#!/usr/bin/env python3
"""
ref_decode.py — Full-context reference offline decoder (Phase A Task A4)
Takes a WAV file, runs the full-audio Wav2Vec2-CTC inference, applies CTC greedy collapse,
and outputs a reference transcript JSON containing phonemes with timestamps.
"""

import sys
import json
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 ref_decode.py <audio.wav> <output_ref.json> [checkpoint]")
        sys.exit(1)

    wav_path = sys.argv[1]
    out_path = sys.argv[2]
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else "./model/"

    print(f"[ref_decode] Loading model from {checkpoint}...")
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint)
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model.eval()

    print(f"[ref_decode] Reading audio from {wav_path}...")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Process audio
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    probs = logits.softmax(dim=-1)[0]
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    confs = probs.max(dim=-1).values.cpu().numpy()

    # Load vocab map
    vocab_map = {}
    if hasattr(processor, "tokenizer"):
        vocab_map = {v: k for k, v in processor.tokenizer.get_vocab().items()}

    # Standard CTC greedy collapse
    blank_id = getattr(model.config, "pad_token_id", 41)
    events = []
    prev_label = blank_id

    for i, (label, conf) in enumerate(zip(pred_ids, confs)):
        # Wav2Vec2 geometry: center of frame i is sample offset 320*i + 200
        t_center_s = (320 * i + 200) / 16000.0

        if label != blank_id and label != prev_label:
            ph_name = vocab_map.get(int(label), f"ph_{label}")
            events.append({
                "frame_idx": int(i),
                "t_center_s": round(t_center_s, 4),
                "label": int(label),
                "phoneme": ph_name,
                "confidence": round(float(conf), 4),
            })
        prev_label = label

    output_data = {
        "audio_file": wav_path,
        "sample_rate": sr,
        "duration_s": round(len(audio) / sr, 3),
        "total_frames": len(pred_ids),
        "total_phonemes": len(events),
        "events": events,
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[ref_decode] Reference transcript written to {out_path} ({len(events)} phonemes)")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
inference/main.py — Real-time model inference worker (Section 5.4)
Pinned to CPU 2, single-threaded PyTorch, zero-copy shm input.
"""

# STRICT RT REQUIREMENT: Threading env vars MUST be set before importing numpy/torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import time
import socket
import numpy as np

# Load config to get CPU pinning configuration
config_path = sys.argv[1] if len(sys.argv) > 1 else "config/hapticsync.json"
with open(config_path, "r") as f:
    cfg = json.load(f)

# Set CPU affinity before importing heavy libraries
model_cpus = cfg.get("cpus", {}).get("model", [2])
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, set(model_cpus))
        print(f"[inference] Pinned to CPU(s): {model_cpus}")
    except Exception as e:
        print(f"[inference] Warning: Could not set CPU affinity: {e}", file=sys.stderr)

import torch
from transformers import Wav2Vec2ForCTC

torch.set_num_threads(len(model_cpus))
torch.set_num_interop_threads(1)

from protocol import (
    unpack_window_msg,
    pack_done_msg,
    pack_result,
    WINDOW_MSG_SIZE,
)
from shm_views import WindowSlotsView


def run_startup_assertions(model, checkpoint_path: str, expected_blank_id: int):
    """Verify geometry formulas and blank token ID (Section 5.4)."""
    print("[inference] Running startup assertions...")

    test_lengths = [800, 1600, 3200, 16000]
    for n_samples in test_lengths:
        expected_frames = (n_samples - 400) // 320 + 1
        dummy_in = torch.zeros(1, n_samples, dtype=torch.float32)
        with torch.inference_mode():
            logits = model(dummy_in).logits
        actual_frames = logits.shape[1]
        assert actual_frames == expected_frames, (
            f"Frame mismatch for {n_samples} samples: got {actual_frames}, expected {expected_frames}"
        )

    # Check pad/blank token ID
    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is not None and expected_blank_id is not None:
        if pad_token_id != expected_blank_id:
            print(f"[inference] Warning: model pad_token_id={pad_token_id} != config blank_id={expected_blank_id}")

    print("[inference] All startup assertions passed.")


def export_vocab(vocab_file: str):
    """Export id -> phoneme mapping from existing model wrapper definition."""
    # From model.py:
    id2phoneme = {
        1: "aa", 2: "ae", 3: "ah", 4: "aw", 5: "ay", 6: "b", 7: "ch", 8: "d", 9: "dh",
        10: "dx", 11: "eh", 12: "er", 13: "ey", 14: "f", 15: "g", 16: "h#", 17: "hh",
        18: "ih", 19: "iy", 20: "jh", 21: "k", 22: "l", 23: "m", 24: "n", 25: "ng",
        26: "ow", 27: "oy", 28: "p", 29: "r", 30: "s", 31: "sh", 32: "t", 33: "th",
        34: "uh", 35: "uw", 36: "v", 37: "w", 38: "y", 39: "z", 0: "|", 40: "[UNK]", 41: "[PAD]"
    }
    with open(vocab_file, "w") as f:
        json.dump({str(k): v for k, v in id2phoneme.items()}, f, indent=2)
    print(f"[inference] Exported vocabulary to {vocab_file}")


def warmup(model):
    """Warm up torch execution and memory allocator (Section 5.4)."""
    print("[inference] Warming up model...")
    warmup_lengths = [3200, 8000, 16000, 32000]
    for _ in range(5):
        for n in warmup_lengths:
            dummy = torch.randn(1, n, dtype=torch.float32)
            with torch.inference_mode():
                _ = model(dummy).logits
    print("[inference] Warmup complete.")


def resolve_model_path(path: str) -> str:
    """Find local model directory or return original path."""
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join("..", path.lstrip("./")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")),
    ]
    for c in candidates:
        if os.path.isdir(c) and (os.path.exists(os.path.join(c, "config.json")) or os.path.exists(os.path.join(c, "pytorch_model.bin"))):
            return c
    return path


def main():
    checkpoint = cfg.get("model", {}).get("checkpoint", "./model/")
    resolved_checkpoint = resolve_model_path(checkpoint)
    blank_id = cfg.get("model", {}).get("blank_id", 41)
    vocab_file = cfg.get("model", {}).get("vocab_json", "vocab.json")

    export_vocab(vocab_file)

    if not os.path.isdir(resolved_checkpoint):
        print(f"\n[inference] ERROR: Model directory not found at '{checkpoint}' (or '../model')!", file=sys.stderr)
        print("Per README Step 01, ensure your 'model' directory is present.", file=sys.stderr)
        sys.exit(1)

    print(f"[inference] Loading Wav2Vec2 model from {resolved_checkpoint}...")
    model = Wav2Vec2ForCTC.from_pretrained(resolved_checkpoint)
    model.eval()

    run_startup_assertions(model, checkpoint, blank_id)
    warmup(model)

    # Open window slots shared memory
    shm_views = WindowSlotsView()

    # Get socket file descriptors
    assem_sock = None
    post_sock = None

    assem_fd = os.getenv("HAPTICSYNC_INFER_FD")
    if assem_fd:
        assem_sock = socket.fromfd(int(assem_fd), socket.AF_UNIX, socket.SOCK_SEQPACKET)
        print(f"[inference] Connected to assembler socket fd={assem_fd}")
    else:
        print("[inference] Warning: HAPTICSYNC_INFER_FD not set")

    post_fd = os.getenv("HAPTICSYNC_POST_FD")
    if post_fd:
        post_sock = socket.fromfd(int(post_fd), socket.AF_UNIX, socket.SOCK_SEQPACKET)
        print(f"[inference] Connected to postproc socket fd={post_fd}")
    else:
        print("[inference] Warning: HAPTICSYNC_POST_FD not set")

    print("[inference] Ready for inference loop.")

    # Inner inference loop
    while True:
        try:
            if assem_sock is None:
                time.sleep(0.1)
                continue

            data = assem_sock.recv(WINDOW_MSG_SIZE)
            if not data or len(data) < WINDOW_MSG_SIZE:
                continue

            msg = unpack_window_msg(data)

            # Map zero-copy numpy slice
            samples = shm_views.get_slot_view(msg["slot"], msg["n_samples"])

            # Forward pass
            t_begin_ns = time.monotonic_ns()
            input_tensor = torch.from_numpy(samples).unsqueeze(0)

            with torch.inference_mode():
                logits = model(input_tensor).logits
                probs = logits.softmax(dim=-1)
                confs, labels = probs.max(dim=-1)

            t_end_ns = time.monotonic_ns()

            confs = confs[0].cpu().numpy()
            labels = labels[0].cpu().numpy()
            n_frames = len(labels)

            frames_data = list(zip(labels.astype(int), confs.astype(float)))

            # Send RESULT to postproc
            if post_sock is not None:
                result_bytes = pack_result(
                    window_id=msg["window_id"],
                    start_sample=msg["start_sample"],
                    t_start_ns=msg["t_start_ns"],
                    t_infer_begin_ns=t_begin_ns,
                    t_infer_end_ns=t_end_ns,
                    n_frames=n_frames,
                    commit_begin=msg["commit_begin"],
                    commit_end=msg["commit_end"],
                    flags=msg["flags"],
                    frames_array=frames_data,
                )
                post_sock.send(result_bytes)

            # Send DONE to assembler
            done_bytes = pack_done_msg(
                window_id=msg["window_id"],
                t_infer_begin_ns=t_begin_ns,
                t_infer_end_ns=t_end_ns,
                status=0,
            )
            assem_sock.send(done_bytes)

        except KeyboardInterrupt:
            break
        except Exception as e:
            # Report error and notify assembler
            print(f"[inference] Error during inference: {e}", file=sys.stderr)
            if assem_sock is not None and 'msg' in locals():
                t_now = time.monotonic_ns()
                done_bytes = pack_done_msg(msg["window_id"], t_now, t_now, status=1)
                assem_sock.send(done_bytes)

    print("[inference] Terminated.")


if __name__ == "__main__":
    main()


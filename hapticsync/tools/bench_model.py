#!/usr/bin/env python3
"""
bench_model.py — Benchmark Wav2Vec2 inference cost curve (Phase A Tasks A1–A5)
Measures t(L) = a + b*L under pinned single-core configuration.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import time
import subprocess
import numpy as np

# Pin to CPU 2
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {2})
        print("[bench] Pinned to CPU 2")
    except Exception as e:
        print(f"[bench] Warning: Could not set CPU affinity: {e}")

import torch
from transformers import Wav2Vec2ForCTC

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_system_telemetry():
    t_str = "N/A"
    th_str = "N/A"
    try:
        t_str = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        th_str = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        pass
    return t_str, th_str


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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/hapticsync.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    checkpoint = cfg.get("model", {}).get("checkpoint", "./model/")
    resolved_checkpoint = resolve_model_path(checkpoint)

    if not os.path.isdir(resolved_checkpoint):
        print(f"\n[bench] ERROR: Model directory not found at '{checkpoint}' (or '../model')!", file=sys.stderr)
        print("Per README Step 01, ensure your 'model' directory is present.", file=sys.stderr)
        print("If it is located in the parent directory, run:\n  ln -s ../model ./model\n", file=sys.stderr)
        sys.exit(1)

    print(f"[bench] Loading model from {resolved_checkpoint}...")
    model = Wav2Vec2ForCTC.from_pretrained(resolved_checkpoint)
    model.eval()

    sample_rate = 16000
    test_durations_ms = [125, 250, 500, 1000, 2000, 4000]
    iterations = 35  # >= 30 iterations each after warmup
    warmup_iters = 5

    results = {}

    print(f"[bench] Starting measurements: {len(test_durations_ms)} lengths x {iterations} iterations...")
    print(f"[bench] Initial state: {get_system_telemetry()}")

    durations_s = []
    mean_times_s = []

    for dur_ms in test_durations_ms:
        n_samples = int((dur_ms * sample_rate) / 1000)
        dur_s = dur_ms / 1000.0

        # Warmup
        for _ in range(warmup_iters):
            dummy = torch.randn(1, n_samples, dtype=torch.float32)
            with torch.inference_mode():
                _ = model(dummy).logits

        timings = []
        for _ in range(iterations):
            dummy = torch.randn(1, n_samples, dtype=torch.float32)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(dummy).logits
            t1 = time.perf_counter()
            timings.append(t1 - t0)

        mean_t = float(np.mean(timings))
        std_t = float(np.std(timings))
        p95_t = float(np.percentile(timings, 95))
        rtf = mean_t / dur_s

        results[str(dur_ms)] = {
            "duration_s": dur_s,
            "samples": n_samples,
            "mean_s": round(mean_t, 5),
            "std_s": round(std_t, 5),
            "p95_s": round(p95_t, 5),
            "rtf": round(rtf, 3),
        }

        durations_s.append(dur_s)
        mean_times_s.append(mean_t)

        print(f"  {dur_ms:4d} ms ({n_samples} samples): mean={mean_t*1000:6.1f} ms | p95={p95_t*1000:6.1f} ms | RTF={rtf:.2f}")

    # Fit linear curve: t(L) = a + b * L
    # mean_times_s = b * durations_s + a
    poly = np.polyfit(durations_s, mean_times_s, 1)
    b_val = float(poly[0])
    a_val = float(poly[1])

    print("\n[bench] ── Model Cost Curve Fit ──────────────────────────────")
    print(f"[bench] t(L) = a + b * L")
    print(f"[bench] a (fixed overhead)      : {a_val*1000:.2f} ms ({a_val:.5f} s)")
    print(f"[bench] b (cost per sec audio)  : {b_val:.4f}")

    # Policy analysis (Section 7.3)
    trim_l = cfg.get("window", {}).get("trim_left_frames", 3)
    trim_r = cfg.get("window", {}).get("trim_right_frames", 3)
    max_w_ms = cfg.get("window", {}).get("max_window_ms", 4000)

    overlap_s = (trim_l + trim_r) * 0.020 + 0.005
    policy_choice = "latest"
    if b_val < 1.0:
        t_star = (a_val + b_val * overlap_s) / (1.0 - b_val)
        l_star = t_star + overlap_s
        if l_star <= (max_w_ms / 1000.0):
            policy_choice = "catchup"
            print(f"[bench] Policy Recommendation   : CATCHUP (L*={l_star:.3f} s, T*={t_star:.3f} s)")
        else:
            print(f"[bench] Policy Recommendation   : LATEST (L*={l_star:.3f} s exceeds max_window {max_w_ms} ms)")
    else:
        print(f"[bench] Policy Recommendation   : LATEST (b >= 1.0; real-time catchup not sustainable on 1 core)")

    # Save to config/bench.json
    bench_data = {
        "a": round(a_val, 6),
        "b": round(b_val, 6),
        "rtf_curve": results,
        "policy_recommendation": policy_choice,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "cpus_model": [2],
            "torch_threads": 1,
            "inference_mode": True
        }
    }

    out_file = "config/bench.json"
    with open(out_file, "w") as f:
        json.dump(bench_data, f, indent=2)

    print(f"[bench] Results saved to {out_file}")
    print(f"[bench] Final state: {get_system_telemetry()}")


if __name__ == "__main__":
    main()


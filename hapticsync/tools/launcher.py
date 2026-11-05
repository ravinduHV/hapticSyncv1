#!/usr/bin/env python3
"""
launcher.py — Supervisor and orchestration service (Section 5.8)
Sets up socketpairs, CPU tasksets, process lifecycle, and clean shutdown.
"""

import os
import sys
import time
import socket
import signal
import subprocess
import json

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config/hapticsync.json"
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

procs = {}
sockpairs = []


def cleanup():
    """Terminate child processes in reverse dependency order."""
    print("\n[launcher] Shutting down pipeline processes...")

    order = ["producer", "inference", "postproc", "telemetry"]
    for name in order:
        p = procs.get(name)
        if p and p.poll() is None:
            print(f"[launcher] Terminating {name} (PID {p.pid})...")
            p.send_signal(signal.SIGTERM)

    time.sleep(0.5)

    for name in order:
        p = procs.get(name)
        if p and p.poll() is None:
            print(f"[launcher] Force killing {name} (PID {p.pid})...")
            p.kill()

    # Close inherited socketpairs
    for s1, s2 in sockpairs:
        try:
            s1.close()
            s2.close()
        except Exception:
            pass

    # Unlink shared memory
    for shm in ["/dev/shm/hapticsync_audio", "/dev/shm/hapticsync_stats",
                "/dev/shm/hapticsync_window", "/dev/shm/hapticsync_events"]:
        if os.path.exists(shm):
            try:
                os.remove(shm)
            except Exception:
                pass

    print("[launcher] Shutdown complete.")


def sig_handler(signum, frame):
    cleanup()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print("==========================================================")
    print("  Haptic-Sync Real-Time Phoneme Streamer Launcher")
    print("==========================================================")

    # 1. Run RT verification check
    check_script = os.path.join(os.path.dirname(__file__), "check_rt.sh")
    if os.path.exists(check_script) and sys.platform.startswith("linux"):
        subprocess.run(["bash", check_script])

    # 2. Create socketpairs (AF_UNIX, SOCK_SEQPACKET)
    # pair 1: assembler <-> inference
    s_assem, s_infer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    # pair 2: inference <-> postproc
    s_post, s_result = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sockpairs.extend([(s_assem, s_infer), (s_post, s_result)])

    # Make sockets inheritable by child processes
    for s in (s_assem, s_infer, s_post, s_result):
        os.set_inheritable(s.fileno(), True)

    # Ensure vocab.json exists before post-process starts
    vocab_file = cfg.get("model", {}).get("vocab_json", "vocab.json")
    if not os.path.exists(vocab_file):
        id2phoneme = {
            1: "aa", 2: "ae", 3: "ah", 4: "aw", 5: "ay", 6: "b", 7: "ch", 8: "d", 9: "dh",
            10: "dx", 11: "eh", 12: "er", 13: "ey", 14: "f", 15: "g", 16: "h#", 17: "hh",
            18: "ih", 19: "iy", 20: "jh", 21: "k", 22: "l", 23: "m", 24: "n", 25: "ng",
            26: "ow", 27: "oy", 28: "p", 29: "r", 30: "s", 31: "sh", 32: "t", 33: "th",
            34: "uh", 35: "uw", 36: "v", 37: "w", 38: "y", 39: "z", 0: "|", 40: "[UNK]", 41: "[PAD]"
        }
        with open(vocab_file, "w") as f:
            json.dump({str(k): v for k, v in id2phoneme.items()}, f, indent=2)
        print(f"[launcher] Created {vocab_file}")

    env_base = os.environ.copy()

    # 3. Start Post-Process (CPU 3)
    post_cpu = cfg.get("cpus", {}).get("post", 3)
    env_post = env_base.copy()
    env_post["HAPTICSYNC_RESULT_FD"] = str(s_result.fileno())

    post_cmd = ["./build/postproc_bin", CONFIG_PATH]
    if sys.platform.startswith("linux"):
        post_cmd = ["taskset", "-c", str(post_cpu)] + post_cmd

    print(f"[launcher] Spawning post-process on CPU {post_cpu}...")
    try:
        procs["postproc"] = subprocess.Popen(
            post_cmd,
            env=env_post,
            pass_fds=[s_result.fileno()],
        )
    except FileNotFoundError:
        print("[launcher] postproc_bin not found. Run CMake build first!")
        sys.exit(1)

    # 4. Start Inference Worker (CPU 2)
    model_cpus = ",".join(str(c) for c in cfg.get("cpus", {}).get("model", [2]))
    env_infer = env_base.copy()
    env_infer["HAPTICSYNC_INFER_FD"] = str(s_infer.fileno())
    env_infer["HAPTICSYNC_POST_FD"] = str(s_post.fileno())

    infer_cmd = [sys.executable, "inference/main.py", CONFIG_PATH]
    if sys.platform.startswith("linux"):
        infer_cmd = ["taskset", "-c", model_cpus] + infer_cmd

    print(f"[launcher] Spawning inference worker on CPU(s) {model_cpus}...")
    procs["inference"] = subprocess.Popen(
        infer_cmd,
        env=env_infer,
        pass_fds=[s_infer.fileno(), s_post.fileno()],
    )

    # Allow model to warm up and report ready
    print("[launcher] Waiting for model warm-up...")
    time.sleep(3.0)

    # 5. Start Producer Process (CPU 1)
    prod_cpu = cfg.get("cpus", {}).get("producer", 1)
    env_prod = env_base.copy()
    env_prod["HAPTICSYNC_ASSEM_FD"] = str(s_assem.fileno())

    prod_cmd = ["./build/producer_bin", CONFIG_PATH]
    if sys.platform.startswith("linux"):
        prod_cmd = ["taskset", "-c", str(prod_cpu)] + prod_cmd

    print(f"[launcher] Spawning producer process on CPU {prod_cpu}...")
    try:
        procs["producer"] = subprocess.Popen(
            prod_cmd,
            env=env_prod,
            pass_fds=[s_assem.fileno()],
        )
    except FileNotFoundError:
        print("[launcher] producer_bin not found. Run CMake build first!")
        cleanup()
        sys.exit(1)

    # 6. Start Telemetry Dashboard (CPU 0)
    if cfg.get("telemetry", {}).get("dashboard", True):
        hk_cpus = ",".join(str(c) for c in cfg.get("cpus", {}).get("housekeeping", [0]))
        telem_cmd = [sys.executable, "telemetry/main.py", CONFIG_PATH]
        if sys.platform.startswith("linux"):
            telem_cmd = ["taskset", "-c", hk_cpus] + telem_cmd

        print(f"[launcher] Spawning telemetry service on CPU(s) {hk_cpus}...")
        procs["telemetry"] = subprocess.Popen(telem_cmd, env=env_base)

    print("[launcher] All processes running. Press Ctrl-C to terminate.\n")

    # Supervision loop
    while True:
        for name, p in list(procs.items()):
            code = p.poll()
            if code is not None:
                print(f"[launcher] WARNING: Process {name} exited with code {code}!")
                if name == "producer":
                    print("[launcher] Producer died. Shutting down entire pipeline.")
                    cleanup()
                    sys.exit(1)
        time.sleep(1.0)


if __name__ == "__main__":
    main()


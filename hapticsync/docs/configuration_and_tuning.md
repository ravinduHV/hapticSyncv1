# Haptic-Sync Configuration & Runtime Tuning Guide

This document details all configuration parameters, window sizing options, telemetry streaming modes, and operational procedures for Haptic-Sync on the Raspberry Pi 5.

---

## 1. Window & Model Configuration (`config/hapticsync.json`)

All window and model parameters are read dynamically at startup by the C++ assembler and Python inference worker. **No C++ recompilation is required** when modifying these values.

```json
  "window": {
    "policy": "auto",
    "trim_left_frames": 3,
    "trim_right_frames": 3,
    "min_window_ms": 300,
    "max_window_ms": 4000,
    "min_new_ms": 200,
    "fixed_window_ms": 1000,
    "normalize_in": "cpp"
  },
  "model": {
    "checkpoint": "./model/",
    "blank_id": 41,
    "vocab_json": "vocab.json",
    "quantize": "none"
  }
```

### Parameter Reference

| Parameter | Type | Default | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `fixed_window_ms` | integer | `1000` | `300` – `4000` ms | **Length of audio slice** delivered to Wav2Vec2 per inference pass. Lower values reduce latency; higher values improve phoneme acoustic context. |
| `policy` | string | `"auto"` | `"latest"`, `"catchup"`, `"fifo"`, `"auto"` | **Queue backlog policy**:<br>• `latest`: Always processes newest audio, discarding stale backlog when model RTF > 1.0.<br>• `fifo`: Processes every window strictly in order (accumulates lag if model cannot keep up).<br>• `catchup`: Expands window size dynamically to consume backlog.<br>• `auto`: Selects between `latest` and `catchup` based on empirical benchmark slope ($b$). |
| `trim_left_frames` | integer | `3` | `0` – `5` frames | Number of 20 ms frames discarded at the beginning of each window to eliminate CTC boundary filter ramp-up artifacts. |
| `trim_right_frames` | integer | `3` | `0` – `5` frames | Number of 20 ms frames discarded at the end of each window. |
| `min_new_ms` | integer | `200` | `100` – `500` ms | Minimum duration of newly arrived audio before the assembler triggers the next inference. |
| `min_window_ms` | integer | `300` | `200` – `1000` ms | Minimum permissible window duration for dynamic windowing (`catchup`). |
| `max_window_ms` | integer | `4000` | `1000` – `4000` ms | Maximum window duration. The POSIX shared memory double-buffer is pre-allocated for up to 4,000 ms (64,000 samples at 16 kHz). |
| `normalize_in` | string | `"cpp"` | `"cpp"`, `"python"` | Whether sample mean/variance standardization is executed in C++ (lock-free assembler) or inside PyTorch. |
| `quantize` | string | `"none"` | `"none"`, `"int8"` | Dynamic post-training quantization on PyTorch Linear layers (`torch.ao.quantization.quantize_dynamic`). Reduces forward pass latency on ARM64 Cortex-A76 cores. |

---

## 2. Frame Grid & Overlap Mathematics

* **Audio Sample Rate:** 16,000 Hz (1 ms = 16 samples).
* **Model Hop / Stride:** 320 samples (20 ms per CTC frame).
* **Model Receptive Field:** 400 samples (25 ms).

### Window Size Calculation (1,000 ms Default)
$$\text{Total Samples} = 1{,}000\text{ ms} \times 16\text{ samples/ms} = 16{,}000\text{ samples}$$
$$\text{Total Frames} = \frac{16{,}000}{320} = 50\text{ frames}$$
$$\text{Committed Frames} = 50 - (\text{trim\_left} + \text{trim\_right}) = 50 - (3 + 3) = 44\text{ frames}$$
$$\text{Committed Duration} = 44 \times 20\text{ ms} = 880\text{ ms}$$

### Low-Latency Configuration Example (500 ms Window)
In `config/hapticsync.json` (or a custom `config/low_latency.json`):
```json
{
  "window": {
    "fixed_window_ms": 500,
    "trim_left_frames": 2,
    "trim_right_frames": 2,
    "min_new_ms": 100
  }
}
```
* **Total Samples:** 8,000 samples.
* **Total Frames:** 25 frames.
* **Committed Frames:** $25 - (2 + 2) = 21\text{ frames}$ (420 ms).

---

## 3. CPU Core Pinning & Task Allocation

The Raspberry Pi 5 features **4 physical ARM Cortex-A76 CPU cores** (`0`, `1`, `2`, `3`). Workload is partitioned in `config/hapticsync.json` under the `"cpus"` block:

```json
  "cpus": {
    "housekeeping": [0],
    "producer": 1,
    "model": [2, 3],
    "post": 1
  }
```

When you update this configuration and run `tools/launcher.py`, the supervisor automatically:
1. Pins child binaries to their assigned cores using Linux `taskset -c <cores>`.
2. Automatically sets PyTorch thread count: `torch.set_num_threads(len(model_cpus))` and `OMP_NUM_THREADS`.

### Core Allocation Reference Table

| Task | Configuration Key | Process | Typical CPU Load | Latency Sensitivity | Recommended Core |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Housekeeping & Telemetry** | `"housekeeping"` | `publisher.py` / `mqtt_publisher.py` | **< 0.5%** | Low | **CPU 0** (absorbs OS interrupts, WiFi/Ethernet, USB) |
| **Audio Capture (ALSA)** | `"producer"` | `producer_bin` | **< 3%** | **High** (delay causes XRUNs) | **CPU 1** (isolated from heavy neural compute) |
| **Window Assembler** | Built into producer | `producer_bin` | **< 1%** | Medium | **CPU 1** |
| **Post-Processor** | `"post"` | `postproc_bin` | **< 2%** (<1 ms/chunk) | Low | **CPU 1** (co-located with producer) |
| **Neural Model Inference** | `"model"` | `inference/main.py` | **100%** on assigned cores | Heavy compute | **CPU 2, 3** (or `[1, 2, 3]`) |

---

### Core Assignment Profiles

#### Profile 1: Dual-Core Model (Default — Balanced & Recommended)
```json
"cpus": {
  "housekeeping": [0],
  "producer": 1,
  "model": [2, 3],
  "post": 1
}
```
* **CPU 0:** System IRQs, network traffic, OS background tasks, and telemetry daemon.
* **CPU 1:** Dedicated entirely to audio capture and CTC merging. Having an un-congested core guarantees **0 ALSA buffer XRUNs**.
* **CPU 2 & 3:** Dedicated 100% to PyTorch neural network forward passes with 2 parallel OpenMP worker threads.

#### Profile 2: Triple-Core Model (Maximum Model Throughput)
```json
"cpus": {
  "housekeeping": [0],
  "producer": 0,
  "model": [1, 2, 3],
  "post": 0
}
```
* Maximizes inference speed by giving PyTorch **3 CPU cores** (`OMP_NUM_THREADS=3`).
* Moves audio capture and CTC collapse to **CPU 0** alongside housekeeping.
* *Note:* Under very heavy WiFi/MQTT network traffic, sharing CPU 0 may occasionally produce ALSA XRUNs.

#### Profile 3: Fully Isolated Stages (Single-Core Model)
```json
"cpus": {
  "housekeeping": [0],
  "producer": 1,
  "model": [2],
  "post": 3
}
```
* Every pipeline stage runs on its own distinct physical CPU core.
* Lowest power consumption and thermal footprint, but inference latency will be slightly higher ($\approx 1.4\times$ real-time).

---

### Verifying CPU Affinities Live on Raspberry Pi

1. **Check thread scheduling and affinities:**
   ```bash
   bash tools/check_rt.sh
   ```
2. **Monitor live per-core load:**
   Run `top`, then press `1` on your keyboard to expand CPU cores into individual meters:
   ```text
   %Cpu0 :  1.2 us,  0.5 sy ... (Housekeeping / Telemetry)
   %Cpu1 :  2.5 us,  0.8 sy ... (Audio Producer / PostProc)
   %Cpu2 : 98.4 us,  1.2 sy ... (PyTorch Model Worker Thread 1)
   %Cpu3 : 97.9 us,  1.5 sy ... (PyTorch Model Worker Thread 2)
   ```
3. **Inspect specific thread affinities:**
   ```bash
   ps -eLo pid,tid,cls,rtprio,psr,comm | grep -Ei 'producer|python|postproc'
   ```
   *(The `psr` column indicates the exact CPU core the thread is currently executing on).*

---

## 4. Telemetry & Monitoring Transport Options

Haptic-Sync offers two remote streaming architectures to eliminate web-server CPU consumption on the Raspberry Pi:

### Architecture Comparison

| Feature | Direct TCP Stream (Default) | MQTT Pub/Sub Mode |
| :--- | :--- | :--- |
| **RPi Process** | `telemetry/publisher.py` | `telemetry/mqtt_publisher.py` |
| **Dependencies** | **Zero** (pure Python standard library) | `paho-mqtt` (`pip install paho-mqtt`) |
| **RPi CPU Usage** | < 0.5% | < 0.5% |
| **Broker Required** | None (RPi listens as TCP server on port 5555) | Eclipse Mosquitto on PC (Port 1883) |
| **Multiple Listeners** | Yes (up to 5 concurrent TCP sockets) | Unlimited (pub/sub) |
| **Auto-reconnection** | Handled by PC receiver loop | Handled natively by MQTT client |

---

## 5. MQTT Setup & Verification

### Running the PC MQTT Broker (Docker Compose)
From the repository root on your PC:
```bash
docker compose up -d
```
The broker listens on:
* **Port `1883`**: MQTT TCP client port.
* **Port `9001`**: MQTT WebSockets port.

### Allowing Port 1883 in Windows Firewall (PC)
Run in PowerShell as Administrator:
```powershell
New-NetFirewallRule -DisplayName "Mosquitto MQTT" -Direction Inbound -LocalPort 1883 -Protocol TCP -Action Allow
```

### Verifying Connectivity from Raspberry Pi
```bash
nc -zv <PC_IP_ADDRESS> 1883
```
Expected output:
```text
Connection to <PC_IP_ADDRESS> 1883 port [tcp/*] succeeded!
```

### MQTT Topics Published by Raspberry Pi

| Topic | Rate | QoS | Payload Content |
| :--- | :--- | :--- | :--- |
| `hapticsync/pipeline_update` | 1 Hz (Retained) | 0 | Full JSON snapshot: hardware thermals, clock, throttle state, bottlenecks, audio source status, waveform preview, and lifetime statistics. |
| `hapticsync/chunk` | Event-driven | 0 | JSON payload per detected phoneme run: chunk ID, phonemes list, audio waveform, RMS levels, and latency measurements. |
| `hapticsync/stats` | 1 Hz | 0 | Raw atomic metrics block unpacked directly from `/dev/shm/hapticsync_stats`. |

---

## 6. How to Run

### Mode A: Zero-Dependency TCP (Default)
1. **On Raspberry Pi:**
   ```bash
   python3 tools/launcher.py --sim
   ```
2. **On PC:**
   ```bash
   python hapticsync/pc_dashboard.py --host 192.168.8.140
   ```

### Mode B: MQTT Pub/Sub
1. **Start broker on PC:**
   ```bash
   docker compose up -d
   ```
2. **Start dashboard on PC:**
   ```bash
   python hapticsync/pc_dashboard.py --mqtt --broker localhost
   ```
3. **On Raspberry Pi:**
   ```bash
   python3 tools/launcher.py --sim --mqtt --mqtt-broker <PC_IP_ADDRESS>
   ```

### Web Access
* **Phoneme Monitor:** `http://localhost:5000`
* **Pipeline Health & Diagnostics:** `http://localhost:5000/pipeline`


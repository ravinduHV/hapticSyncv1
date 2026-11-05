# Haptic-Sync Operational Runbook

## 1. Prerequisites (Raspberry Pi 5)
Ensure the following system packages are installed:
```bash
sudo apt update
sudo apt install -y build-essential cmake libasound2-dev nlohmann-json3-dev rt-tests stress-ng linux-perf
```

Ensure Python dependencies are installed in your active virtual environment:
```bash
pip install torch transformers numpy scipy flask flask-socketio soundfile pytest
```

## 2. RT Setup
Before starting the pipeline or running benchmarks, execute the RT setup script as root:
```bash
sudo bash tools/rt_setup.sh
```
Verify RT status:
```bash
bash tools/check_rt.sh
```

## 3. Build C++ Binaries
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
ctest --output-on-failure
cd ..
```

## 4. Phase A: Model Cost Curve Benchmark
Measure the model execution time on the pinned single core:
```bash
python3 tools/bench_model.py
```
This generates `config/bench.json` which determines the optimal window policy (`catchup` vs. `latest`).

## 5. Running the Complete Pipeline

### Mode A: Lightweight Telemetry Publisher (Recommended for RPi)
Offloads web serving, WebSockets, and UI rendering entirely to your PC:

1. **Start pipeline on Raspberry Pi (with simulated audio or real mic):**
   ```bash
   # Using I2S software simulator:
   python3 tools/launcher.py --sim

   # Or using real hardware I2S mic:
   python3 tools/launcher.py
   ```

2. **Run Web Dashboard locally on your PC:**
   ```bash
   python hapticsync/pc_dashboard.py --host <RASPBERRY_PI_IP>
   ```
   Open `http://localhost:5000` (Phoneme Monitor) or `http://localhost:5000/pipeline` (Pipeline Health).

### Mode B: MQTT Streaming Mode
Streams telemetry events to an MQTT broker running on your PC:

1. **Start MQTT broker on your PC:**
   ```bash
   docker compose up -d
   ```
2. **Start dashboard on your PC:**
   ```bash
   python hapticsync/pc_dashboard.py --mqtt --broker localhost
   ```
3. **Launch pipeline on Raspberry Pi:**
   ```bash
   python3 tools/launcher.py --sim --mqtt --mqtt-broker <YOUR_PC_IP>
   ```

### Mode C: Legacy On-Pi Web Server
Runs the Flask web server directly on Raspberry Pi CPU 0 (note: CPU load will be higher):
```bash
python3 tools/launcher.py --web --sim
```
Access directly at `http://<RASPBERRY_PI_IP>:5000`.

## 6. Runtime Tuning & Custom Configurations
Refer to `docs/configuration_and_tuning.md` for full parameter definitions.
To run with a custom profile (e.g. low-latency 500 ms window):
```bash
python3 tools/launcher.py config/low_latency.json --sim
```


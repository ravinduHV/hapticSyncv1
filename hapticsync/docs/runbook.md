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
Launch the pipeline with the supervisor:
```bash
python3 tools/launcher.py
```

Monitor live phoneme events in another terminal:
```bash
python3 tools/phoneme_tail.py
```

Access the live browser dashboard:
```
http://<raspberry-pi-ip>:5000
```


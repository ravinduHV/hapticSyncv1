# Gate A: Model Cost Curve & Policy Determination

## 1. Measured Cost Curve on Pinned Single Core (CPU 2)
- **Platform**: Raspberry Pi 5 (Cortex-A76)
- **Model**: Wav2Vec2ForCTC (FP32, PyTorch `torch.inference_mode()`, single thread)
- **CPU Pinning**: Core ID 2 (`taskset -c 2`, `torch.set_num_threads(1)`)
- **Iterations**: 35 per duration (after 5 warm-up iterations)

### Linear Fit
$$t(L) = a + b \cdot L$$
- **Fixed overhead ($a$)**: `0.40177 s` ($401.77\text{ ms}$)
- **Cost per second of audio ($b$)**: `1.3993` ($1.40\text{ s compute per 1.0 s audio}$)

### Empirical RTF Table
| Audio Duration ($L$) | Samples (16 kHz) | Mean Inference ($t$) | p95 Latency | Real-Time Factor (RTF) |
|---|---|---|---|---|
| **125 ms** | 2,000 | 634.1 ms | 636.2 ms | 5.07 |
| **250 ms** | 4,000 | 818.3 ms | 823.7 ms | 3.27 |
| **500 ms** | 8,000 | 1,104.8 ms | 1,108.2 ms | 2.21 |
| **1,000 ms** | 16,000 | 1,715.5 ms | 1,718.8 ms | 1.72 |
| **2,000 ms** | 32,000 | 3,086.6 ms | 3,094.0 ms | 1.54 |
| **4,000 ms** | 64,000 | 6,070.9 ms | 6,076.1 ms | 1.52 |

---

## 2. Gate A Policy Decision (Per Section 7.3 & Section 14 Item 8)
- **Condition**: $b = 1.3993 \ge 1.0$.
- **Finding**: A single Cortex-A76 core takes $\approx 1.40\text{ seconds}$ to compute $1.0\text{ second}$ of audio forward pass. Therefore, sustainable 100% audio coverage (`catchup` mode) cannot keep up with real-time audio on 1 core.
- **Selected Policy**: **`latest`** (Policy B).
  - The pipeline fixes the analysis window (e.g. $L_{\text{fix}} = 1000\text{ ms}$).
  - It continuously processes the newest audio to maintain freshness and low lag.
  - Backlog audio that accumulated during the forward pass is cleanly skipped, counted, and reported in the stats block (`windows_dropped`).
  - Memory usage remains strictly bounded, and no pipeline stalls occur.
- **Option for More Cores**: If the human supervisor desires 100% audio coverage without quantization, `config.cpus.model` can be expanded from `[2]` to `[2, 3]` (or more cores) in `config/hapticsync.json`.

---

## 3. Thermal & System Stability
- **Initial Temp**: $41.7^\circ\text{C}$, `throttled=0x0`
- **End Temp**: $49.9^\circ\text{C}$, `throttled=0x0`
- **Throttling**: Zero thermal or voltage throttling observed across 210 full model inferences.


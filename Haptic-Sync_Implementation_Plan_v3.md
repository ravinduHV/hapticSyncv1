# Haptic-Sync Phoneme Streamer — Implementation Plan v3 (Agent-Ready)

**Supersedes:** v1 and v2.
**Audience:** an agentic coding model (and the human engineer supervising it). This document is intended to be self-contained: it states the scope, the decisions and the reasons for them, the architecture, interfaces, algorithms, build layout, phased tasks, tests and acceptance criteria.
**Target:** Raspberry Pi 5, custom **PREEMPT_RT** Linux kernel, I2S/ALSA microphone (or `snd-aloop` for testing).

> **Tags used in this document**
> **[Fixed]** = a project decision from the human owner; do not change it.
> **[Decision]** = a design choice made in this plan, with the reason given; may be revisited only with measured evidence.
> **[Measured]** = from existing telemetry. **[Hypothesis]** = must be tested before relying on it.
> **[Verify]** = depends on the target system; check on the device and record the result.

---

## 0. How the Agent Must Use This Document

1. Read Sections 1–4 fully before writing code. Sections 5–8 are the specification. Section 10 is the work order. Section 16 lists hard rules.
2. Work **phase by phase**. Do not start a phase before the previous phase's gate is met and its results are recorded in `docs/results/`.
3. Every number marked **[Estimate]** or **[Hypothesis]** must be replaced by a measurement. Record measurements, do not guess.
4. If a **[Verify]** item cannot be confirmed on the device, stop and ask the human (Section 14 lists the known blockers).
5. Do **not** expand scope (Section 1.2). Ideas outside scope go into `docs/handoff.md`, not into the code.

---

## 1. Context and Scope

### 1.1 Project
A hearing-aid support system that detects phonemes from live audio. A later project phase will turn phonemes into vibration cues. The original proposal targeted <30 ms end-to-end latency with a small INT8 model and hardware PWM.

### 1.2 Current scope **[Fixed]**
| # | Decision | Consequence |
|---|---|---|
| F1 | **Keep the model as it is** (Wav2Vec2-CTC, FP32, PyTorch, same checkpoint and vocabulary as the current system). | No quantization, ONNX/TFLite conversion, pruning, distillation or retraining. Only the code *around* the model changes. |
| F2 | **No PWM / haptic output.** | The pipeline ends at *phoneme extraction + push to an event queue*. |
| F3 | **The <30 ms target is withdrawn.** | Do what is achievable. Latency is measured and minimized, not pass/fail against 30 ms. |
| F4 | **RT kernel; core pinning:** producer + all pre-model work on the **2nd core**, model on the **3rd core**, post-model work on the **4th core**. | See Section 4. Core numbering interpretation is in Section 2. |
| F5 | **Language:** C or C++ wherever possible (maximum compatibility); Python only where C/C++ is not practical. | Producer, pre-model and post-model stages in C++17. Model inference, telemetry, launcher and tools in Python. |
| F6 | **Follow the RT-programming slides** (Ogness, Embedded Recipes 2025). | Section 3 maps each slide to a decision. |

### 1.3 In scope
Continuous gap-free capture; pre-model window assembly; unchanged-model inference; CTC decoding with timestamps; overlap merging; phoneme-event queue with a defined contract; non-blocking telemetry; RT validation tooling; measurement harness.

### 1.4 Out of scope (deferred; see Section 15)
Model optimization, haptic consumer/PWM/motors, phoneme→pattern mapping, the <30 ms goal, networking/TSN.

---

## 2. Platform and Core-Numbering Assumption

- Raspberry Pi 5: 4 × Cortex-A76 cores. Custom **PREEMPT_RT** kernel **[Verify]**: `uname -a` shows `PREEMPT_RT`; `/sys/kernel/realtime` contains `1`.
- **[Decision] Core numbering:** the human's "2nd / 3rd / 4th core" is interpreted as ordinals, i.e. Linux CPU ids **1, 2, 3**. CPU **0** is the housekeeping core (OS, remaining IRQs, telemetry, dashboard). All CPU ids are configuration values (`config.cpus`), so this can be changed without code changes. **Confirm this interpretation with the human at Gate P0.**

| CPU id | Role | Contents |
|---|---|---|
| 0 | Housekeeping | systemd and all other services, non-pinned IRQs, launcher, telemetry/dashboard (Flask-SocketIO), tools |
| 1 ("2nd core") | **Capture + pre-model** | Audio IRQ and its IRQ kthread, **producer** thread (RT), **assembler** thread (window build, normalization, scheduling policy) |
| 2 ("3rd core") | **Model** | Python inference process, PyTorch with exactly **1 thread** |
| 3 ("4th core") | **Post-model** | C++ post-processor: CTC collapse, overlap merge, event emission |

### 2.1 Important consequence of F4 **[Hypothesis]**
The existing measurement (790 ms inference per 125 ms chunk, RTF 6.33) was taken without any thread limit. Running PyTorch on **one** core will most likely be **slower, not faster**. A lossless real-time pipeline is therefore **not guaranteed**. The design handles this explicitly with a backlog policy (Section 7.3) chosen from measurements taken **under the pinned single-core configuration** (Phase A). The model core count is a configuration value (`config.cpus.model` is a list) so the human can later allow more cores without code changes.

---

## 3. Decisions Derived from the RT Slides

Each row maps a slide topic to a concrete decision in this project.

| Slide topic | Slide guidance | **[Decision] for this project** |
|---|---|---|
| Memory in physical RAM | Linux maps memory on demand and recycles it; solution is `mlockall(2)`, pre-faulting, `mallopt(3)` tuning | Producer/assembler process calls `mlockall(MCL_CURRENT\|MCL_FUTURE)` after all setup; pre-fault stack and ring buffer; `mallopt(M_MMAP_MAX, 0)` and `mallopt(M_TRIM_THRESHOLD, -1)` before allocating; no allocation in the RT loop. Test by counting minor page faults in steady state (must be 0 growth). |
| RT policies and priorities | `SCHED_FIFO`, `SCHED_RR`; priorities 1–99; `chrt`, `sched_setscheduler`; `SCHED_DEADLINE` is complex to combine with priorities | **`SCHED_FIFO` only**, set via `sched_setscheduler`/`pthread_setschedparam`. **`SCHED_DEADLINE` is not used.** |
| Real-time work only | Only use RT priority for real-time work; do not abuse priorities for "scheduler tuning" | **Only the producer thread is `SCHED_FIFO`** (hardware-paced audio capture). Assembler, inference, post-processor, telemetry are **`SCHED_OTHER`**; separation is done with CPU pinning, not priorities. The post-processor becomes RT only when a real RT consumer exists. |
| Event flow | Transition from higher to lower priorities; hardware interrupt is highest "priority" in an event chain | Chain: HW interrupt → audio IRQ kthread (**FIFO 60**) → producer (**FIFO 50**) → assembler / inference / post (**SCHED_OTHER**). Priorities strictly decrease along the chain. |
| RT-safe sleeping | Consider the waker in all scenarios; the waker is the parent in an event flow | RT→non-RT wake-ups are one-directional (producer never waits on a non-RT thread). Control messages from the producer side use **non-blocking** writes; the producer never blocks on locks, sockets or queues. The future haptic consumer must not depend on a non-RT waker (Section 15). |
| Synchronization | `pthread_mutex_t` with `PTHREAD_PRIO_INHERIT`; `pthread_cond_t`; `librtpi` | **Design goal: no lock shared between RT and non-RT threads** (lock-free rings/atomics). If a mutex is unavoidable, it must be `PTHREAD_PRIO_INHERIT`; condition variables on that path use `librtpi`. |
| Cyclic tasks | Avoid `timerfd` and POSIX timers (RT-unsafe). Correct: dedicated thread + `clock_nanosleep(2)` + `CLOCK_MONOTONIC` + `TIMER_ABSTIME` | Capture is paced by the **ALSA hardware period** (blocking read), not a timer. The file-replay source (testing) and any periodic RT-priority work use `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` on a dedicated thread. **No `timerfd`/POSIX timers in RT-priority threads.** All timestamps use `CLOCK_MONOTONIC`. |
| CPU affinities | `cpuset(7)`/cgroups; `isolcpus` is deprecated; `taskset`, `sched_setaffinity`, `/proc/irq/N/smp_affinity` | Use **cpuset/cgroup** isolation (not `isolcpus`) plus `sched_setaffinity` in each worker; move all IRQs except the audio IRQ to CPU 0 via `smp_affinity`; pin the audio IRQ to CPU 1. |
| Library caution | Libraries (even libc) may not be RT-safe | RT thread restricted to a short audited list of calls (Section 5.2.3). Python is **never** used in an RT thread. PortAudio/`sounddevice` is **removed** from the capture path (it runs a callback in a library-owned thread and, in Python, takes the GIL). |
| Monitoring | RV monitor `rtapp` (pagefault and sleep monitors) | Used as a validation gate (Section 11.3): zero `rv:error_sleep` and zero `rv:error_pagefault` events attributable to the producer thread. |
| PipeWire example | "Do not post non-real-time work to real-time loop" | FFT, waveform, dashboard, logging and file I/O never run in the producer thread; telemetry reads shared memory from CPU 0. Avoid sound servers (PipeWire/PulseAudio) in the capture path; use ALSA `hw:` directly. |
| RT networking / TSN | PTP, 802.1Qbv, RTC-Testbench | **Not applicable** to this scope. The dashboard network traffic is non-RT and stays on CPU 0. |

---

## 4. Architecture

### 4.1 Processes, cores, priorities, languages

```
CPU 1 ── PRODUCER PROCESS (C++17) ──────────────────────────────────────────────────────────────
  [HW I2S IRQ] → [irq/NN-name kthread, FIFO 60] → [producer thread, FIFO 50]
       ALSA hw read (160-frame period) → mono downmix → float32 audio ring (shared mem, lock-free)
  [assembler thread, SCHED_OTHER]
       window policy → normalize → write window slot (shared mem) → send WINDOW msg ──┐
CPU 2 ── INFERENCE PROCESS (Python 3 + PyTorch, 1 thread, SCHED_OTHER) ◄───────────────┘
       read slot → Wav2Vec2ForCTC forward (UNCHANGED) → softmax/argmax per frame
       send DONE msg → assembler;   send RESULT msg ──┐
CPU 3 ── POST-PROCESS PROCESS (C++17, SCHED_OTHER) ◄──┘
       commit-region tiling → CTC collapse (stateful) → PhonemeEvent → EVENT RING (shared mem)
CPU 0 ── TELEMETRY / DASHBOARD (Python), LAUNCHER (Python), TOOLS
       read-only taps of audio ring, stats block, event ring (never write to RT structures)
```

### 4.2 Data flow and transports (control = messages, bulk = shared memory)

| Link | Transport | Direction | Notes |
|---|---|---|---|
| Producer → ring | In-process write to **shared-memory audio ring** | — | Lock-free single writer; absolute sample index |
| Assembler → Inference (window data) | **Shared-memory window slots** (2 slots) | C++ → Py | Normalized float32 samples |
| Assembler → Inference (control) | `AF_UNIX` `SOCK_SEQPACKET` socketpair, **non-blocking** write | C++ → Py | `WINDOW` message |
| Inference → Assembler | Same socketpair, other direction | Py → C++ | `DONE` message (model idle, timings) |
| Inference → Post | `AF_UNIX` `SOCK_SEQPACKET` socketpair | Py → C++ | `RESULT` message (small per-frame arrays) |
| Post → consumers | **Shared-memory event ring** (single writer, multi-reader, sequence-numbered) | C++ → any | The "phoneme queue" |
| All → telemetry | Read-only mapping of rings plus a **stats block** | — | Telemetry tolerates tearing (sequence check) |

**[Decision] Why messages for control and shared memory for bulk data:** Python cannot do portable atomics/memory barriers; a syscall (`write`/`read`) per hand-off provides ordering and keeps the Python side simple and safe, while large audio data avoids copying through the socket. Sockets are created by the launcher via `socketpair()` and inherited by the workers (fd numbers passed in environment variables).

### 4.3 Priority and policy table

| Thread/process | CPU | Policy | Priority | Reason |
|---|---|---|---|---|
| Audio IRQ kthread (`irq/NN-…`) | 1 | `SCHED_FIFO` | 60 (set by `rt_setup.sh`) | Event-chain parent must be above the producer |
| Producer thread | 1 | `SCHED_FIFO` | 50 | Hardware-paced capture; the only RT-deadline work |
| Assembler thread | 1 | `SCHED_OTHER` | nice 0 | Not real-time; shares CPU 1 and is preempted by the producer |
| Inference process | 2 | `SCHED_OTHER` | nice 0 | Long, compute-bound, non-RT; isolation via CPU pinning |
| Post-process | 3 | `SCHED_OTHER` | nice 0 | Not RT until an RT consumer exists |
| Launcher, telemetry, tools | 0 | `SCHED_OTHER` | — | Housekeeping |

**[Verify]** Privilege: setting `SCHED_FIFO` and IRQ affinity needs root or `CAP_SYS_NICE` (and an `rtprio` limit in `/etc/security/limits.d/`). The process must **fail fast with a clear error** if `require_rt=true` and RT could not be obtained.

---

## 5. Component Specifications

### 5.1 RT platform setup (`tools/rt_setup.sh`, `tools/check_rt.sh`)

`rt_setup.sh` (run as root at boot or before launch; idempotent):
1. CPU governor `performance` on CPUs 0–3 (`/sys/devices/system/cpu/cpuN/cpufreq/scaling_governor`).
2. IRQ placement: write mask `1` (CPU 0) to `/proc/irq/default_smp_affinity` and to every `/proc/irq/*/smp_affinity` (ignore per-CPU/immovable IRQs); then set the **audio device IRQ** to mask `2` (CPU 1). Identify it from `/proc/interrupts` and the sound card name. **[Verify]** On Pi 5 the I2S/audio interrupt path goes through the RP1 chip and may not be movable; if so, record this in `docs/results/p0.md` and continue (the cross-core wake costs microseconds, measure it).
3. Raise the audio IRQ kthread to `SCHED_FIFO` 60 (`chrt -f -p 60 <pid of irq/NN-name>`), if such a thread exists.
4. Disable `irqbalance` if installed.
5. Create cpuset cgroups (cgroup v2): housekeeping `cpuset.cpus=0`; `rt-capture` = 1; `rt-model` = 2; `rt-post` = 3. Move user services/shells to the housekeeping set (systemd `AllowedCPUs=0` for `system.slice`/`user.slice` is the preferred mechanism). **Do not use `isolcpus`** (deprecated per slides). **[Verify]** cgroup v2 mounted with `cpuset` controller enabled.
6. Print a verification report (governor, IRQ affinities, cpuset membership, `vcgencmd get_throttled`).

`check_rt.sh` verifies and prints: kernel is PREEMPT_RT; governor; IRQ affinities; each worker's `Cpus_allowed_list` (from `/proc/<pid>/task/*/status`) and policy/priority (`chrt -p`); absence of other busy threads on CPUs 1–3 (`top -H`/`ps -eLo`).

Optional kernel boot options (`nohz_full=`, `rcu_nocbs=`) are **not** part of the baseline. Adopt only if Phase P0 measurements show a need and the kernel was built with support.

### 5.2 Producer process (C++17) — `producer/`

#### 5.2.1 Threads
- **Producer thread** (`SCHED_FIFO` 50, CPU 1): ALSA capture → downmix → ring write → publish counters. Nothing else.
- **Assembler thread** (`SCHED_OTHER`, CPU 1): builds windows and talks to the inference process (Section 5.3).
- **Log drain thread** (`SCHED_OTHER`, CPU 0 allowed or CPU 1): drains a lock-free log ring written by the producer; the producer never calls `printf`/`iostream`/syslog.

#### 5.2.2 Capture
- **[Decision]** Use **alsa-lib directly** (`snd_pcm_*`), device `hw:<card>,<dev>` from config. Do not go through PortAudio, PulseAudio or PipeWire.
- Parameters: 16 000 Hz, 2 channels, access `RW_INTERLEAVED`, sample format as the hardware provides (commonly `S32_LE` for I2S MEMS mics, or `S16_LE`); convert to float32 in [-1, 1] (divide by full scale of the *valid bits*; **[Verify]** the mic's valid bit width). **Period = 160 frames (10 ms)**, buffer = 8 periods (config). `avail_min` = 1 period.
- **[Verify]** If the hardware cannot do 16 kHz directly, first try `plughw:` (ALSA does the conversion in the caller's context). Run the RT validation (Section 11.3) with that device; if it fails, add a dedicated resampling stage in the assembler instead (not in the producer thread).
- Capture timestamps: set `SND_PCM_TSTAMP_TYPE_MONOTONIC` and read `snd_pcm_status_get_htstamp()` per period; store `t_ns` (CLOCK_MONOTONIC) for the **last sample of each period** in the ring's period table.
- Downmix: `channel_mode` = `mean` (default; identical to the current system's `np.mean(axis=1)`), `left` or `right`. **[Verify]** if the I2S mic only drives one channel, `mean` halves the amplitude; use the relevant single channel and record the choice.
- Xrun handling: on `-EPIPE`/`-ESTRPIPE`, `snd_pcm_prepare()`, increment the `xruns` counter, record a **gap marker** `{sample_idx, estimated_lost_frames}` in the ring header, advance the absolute sample counter by the estimated lost frames (so timestamps stay correct) and **flag the next window as `first_after_gap`**.
- Alternative sources (same ring interface; selected by `source` config): `alsa`, `wav` (file paced in real time using `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` on its own dedicated thread; used for all repeatable tests), `synthetic` (sine/click generator for latency tests). Both `snd-aloop` and `wav` sources must work.

#### 5.2.3 RT-thread rules (the producer thread may only…)
Allowed: `snd_pcm_readi`/`snd_pcm_status*`/`snd_pcm_prepare`, `clock_gettime` (vDSO), arithmetic, writes into pre-allocated memory, atomic operations, `write()` of fixed-size records to the log ring (memory only), non-blocking `write()` to the wake pipe (optional).
Forbidden: `new`/`malloc`/`free`, `std::string`/`std::vector` growth, iostream/printf/syslog, exceptions on the hot path, any mutex shared with non-RT threads, `timerfd`, POSIX timers, file I/O, Python, any call not on the allowed list. Review every new call against this list.

#### 5.2.4 RT setup sequence (`common/rt_util.h`)
```
1. set_affinity(cpu)                       // sched_setaffinity, thread-level
2. mallopt(M_MMAP_MAX, 0); mallopt(M_TRIM_THRESHOLD, -1);
3. allocate/map everything: ALSA handles, shm rings, log ring (touch every page: memset)
4. prefault_stack(1 MiB)                   // volatile char buf[...]; memset
5. mlockall(MCL_CURRENT | MCL_FUTURE)
6. set_fifo(priority)                      // pthread_setschedparam(SCHED_FIFO)
7. read back affinity/policy/priority; log; abort if require_rt && mismatch
8. enter loop
```

#### 5.2.5 Audio ring (`common/ring.h`, shared memory `/hapticsync_audio`)
- Mono float32, capacity `ring_seconds` (default 60 s ≈ 3.8 MB). Power-of-two sample capacity recommended.
- Header (cache-line aligned): `magic, version, sample_rate, capacity, atomic<uint64> write_idx` (absolute sample count; never resets), `period_table[]` (ring of `{sample_idx_end, t_ns}`), `gap_table[]`, `xruns`, `overruns`.
- Single writer (producer), readers are the assembler (same process) and telemetry (read-only, other process). Reader validity: a reader copying `[a, b)` must re-read `write_idx` afterwards and discard the copy if `write_idx - a > capacity`.

### 5.3 Assembler (pre-model work) — `producer/assembler.*`

Responsibilities (all on CPU 1, `SCHED_OTHER`): decide **when** to send a window and **which** one, normalize it, publish it, and track the inference process' state.

1. Wait on a `poll()` of (a) the `DONE` socket and (b) a short timeout (5 ms). (The producer does not wake it; see Section 3, RT-safe sleeping.)
2. When the model is **idle** and enough new audio exists, build the next window per the policy in Section 7.3 and the frame-grid rules in Section 7.1–7.2.
3. **Normalize** the samples exactly as the Hugging Face `Wav2Vec2FeatureExtractor` does for this checkpoint: `y = (x − mean(x)) / sqrt(var(x) + 1e-7)` over the whole window (float32). **[Verify]** `do_normalize` in the checkpoint's `preprocessor_config.json`; the C++ implementation must match the Python processor output within `1e-5` on test audio (unit test, Section 11.1). Config flag `normalize_in = cpp|python` allows falling back to the Python processor without code changes.
4. Write the normalized samples into the free **window slot** (two slots, double-buffered), then send the `WINDOW` message (non-blocking).
5. Maintain stats: windows sent, windows dropped (policy B), model idle time, queue state.

### 5.4 Inference process (Python) — `inference/`

- **First statements of `main.py`, before importing torch/numpy:** `os.environ["OMP_NUM_THREADS"]="1"`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; then `os.sched_setaffinity(0, {cfg.cpus.model...})` (applies to the calling thread and is inherited by threads created afterwards); then import torch; `torch.set_num_threads(len(cpus.model))`, `torch.set_num_interop_threads(1)`.
- Launcher also starts the process with `taskset -c <cpu>` as a second safeguard.
- **Model:** load the *same* `Wav2Vec2ForCTC` checkpoint (path/id from config, **[Verify]** identical to the current system), `model.eval()`, FP32, no modification of any kind. Use `torch.inference_mode()`.
- **Warm-up before announcing "ready":** run several inferences on dummy windows of different lengths (shapes affect allocation) until timings stabilize.
- **Per message:** map the window slot as a NumPy view (no copy) → `torch.from_numpy(...).unsqueeze(0)` → `model(input_values).logits` → `probs = logits.softmax(-1)` → `conf, label = probs.max(-1)` → pack `n_frames` × (label `uint16`, conf `float32`) → send `RESULT` to post-process → send `DONE` to the assembler.
- **Startup assertions:** (a) `n_frames == floor((n_samples − 400)/320) + 1` for several window lengths (frame-grid contract, Section 7.1); (b) blank token id equals `config.blank_id` (HF CTC models use the pad token as blank; **[Verify]**); (c) export the `id → phoneme` map (the existing `id2phoneme` data) to `vocab.json` for the post-processor.
- Timings: record `t_infer_begin_ns`/`t_infer_end_ns` (`time.monotonic_ns()`, same clock as C++) and put them in `RESULT`/`DONE`.
- The inference process contains **no** FFT, dashboard, printing in the loop (log through the stats block or a queue to telemetry) and no processor call in the default configuration.

### 5.5 Post-process (C++17) — `postproc/`

- CPU 3, `SCHED_OTHER`; written in the same RT-safe *style* (no allocation after start-up, no exceptions in the loop), because it will later feed a real-time consumer.
- Loop: blocking `recv()` on the `RESULT` socket → for each frame in the **commit range** (`commit_begin..commit_end−1`) apply the stateful CTC collapse (Section 7.4) → write `PhonemeEvent` records to the event ring → update stats.
- On `first_after_gap` flag: reset collapse state (`prev_label = blank`) and record a gap in stats.

### 5.6 Phoneme event ring ("the queue") — `common/events.h`, shared memory `/hapticsync_events`

Single writer (post-process), any number of independent readers, each with its own cursor. The writer **never blocks** and never waits for readers; slow readers detect being lapped.

```c
// 64-byte fixed slot, little endian
struct PhonemeEvent {
  uint64_t seq;          // slot sequence: odd = being written, even = valid (2*n+2)
  uint64_t event_id;     // monotonically increasing, starts at 1
  int64_t  t_frame_ns;   // CLOCK_MONOTONIC capture time of the frame center
  int64_t  t_emit_ns;    // time written to the ring
  uint64_t abs_frame;    // absolute 20 ms frame index since stream start
  uint32_t window_id;
  uint16_t phoneme_id;   // model token id
  uint16_t flags;        // bit0: first event after a gap
  float    confidence;   // softmax probability of the frame label
  char     phoneme[12];  // NUL-terminated label from vocab.json
};
```
- Header: `magic, version, slot_count (default 1024), atomic<uint64> head (next event_id), stream_start_ns`.
- Writer protocol: `slot.seq = 2*id+1` (release) → fill fields → `slot.seq = 2*id+2` (release) → `head = id+1`.
- Reader protocol: read `seq1`; copy; read `seq2`; accept only if `seq1 == seq2` and even; if `head − cursor > slot_count`, the reader was lapped (count and resync).
- **Event semantics [Decision]:** one event per CTC *run onset* (the first frame where a non-blank label starts), emitted as soon as that frame is committed. This minimizes lag. `lag = t_emit_ns − t_frame_ns`.
- A Python reader (`tools/events_reader.py`) and a CLI (`tools/phoneme_tail.py`) are delivered; they double as the reference for the future haptic consumer.

### 5.7 Telemetry / dashboard (Python, CPU 0)
- Replaces the current terminal table and Flask-SocketIO dashboard; same information (waveform, spectrum, phonemes, latency), now **read-only taps** of shared memory.
- Per-channel FFT is computed here from a copy of recent samples (stereo FFT is a telemetry-only feature; if the dashboard still needs L/R, add an optional raw-stereo debug ring that the producer writes only when `debug_stereo=true`).
- Update rate 10–20 Hz. If telemetry is slow it simply skips; it can never stall the pipeline.
- Shows: coverage %, xruns, gaps, RTF, queue/state, lag distribution, events per second, CPU/temperature (`vcgencmd`), throttle flags.

### 5.8 Launcher (`tools/launcher.py`)
1. Run `check_rt.sh`; abort on hard failures.
2. Create shared memory, socketpairs, directories.
3. Start: post-process (CPU 3) → inference (CPU 2, wait for "ready") → producer (CPU 1) → telemetry (CPU 0).
4. Supervise; restart policy for inference and post (they must be restartable without restarting the producer; the producer must keep capturing when the inference process is down, and then count drops by policy).
5. Clean shutdown on SIGINT/SIGTERM (reverse order, unlink shared memory).

---

## 6. Message Formats (`common/protocol.h`, mirrored in `inference/protocol.py`)

All fields little-endian, packed, explicit size and a `version` field. Messages over `SOCK_SEQPACKET` (one message per datagram).

```c
enum MsgType : uint32_t { MSG_WINDOW = 1, MSG_DONE = 2, MSG_RESULT = 3 };

struct WindowMsg {          // assembler -> inference
  uint32_t type, version;
  uint64_t window_id;
  uint32_t slot;            // 0 or 1
  uint32_t n_samples;       // = (n_frames-1)*320 + 400
  uint64_t start_sample;    // absolute, multiple of 320
  int64_t  t_start_ns;      // capture time of start_sample
  int64_t  t_sent_ns;
  uint32_t n_frames;        // expected model output frames
  uint32_t commit_begin;    // frame index inside the window (inclusive)
  uint32_t commit_end;      // frame index inside the window (exclusive)
  uint32_t flags;           // bit0: first_after_gap
};

struct DoneMsg {            // inference -> assembler
  uint32_t type, version;
  uint64_t window_id;
  int64_t  t_infer_begin_ns, t_infer_end_ns;
  uint32_t status;          // 0 ok, nonzero = error (window discarded)
};

struct ResultHeader {       // inference -> post, followed by n_frames * FrameOut
  uint32_t type, version;
  uint64_t window_id;
  uint64_t start_sample;
  int64_t  t_start_ns;
  int64_t  t_infer_begin_ns, t_infer_end_ns;
  uint32_t n_frames, commit_begin, commit_end, flags;
};
struct FrameOut { uint16_t label; uint16_t pad; float conf; };
```
Rules: unknown `version` → drop and count. The `RESULT` echoes the commit range from the `WINDOW` so the post-process needs no knowledge of the scheduling policy.

---

## 7. Algorithms

### 7.1 Frame grid (Wav2Vec2 geometry)
The Wav2Vec2 feature encoder (conv kernels 10,3,3,3,3,2,2; strides 5,2,2,2,2,2,2; no padding) maps audio to frames with **hop = 320 samples (20 ms)** and **receptive field = 400 samples (25 ms)**:

```
n_frames(n_samples) = floor((n_samples − 400) / 320) + 1
n_samples(n_frames) = (n_frames − 1) * 320 + 400
frame i of a window starting at sample s begins at sample s + 320*i; its center is s + 320*i + 200
```
**[Decision] Grid alignment:** every window start `s` is a multiple of 320, so frame indices are comparable across windows: **absolute frame index `F = s/320 + i`**. The model's frame count is asserted at start-up (Section 5.4); if it differs, stop and report.

### 7.2 Overlap-save tiling (replaces centrality merging from earlier plans)
Parameters: `trimL` and `trimR` (frames, default 3 each ≈ 60 ms) discard the unreliable edge frames of every window.
- First window (or after a gap): commit frames `[0, n − trimR)`.
- Every later window w with `n` frames: commit frames `[trimL, n − trimR)`.
- The next window's start is chosen so its committed region begins exactly where the previous one ended:

```
start_frame(next) = start_frame(prev) + n_prev − trimR − trimL      // in 20 ms grid units
```
Committed regions then **tile the timeline without gaps or duplicates**, no voting or centrality logic is needed, and every committed frame is final immediately. Lag cost of the right trim = `trimR × 20 ms`.
If a window was **dropped** (policy B), the tiling is broken: the next window is flagged `first_after_gap` and commits from frame 0.

### 7.3 Backlog policy (how to keep up with real time)
Measure on the **pinned single-core configuration** (Phase A): `t(L) = a + b·L` (seconds), where `a` = fixed overhead, `b` = cost per second of audio. Overlap per window is `O ≈ (trimL + trimR)·0.02 + 0.005` s.

| Policy | Rule | Coverage | Lag | Use when |
|---|---|---|---|---|
| `catchup` | Next window = all audio available since the last commit end (plus overlap), capped at `max_window_ms` | 100% if sustainable | ≈ steady-state window + inference time | `b < 1` |
| `latest` | Fixed window length `L_fix`; always process the newest audio; older unprocessed audio is dropped and counted | `(L_fix − O)/t(L_fix)` (capped at 1) | ≈ `L_fix` + `t(L_fix)` | `b ≥ 1`, or `catchup` exceeds `max_window_ms` |
| `fifo` | Fixed windows in order, no dropping | 100% | Grows without bound if `t > window` | Only for tests |

**Steady state of `catchup`** exists iff `b < 1`: cycle time `T* = (a + b·O) / (1 − b)`, window length `L* = T* + O`. If `L* > max_window_ms`, switch to `latest`. `policy: "auto"` (default) computes this from the Phase A numbers stored in `config/bench.json` and picks the policy; the choice and numbers are logged at start-up. **Drops are always counted and reported, never silent.**

### 7.4 CTC collapse and event emission (post-process)
State: `prev_label` (initial = blank), `abs_frame` of the next frame.
For each committed frame with `(label, conf)`:
```
if label != blank and label != prev_label:  emit PhonemeEvent(label, conf, abs_frame, t_frame_ns)
prev_label = label
```
Standard greedy-CTC semantics (repeated labels merge unless separated by a blank), carried across window boundaries by the state. `t_frame_ns = t_start_ns(window) + ((i·320 + 200) / 16000) · 1e9`, using the window's capture anchor. An optional `min_confidence` filter and an optional silence gate (RMS threshold computed by the assembler and carried in a flag bit) exist in config, **disabled by default**.

---

## 8. Configuration (`config/hapticsync.json`)

Read by C++ (`nlohmann::json`) and Python. Defaults:

```json
{
  "cpus": { "housekeeping": [0], "producer": 1, "model": [2], "post": 3 },
  "rt": { "require_rt": true, "irq_thread_prio": 60, "producer_prio": 50 },
  "audio": { "source": "alsa", "alsa_device": "hw:0,0", "rate": 16000, "channels": 2,
             "period_frames": 160, "buffer_periods": 8, "channel_mode": "mean",
             "wav_path": null, "ring_seconds": 60 },
  "window": { "policy": "auto", "trim_left_frames": 3, "trim_right_frames": 3,
              "min_window_ms": 300, "max_window_ms": 4000, "min_new_ms": 200,
              "fixed_window_ms": 1000, "normalize_in": "cpp" },
  "model": { "checkpoint": "<same as current system>", "blank_id": null, "vocab_json": "vocab.json" },
  "events": { "slot_count": 1024, "min_confidence": 0.0 },
  "telemetry": { "rate_hz": 10, "dashboard": true, "port": 5000 }
}
```
`<same as current system>` and `blank_id` must be filled from the existing code on the first task of Phase P0 (**[Verify]**).

---

## 9. Repository Layout and Build

```
hapticsync/
├─ CMakeLists.txt                 # C++17, -O2, -Wall -Wextra -Werror (RT code), aarch64 native
├─ config/hapticsync.json, bench.json
├─ common/    rt_util.h  ring.h  events.h  protocol.h  shm.h  config.h  logring.h  stats.h
├─ producer/  main.cpp  alsa_source.cpp  wav_source.cpp  synth_source.cpp  assembler.cpp
├─ postproc/  main.cpp  ctc_merge.cpp  vocab.cpp
├─ inference/ main.py  worker.py  protocol.py  shm_views.py
├─ telemetry/ main.py  dashboard.py
├─ tools/     launcher.py  rt_setup.sh  check_rt.sh  bench_model.py  replay.py
│             phoneme_tail.py  events_reader.py  latency_report.py  ref_decode.py  agreement.py
├─ tests/     cpp/ (doctest)   py/ (pytest)
└─ docs/      architecture.md  handoff.md  results/  runbook.md
```
Dependencies (Raspberry Pi OS 64-bit or equivalent): `build-essential cmake libasound2-dev nlohmann-json3-dev doctest-dev rt-tests stress-ng linux-perf python3-venv`; Python venv: `torch` (aarch64), `transformers`, `numpy`, `scipy`, `flask`, `flask-socketio`, `soundfile`, `pytest`.
`librtpi` is added **only** if a PI condition variable is needed (not expected).

---

## 10. Phased Work Plan

**Timing is given as relative effort** (the original 15-week schedule was withdrawn). Each phase ends with a gate; record results in `docs/results/<phase>.md`.

### Phase P0 — Platform and RT Validation (~2–3 days)
| ID | Task | Output |
|---|---|---|
| P0.1 | Confirm kernel is PREEMPT_RT; kernel/cgroup/debugfs capabilities (cpuset, RV monitors) | `p0.md` |
| P0.2 | Locate the current system's model checkpoint, `id2phoneme`, blank id, processor config; fill `config` placeholders | config updated |
| P0.3 | Identify the ALSA capture device, supported rates/formats/period sizes (`arecord -D hw:… --dump-hw-params`) | `p0.md` |
| P0.4 | Write and run `rt_setup.sh`, `check_rt.sh`; record IRQ affinity outcome for the audio IRQ | report |
| P0.5 | Baseline scheduling latency: `cyclictest -m -S -p80 -i1000 -D 10m -a 1,2,3` while `stress-ng --cpu 4` runs (on all cores, then on CPUs 0, 2, 3 only) | histogram, max, p99 |
| P0.6 | Confirm human's core-numbering interpretation (Section 2) | decision recorded |

**Gate P0:** RT kernel confirmed; CPUs 1–3 show bounded scheduling latency under load (record max; investigate if > ~100 µs on CPU 1, since this bounds capture reliability); open questions in Section 14 resolved or explicitly deferred.

### Phase A — Model Cost Curve on the Pinned Core (~2–3 days)
| ID | Task |
|---|---|
| A1 | `bench_model.py`: run under `taskset -c 2`, `torch.set_num_threads(1)`, `inference_mode`; input lengths 125 / 250 / 500 / 1000 / 2000 / 4000 ms; ≥ 30 iterations each after warm-up |
| A2 | Separate first-call (warm-up) cost from steady state; fit `t(L) = a + b·L`; also run with 2 and 4 threads for information (not for use) |
| A3 | Log temperature, frequency, `get_throttled` during runs |
| A4 | Produce `ref_decode.py`: full-context decode of recorded test audio → **reference transcript** (labels with frame times) |
| A5 | Write `config/bench.json` `{a, b, rtf_curve}`; run the policy calculator (Section 7.3) and record the chosen policy |

**Gate A:** `a`, `b` measured on the pinned core; policy chosen and justified; reference transcripts exist. **If `b ≥ 1`, report this to the human before proceeding** (coverage–freshness trade-off must be accepted: policy `latest`, or allow more cores).

### Phase B — Producer, Ring and RT Hygiene (~1 week)
| ID | Task |
|---|---|
| B1 | `common/` utilities: `rt_util.h` (Section 5.2.4), shared memory helpers, lock-free log ring, stats block |
| B2 | Audio ring (Section 5.2.5) with unit tests (wraparound, absolute index, reader invalidation) |
| B3 | `wav_source` (paced with `clock_nanosleep` ABSTIME) and `synth_source` |
| B4 | `alsa_source`: configuration, blocking read loop, timestamps, xrun handling, gap markers |
| B5 | Producer main with RT setup sequence; stats export |
| B6 | RT validation (Section 11.3): rtapp monitor, minor-fault count, loop-jitter histogram, 30-min soak with `stress-ng` on CPUs 0, 2, 3 |

**Gate B:** zero gaps over 30 min under load; zero xruns; zero `rv:error_sleep` / `rv:error_pagefault` for the producer thread; minor-fault count constant after warm-up; producer loop jitter p99.9 recorded.

### Phase C — Assembler, Protocol and Inference Worker (~1–1.5 weeks)
| ID | Task |
|---|---|
| C1 | `protocol.h` / `protocol.py` with round-trip tests |
| C2 | Frame-grid helpers and **tiling tests** (Section 11.1) |
| C3 | Assembler: policy engine (`catchup`/`latest`/`fifo`), normalization in C++, slot writing, `WINDOW`/`DONE` handling |
| C4 | Normalization parity test vs. HF processor (tolerance 1e-5) |
| C5 | Inference worker (Section 5.4) with warm-up, start-up assertions, timing fields |
| C6 | Fault tests: inference killed → producer keeps capturing; inference slow → policy behaves as specified; message of unknown version dropped |

**Gate C:** end-to-end (producer → inference → a debug printer) runs 10 min on file replay; no deadlocks; counters consistent (`windows_sent = windows_done + dropped`); the producer thread's RT validation is unchanged (re-run B6 with inference active).

### Phase D — Post-Process, Event Ring, Telemetry (~1 week)
| ID | Task |
|---|---|
| D1 | Event ring (writer + reader library in C++ and Python) with tests (lapping, torn-read detection) |
| D2 | Post-process: stateful CTC collapse, commit-range handling, gap reset, vocab loading |
| D3 | `phoneme_tail.py` and `events_reader.py` |
| D4 | Telemetry process and dashboard rewritten as read-only taps |
| D5 | Launcher with supervision and clean shutdown |

**Gate D:** live and replay runs produce a stable event stream; replayed events match the offline reference decode per Section 12 (agreement metric); the dashboard shows data without changing producer timing (re-run B6 with dashboard on).

### Phase E — Tuning (~1 week)
| ID | Task |
|---|---|
| E1 | Sweep `trimL/trimR`, `max_window_ms` or `fixed_window_ms`, `min_new_ms`; record RTF, lag distribution, agreement |
| E2 | Normalization/silence behavior check: does per-window normalization create spurious phonemes in silence? Try the silence gate; keep default off unless clearly beneficial |
| E3 | Thermal and long-run stability (≥ 60 min), including throttle flags |
| E4 | Select and document the default configuration with its trade-offs |

**Gate E:** default configuration chosen; Must tier (Section 12) met; Should tier met or the shortfall documented with data.

### Phase F — Evaluation and Hand-off (~3–4 days)
Final measurement run and report; comparison against the baseline (Section 12.3); documentation (`architecture.md`, `runbook.md`, `handoff.md`); hand-off notes for the haptic consumer and the model-optimization work (Section 15).

---

## 11. Test and Validation Plan

### 11.1 Unit tests (must exist before integration)
- **Ring buffer:** wraparound; absolute indexing; reader invalidation when lapped.
- **Frame grid:** `n_frames`/`n_samples` formulas vs. the real model output length on several lengths.
- **Tiling:** simulate random window sizes/timings; assert committed frames form a contiguous range with no gaps or duplicates; assert reset after a simulated drop.
- **CTC state:** same label across a commit boundary emits once; blank resets; gap resets.
- **Normalization parity:** C++ vs. Hugging Face feature extractor, tolerance 1e-5, on random and real audio, lengths 125 ms – 4 s.
- **Protocol:** round-trip of every message; version mismatch handling.
- **Event ring:** writer/reader sequence protocol, lapping, torn-read rejection.

### 11.2 Integration tests
- **File replay at real time** (`wav` source): compare the pipeline's event stream with the full-context reference transcript (`agreement.py`).
- **Fault injection:** kill/restart the inference process; stall it artificially (sleep); fill the event ring faster than a reader consumes it. Producer must keep writing the audio ring in all cases.
- **Policy tests:** feed synthetic `t(L)` into the policy engine and verify the predicted steady state and drop counts.
- **Latency probe:** `synthetic` click source → event with the click's capture time; verify `t_frame_ns` accuracy and report lag.

### 11.3 RT validation (re-run after every change touching the producer)
```bash
# Scheduling latency baseline (P0) and under pipeline load
cyclictest -m -S -p80 -i1000 -D 10m -a 1,2,3 -h 400 -q
stress-ng --cpu 4 --vm 1 --vm-bytes 256M --timeout 10m

# RV monitor from the RT slides (needs a kernel that provides it; check the directory first)
ls /sys/kernel/debug/tracing/rv/monitors/
echo 1       > /sys/kernel/debug/tracing/rv/monitors/rtapp/enable
echo printk  > /sys/kernel/debug/tracing/rv/monitors/rtapp/reactors
perf record -g --call-graph dwarf -a -e rv:error_sleep -e rv:error_pagefault -- sleep 600
perf script        # pass criterion: no events for the producer thread

# Page faults of the producer after warm-up must not grow
grep -E 'minflt|majflt' /proc/<producer_pid>/stat     # sample at t=60 s and t=1800 s

# Affinity / policy / priority
ps -eLo pid,tid,cls,rtprio,psr,comm | grep -E 'producer|assembler|infer|post|irq/'
```
**[Verify]** If the `rtapp` monitor is not present in the custom kernel, substitute `perf trace -F all`/ftrace `sched_switch` + `exceptions:page_fault_user` filtered on the producer TID, and note the kernel version in `p0.md`.
- **Producer loop jitter:** record the interval between consecutive `snd_pcm_readi` returns (expect 10 ms ± small); report histogram, p99, p99.9 and max.
- **Cross-core wake cost:** if the audio IRQ stays on CPU 0 because it cannot be moved, measure the added delay versus an IRQ-on-CPU-1 run (if possible).

---

## 12. Metrics, Acceptance and Expected Results

### 12.1 Metrics
| Metric | Definition |
|---|---|
| Coverage | Audio samples analyzed ÷ samples captured (drops reported separately) |
| Capture gaps / xruns | From the ring's gap table and `xruns` counter |
| RTF | Inference time ÷ audio duration advanced per cycle |
| Lag | `t_emit_ns − t_frame_ns` per event; report mean, median, p95, p99, max |
| Queue stability | Max and trend of window backlog and event-ring lag over a long run |
| Agreement | Event-level agreement with the full-context reference decode (precision, recall of phoneme onsets within ±40 ms, plus label-sequence edit distance); PER vs. ground truth if labelled data exist |
| RT quality | Scheduling latency (cyclictest), producer loop jitter, rtapp violations, minor faults |
| Resources | Per-core load, memory, temperature, throttle flags |

### 12.2 Tiers (no absolute latency number is promised; it is set after Phase A)
| Tier | Criteria |
|---|---|
| **Must** | Capture has **zero unintended gaps** over 60 min under load; producer RT validation clean; unchanged model; event contract implemented; every drop counted and reported; no unbounded queue growth; pinning/priorities as specified in Section 4 |
| **Should** | Policy matches measured `a`, `b`; coverage 100% (or a documented drop rate if `b ≥ 1`); lag substantially below the 962 ms baseline cycle; agreement measured across settings; no thermal throttling |
| **Stretch** | Lowest lag allowed by the chosen window and trims; silence gating reduces spurious events; documented evidence on what extra cores for the model would buy |

### 12.3 Baseline for comparison **[Measured]**
Chunk 125 ms; cycle 962 ms; inference 790 ms; RTF 6.33; coverage ~13%; ~46 ms capture start-up per call (stream reopened each time). The final report must contain this table with the new measured values.

### 12.4 Expected lag floor
Lag cannot go below roughly: time to accumulate the window's last needed frame + one inference time + `trimR × 20 ms`. With the model unchanged and one core, expect **hundreds of milliseconds to seconds**, not tens of milliseconds. The gains over the baseline come from removing the capture blind spot and start-up cost, overlapping capture with inference, and tuning window policy.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Single-core FP32 Wav2Vec2 has `b ≥ 1` | Cannot be lossless and fresh at once | Policy `latest` with counted drops; report to human; config allows more model cores later |
| Audio IRQ cannot be moved to CPU 1 (RP1 interrupt routing) | Slightly higher wake latency | Measure; keep producer on CPU 1; document |
| Producer thread page-faults or sleeps in an RT-unsafe way | Xruns / gaps | RT rules (5.2.3), `mlockall`, pre-faulting, rtapp gate (B6) |
| `hw:` device does not support 16 kHz / needed format | Capture failure | Try `plughw:`; else resample in assembler; re-validate RT |
| C++ normalization differs from HF processor | Different model input → different phonemes | Parity test; `normalize_in=python` fallback |
| Short windows reduce phoneme quality | Poor events | Larger windows/trims; measure agreement; do not alter model |
| Per-window normalization amplifies noise in silence | Spurious phonemes | Silence gate (E2) |
| Python socket/shm timing noise in inference process | Extra lag | Acceptable: non-RT stage on its own core; measure |
| Thermal throttling during long inference | Latency spikes | Active cooling, `get_throttled` logging |
| Dashboard load interferes | Producer jitter | Dashboard confined to CPU 0; read-only taps; re-run B6 with dashboard |
| Scope drift into PWM/model work | Schedule slip | Section 1.2 and Section 16 are hard boundaries |

---

## 14. Open Items the Agent Must Resolve or Ask About

1. **Core numbering** (Section 2): confirm CPU ids 1/2/3 for the "2nd/3rd/4th core".
2. **Model checkpoint** identifier/path, vocabulary, blank id, `preprocessor_config.json` (`do_normalize`, sampling rate) from the existing code.
3. **Audio device:** ALSA card/device, native rate, format, valid bit depth, which channel(s) carry data; whether `snd-aloop` is used in the lab.
4. **Kernel facts:** exact version, PREEMPT_RT confirmed, cgroup v2 + cpuset, debugfs mounted, RV `rtapp` monitor available, `CONFIG_NO_HZ_FULL` (only relevant if boot options are ever tried).
5. **Privileges:** root or capabilities for `SCHED_FIFO`, `mlockall` limits (`ulimit -l`), IRQ affinity writes.
6. **RAM/swap:** Pi 5 RAM size; swap/zram status (swap should be off or irrelevant to RT processes).
7. **Test audio:** availability of labelled phoneme audio (e.g. TIMIT) for PER; otherwise only agreement-vs-reference can be reported.
8. **If Phase A shows `b ≥ 1`:** human decision on accepting policy `latest`, or allowing the model more than one core.

---

## 15. Deferred Work and Hand-off Hooks

| Deferred item | Hook already provided |
|---|---|
| Model optimization (INT8, ONNX, smaller model) | Inference stage is isolated behind the `WINDOW`/`RESULT` protocol; any model that returns per-frame labels with the 20 ms grid can be swapped in. `bench.json` gives the cost curve to beat. |
| Haptic consumer / PWM | **Event ring** + reader library. **Design rule from the RT slides:** an RT consumer must **not** block waiting for a wake-up from the non-RT post-process (the pattern that `rtapp` flags as a priority-inversion hint). It should poll the ring from a dedicated thread using `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`, or the post-process must be raised to RT priority above/equal to the consumer with proper PI handling. |
| Latency targets (<30 ms) | `t_frame_ns`/`t_emit_ns` on every event and the measured lag distributions. |
| Phoneme → pattern mapping | Events carry the model token id, label and confidence. |
| More cores for the model | `config.cpus.model` is a list; the policy calculator will be re-run with new `bench.json`. |

---

## 16. Hard Rules for the Agent

1. **Never modify the model** or its weights, and never change quantization/format (F1).
2. **No PWM/GPIO/motor code** (F2).
3. **No Python, no allocation, no logging I/O, no `timerfd`/POSIX timers, no shared non-PI mutex** in an RT-priority thread (Sections 3, 5.2.3).
4. Only the producer thread is `SCHED_FIFO` (and the audio IRQ kthread, set by script). Do not use priorities as a substitute for CPU pinning.
5. No `isolcpus`; use cpuset/cgroups and `sched_setaffinity`. No `SCHED_DEADLINE`.
6. Producer/assembler/post in C++17; inference/telemetry/launcher/tools in Python (F5). Do not add dependencies without need.
7. Every drop, gap, xrun, lapped read and protocol error must be **counted and surfaced** in the stats block and dashboard.
8. Use `CLOCK_MONOTONIC` everywhere (C++ `steady_clock`/`clock_gettime`, Python `time.monotonic_ns()`).
9. Record measurements in `docs/results/` and replace every **[Estimate]/[Hypothesis]** with data before claiming a result.
10. Stop and ask the human when a **[Verify]** item blocks progress (Section 14) rather than guessing.

---

## Appendix A — RT Utility Sketch (`common/rt_util.h`)

```cpp
// Illustrative; keep the real version free of allocation and exceptions on RT paths.
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <malloc.h>
#include <cstring>

inline bool set_affinity(int cpu) {
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
  return pthread_setaffinity_np(pthread_self(), sizeof(s), &s) == 0;
}
inline bool set_fifo(int prio) {
  sched_param p{}; p.sched_priority = prio;
  return pthread_setschedparam(pthread_self(), SCHED_FIFO, &p) == 0;
}
inline void rt_malloc_setup() { mallopt(M_MMAP_MAX, 0); mallopt(M_TRIM_THRESHOLD, -1); }
inline void prefault_stack(size_t n = 1 << 20) {
  volatile unsigned char buf[1 << 20]; (void)n;   // fixed size; touch every page
  std::memset((void*)buf, 0, sizeof(buf));
}
inline bool lock_all() { return mlockall(MCL_CURRENT | MCL_FUTURE) == 0; }
```

## Appendix B — Producer Loop Sketch (RT thread)

```cpp
while (running.load(std::memory_order_relaxed)) {
  snd_pcm_sframes_t n = snd_pcm_readi(pcm, raw, period_frames);    // blocks on HW period IRQ
  if (n < 0) { handle_xrun(n); continue; }                          // prepare, count, gap marker
  int64_t t_ns = period_end_timestamp(pcm);                         // MONOTONIC htstamp
  for (int i = 0; i < n; ++i) mono[i] = downmix(raw, i, mode);      // float32
  ring.write(mono, n, t_ns);                                        // memcpy + release-store write_idx
  stats.periods.fetch_add(1, std::memory_order_relaxed);
}
```

## Appendix C — Pass/Fail Checklist for the RT Gate (B6)

- [ ] `check_rt.sh` clean; producer `psr`=1, `cls`=FF, `rtprio`=50
- [ ] Audio IRQ kthread `rtprio`=60 (or documented why not)
- [ ] 30 min with `stress-ng` on CPUs 0, 2, 3: **0 xruns, 0 gaps**
- [ ] `rv:error_sleep` = 0 and `rv:error_pagefault` = 0 for the producer TID
- [ ] Minor faults identical at t=60 s and end of run
- [ ] Producer loop interval p99.9 and max recorded and below the period (10 ms)
- [ ] Result file written to `docs/results/b6.md`

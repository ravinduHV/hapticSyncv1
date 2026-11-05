#!/bin/bash
# rt_setup.sh — Platform Real-Time Setup for Raspberry Pi 5 (Section 5.1)
# Run as root before starting Haptic-Sync. Idempotent.

set -euo pipefail

echo "=========================================================="
echo "  Haptic-Sync RT Platform Setup (Raspberry Pi 5)"
echo "=========================================================="

# 1. Verify PREEMPT_RT kernel
if [ ! -f /sys/kernel/realtime ] || [ "$(cat /sys/kernel/realtime)" != "1" ]; then
    echo "WARNING: Kernel does not report PREEMPT_RT (/sys/kernel/realtime != 1)"
    uname -a
else
    echo "[OK] PREEMPT_RT kernel confirmed: $(uname -r)"
fi

# 2. Set CPU governor to 'performance' on all cores (0-3)
echo "Setting CPU governor to performance on CPUs 0-3..."
for cpu in /sys/devices/system/cpu/cpu[0-3]; do
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        echo "performance" > "$cpu/cpufreq/scaling_governor" || true
    fi
done

# 3. Disable irqbalance if active
if systemctl is-active --quiet irqbalance 2>/dev/null; then
    echo "Stopping irqbalance service..."
    systemctl stop irqbalance || true
    systemctl disable irqbalance || true
fi

# 4. IRQ Placement: move default IRQs to CPU 0 (mask 1)
echo "Directing default interrupts to CPU 0..."
if [ -w /proc/irq/default_smp_affinity ]; then
    echo 1 > /proc/irq/default_smp_affinity || true
fi

for aff in /proc/irq/*/smp_affinity; do
    if [ -w "$aff" ]; then
        echo 1 > "$aff" 2>/dev/null || true
    fi
done

# 5. Locate Audio IRQ and pin to CPU 1 (mask 2)
# Check /proc/interrupts for sound/i2s/rp1-i2s
AUDIO_IRQ=$(grep -Ei 'i2s|sound|snd|bcm2835-i2s|rp1-i2s' /proc/interrupts | head -n1 | awk -F: '{print $1}' | tr -d ' ' || true)

if [ -n "$AUDIO_IRQ" ] && [ -w "/proc/irq/$AUDIO_IRQ/smp_affinity" ]; then
    echo "Setting audio IRQ $AUDIO_IRQ affinity to CPU 1 (mask 2)..."
    echo 2 > "/proc/irq/$AUDIO_IRQ/smp_affinity" 2>/dev/null || echo "Audio IRQ $AUDIO_IRQ affinity not movable (RP1 chip path)"

    # Raise IRQ kthread priority if exists
    IRQ_KTHREAD_PID=$(pgrep -f "irq/$AUDIO_IRQ-" || true)
    if [ -n "$IRQ_KTHREAD_PID" ]; then
        echo "Raising audio IRQ kthread $IRQ_KTHREAD_PID to SCHED_FIFO 60..."
        chrt -f -p 60 "$IRQ_KTHREAD_PID" || true
    fi
else
    echo "Audio IRQ not identified in /proc/interrupts; will rely on CPU pinning."
fi

# 6. Setup cgroup v2 / cpuset isolation
CG_DIR="/sys/fs/cgroup/hapticsync"
if [ -d /sys/fs/cgroup ] && [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    echo "Configuring cgroup v2 cpusets..."
    mkdir -p "$CG_DIR" || true
    echo "+cpuset" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true

    mkdir -p "$CG_DIR/housekeeping" "$CG_DIR/rt-capture" "$CG_DIR/rt-model" "$CG_DIR/rt-post" || true

    echo 0 > "$CG_DIR/housekeeping/cpuset.cpus" 2>/dev/null || true
    echo 1 > "$CG_DIR/rt-capture/cpuset.cpus" 2>/dev/null || true
    echo 2 > "$CG_DIR/rt-model/cpuset.cpus" 2>/dev/null || true
    echo 3 > "$CG_DIR/rt-post/cpuset.cpus" 2>/dev/null || true
fi

# 7. Verification Report
echo ""
echo "── RT Verification Report ─────────────────────────────────"
echo "Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
if command -v vcgencmd &>/dev/null; then
    echo "Throttled: $(vcgencmd get_throttled || echo 'N/A')"
    echo "Temp:      $(vcgencmd measure_temp || echo 'N/A')"
fi
echo "RT setup completed successfully."


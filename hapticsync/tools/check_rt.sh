#!/bin/bash
# check_rt.sh — Verify system and pipeline RT compliance (Section 5.1)

set -uo pipefail

echo "── System Kernel Check ────────────────────────────────────"
if [ -f /sys/kernel/realtime ] && [ "$(cat /sys/kernel/realtime 2>/dev/null)" = "1" ]; then
    echo "[PASS] Kernel is PREEMPT_RT (via /sys/kernel/realtime): $(uname -r)"
elif uname -v | grep -q "PREEMPT_RT" || uname -a | grep -q "PREEMPT_RT"; then
    echo "[PASS] Kernel is PREEMPT_RT (via uname): $(uname -r)"
else
    echo "[WARN] Kernel does not report PREEMPT_RT"
fi

echo ""
echo "── CPU Scaling Governors ──────────────────────────────────"
for cpu in /sys/devices/system/cpu/cpu[0-3]; do
    cname=$(basename "$cpu")
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        gov=$(cat "$cpu/cpufreq/scaling_governor")
        echo "$cname: $gov"
    fi
done

echo ""
echo "── Worker Process Affinities & Scheduler Status ───────────"
ps -eLo pid,tid,cls,rtprio,psr,comm | grep -Ei 'producer|assembler|python|irq/' | grep -v grep || echo "No pipeline workers running yet."

echo ""
echo "── System Thermal & Throttle Status ───────────────────────"
if command -v vcgencmd &>/dev/null; then
    echo "Temperature: $(vcgencmd measure_temp 2>/dev/null || echo 'N/A')"
    echo "Throttled:   $(vcgencmd get_throttled 2>/dev/null || echo 'N/A')"
fi


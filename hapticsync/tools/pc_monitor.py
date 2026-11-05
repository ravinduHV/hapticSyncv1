#!/usr/bin/env python3
"""
tools/pc_monitor.py — Terminal Pipeline & Phoneme Streamer (Runs on PC or anywhere)
Requires ZERO external dependencies (pure Python standard library).
Connects to Raspberry Pi publisher (port 5555) and prints live pipeline stats and phonemes.
"""

import sys
import time
import json
import struct
import socket
import argparse

parser = argparse.ArgumentParser(description="HapticSync Terminal Monitor")
parser.add_argument("--host", default="192.168.8.140", help="Raspberry Pi IP (default: 192.168.8.140)")
parser.add_argument("--port", type=int, default=5555, help="Publisher port (default: 5555)")
args = parser.parse_args()

RPI_HOST = args.host
RPI_PORT = args.port

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def recv_exact(sock, n_bytes):
    data = bytearray()
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def main():
    print(f"{BOLD}{CYAN}=========================================================={RESET}")
    print(f"{BOLD}{CYAN}  HapticSync Terminal Pipeline & Phoneme Streamer{RESET}")
    print(f"  Connecting to {RPI_HOST}:{RPI_PORT}...")
    print(f"{BOLD}{CYAN}=========================================================={RESET}\n")

    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((RPI_HOST, RPI_PORT))
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            print(f"{GREEN}[CONNECTED]{RESET} Receiving real-time telemetry from Raspberry Pi!\n")

            while True:
                hdr = recv_exact(sock, 4)
                if not hdr:
                    print(f"{YELLOW}[DISCONNECTED]{RESET} Connection closed by Raspberry Pi.")
                    break

                pkt_len = struct.unpack("<I", hdr)[0]
                raw_data = recv_exact(sock, pkt_len)
                if not raw_data:
                    break

                msg = json.loads(raw_data.decode("utf-8"))
                mtype = msg.get("type")
                data = msg.get("data", {})

                if mtype == "new_chunk":
                    ph_list = data.get("phonemes", [])
                    ph_str = " ".join(ph_list) if ph_list else "(silence)"
                    inf_ms = data.get("inference_ms", 0.0)
                    lag_ms = data.get("timing", {}).get("total", 0.0)
                    chunk_id = data.get("id", "chunk")

                    print(f"[{chunk_id}] {BOLD}{GREEN}PHONEMES:{RESET} [ {BOLD}{CYAN}{ph_str}{RESET} ] "
                          f"| Infer: {inf_ms:.1f}ms | Lag: {lag_ms:.1f}ms")

                elif mtype == "pipeline_update":
                    st = data.get("stats", {})
                    sys_info = data.get("system", {})
                    bneck = data.get("bottlenecks", {})

                    rtf = bneck.get("rtf", 0.0)
                    rtf_color = GREEN if rtf < 1.0 else (YELLOW if rtf < 1.5 else RED)
                    temp = sys_info.get("temp_c", 0.0)
                    temp_color = RED if temp > 75 else (YELLOW if temp > 60 else GREEN)

                    status = bneck.get("overall_status", "healthy").upper()
                    status_color = GREEN if status == "HEALTHY" else (YELLOW if status == "WARNING" else RED)

                    sys.stdout.write(
                        f"\r{BOLD}[STATUS: {status_color}{status}{RESET}{BOLD}]{RESET} "
                        f"RTF: {rtf_color}{rtf:.2f}x{RESET} | "
                        f"Infer: {st.get('last_infer_ms', 0):.0f}ms | "
                        f"Lag: {st.get('last_lag_ms', 0):.0f}ms | "
                        f"Sent/Done/Drop: {st.get('windows_sent', 0)}/{st.get('windows_done', 0)}/{st.get('windows_dropped', 0)} | "
                        f"Temp: {temp_color}{temp:.1f}°C{RESET} | "
                        f"Events: {st.get('events_emitted', 0)}  "
                    )
                    sys.stdout.flush()

        except Exception as e:
            print(f"{YELLOW}[WAITING]{RESET} Could not connect ({e}). Retrying in 2 seconds...")
            time.sleep(2.0)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
run_pc_dashboard.py — Launch the Remote Web Dashboard on your PC
Connects to Raspberry Pi (default: 192.168.8.140:5555) and serves the
web UI locally at http://localhost:5000 and http://localhost:5000/pipeline
"""

import sys
import os

# Add hapticsync to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hapticsync"))

from pc_dashboard import main

if __name__ == "__main__":
    main()


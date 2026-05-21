"""
main.py — SpikeVoice OS Entry Point
=====================================
Run this file to launch the full application.

Usage:
    python main.py

If no trained weights exist, the model will run with random weights
(classification will be random). Train first:
    python src/train.py
"""

import os
import sys
import io

# Reconfigure stdout/stderr to use UTF-8 if they aren't already, preventing UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
else:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dashboard import SpikeVoiceDashboard

if __name__ == "__main__":
    print("=" * 55)
    print("  ⚡ SpikeVoice OS — Neuromorphic Voice Control")
    print("=" * 55)
    print("  Press Ctrl+C or close the window to exit.\n")

    app = SpikeVoiceDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

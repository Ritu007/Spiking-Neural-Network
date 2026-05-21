"""
inference.py — Real-Time Sliding-Window Inference Engine for SpikeVoice OS
===========================================================================
Opens a microphone stream via sounddevice, fills a 1-second rolling buffer,
and runs the SNN every 100ms.

Puts results into a thread-safe queue consumed by the dashboard and the
action handler.

Queue message format:
    {
        "label":      str,              # detected keyword or "silence"
        "confidence": float,            # normalised spike count [0, 1]
        "mem_trace":  list[Tensor],     # membrane potentials per time-step
        "spike_trace":list[Tensor],     # hidden layer spikes per time-step
        "raw_audio":  np.ndarray,       # current 1-second audio buffer
    }
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

import time
import queue
import threading
import numpy as np
import torch
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoder import encode_audio, SAMPLE_RATE
from src.model   import SpikeVoiceModel

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 1024           # samples per audio callback (~64ms at 16kHz)
BUFFER_DURATION   = 1.0            # seconds of audio held in the rolling buffer
INFER_INTERVAL    = 0.1            # run inference every 100ms
CONFIDENCE_THRESH = 0.65           # minimum normalised spike rate to confirm
CONFIRM_WINDOWS   = 1              # number of consecutive confident windows needed (lowered for responsiveness)
RMS_NOISE_THRESH  = 0.0030         # energy gate threshold to filter ambient silence
WEIGHTS_PATH      = os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "weights", "spikevoice_v1.pt")
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FALLBACK_KEYWORDS = ["mute", "up", "down", "stop", "bed", "silence"]


class InferenceEngine:
    """
    Manages microphone capture, sliding-window encoding,
    SNN inference, and result dispatch.
    """

    def __init__(self, result_queue: queue.Queue):
        self.q               = result_queue
        self.buffer          = np.zeros(int(SAMPLE_RATE * BUFFER_DURATION),
                                        dtype=np.float32)
        self._lock           = threading.Lock()
        self._running        = False
        self._confirm_count  = 0
        self._last_label     = "silence"

        # Load model
        self.model, self.keywords = self._load_model()
        self.model.eval()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        model    = SpikeVoiceModel(num_classes=6)
        keywords = FALLBACK_KEYWORDS

        if os.path.exists(WEIGHTS_PATH):
            ckpt     = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            keywords = ckpt.get("keywords", FALLBACK_KEYWORDS) + ["silence"]
            print(f"✅ Loaded weights from {WEIGHTS_PATH}")
        else:
            print("⚠️  No weights found — running with random weights (train first!).")

        return model.to(DEVICE), keywords

    # ── Audio callback (called by sounddevice in a separate thread) ───────────
    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].astype(np.float32)  # mono
        with self._lock:
            # Shift buffer left and append new chunk
            self.buffer = np.roll(self.buffer, -len(chunk))
            self.buffer[-len(chunk):] = chunk

    # ── Inference step ────────────────────────────────────────────────────────
    def _run_inference(self):
        with self._lock:
            audio_snapshot = self.buffer.copy()

        # Calculate Root-Mean-Square (RMS) amplitude to detect silence/background noise
        rms = float(np.sqrt(np.mean(audio_snapshot**2)))

        if rms < RMS_NOISE_THRESH:
            # Silence event-driven shortcut: bypass running the SNN entirely
            self._confirm_count = 0
            self.q.put({
                "label":       "silence",
                "confidence":  1.0,
                "confirmed":   False,
                "mem_trace":   [],
                "spike_trace": [],
                "raw_audio":   audio_snapshot,
                "rms":         rms,
            })
            return

        spikes = encode_audio(audio_snapshot)              # [T, 1, 128, F/T]
        spikes = spikes.unsqueeze(1).to(DEVICE)            # [T, 1, 1, 128, F/T]

        with torch.no_grad():
            counts, mem_rec, spike_rec = self.model(spikes)

        # Normalise spike counts to [0,1]
        probs      = torch.softmax(counts.squeeze(0), dim=0)
        confidence = probs.max().item()
        class_idx  = probs.argmax().item()
        label      = self.keywords[class_idx] if class_idx < len(self.keywords) else "silence"

        # Confirmation logic
        if label != "silence" and confidence >= CONFIDENCE_THRESH:
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        confirmed = self._confirm_count >= CONFIRM_WINDOWS

        self.q.put({
            "label":       label,
            "confidence":  confidence,
            "confirmed":   confirmed,
            "mem_trace":   mem_rec,
            "spike_trace": spike_rec,
            "raw_audio":   audio_snapshot,
            "rms":         rms,
        })

        if confirmed:
            self._confirm_count = 0   # reset after firing

    # ── Main loop ─────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
            dtype="float32",
        )
        stream.start()
        print("🎙️  Microphone stream started.")

        try:
            while self._running:
                self._run_inference()
                time.sleep(INFER_INTERVAL)
        finally:
            stream.stop()
            stream.close()
            print("🔇 Microphone stream stopped.")

    def stop(self):
        self._running = False


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = queue.Queue()
    engine = InferenceEngine(q)

    def monitor():
        while True:
            msg = q.get()
            bar = "█" * int(msg["confidence"] * 20)
            rms_val = msg.get("rms", 0.0)
            print(f"[{msg['label']:10s}] conf={msg['confidence']:.2f}  rms={rms_val:.6f}  {bar}"
                  f"{'  ✅ CONFIRMED' if msg['confirmed'] else ''}")

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    engine.start()

"""
dashboard.py — SpikeVoice OS Neuromorphic Dashboard
====================================================
A customtkinter GUI that visualises the SNN's internals in real-time:
  • Audio Waveform panel    — raw mic buffer
  • Spike Raster Plot       — binary fire events from hidden layer neurons
  • Membrane Voltage Chart  — output neuron potential building toward threshold
  • Command Badge           — flashes with the detected keyword

Threading model:
  - Inference engine runs in a daemon thread → pushes to a queue.Queue
  - Dashboard polls the queue every 100ms via CTk's after() mechanism.
"""

import os
import sys
import queue
import threading
import time
import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import InferenceEngine
from src.actions   import dispatch, is_in_cooldown

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

ACCENT      = "#00e5ff"        # electric cyan
BG_DARK     = "#0d1117"
BG_CARD     = "#161b22"
TEXT_WHITE  = "#e6edf3"
GREEN_SPIKE = "#39ff14"        # neon green for spikes
ORANGE_MEM  = "#ff9500"        # membrane voltage colour
RASTER_NEURONS = 64            # number of neurons to show in raster


class SpikeVoiceDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SpikeVoice OS — Neuromorphic Dashboard")
        self.geometry("1200x750")
        self.configure(fg_color=BG_DARK)
        self.resizable(True, True)

        self.result_queue = queue.Queue(maxsize=5)

        # Internal state buffers
        self._audio_buf     = np.zeros(16000)
        self._raster_buf    = []          # list of spike arrays [neurons]
        self._mem_buf       = []          # list of float values (max output mem)
        self._last_label    = "—"
        self._label_timer   = 0

        self._build_ui()
        self._start_inference()
        self.after(100, self._poll_queue)

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=60)
        header.pack(fill="x", padx=0, pady=0)
        ctk.CTkLabel(header, text="⚡ SpikeVoice OS",
                     font=("Inter", 22, "bold"), text_color=ACCENT).pack(
                         side="left", padx=20, pady=15)
        self.status_label = ctk.CTkLabel(
            header, text="● LISTENING", font=("Inter", 13),
            text_color="#2ea043")
        self.status_label.pack(side="left", padx=10)

        # ── Cooldown indicator ──
        self.cooldown_label = ctk.CTkLabel(
            header, text="", font=("Inter", 12), text_color="#8b949e")
        self.cooldown_label.pack(side="right", padx=20)

        # ── Main layout (left column | right column) ──
        main = ctk.CTkFrame(self, fg_color=BG_DARK)
        main.pack(fill="both", expand=True, padx=15, pady=10)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # ── LEFT TOP: Audio Waveform ──
        wave_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        wave_card.grid(row=0, column=0, padx=(0, 8), pady=(0, 8), sticky="nsew")
        ctk.CTkLabel(wave_card, text="Audio Waveform",
                     font=("Inter", 13, "bold"), text_color=TEXT_WHITE).pack(
                         anchor="w", padx=15, pady=(10, 0))
        self.fig_wave, self.ax_wave = plt.subplots(figsize=(6, 2),
                                                    facecolor=BG_CARD)
        self.ax_wave.set_facecolor(BG_DARK)
        self.ax_wave.tick_params(colors=TEXT_WHITE, labelsize=7)
        for spine in self.ax_wave.spines.values():
            spine.set_edgecolor("#30363d")
        self.wave_line, = self.ax_wave.plot(
            np.zeros(16000), color=ACCENT, linewidth=0.6)
        self.ax_wave.set_ylim(-1, 1)
        self.ax_wave.set_xlim(0, 16000)
        self.ax_wave.set_xlabel("Samples", color=TEXT_WHITE, fontsize=7)
        self.fig_wave.tight_layout(pad=0.5)
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, wave_card)
        self.canvas_wave.get_tk_widget().pack(fill="both", expand=True,
                                              padx=10, pady=5)

        # ── LEFT BOTTOM: Spike Raster ──
        raster_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        raster_card.grid(row=1, column=0, padx=(0, 8), pady=(0, 0), sticky="nsew")
        ctk.CTkLabel(raster_card, text="Spike Raster Plot (Hidden Layer 2)",
                     font=("Inter", 13, "bold"), text_color=TEXT_WHITE).pack(
                         anchor="w", padx=15, pady=(10, 0))
        self.fig_raster, self.ax_raster = plt.subplots(figsize=(6, 2.5),
                                                        facecolor=BG_CARD)
        self.ax_raster.set_facecolor(BG_DARK)
        self.ax_raster.tick_params(colors=TEXT_WHITE, labelsize=7)
        for spine in self.ax_raster.spines.values():
            spine.set_edgecolor("#30363d")
        self.ax_raster.set_xlabel("Time-step", color=TEXT_WHITE, fontsize=7)
        self.ax_raster.set_ylabel("Neuron index", color=TEXT_WHITE, fontsize=7)
        self.ax_raster.set_xlim(0, 25)
        self.ax_raster.set_ylim(0, RASTER_NEURONS)
        self.raster_scatter = self.ax_raster.scatter(
            [], [], s=3, c=GREEN_SPIKE, marker=".", alpha=0.8)
        self.fig_raster.tight_layout(pad=0.5)
        self.canvas_raster = FigureCanvasTkAgg(self.fig_raster, raster_card)
        self.canvas_raster.get_tk_widget().pack(fill="both", expand=True,
                                                padx=10, pady=5)

        # ── RIGHT TOP: Membrane Voltage ──
        mem_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        mem_card.grid(row=0, column=1, padx=(0, 0), pady=(0, 8), sticky="nsew")
        ctk.CTkLabel(mem_card, text="Output Membrane Potential V(t)",
                     font=("Inter", 13, "bold"), text_color=TEXT_WHITE).pack(
                         anchor="w", padx=15, pady=(10, 0))
        self.fig_mem, self.ax_mem = plt.subplots(figsize=(4, 2), facecolor=BG_CARD)
        self.ax_mem.set_facecolor(BG_DARK)
        self.ax_mem.tick_params(colors=TEXT_WHITE, labelsize=7)
        for spine in self.ax_mem.spines.values():
            spine.set_edgecolor("#30363d")
        self.ax_mem.axhline(y=1.0, color="#ff4444", linewidth=0.8,
                            linestyle="--", label="Threshold")
        self.mem_line, = self.ax_mem.plot([], [], color=ORANGE_MEM, linewidth=1.2)
        self.ax_mem.set_ylim(-0.2, 2.5)
        self.ax_mem.set_xlim(0, 25)
        self.ax_mem.set_xlabel("Time-step", color=TEXT_WHITE, fontsize=7)
        self.ax_mem.set_ylabel("V(t)", color=TEXT_WHITE, fontsize=7)
        self.fig_mem.tight_layout(pad=0.5)
        self.canvas_mem = FigureCanvasTkAgg(self.fig_mem, mem_card)
        self.canvas_mem.get_tk_widget().pack(fill="both", expand=True,
                                             padx=10, pady=5)

        # ── RIGHT BOTTOM: Command Badge ──
        badge_card = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        badge_card.grid(row=1, column=1, padx=(0, 0), pady=(0, 0), sticky="nsew")
        ctk.CTkLabel(badge_card, text="Detected Command",
                     font=("Inter", 13, "bold"), text_color=TEXT_WHITE).pack(
                         anchor="w", padx=15, pady=(10, 0))
        self.cmd_label = ctk.CTkLabel(
            badge_card, text="—",
            font=("Inter", 52, "bold"), text_color=ACCENT)
        self.cmd_label.pack(expand=True)
        self.conf_label = ctk.CTkLabel(
            badge_card, text="confidence: —",
            font=("Inter", 12), text_color="#8b949e")
        self.conf_label.pack(pady=(0, 15))

    # ── Inference thread ──────────────────────────────────────────────────────
    def _start_inference(self):
        self.engine = InferenceEngine(self.result_queue)
        t = threading.Thread(target=self.engine.start, daemon=True)
        t.start()

    # ── Queue polling (runs on GUI thread every 100ms) ────────────────────────
    def _poll_queue(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                self._update_ui(msg)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def _update_ui(self, msg: dict):
        label      = msg["label"]
        confidence = msg["confidence"]
        confirmed  = msg["confirmed"]
        audio      = msg["raw_audio"]
        mem_trace  = msg["mem_trace"]
        spike_trace= msg["spike_trace"]

        # ── Dispatch OS action if confirmed ──
        if confirmed:
            fired = dispatch(label)
            if fired:
                self.cmd_label.configure(text_color="#39ff14")
                self.after(600, lambda: self.cmd_label.configure(text_color=ACCENT))

        # ── Waveform ──
        self.wave_line.set_ydata(audio)
        self.canvas_wave.draw_idle()

        # ── Raster ──
        if spike_trace:
            xs, ys = [], []
            for t_idx, spk in enumerate(spike_trace):
                # spk: [1, 1, 32, ...] — flatten to find firing neurons
                flat = spk.view(-1).numpy()
                n    = min(len(flat), RASTER_NEURONS)
                for n_idx in range(n):
                    if flat[n_idx] > 0:
                        xs.append(t_idx)
                        ys.append(n_idx)
            if xs:
                self.raster_scatter.set_offsets(np.column_stack([xs, ys]))
            else:
                self.raster_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.raster_scatter.set_offsets(np.empty((0, 2)))
        self.canvas_raster.draw_idle()

        # ── Membrane voltage ──
        if mem_trace:
            vals = [m.squeeze().max().item() for m in mem_trace]
            self.mem_line.set_data(range(len(vals)), vals)
            self.ax_mem.set_xlim(0, max(len(vals), 1))
        else:
            self.mem_line.set_data(range(25), [0.0]*25)
            self.ax_mem.set_xlim(0, 25)
        self.canvas_mem.draw_idle()

        # ── Command badge ──
        display = label.upper() if label != "silence" else "—"
        self.cmd_label.configure(text=display)
        self.conf_label.configure(
            text=f"confidence: {confidence:.2f}")

        # ── Cooldown indicator ──
        if is_in_cooldown():
            self.cooldown_label.configure(text="⏳ Cooldown active",
                                          text_color="#e3a800")
        else:
            self.cooldown_label.configure(text="✅ Ready",
                                          text_color="#2ea043")

    # ── Clean shutdown ────────────────────────────────────────────────────────
    def on_close(self):
        self.engine.stop()
        self.destroy()


if __name__ == "__main__":
    app = SpikeVoiceDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

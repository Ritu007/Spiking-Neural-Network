# ⚡ SpikeVoice OS: Neuromorphic Voice Control

**SpikeVoice OS** is a real-time, end-to-end application that leverages Spiking Neural Networks (SNNs) for ultra-low-power keyword spotting and OS-level command execution. It bridges the gap between abstract neuromorphic research and practical, "living" software.

---

## 🚀 Mission
To demonstrate that event-driven, brain-inspired computation can provide "always-on" intelligence on commodity hardware with minimal energy consumption and high interpretability.

## ✨ Core Features
- **Real-Time SNN Inference**: Processes 16kHz audio streams through a 3-layer Convolutional SNN.
- **Neuromorphic Dashboard**: Live visualization of hidden-layer spikes (Raster Plots) and output neuron membrane potentials.
- **OS Command Integration**: Automate system volume, mute, and sleep states via voice.
- **Delta-Modulation Encoding**: High-sparsity audio-to-spike conversion for energy efficiency.
- **Explainable AI**: Observe the "tension" build-up in neurons as they recognize your voice.

---

## 🛠️ Quick Start

### 1. Environment Setup
The project leverages PyTorch with GPU/CUDA acceleration. Use the pre-configured **`pyt`** Conda environment which contains CUDA libraries, PyTorch, and all necessary deep learning dependencies.

**Activate `pyt` Environment:**
```powershell
conda activate pyt
```

**Key Dependencies:**
- **SNN Engine**: `snntorch` (v0.9.4+)
- **Deep Learning**: `torch` (v2.1+), `torchaudio`
- **Audio Processing**: `librosa`, `pyaudio`, `sounddevice`
- **GUI & Viz**: `customtkinter`, `matplotlib`
- **Utilities**: `numpy`, `scipy`, `tqdm`, `pyautogui`

### 2. Install Dependencies
```bash
pip install -r "SpikeVoice OS/requirements.txt"
```

### 3. Training the Brain
Before running the OS, you must train the model on the Google Speech Commands dataset.
```powershell
python "SpikeVoice OS/src/train.py"
```
*Note: This will auto-download the dataset and save the best weights to `weights/spikevoice_v1.pt`.*

### 4. Launch SpikeVoice OS
```powershell
python "SpikeVoice OS/main.py"
```

---

## 🧠 Technical Architecture

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Encoder** | Delta Modulation | Converts Mel-spectrogram changes into binary spikes. |
| **Neuron Model** | LIF (Leaky Integrate-and-Fire) | Uses `snnTorch` with learnable leak factors ($\beta$). |
| **Learning Rule** | BPTT + Surrogate Gradients | Fast-sigmoid approximation for non-differentiable spikes. |
| **Frontend** | CustomTkinter + Matplotlib | Dark-mode GUI with real-time matplotlib embedding. |
| **Inference** | Sliding Window (100ms) | 1-second rolling buffer with confidence gating. |

---

## 📂 Repository Structure
- `SpikeVoice OS/main.py`: Main entry point.
- `SpikeVoice OS/src/`: Core logic (encoder, model, train, inference, dashboard).
- `SNN_Master_Notebook.md`: Technical ledger and research breakthroughs.
- `SpikeVoice_OS_Knowledge_Base.md`: Comprehensive documentation of methodology and results.

---

## 🔬 Research Context
SpikeVoice OS specifically addresses the **Temporal Gradient Problem** in SNNs by optimizing surrogate gradient slopes and utilizing hybrid state-space dynamics to reduce approximation errors over long audio sequences.

## 📈 Roadmap
- [ ] Integration of **Spiking Mamba** for even longer context recognition.
- [ ] On-device **STDP adaptation** for user-specific voice fine-tuning.
- [ ] Deployment to **Intel Loihi 2** neuromorphic hardware.

---
*Developed by the SpikeVoice Team (2026)*

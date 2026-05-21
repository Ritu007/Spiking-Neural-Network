# SNN Master Notebook
*This document serves as a persistent ledger of all research insights, video notes, and technical breakthroughs.*

## [2026-05-12] - Project Kickoff & Structure
- **Objective**: Establish a standalone research and implementation pipeline for Spiking Neural Networks.
- **Current Focus**: Identifying modern neuron models and high-impact applications in event-based vision.

---

## Technical Glossary (Initial Entries)
| Term | Definition |
| :--- | :--- |
| **LIF** | Leaky Integrate-and-Fire; the standard neuron model for hardware-friendly SNNs. |
| **STDP** | Spike-Timing-Dependent Plasticity; a biological learning rule based on the temporal order of spikes. |
| **Surrogate Gradient** | A technique to overcome the non-differentiability of the heaviside step function in SNNs during backpropagation. |

---

## Video & Research Log
*New entries will be appended here.*

### [2026-05-13] - SNN Modern Architectures & Frameworks
**Modern Architectures:**
- **Spiking Transformers**: Integration of Transformer architectures into the spiking domain (e.g., Spike-driven Transformers, 2D-Spiking Transformers) to handle complex temporal and spatial data.
- **Hybrid Models**: Combining the accuracy of traditional deep learning with the efficiency of spiking neurons. Often involves training conventional ANN models and converting them into SNNs, or using hybrid training schemes.
- **Deep Spiking Networks**: Constructing deep SNNs from ANN counterparts using knowledge distillation.

**Key Applications:**
- **Computer Vision & Robotics**: Well-suited for edge computing, low-power hardware, and tasks requiring rapid temporal processing (e.g. processing asynchronous data from event-based cameras).
- **Brain-Machine Interfaces (BMIs)**: Decoding neural signals in healthcare, such as EEG-based interfaces, requiring low-latency and high energy efficiency.
- **Edge AI**: Ultra-low-power inference on neuromorphic hardware (e.g., Intel Loihi 2, IBM TrueNorth). 

**Software Ecosystem (2024-2026):**
- **snnTorch**: A PyTorch-native framework that treats spiking neurons as simple activation layers. Highly intuitive for those familiar with `torch.nn`. Supports gradient-based learning via surrogate gradients.
- **SpikingJelly**: A robust PyTorch-based framework optimized for large-scale SNN training, featuring Triton backends and memory-efficient training techniques (e.g., gradient checkpointing).
- **Nengo**: A comprehensive ecosystem for building, testing, and deploying neural networks across various backends, including diverse neuromorphic hardware.
- **Lava / Fugu**: Frameworks gaining attention for composing neural algorithms and managing hardware interoperability on specialized neuromorphic chips.

---

## [2026-05-13] - Technical Kickoff: SpikeVoice OS

**Project**: A real-time, standalone neuromorphic voice-command application powered by a 3-layer Convolutional SNN trained on keyword spotting.

### Keyword Command Set (v1)
| Index | Keyword   | Triggered OS Action            |
| :---- | :-------- | :----------------------------- |
| 0     | `mute`    | Toggle system audio mute       |
| 1     | `up`      | Volume +10%                    |
| 2     | `down`    | Volume -10%                    |
| 3     | `sleep`   | Put system to sleep            |
| 4     | `stop`    | Pause/stop media playback      |
| 5     | `silence` | Negative class — no action     |

### Architecture Summary
- **Neuron Model**: Leaky Integrate-and-Fire (LIF), β=0.9, threshold=1.0.
- **Layers**: 3 × (Conv2d → BatchNorm → LIF → AvgPool) + Linear readout.
- **Training**: BPTT with `snn.surrogate.fast_sigmoid()`, 25 time-steps, Adam lr=1e-3.
- **Inference**: 100ms sliding window, 1-second rolling audio buffer at 16 kHz.
- **Dashboard**: customtkinter GUI with live spike raster plot + membrane voltage gauge.

---

## Encoding Strategies

### Delta-Modulation Encoder (Primary Strategy for SpikeVoice OS)

**Motivation**: SNNs are binary event processors — they cannot directly consume floating-point Mel spectrogram values. The delta-modulator converts the continuously-valued feature map into a sparse binary spike tensor, perfectly matching the event-driven nature of LIF neurons.

**Algorithm**:
For each time-step `t` and frequency bin `f` in the log-Mel spectrogram:

```
s(t, f) = 1   if |I(t, f) - I(t-1, f)| > θ
s(t, f) = 0   otherwise
```

Where:
- `I(t, f)` = log-Mel intensity at time `t`, frequency bin `f`.
- `θ` = threshold (default: `0.1`). Higher values → sparser spikes (more energy efficient). Lower values → denser spikes (more informative).
- Output shape: `[T, 1, 128, 101]` where T = 25 unrolled time-steps.

**Key Properties**:
- Produces a **sparse** binary tensor (typically 5–15% non-zero), ensuring the SNN's energy advantage is preserved.
- Encodes **changes** in audio energy, making it naturally robust to slowly-varying background noise (ambient hum produces very few spikes; transient speech events produce many).
- Threshold `θ` is a tunable hyperparameter that controls the sparsity/accuracy trade-off.

**Alternative Strategies** (for future experiments):
| Strategy         | Description                                           | Best For                   |
| :--------------- | :---------------------------------------------------- | :------------------------- |
| Rate Coding      | Spike probability ∝ feature intensity                 | Dense, high-accuracy tasks |
| Latency Coding   | Earlier spike = stronger signal                       | Ultra-fast, 1-spike tasks  |
| Phase Coding     | Spike phase relative to a global oscillation          | Precise temporal tasks     |
| Delta Modulation | Spike on significant feature *change* (our approach)  | Real-time streaming audio  |

---

## [2026-05-18] - Phase 2: Resolving SNN Underfitting & Dying Spike Collapse
We diagnosed a severe underfitting bottleneck where the validation accuracy collapsed to **26.3%**. A systematic deep telemetry investigation revealed the following critical bugs and architectural bottlenecks:

### 1. The "Dying Spike" Evaluation Phenomenon (Catastrophic Failure)
- **Problem**: Layer-by-layer spike analysis revealed that when running `model.eval()`, spike rates in Layer 3 collapsed to exactly **0.00%**.
- **Cause**: PyTorch's `BatchNorm1d` tracks running statistics across batches during training, but in evaluation mode, it freezes these stats. For SNNs, which feature sparse binary activations, these running statistics completely drift, causing the normalized potentials to fall far below the LIF firing threshold.
- **Fix**: Replaced all `BatchNorm1d` layers with **GroupNorm** (per-sample normalization). Because GroupNorm operates per-sample on the spatial/frequency dimension, it does not rely on cross-batch history and maintains a stable **~25% spike rate** across all SNN layers in both training and evaluation modes.

### 2. Chaotic Temporal Dropout (Unstable Gradients)
- **Problem**: Temporal sequence integration was corrupted.
- **Cause**: Using `nn.Dropout1d` inside the temporal unrolling loop (`for t in range(T)`) meant that *different frequency channels were dropped out at different time steps*. For stateful LIF neurons, this randomized spatial masking acted as severe high-frequency noise, preventing stable temporal membrane potential accumulation.
- **Fix**: Removed all dropout layers entirely, relying on GroupNorm for regularisation.

### 3. Encoder Upgrades: 2-Channel ON/OFF Delta Modulation (Spectacular Breakthrough)
- **Problem**: The absolute difference delta encoder `|I(t) - I(t-1)| > 0.1` destroyed the polarity of audio energy changes (increasing attack vs. decaying release looked identical).
- **Fix**: Upgraded the encoder to output two channels:
  - **ON Channel**: `diff > 0.05` (intensity increased)
  - **OFF Channel**: `diff < -0.05` (intensity decreased)
- **Hop Adjustment**: Switched `hop_length` to 640. At 16kHz, this natively generates exactly `T=25` frames per 1 second of audio, eliminating lossy temporal max-pooling.
- **Impact**: Validation accuracy soared from **26% to 75.6%**!

---

## [2026-05-19] - Phase 3: Accuracy Maximization (83.9% Breakthrough)
To push validation accuracy closer to the 90% target, we addressed the remaining underfitting bottlenecks.

### 1. Wider Spectral Receptive Field
- **Problem**: Mel spectrogram speech features are broad, but `kernel_size=3` inside `conv1` only captured a tiny 2% frequency band (3 bins out of 128).
- **Fix**: Upgraded `conv1` to `kernel_size=5` (with `padding=2`), widening the receptive field so the first layer can extract complex spectro-temporal phoneme tracks.

### 2. Training Convergence & Augmented Underfitting Resolution
- **Problem**: The model's training accuracy (70%) lagged behind validation (75%), showing it was heavily underfitting the complex augmented training samples (which feature random background noise and shifts).
- **Fix**: Scaled training epochs from **30 to 80** and adjusted the Cosine Annealing scheduler.
- **Impact**: Validation accuracy surged to an outstanding **83.93%** (Best: **83.93%**), under full data augmentation and background noise!


"""
encoder.py — 2-Channel ON/OFF Delta-Modulation Audio Encoder for SpikeVoice OS
=============================================================
Converts raw 16kHz audio frames into binary spike tensors using
a delta-modulation scheme on Log-Mel Spectrogram features.

Encoding rule:
    ON  spikes: s+(t, f) = 1 if I(t, f) − I(t−1, f) > θ
    OFF spikes: s-(t, f) = 1 if I(t, f) − I(t−1, f) < -θ

Output shape: [T, 2, n_mels]   (T = num_steps, 2 = ON/OFF channels)
"""

import numpy as np
import librosa
import torch

# ── Configuration ────────────────────────────────────────────────────────────
NUM_STEPS     = 25              # Target number of time steps for SNN
TARGET_SR     = 16000           # 16kHz
SAMPLE_RATE   = TARGET_SR       # Alias for backward compatibility
N_MELS        = 128             # Frequency bands
HOP_LENGTH    = 640             # 40ms hop -> exactly 25 frames for 1s audio
WIN_LENGTH    = 640             # 40ms window
F_MIN         = 20
F_MAX         = 8000
DELTA_THRESH  = 0.05            # Lowered threshold for ON/OFF spikes


def audio_to_mel(audio_np: np.ndarray) -> np.ndarray:
    """Converts 1D audio waveform into a log-Mel spectrogram."""
    # Ensure audio is exactly 1 second (16000 samples) to avoid librosa issues
    if len(audio_np) < TARGET_SR:
        audio_np = np.pad(audio_np, (0, TARGET_SR - len(audio_np)))
    else:
        audio_np = audio_np[:TARGET_SR]

    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=TARGET_SR,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Map from [-80dB, 0dB] to [0.0, 1.0]
    log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
    
    return log_mel


def delta_encode(log_mel: np.ndarray) -> np.ndarray:
    """
    Applies 2-Channel Delta Modulation to the log-Mel spectrogram.
    Returns ON spikes (volume increased) and OFF spikes (volume decreased).
    """
    diff = np.diff(log_mel, axis=1, prepend=log_mel[:, :1])
    
    on_spikes = (diff > DELTA_THRESH).astype(np.float32)
    off_spikes = (diff < -DELTA_THRESH).astype(np.float32)
    
    # Combine into 2 channels: [2, N_MELS, n_frames]
    binary = np.stack([on_spikes, off_spikes], axis=0)
    return binary


def encode_audio(audio_np: np.ndarray, num_steps: int = NUM_STEPS) -> torch.Tensor:
    """
    Full pipeline: Audio -> Mel -> ON/OFF Spikes -> Tensor.
    Output Shape: [T, 2, N_MELS]
    """
    log_mel = audio_to_mel(audio_np)
    binary = delta_encode(log_mel)
    
    # Ensure exactly `num_steps` frames
    n_channels, n_mels, n_frames = binary.shape
    if n_frames < num_steps:
        binary = np.pad(binary, ((0, 0), (0, 0), (0, num_steps - n_frames)))
    else:
        binary = binary[:, :, :num_steps]
        
    # Transpose to [T, Channels, Mels] -> [25, 2, 128]
    binary = binary.transpose(2, 0, 1)
    
    spikes = torch.from_numpy(binary)
    return spikes


# ── Quick sanity-check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_audio = np.random.randn(TARGET_SR).astype(np.float32)
    spikes      = encode_audio(dummy_audio)
    sparsity    = spikes.mean().item()

    print(f"Spike tensor shape : {tuple(spikes.shape)}")
    print(f"Dtype              : {spikes.dtype}")
    print(f"Sparsity (mean)    : {sparsity:.4f}  ({sparsity*100:.1f}% active)")
    print(f"Values unique      : {spikes.unique().tolist()}")
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0}), "Non-binary values found!"
    print("[PASS] Encoder sanity check passed.")

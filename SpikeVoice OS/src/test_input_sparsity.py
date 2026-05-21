import os
import sys
import torch
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import KeywordDataset

def encode_audio_original(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=1024, hop_length=160, win_length=640, n_mels=128, fmin=20, fmax=8000
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)
    
    delta = np.abs(np.diff(log_mel, axis=1, prepend=log_mel[:, :1]))
    binary = (delta > 0.1).astype(np.float32)
    
    if binary.shape[1] < 100:
        binary = np.pad(binary, ((0, 0), (0, 100 - binary.shape[1])))
    else:
        binary = binary[:, :100]
        
    binary = binary.reshape(128, 25, 4)
    binary = binary.transpose(1, 0, 2)
    spikes = torch.from_numpy(binary[:, np.newaxis, :, :])
    return spikes

class OriginalDataset(KeywordDataset):
    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[self.indices[idx]]
        audio = waveform.squeeze(0).numpy()
        spikes = encode_audio_original(audio)
        class_id = self.LABEL_MAP.get(label, 5)
        return spikes, class_id

dataset = OriginalDataset(subset="training")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
spikes, labels = next(iter(loader))

print(f"Spikes shape: {spikes.shape}")
print(f"Sparsity: {spikes.mean().item() * 100:.2f}%")

# Let's check how many samples are COMPLETELY empty
empty_samples = 0
for i in range(32):
    if spikes[i].sum() == 0:
        empty_samples += 1
print(f"Empty samples in batch: {empty_samples}/32")

# Let's check sparsity per class
for c in range(6):
    mask = labels == c
    if mask.sum() > 0:
        print(f"Class {c} sparsity: {spikes[mask].mean().item() * 100:.2f}%")

"""
train.py — Training Pipeline for SpikeVoice OS
================================================
Uses Google Speech Commands dataset (subset of 6 keywords) to train
the 3-layer Conv-SNN via BPTT with fast_sigmoid surrogate gradients.

Usage:
    python src/train.py

Downloads the dataset automatically via torchaudio if not present.
Saves weights to:  weights/spikevoice_v1.pt
"""

import os
print("SCRIPT STARTING...")
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchaudio
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoder import encode_audio, TARGET_SR
from src.model   import SpikeVoiceModel

# ── Config ────────────────────────────────────────────────────────────────────
KEYWORDS      = ["mute", "up", "down", "stop", "bed"]  # 5 real + silence=background
NUM_CLASSES   = 6                 # 5 keywords + silence
BATCH_SIZE    = 64
EPOCHS        = 80
LR            = 1e-3
DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "data", "speech_commands")
WEIGHTS_PATH  = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "weights", "spikevoice_v1.pt")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROGRESS_LOG  = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "training_progress.json")

def log_progress(data):
    with open(PROGRESS_LOG, "w") as f:
        json.dump(data, f)

# ── Dataset wrapper ───────────────────────────────────────────────────────────
class KeywordDataset(torch.utils.data.Dataset):
    """
    Wraps torchaudio.datasets.SPEECHCOMMANDS.
    Filters to KEYWORDS + '_silence_' class and returns spike tensors.
    """
    LABEL_MAP = {kw: i for i, kw in enumerate(KEYWORDS)}
    LABEL_MAP["_silence_"] = len(KEYWORDS)          # class 5 = silence
    LABEL_MAP["_unknown_"] = len(KEYWORDS)           # map unknowns to silence

    def __init__(self, subset: str = "training"):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=DATA_DIR, url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=True, subset=subset
        )
        self.subset = subset
        self.indices = []
        
        # Load background noise for augmentation
        self.bg_noise = []
        bg_dir = os.path.join(DATA_DIR, "SpeechCommands", "speech_commands_v0.02", "_background_noise_")
        if subset == "training" and os.path.exists(bg_dir):
            for file in os.listdir(bg_dir):
                if file.endswith(".wav"):
                    noise_path = os.path.join(bg_dir, file)
                    noise, _ = torchaudio.load(noise_path)
                    if noise.shape[1] > TARGET_SR:
                        self.bg_noise.append(noise.squeeze(0).numpy())

        print(f"Filtering {len(self.dataset)} samples for subset '{subset}'...")
        # Collect unknown samples to balance the dataset
        unknown_pool = []
        for i, path in enumerate(self.dataset._walker):
            label = os.path.basename(os.path.dirname(path))
            if label in self.LABEL_MAP and label != "_unknown_":
                self.indices.append(i)
            else:
                unknown_pool.append(i)
                
        # Randomly sample unknowns to match roughly 10% of the dataset
        target_unknowns = int(len(self.indices) * 0.1)
        if target_unknowns > 0 and unknown_pool:
            np.random.seed(42 if subset != "training" else None)
            sampled_unknowns = np.random.choice(unknown_pool, size=min(target_unknowns, len(unknown_pool)), replace=False)
            self.indices.extend(sampled_unknowns.tolist())
            
        print(f"  Done. Kept {len(self.indices)} samples (incl. unknowns).")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[self.indices[idx]]
        # Resample if needed
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        
        audio = waveform.squeeze(0).numpy()          # [16000]
        
        # Data Augmentation (only during training)
        if self.subset == "training":
            # 1. Random Gain (± 20%)
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
            
            # 2. Add Background Noise (50% chance)
            if self.bg_noise and np.random.rand() > 0.5:
                noise_idx = np.random.randint(0, len(self.bg_noise))
                noise_src = self.bg_noise[noise_idx]
                if len(noise_src) > len(audio):
                    start = np.random.randint(0, len(noise_src) - len(audio))
                    noise_crop = noise_src[start:start+len(audio)]
                    # Mix with a random signal-to-noise ratio
                    noise_level = np.random.uniform(0.01, 0.1)
                    audio = audio + noise_crop * noise_level
            
            # 3. Random Time Shift (± 10%)
            shift_max = int(TARGET_SR * 0.1)
            shift = np.random.randint(-shift_max, shift_max)
            if shift > 0:
                audio = np.pad(audio, (shift, 0))[:-shift]
            elif shift < 0:
                audio = np.pad(audio, (0, -shift))[-shift:]

        # Ensure no clipping
        audio = np.clip(audio, -1.0, 1.0)
        
        spikes   = encode_audio(audio)                  # [T, 2, 128]
        class_id = self.LABEL_MAP.get(label, len(KEYWORDS))
        return spikes, class_id


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")
    print("Loading datasets …")
    train_ds = KeywordDataset("training")
    val_ds   = KeywordDataset("validation")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    model     = SpikeVoiceModel(num_classes=NUM_CLASSES).to(DEVICE)
    best_acc  = 0.0
    start_epoch = 1

    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading existing weights from {WEIGHTS_PATH}...")
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        best_acc = checkpoint.get("val_acc", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch} (best_acc={best_acc:.3f})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for spikes, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            # spikes: [B, T, 2, 128]  — DataLoader adds batch dim at front
            # Permute to [T, B, 2, 128] for the SNN forward loop
            spikes = spikes.permute(1, 0, 2, 3).to(DEVICE, dtype=torch.float32)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            counts, _, _ = model(spikes)     # [B, num_classes]
            loss = criterion(counts, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (counts.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for spikes, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val] "):
                spikes = spikes.permute(1, 0, 2, 3).to(DEVICE, dtype=torch.float32)
                labels = labels.to(DEVICE)
                counts, _, _ = model(spikes)
                val_correct += (counts.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        print(f"  Loss: {total_loss/total:.4f}  |  Train Acc: {train_acc:.3f}"
              f"  |  Val Acc: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "keywords":    KEYWORDS,
                        "val_acc":     val_acc,
                        "epoch":       epoch},
                       WEIGHTS_PATH)
            print(f"  [SAVED] Best model -> {WEIGHTS_PATH}  (val_acc={val_acc:.3f})")

        # Log progress to file for IDE visibility
        log_progress({
            "epoch": epoch,
            "total_epochs": EPOCHS,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "loss": total_loss / total,
            "best_acc": best_acc
        })

    print(f"\n[DONE] Training complete. Best validation accuracy: {best_acc:.3f}")
    if best_acc < 0.90:
        print("[WARNING] Target accuracy (<90%) not yet reached. "
              "Consider training more epochs or tuning hyperparameters.")


if __name__ == "__main__":
    train()

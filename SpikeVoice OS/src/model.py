"""
model.py — Spiking Neural Network Architecture for SpikeVoice OS
==============================================================
A 3-layer Convolutional Spiking Neural Network (CSNN) built with snnTorch.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ── Hyperparameters ───────────────────────────────────────────────────────────
BETA      = 0.9     # LIF leak factor
THRESHOLD = 0.8     # Spike threshold voltage


class SpikeVoiceModel(nn.Module):
    def __init__(self, num_classes: int = 6,
                 beta: float = BETA, threshold: float = THRESHOLD):
        super().__init__()
        
        # Surrogate gradient for BPTT through discrete spikes
        spike_grad = surrogate.fast_sigmoid(slope=10)

        # ── Convolutional feature extractor ──────────────────────────────────
        # Block 1 (Input channels = 2 for ON/OFF Delta Encoder)
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, padding=2, bias=False)
        self.gn1   = nn.GroupNorm(8, 32)
        self.lif1  = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=True)
        self.pool1 = nn.AvgPool1d(2)

        # Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(16, 64)
        self.lif2  = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=True)
        self.pool2 = nn.AvgPool1d(2)

        # Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.gn3   = nn.GroupNorm(32, 128)
        self.lif3  = snn.Leaky(beta=beta, threshold=threshold,
                               spike_grad=spike_grad, learn_beta=True)
        self.gap   = nn.AdaptiveAvgPool1d(1)     # global average → [B, 128, 1]

        # ── Classifier ────────────────────────────────────────────────────────
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Spike tensor of shape [T, B, 2, n_mels]
        Returns:
            spike_counts: Tensor of shape [B, num_classes] (Sum of logits over time)
            mem_rec:      Membrane potentials of Layer 3 over time-steps
            hidden_spikes:Spikes from Layer 2 over time-steps
        """
        T = x.size(0)

        # Initialize hidden states (membrane potentials)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spike_counts = None
        mem_rec = []
        hidden_spikes = []

        for t in range(T):
            xt = x[t]            # [B, 2, n_mels]

            # Block 1
            out = self.pool1(self.gn1(self.conv1(xt)))
            spk1, mem1 = self.lif1(out, mem1)

            # Block 2
            out = self.pool2(self.gn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(out, mem2)

            # Block 3
            out = self.gap(self.gn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(out, mem3)

            # Readout - We use the robust Summed Logits approach
            flat  = spk3.view(spk3.size(0), -1)     # [B, 128]
            logit = self.fc(flat)                   # [B, num_classes]
            
            if spike_counts is None:
                spike_counts = logit
            else:
                spike_counts = spike_counts + logit

            # Collect traces for real-time visualization (move to CPU to save memory/avoid device errors in dashboard)
            mem_rec.append(mem3.detach().cpu())
            hidden_spikes.append(spk2.detach().cpu())

        # Return format matching train.py expectations
        return spike_counts, mem_rec, hidden_spikes


if __name__ == "__main__":
    # Test forward pass with the new 2-channel encoder shape
    dummy_input = torch.rand(25, 4, 2, 128) # [T, B, C, F]
    model = SpikeVoiceModel()
    counts, _, _ = model(dummy_input)

    print(model)
    print(f"\nOutput shape : {counts.shape}")
    print("[PASS] Model forward pass OK.")

import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import KeywordDataset

class NoBNModel(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, use_gn=False):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=10)
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(8, 32) if use_gn else nn.Identity()
        self.lif1  = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, learn_beta=True)
        self.pool1 = nn.AvgPool1d(2)
        self.drop1 = nn.Dropout1d(0.2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(16, 64) if use_gn else nn.Identity()
        self.lif2  = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, learn_beta=True)
        self.pool2 = nn.AvgPool1d(2)
        self.drop2 = nn.Dropout1d(0.2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=True)
        self.gn3 = nn.GroupNorm(32, 128) if use_gn else nn.Identity()
        self.lif3  = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, learn_beta=True)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.drop3 = nn.Dropout1d(0.2)

    def forward(self, x):
        T = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk1_rates = []
        spk2_rates = []
        spk3_rates = []
        
        for t in range(T):
            xt = x[t]
            out = self.drop1(self.pool1(self.gn1(self.conv1(xt))))
            spk1, mem1 = self.lif1(out, mem1)
            spk1_rates.append(spk1.mean().item())
            
            out = self.drop2(self.pool2(self.gn2(self.conv2(spk1))))
            spk2, mem2 = self.lif2(out, mem2)
            spk2_rates.append(spk2.mean().item())
            
            out = self.drop3(self.gap(self.gn3(self.conv3(spk2))))
            spk3, mem3 = self.lif3(out, mem3)
            spk3_rates.append(spk3.mean().item())
            
        return (
            sum(spk1_rates)/len(spk1_rates),
            sum(spk2_rates)/len(spk2_rates),
            sum(spk3_rates)/len(spk3_rates)
        )

def run_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KeywordDataset(subset="validation")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    spikes, _ = next(iter(loader))
    spikes = spikes.permute(1, 0, 2, 3).to(device, dtype=torch.float32)
    
    print("--- NO BN, NO GN ---")
    for thresh in [1.0, 0.5, 0.3]:
        model = NoBNModel(threshold=thresh, use_gn=False).to(device)
        model.eval()
        with torch.no_grad():
            r1, r2, r3 = model(spikes)
        print(f"Thresh: {thresh:.2f} -> L1: {r1*100:.2f}%, L2: {r2*100:.2f}%, L3: {r3*100:.2f}%")
        
    print("\n--- WITH GROUP NORM ---")
    for thresh in [1.0, 0.5, 0.3]:
        model = NoBNModel(threshold=thresh, use_gn=True).to(device)
        model.eval()
        with torch.no_grad():
            r1, r2, r3 = model(spikes)
        print(f"Thresh: {thresh:.2f} -> L1: {r1*100:.2f}%, L2: {r2*100:.2f}%, L3: {r3*100:.2f}%")

if __name__ == "__main__":
    run_tests()

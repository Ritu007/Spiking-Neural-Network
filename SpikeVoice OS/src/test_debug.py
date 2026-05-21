import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import KeywordDataset

class DebugModel(nn.Module):
    def __init__(self, beta=0.9, threshold=0.8):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=10)
        
        # Reduced capacity just to see if it learns AT ALL
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(8, 16)
        self.lif1  = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.pool1 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(16, 32)
        self.lif2  = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.pool2 = nn.AvgPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False)
        self.gn3   = nn.GroupNorm(32, 64)
        self.lif3  = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.gap   = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        T = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spike_counts = None
        
        for t in range(T):
            xt = x[t]
            out = self.pool1(self.gn1(self.conv1(xt)))
            spk1, mem1 = self.lif1(out, mem1)
            
            out = self.pool2(self.gn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(out, mem2)
            
            out = self.gap(self.gn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(out, mem3)
            
            flat = spk3.view(spk3.size(0), -1)
            logit = self.fc(flat)
            
            if spike_counts is None:
                spike_counts = logit
            else:
                spike_counts = spike_counts + logit
                
        return spike_counts

def test_overfit_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KeywordDataset(subset="training")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Take ONE batch
    spikes, labels = next(iter(loader))
    spikes = spikes.permute(1, 0, 2, 3).to(device, dtype=torch.float32)
    labels = labels.to(device)
    
    model = DebugModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Trying to overfit 1 batch (size 32) without Dropout, using Summed Logits readout...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        counts = model(spikes)
        loss = criterion(counts, labels)
        loss.backward()
        optimizer.step()
        
        acc = (counts.argmax(1) == labels).float().mean().item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {acc*100:.1f}%")
            
    print(f"Final: Loss = {loss.item():.4f}, Acc = {acc*100:.1f}%")

if __name__ == "__main__":
    test_overfit_batch()

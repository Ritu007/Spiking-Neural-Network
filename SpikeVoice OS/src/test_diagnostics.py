import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoder import encode_audio
from src.model import SpikeVoiceModel
from src.train import KeywordDataset

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running diagnostics on {device}...")

    # Load dataset and get one batch
    dataset = KeywordDataset(subset="validation")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    spikes, labels = next(iter(loader))
    spikes = spikes.permute(1, 0, 2, 3).to(device, dtype=torch.float32)
    
    # Load model
    model = SpikeVoiceModel(num_classes=6).to(device)
    model.eval()
    
    # Forward pass manually to inspect intermediate states
    T = spikes.size(0)
    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()
    mem3 = model.lif3.init_leaky()
    
    spk1_rates = []
    spk2_rates = []
    spk3_rates = []
    
    with torch.no_grad():
        for t in range(T):
            xt = spikes[t]
            
            # Block 1
            out = model.pool1(model.gn1(model.conv1(xt)))
            spk1, mem1 = model.lif1(out, mem1)
            spk1_rates.append(spk1.mean().item())
            
            # Block 2
            out = model.pool2(model.gn2(model.conv2(spk1)))
            spk2, mem2 = model.lif2(out, mem2)
            spk2_rates.append(spk2.mean().item())
            
            # Block 3
            out = model.gap(model.gn3(model.conv3(spk2)))
            spk3, mem3 = model.lif3(out, mem3)
            spk3_rates.append(spk3.mean().item())
            
    print(f"Layer 1 Spike Rate: {sum(spk1_rates)/len(spk1_rates):.6f}")
    print(f"Layer 2 Spike Rate: {sum(spk2_rates)/len(spk2_rates):.6f}")
    print(f"Layer 3 Spike Rate: {sum(spk3_rates)/len(spk3_rates):.6f}")

if __name__ == "__main__":
    run_diagnostics()

import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import KeywordDataset
from test_hparams import DiagnosticModel

def run_eval_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KeywordDataset(subset="validation")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    spikes, _ = next(iter(loader))
    spikes = spikes.permute(1, 0, 2, 3).to(device, dtype=torch.float32)
    
    configs = [
        {"beta": 0.9, "threshold": 1.0},
        {"beta": 0.9, "threshold": 0.5},
        {"beta": 0.9, "threshold": 0.3},
        {"beta": 0.95, "threshold": 0.5},
        {"beta": 0.95, "threshold": 0.3},
    ]
    
    for cfg in configs:
        model = DiagnosticModel(beta=cfg["beta"], threshold=cfg["threshold"]).to(device)
        model.eval() # Set to eval mode!
        with torch.no_grad():
            r1, r2, r3 = model(spikes)
        print(f"EVAL MODE - Beta: {cfg['beta']:.2f}, Thresh: {cfg['threshold']:.2f} -> "
              f"L1: {r1*100:.2f}%, L2: {r2*100:.2f}%, L3: {r3*100:.2f}%")

if __name__ == "__main__":
    run_eval_tests()

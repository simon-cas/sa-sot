"""Test SA-SOT model forward pass"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.sasot_model import SASOTModel


def test_sasot_model():
    B, T, F = 2, 200, 80
    N = 20
    V = 3730

    feats = torch.randn(B, T, F)
    feat_lens = torch.tensor([200, 180])
    prev_tokens = torch.randint(0, V, (B, N))

    model = SASOTModel(vocab_size=V)
    out = model(
        feats,
        feat_lens,
        prev_tokens,
        target_len=torch.tensor([N, N]),
        return_intermediate=True
    )

    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"{k}: {v.shape}")

    print("✓ SA-SOT model test passed")


if __name__ == "__main__":
    test_sasot_model()


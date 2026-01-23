"""Test CIF (Continuous Integrate-and-Fire) module"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.encoder.cif import CIF


def test_cif():
    B, T, D = 1, 50, 256
    h = torch.randn(B, T, D, requires_grad=True)
    pre_alpha = torch.randn(B, T, requires_grad=True)
    alpha = torch.sigmoid(pre_alpha)
    target_len = torch.tensor([10])

    cif = CIF()
    c, token_lens, boundaries = cif(h, alpha, target_len)

    print(f"Token lengths: {token_lens}")
    print("Boundaries:")
    for i, (s, e) in enumerate(boundaries[0]):
        print(f"  token {i}: frames [{s}, {e})")

    loss = c.sum()
    loss.backward()
    print(f"Gradient check: {pre_alpha.grad.abs().mean().item():.6f}")
    assert pre_alpha.grad is not None, "Gradient should not be None"
    print("✓ CIF test passed")


if __name__ == "__main__":
    test_cif()


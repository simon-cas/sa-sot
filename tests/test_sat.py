"""Test SAT (Speaker-Aware Training) loss"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from loss.asr_loss import masked_ce_loss


def test_sat_loss():
    B, N, V = 1, 6, 100

    # Fake decoder output
    logits = torch.randn(B, N, V, requires_grad=True)

    # t-SOT tokens
    tokens = torch.tensor([[10, 20, 3, 40, 50, 3]])  # 3 = <cc>
    loss_mask = torch.tensor([[1, 1, 0, 0, 1, 0]]).bool()

    loss = masked_ce_loss(logits, tokens, loss_mask)
    loss.backward()

    print(f"SAT loss: {loss.item():.6f}")
    print(f"Gradient check: {logits.grad.abs().mean().item():.6f}")
    assert logits.grad is not None, "Gradient should not be None"
    print("✓ SAT loss test passed")


if __name__ == "__main__":
    test_sat_loss()


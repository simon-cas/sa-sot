"""Test speaker feature pooling"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.encoder.cif import CIF
from model.sasot_model import SASOTModel


def test_speaker_pooling():
    B, T, D = 1, 50, 256

    # Fake encoder output
    h = torch.randn(B, T, D, requires_grad=True)
    pre_alpha = torch.randn(B, T, requires_grad=True)
    alpha = torch.sigmoid(pre_alpha)
    target_len = torch.tensor([10])

    # CIF
    cif = CIF()
    c, token_lens, boundaries = cif(h, alpha, target_len)

    # Use SASOTModel's boundary pooling
    model = SASOTModel(vocab_size=100)
    spk_feat = torch.randn(B, T, 256)
    token_spk = model._boundary_pooling(spk_feat, boundaries)

    print(f"Token speaker features shape: {token_spk.shape}")

    loss = token_spk.sum()
    loss.backward()

    print(f"Gradient check: {h.grad.abs().mean().item():.6f}")
    assert h.grad is not None, "Gradient should not be None"
    print("✓ Speaker pooling test passed")


if __name__ == "__main__":
    test_speaker_pooling()


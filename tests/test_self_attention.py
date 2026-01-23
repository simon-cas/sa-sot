"""Test self-attention decoder"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.encoder.cif import CIF
from model.sasot_model import SASOTModel
from model.decoder.asr_decoder import ASRDecoder


def test_self_attention():
    B, T, D = 1, 50, 256
    V = 100

    h = torch.randn(B, T, D, requires_grad=True)
    pre_alpha = torch.randn(B, T, requires_grad=True)
    alpha = torch.sigmoid(pre_alpha)
    target_len = torch.tensor([10])

    # CIF
    cif = CIF()
    c, token_lens, boundaries = cif(h, alpha, target_len)

    # Use SASOTModel's boundary pooling
    model = SASOTModel(vocab_size=V)
    spk_feat = torch.randn(B, T, D)
    token_spk = model._boundary_pooling(spk_feat, boundaries)

    # Fake text input
    prev_tokens = torch.randint(0, V, (B, token_lens.item()))

    # ASR decoder
    decoder = ASRDecoder(vocab_size=V, d_model=D, spk_dim=D)
    logits = decoder(c, prev_tokens, token_spk)

    print(f"Logits shape: {logits.shape}")

    loss = logits.sum()
    loss.backward()

    print(f"Gradient check: {h.grad.abs().mean().item():.6f}")
    assert h.grad is not None, "Gradient should not be None"
    print("✓ Self-attention decoder test passed")


if __name__ == "__main__":
    test_self_attention()


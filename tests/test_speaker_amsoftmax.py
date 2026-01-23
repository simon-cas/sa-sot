"""Test speaker AM-Softmax loss"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.encoder.cif import CIF
from model.sasot_model import SASOTModel
from loss.speaker_loss import SpeakerAMSoftmax


def test_speaker_amsoftmax():
    B, T, D = 2, 60, 256
    num_spk = 5

    h = torch.randn(B, T, D, requires_grad=True)
    pre_alpha = torch.randn(B, T, requires_grad=True)
    alpha = torch.sigmoid(pre_alpha)
    target_len = torch.tensor([10, 12])
    speaker_ids = torch.tensor([1, 3])

    # CIF
    cif = CIF()
    _, token_lens, boundaries = cif(h, alpha, target_len)

    # Use SASOTModel's boundary pooling method
    model = SASOTModel(vocab_size=100, num_speakers=num_spk)
    
    # Create dummy speaker features
    spk_feat = torch.randn(B, T, D)
    token_spk = model._boundary_pooling(spk_feat, boundaries)

    # AM-Softmax loss
    spk_loss_fn = SpeakerAMSoftmax(embed_dim=D, num_speakers=num_spk)
    
    # Expand speaker_ids to match token_spk shape
    expanded_speaker_ids = torch.zeros(B, token_spk.size(1), dtype=torch.long)
    for b in range(B):
        expanded_speaker_ids[b, :token_lens[b]] = speaker_ids[b]
    
    loss = spk_loss_fn(token_spk, expanded_speaker_ids)

    loss.backward()
    print(f"AM-Softmax speaker loss: {loss.item():.6f}")
    print(f"Gradient check: {h.grad.abs().mean().item():.6f}")
    assert h.grad is not None, "Gradient should not be None"
    print("✓ Speaker AM-Softmax test passed")


if __name__ == "__main__":
    test_speaker_amsoftmax()


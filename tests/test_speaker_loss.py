"""Test speaker classification loss"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.encoder.cif import CIF
from model.sasot_model import SASOTModel
from model.decoder.speaker_decoder import SpeakerDecoder
from torch.nn.functional import cross_entropy


def test_speaker_loss():
    B, T, D = 1, 50, 256
    S = 10

    h = torch.randn(B, T, D, requires_grad=True)
    pre_alpha = torch.randn(B, T, requires_grad=True)
    alpha = torch.sigmoid(pre_alpha)
    target_len = torch.tensor([10])
    speaker_ids = torch.tensor([3])

    # CIF
    cif = CIF()
    c, token_lens, boundaries = cif(h, alpha, target_len)

    # Use SASOTModel's boundary pooling
    model = SASOTModel(vocab_size=100, num_speakers=S)
    spk_feat = torch.randn(B, T, D)
    token_spk = model._boundary_pooling(spk_feat, boundaries)

    # Speaker classifier
    spk_decoder = SpeakerDecoder(input_dim=D, num_speakers=S)
    logits = spk_decoder(token_spk)

    loss = cross_entropy(
        logits.view(-1, S),
        speaker_ids.unsqueeze(1).expand(B, token_lens.item()).reshape(-1)
    )

    loss.backward()
    print(f"Speaker loss: {loss.item():.6f}")
    print(f"Gradient check: {h.grad.abs().mean().item():.6f}")
    assert h.grad is not None, "Gradient should not be None"
    print("✓ Speaker loss test passed")


if __name__ == "__main__":
    test_speaker_loss()


"""Test ASR decoder forward pass"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.decoder.asr_decoder import ASRDecoder


def test_decoder_forward():
    B, N, D = 2, 10, 512
    V = 100

    c = torch.randn(B, N, D)
    prev_tokens = torch.randint(0, V, (B, N))
    speaker_embed = torch.randn(B, N, D)

    decoder = ASRDecoder(V)
    logits = decoder(c, prev_tokens, speaker_embed)

    print(f"Logits shape: {logits.shape}")  # (B,N,V)
    assert logits.shape == (B, N, V), f"Expected shape ({B}, {N}, {V}), got {logits.shape}"
    
    logits.sum().backward()
    print("✓ Decoder forward test passed")


if __name__ == "__main__":
    test_decoder_forward()


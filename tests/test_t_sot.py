"""Test t-SOT (tokenized Serialized Output Training) label building"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from loss.asr_loss import masked_ce_loss


def test_t_sot():
    spk_segments = {
        0: [10, 11, 12],
        1: [20, 21]
    }

    D = 256
    spk_embeds = torch.randn(2, D)  # One embedding per speaker
    target_spk = 0
    target_embed = spk_embeds[target_spk]

    sims = F.cosine_similarity(
        spk_embeds,
        target_embed.unsqueeze(0),
        dim=1
    )

    order = torch.argsort(sims, descending=True)
    print(f"Speaker order: {order.tolist()}")

    CC_ID = 3  # <cc>

    def build_t_sot(speaker_segments, order, cc_id):
        tokens = []
        speaker_ids = []

        for i, spk in enumerate(order.tolist()):
            if i > 0:
                tokens.append(cc_id)
                speaker_ids.append(-1)

            for t in speaker_segments[spk]:
                tokens.append(t)
                speaker_ids.append(spk)

        return tokens, speaker_ids

    tokens, speaker_ids = build_t_sot(spk_segments, order, CC_ID)
    print(f"t-SOT tokens: {tokens}")
    print(f"t-SOT speaker ids: {speaker_ids}")

    MASK_ID = 0  # <mask>

    def build_masked_t_sot(tokens, speaker_ids, target_spk, mask_id, cc_id):
        masked_tokens = []
        loss_mask = []

        for t, s in zip(tokens, speaker_ids):
            if t == cc_id:
                masked_tokens.append(t)
                loss_mask.append(False)
            elif s == target_spk:
                masked_tokens.append(t)
                loss_mask.append(True)
            else:
                masked_tokens.append(mask_id)
                loss_mask.append(False)

        return masked_tokens, torch.tensor(loss_mask).bool()

    masked_tokens, loss_mask = build_masked_t_sot(
        tokens, speaker_ids, target_spk, MASK_ID, CC_ID
    )

    print(f"Masked tokens: {masked_tokens}")
    print(f"Loss mask: {loss_mask.tolist()}")

    B = 1
    N = len(tokens)
    V = 100

    logits = torch.randn(B, N, V, requires_grad=True)
    targets = torch.tensor(tokens).unsqueeze(0)

    loss = masked_ce_loss(
        logits,
        targets,
        loss_mask.unsqueeze(0)
    )

    loss.backward()
    print(f"t-SOT SAT loss: {loss.item():.6f}")
    print(f"Gradient check: {logits.grad.abs().mean().item():.6f}")
    assert logits.grad is not None, "Gradient should not be None"
    print("✓ t-SOT test passed")


if __name__ == "__main__":
    test_t_sot()


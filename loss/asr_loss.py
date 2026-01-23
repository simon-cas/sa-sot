import torch
import torch.nn.functional as F


def masked_ce_loss(logits, targets, loss_mask):
    """
    Args:
        logits:    (B, N, V)
        targets:   (B, N)
        loss_mask: (B, N) bool
    """
    B, N, V = logits.size()

    logits = logits.view(B * N, V)
    targets = targets.view(B * N)
    loss_mask = loss_mask.view(B * N)

    logits = logits[loss_mask]
    targets = targets[loss_mask]

    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    return F.cross_entropy(logits, targets)


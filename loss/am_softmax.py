import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax Loss
    """
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.2):
        super().__init__()
        self.s = s  # Scale factor (default 30.0, typical range: 10-64)
        self.m = m  # Margin (default 0.2, typical range: 0.1-0.5)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        embeddings: (B, N, E) or (B*N, E)
        labels:     (B,) or (B*N,)
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))

        # normalize
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        logits = F.linear(embeddings, weight)  # (B*N, C)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits_m = logits - one_hot * self.m
        logits_m = logits_m * self.s

        loss = F.cross_entropy(logits_m, labels.view(-1))
        return loss


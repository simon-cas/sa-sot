import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerDecoder(nn.Module):
    """
    Paper-equivalent speaker decoder:
    FC(256, ReLU) -> FC(num_speakers)

    Note:
    - Input token-level speaker embeddings are 512-d
    - This decoder is ONLY used to produce logits for speaker loss
    - For AM-Softmax, you can bypass this module
    """
    def __init__(self, input_dim=512, hidden_dim=256, num_speakers=2341):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_speakers)

    def forward(self, token_spk):
        """
        Args:
            token_spk: (B, N, 512)
        Returns:
            logits:    (B, N, num_speakers)
        """
        x = F.relu(self.fc1(token_spk))
        return self.fc2(x)


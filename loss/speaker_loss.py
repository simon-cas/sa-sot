import torch.nn as nn
from loss.am_softmax import AMSoftmaxLoss


class SpeakerAMSoftmax(nn.Module):
    def __init__(self, embed_dim, num_speakers):
        super().__init__()
        self.loss_fn = AMSoftmaxLoss(embed_dim, num_speakers)

    def forward(self, token_spk, speaker_ids):
        """
        token_spk:   (B, N, E)
        speaker_ids:(B,) or (B, N)
        """
        B, N, E = token_spk.size()

        if speaker_ids.dim() == 1:
            speaker_ids = speaker_ids.unsqueeze(1).expand(B, N)

        return self.loss_fn(
            token_spk.reshape(B * N, E),
            speaker_ids.reshape(B * N)
        )


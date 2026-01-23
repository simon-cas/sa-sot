class SpeakerAwareAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, speaker_embed):
        """
        q,k,v: [T, B, D]
        speaker_embed: [B, T, D]
        """
        T, B, D = q.size()

        q = q.transpose(0, 1)  # [B, T, D]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)

        # speaker similarity
        sim = F.cosine_similarity(
            speaker_embed.unsqueeze(2),
            speaker_embed.unsqueeze(1),
            dim=-1
        )  # [B, T, T]

        sim = (sim + 1.0) / 2.0  # [-1,1] -> [0,1]

        attn = attn * sim
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(attn, v)
        return out.transpose(0, 1)


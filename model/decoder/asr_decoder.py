import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        """
        x:      (T,B,D)  token embeddings
        memory: (S,B,D)  CIF outputs
        """
        # self-attention
        res = x
        x, _ = self.self_attn(x, x, x)
        x = self.norm1(res + x)

        # cross-attention
        res = x
        x, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(res + x)

        # FFN
        return self.norm3(x + self.ffn(x))


class ASRDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, spk_dim=512, num_layers=2):
        super().__init__()

        # Special token IDs for t-SOT
        # mask_id: token used to mask other speakers' tokens
        # cc_id: change-of-character token (speaker change marker)
        # These should be in vocab, typically 0 and a small number like 3
        self.mask_id = 0  # <mask> token
        self.cc_id = 3    # <cc> token (speaker change marker)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.spk_fusion = nn.Linear(d_model + spk_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                ffn_dim=2048
            )
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, c, prev_tokens, speaker_embed=None):
        """
        Args:
            c:              (B, N, D)   CIF outputs
            prev_tokens:    (B, N)
            speaker_embed:  (B, N, D) or None
        """
        x = self.embed(prev_tokens)  # (B,N,D)

        if speaker_embed is not None:
            x = torch.cat([x, speaker_embed], dim=-1)
            x = self.spk_fusion(x)

        # Transformer wants (T,B,D)
        x = x.transpose(0, 1)
        memory = c.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, memory)

        x = x.transpose(0, 1)
        return self.output(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Conv front-end (paper: 2 conv, 128 ch, 1/4 downsample)
# -----------------------------
class ConvSubsampling(nn.Module):
    def __init__(self, in_dim=80, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Linear(out_dim * (in_dim // 4), 512)

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)  # (B,1,T,F)
        x = self.conv(x)    # (B,128,T/4,F/4)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        return self.out(x)  # (B,T/4,512)


# -----------------------------
# Conformer blocks (minimal, paper-equivalent)
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, dim=512, hidden=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (T,B,D)
        res = x
        x, _ = self.mha(x, x, x)
        return self.norm(res + self.dropout(x))


class ConformerLayer(nn.Module):
    def __init__(self, dim=512, heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForward(dim, ffn_dim, dropout)
        self.mha = MultiHeadSelfAttention(dim, heads, dropout)
        self.ffn2 = FeedForward(dim, ffn_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (T,B,D)
        x = x + 0.5 * self.ffn1(x.transpose(0,1)).transpose(0,1)
        x = self.mha(x)
        x = x + 0.5 * self.ffn2(x.transpose(0,1)).transpose(0,1)
        return self.norm(x)


# -----------------------------
# ASR Encoder (paper config)
# -----------------------------
class ASREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.subsampling = ConvSubsampling()
        self.layers = nn.ModuleList([
            ConformerLayer(dim=512, heads=8, ffn_dim=2048)
            for _ in range(18)
        ])

    def forward(self, feats, feat_lens=None):
        # feats: (B,T,80)
        x = self.subsampling(feats)   # (B,T/4,512)
        x = x.transpose(0, 1)         # (T,B,D)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(0, 1)      # (B,T/4,512)


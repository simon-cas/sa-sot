import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic ResNet blocks (channels halved, no global pooling)
# -----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18Half(nn.Module):
    """
    ResNet-18 with channels halved and WITHOUT global average pooling
    """
    def __init__(self):
        super().__init__()
        self.in_planes = 32  # halved from 64

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=1)  # no extra downsampling

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)              # (B,1,T,F)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # output: (B,C,T/4,F/4)
        b, c, t, f = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, c * f)


# -----------------------------
# Conformer layers (same setting as ASR encoder)
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
        x = x + 0.5 * self.ffn1(x.transpose(0, 1)).transpose(0, 1)
        x = self.mha(x)
        x = x + 0.5 * self.ffn2(x.transpose(0, 1)).transpose(0, 1)
        return self.norm(x)


# -----------------------------
# Speaker Encoder (paper-equivalent)
# -----------------------------
class SpeakerEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.resnet = ResNet18Half()
        # ResNet output: (B, T/4, 256*10) = (B, T/4, 2560)
        # After 3 stride=2 downsampling steps: 80 -> 40 -> 20 -> 10
        self.proj = nn.Linear(256 * 10, out_dim)  # 2560 -> 512
        self.conformers = nn.ModuleList([
            ConformerLayer(dim=out_dim, heads=8, ffn_dim=2048)
            for _ in range(2)
        ])

    def forward(self, feats):
        # feats: (B,T,80)
        x = self.resnet(feats)      # (B,T/4,*)
        x = self.proj(x)            # (B,T/4,512)
        x = x.transpose(0, 1)       # (T,B,512)
        for layer in self.conformers:
            x = layer(x)
        return x.transpose(0, 1)    # (B,T/4,512)


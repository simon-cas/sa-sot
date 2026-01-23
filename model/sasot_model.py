"""
SA-SOT Model Implementation

Copyright (c) 2026 Simon Fang
Email: fangshuming519@gmail.com

MIT License
"""

import torch
import torch.nn as nn

from model.encoder.asr_encoder import ASREncoder
from model.encoder.speaker_encoder import SpeakerEncoder
from model.decoder.asr_decoder import ASRDecoder
from model.decoder.speaker_decoder import SpeakerDecoder
from model.encoder.cif import CIF


class SASOTModel(nn.Module):
    """
    Full SA-SOT model (paper-aligned)
    --------------------------------
    SAT is implemented by running ASR decoder
    once per speaker with masked t-SOT inputs.
    """

    def __init__(self,
                 vocab_size: int,
                 num_speakers: int = 2341,
                 asr_dim: int = 512,
                 spk_dim: int = 512,
                 cif_beta: float = 1.0):
        super().__init__()

        self.weight_estimator = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Encoders
        self.asr_encoder = ASREncoder()
        self.spk_encoder = SpeakerEncoder(out_dim=spk_dim)

        # CIF
        self.cif = CIF(threshold=cif_beta)

        # Decoders
        self.asr_decoder = ASRDecoder(
            vocab_size=vocab_size,
            d_model=asr_dim,
            spk_dim=spk_dim
        )
        self.spk_decoder = SpeakerDecoder(
            input_dim=spk_dim,
            hidden_dim=256,
            num_speakers=num_speakers
        )

        # special ids (must exist in vocab)
        self.mask_id = self.asr_decoder.mask_id
        self.cc_id = self.asr_decoder.cc_id

    def forward(self,
                feats,
                feat_lens,
                prev_tokens,
                target_len=None,
                speaker_ids=None,
                return_intermediate=False):
        """
        Args:
            feats:        (B, T, 80)
            feat_lens:    (B,)
            prev_tokens: (B, N)
            target_len:  (B,)
            speaker_ids: (B, N)
        """
        out = {}

        # ========= ASR encoder =========
        h = self.asr_encoder(feats, feat_lens)   # (B, T', 512)

        we = self.weight_estimator(
            h.transpose(1, 2)
        ).transpose(1, 2)

        alpha = we.squeeze(-1)

        # ========= CIF =========
        c, token_lens, boundaries = self.cif(h, alpha, target_len)

        out["enc_out"] = h
        out["cif_out"] = c
        out["token_lens"] = token_lens
        out["boundaries"] = boundaries

        # ========= Speaker encoder =========
        spk_frame = self.spk_encoder(feats)      # (B, T', D)
        token_spk = self._boundary_pooling(spk_frame, boundaries)

        out["token_spk"] = token_spk

        # ========= ASR decoder =========
        # ---- Inference or no SAT ----
        if (not self.training) or (speaker_ids is None):
            asr_logits = self.asr_decoder(c, prev_tokens, token_spk)
            out["asr_logits"] = asr_logits

        # ---- Training with SAT ----
        else:
            B, N = prev_tokens.size()
            sat_logits = {}

            for b in range(B):
                uniq_spks = torch.unique(speaker_ids[b])
                for spk in uniq_spks:
                    # build masked t-SOT input
                    masked_prev = prev_tokens[b].clone()
                    mask = (speaker_ids[b] != spk) & (masked_prev != self.cc_id)
                    masked_prev[mask] = self.mask_id

                    logits = self.asr_decoder(
                        c[b:b+1],
                        masked_prev.unsqueeze(0),
                        token_spk[b:b+1]
                    )

                    sat_logits[(b, int(spk))] = logits

            out["asr_logits"] = sat_logits

        # ========= Speaker decoder =========
        if speaker_ids is not None:
            out["spk_logits"] = self.spk_decoder(token_spk)

        if return_intermediate:
            return out
        return out["asr_logits"]

    @staticmethod
    def _boundary_pooling(spk_feat, boundaries):
        """
        Mean pooling speaker features by CIF boundaries
        """
        B, T, D = spk_feat.size()
        max_tokens = max(len(boundaries[b]) for b in range(B)) if B > 0 else 0

        token_spk = torch.zeros(
            B, max_tokens, D,
            device=spk_feat.device,
            dtype=spk_feat.dtype
        )

        for b in range(B):
            feats = []
            for s, e in boundaries[b]:
                if s is None:
                    s = 0
                if e is None:
                    e = T
                s = max(0, min(s, T))
                e = max(s, min(e, T))

                if e > s:
                    feats.append(spk_feat[b, s:e].mean(dim=0))
                else:
                    feats.append(torch.zeros(D, device=spk_feat.device))

            if len(feats) > 0:
                token_spk[b, :len(feats)] = torch.stack(feats)

        return token_spk


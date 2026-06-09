# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
SigMamba: Mamba-based weakly-supervised video anomaly detection model.
"""

import torch
import torch.nn as nn
import torch.nn.init as torch_init

from sigmamba.mamba import MambaEncoder

torch.set_default_dtype(torch.float32)


def _xavier_init(m: nn.Module):
    """Applies Xavier uniform initialisation to Conv and Linear layers."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Model(nn.Module):
    """
    Mamba-based anomaly detection model for weakly-supervised MIL training.

    During training, the input batch is expected to be a concatenation of
    normal bags (first half) and abnormal bags (second half),
    matching the dataloader setup in train.py.

    Args:
        feature_dim: Dimension of input visual features (default 1024 for SigLIP2-large).
        batch_size:  Per-class batch size used to split normal / abnormal halves.
        seg_num:     Number of fixed temporal segments per video.
        d_model:     Hidden dimension of the Mamba encoder.
        depth:       Number of stacked Mamba layers.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        batch_size: int = 64,
        seg_num: int = 32,
        d_model: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_segments = seg_num
        self.k_top = seg_num // 10

        self.mamba_encoder = MambaEncoder(input_dim=feature_dim, d_model=d_model, depth=depth)

        self.fc_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LeakyReLU(negative_slope=5e-2),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.fc_head.apply(_xavier_init)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input features of shape (B, T, C).

        Returns:
            Tuple of (score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores).
            The first four elements are None when B == 1 (inference mode).
        """
        B, T, _ = x.shape

        features = self.mamba_encoder(x)
        scores = self.fc_head(features).view(B, T, 1)

        if B == 1:
            return None, None, None, None, scores

        normal_features = features[: self.batch_size]
        abnormal_features = features[self.batch_size :]
        normal_scores = scores[: self.batch_size]
        abnormal_scores = scores[self.batch_size :]

        feat_magnitudes = torch.norm(features, p=2, dim=2, keepdim=True)
        nfea_magnitudes = feat_magnitudes[: self.batch_size]
        afea_magnitudes = feat_magnitudes[self.batch_size :]

        idx_abn = torch.topk(afea_magnitudes, self.k_top, dim=1)[1]
        feat_select_abn = torch.gather(abnormal_features, 1, idx_abn.expand(-1, -1, features.shape[2]))
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn.expand(-1, -1, 1)), dim=1)

        idx_nor = torch.topk(nfea_magnitudes, self.k_top, dim=1)[1]
        feat_select_normal = torch.gather(normal_features, 1, idx_nor.expand(-1, -1, features.shape[2]))
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_nor.expand(-1, -1, 1)), dim=1)

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores

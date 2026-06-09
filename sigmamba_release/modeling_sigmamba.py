# Copyright (c) 2026 Vinay Umrethe
# This project is licensed under the MIT License - see the LICENSE file for details.

"""
SigMamba: Unified Video Anomaly Detection Model.
Combines SigLIP2 vision encoder with Mamba temporal reasoning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_utils import PreTrainedModel

from .configuration_sigmamba import SigMambaConfig

try:
    from mamba_ssm import Mamba

    IS_OFFICIAL_MAMBA = True
except ImportError:
    IS_OFFICIAL_MAMBA = False
    Mamba = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MambaSSM(nn.Module):
    """
    Pure PyTorch Mamba SSM implementation.
    Fallback for systems without official kernels.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, int(self.d_inner // 16) + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(int(self.d_inner // 16), self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initializes projection weights and dt_proj bias for stable training."""
        dt_min, dt_max = 0.001, 0.1
        dt_init_std = 0.02

        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=1e-4
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj)[:, :, :seq_len]
        x_proj = x_proj.transpose(1, 2)
        x_proj = F.silu(x_proj)

        x_dbl = self.x_proj(x_proj)
        d_rank = int(self.d_inner // 16)
        delta, B, C = torch.split(x_dbl, [d_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)

        y = self.selective_scan_seq(x_proj, delta, A, B, C, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan_seq(self, u, delta, A, B, C, D):
        """Performs sequential selective scan using S6 recurrence."""
        b_size, seq_len, d_in = u.shape
        d_state = A.shape[1]
        h = torch.zeros(b_size, d_in, d_state, device=u.device)
        ys = []

        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        for i in range(seq_len):
            h = h * deltaA[:, i] + deltaB_u[:, i]
            y = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        return y + u * D


class MambaBlock(nn.Module):
    """Single Mamba layer with residual connection."""

    def __init__(self, config, d_model, depth_idx=0):
        super().__init__()
        self.norm = RMSNorm(d_model)

        if IS_OFFICIAL_MAMBA:
            self.mixer = Mamba(d_model=d_model, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
        else:
            self.mixer = MambaSSM(d_model=d_model, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaEncoder(nn.Module):
    """Stacked Mamba blocks used for encoding temporal sequences."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Linear(config.feature_dim, config.d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(config, d_model=config.d_model, depth_idx=i) for i in range(config.depth)]
        )
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class SigMambaPreTrainedModel(PreTrainedModel):
    """Base class for SigMamba models."""

    config_class = SigMambaConfig
    base_model_prefix = "sigmamba"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SigMambaForVideoClassification(SigMambaPreTrainedModel):
    """
    SigMamba model for video anomaly detection.

    Supports two input modes:
        - features: Pre-extracted embeddings (B, T, 1024)
        - pixel_values: Raw video frames (B, T, C, H, W)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Structure only, weights loaded from checkpoint.
        vision_config = AutoConfig.from_pretrained(config.vision_model_id)
        self.vision_model = AutoModel.from_config(vision_config)

        # Temporal encoder.
        self.mamba_encoder = MambaEncoder(config)

        # Classification head.
        self.fc_head = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.LeakyReLU(negative_slope=5e-2),
            nn.Dropout(0.2),
            nn.Linear(128, config.num_classes),
            nn.Sigmoid(),
        )

        self.post_init()

    def forward(self, features=None, pixel_values=None):
        """
        Args:
            features: Pre-extracted features (B, T, 1024)
            pixel_values: Raw video frames (B, T, C, H, W)

        Returns:
            scores: Anomaly scores (B, T, 1)
        """
        # Path A: Unified mode (pixels -> features -> scores).
        if pixel_values is not None and features is None:
            if pixel_values.dim() == 5:
                b, t, c, h, w = pixel_values.shape
                flat_pixels = pixel_values.view(b * t, c, h, w)
            else:
                flat_pixels = pixel_values
                b, t = flat_pixels.shape[0], 1

            # Extract and normalize features.
            if hasattr(self.vision_model, "get_image_features"):
                flat_features = self.vision_model.get_image_features(pixel_values=flat_pixels)
                if not isinstance(flat_features, torch.Tensor):
                    flat_features = getattr(
                        flat_features, "pooler_output", getattr(flat_features, "image_embeds", flat_features[0])
                    )
                flat_features = flat_features / flat_features.norm(p=2, dim=-1, keepdim=True)
            else:
                vision_outputs = self.vision_model(pixel_values=flat_pixels)
                flat_features = getattr(
                    vision_outputs, "pooler_output", getattr(vision_outputs, "image_embeds", vision_outputs[0])
                )

            # Reshape to (B, T, D).
            features = flat_features.view(b, t, -1) if pixel_values.dim() == 5 else flat_features.unsqueeze(1)

        # Path B: Modular mode (features -> scores).
        if features is None:
            raise ValueError("You must provide either 'features' or 'pixel_values'")

        # Encode and classify.
        x = self.mamba_encoder(features)
        scores = self.fc_head(x)

        return scores

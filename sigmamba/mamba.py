# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
Mamba SSM module for temporal sequence modelling.

Falls back to a pure-PyTorch selective-scan implementation when the
official mamba_ssm package is not installed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba

    IS_OFFICIAL_MAMBA = True
except ImportError:
    IS_OFFICIAL_MAMBA = False
    Mamba = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaSSM(nn.Module):
    """
    Pure-PyTorch Mamba Selective State Space Model.

    Architecture follows the original Mamba paper (Gu & Dao, 2023)
    https://arxiv.org/abs/2312.00752.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.d_rank = int(self.d_inner // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        dt_min, dt_max, dt_init_std = 0.001, 0.1, 0.02
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=1e-4
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        x_gate, z = self.in_proj(x).chunk(2, dim=-1)

        x_gate = self.conv1d(x_gate.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_gate = F.silu(x_gate)

        delta, B_ssm, C = torch.split(self.x_proj(x_gate), [self.d_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_gate, delta, A, B_ssm, C, self.D)
        return self.out_proj(y * F.silu(z))

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """S6 recurrence: sequential selective scan over the time axis."""
        B_size, L, d_in = u.shape
        h = torch.zeros(B_size, d_in, A.shape[1], device=u.device)

        dA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        dBu = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        ys = []
        for i in range(L):
            h = h * dA[:, i] + dBu[:, i]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, i]))

        return torch.stack(ys, dim=1) + u * D


class MambaBlock(nn.Module):
    """Single Mamba layer with pre-norm and residual connection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = (
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if IS_OFFICIAL_MAMBA else MambaSSM(d_model=d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MambaEncoder(nn.Module):
    """
    Stacked Mamba blocks for temporal sequence encoding.

    Projects input features to d_model, applies depth Mamba layers,
    then applies a final RMSNorm.
    """

    def __init__(self, input_dim: int = 1024, d_model: int = 512, depth: int = 3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(depth)])
        self.norm_f = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

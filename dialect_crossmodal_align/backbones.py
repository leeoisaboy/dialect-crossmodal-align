from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DepthwiseSeparableConvBlock(nn.Module):
    """Padding-safe dilated depthwise separable 1D conv block for syllable & tonal modeling."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric temporal padding")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        valid = frame_mask.bool().unsqueeze(-1)
        safe_hidden = hidden.masked_fill(~valid, 0.0)
        conv_out = self.depthwise(safe_hidden.transpose(1, 2)).transpose(1, 2)
        conv_out = conv_out.masked_fill(~valid, 0.0)
        updated = safe_hidden + self.pointwise(self.norm(conv_out))
        return updated.masked_fill(~valid, 0.0)


class LinearDialectAdapter(nn.Module):
    """Multi-scale dilated dialect feature adapter with residual protection.

    Stacks padding-safe dilated depthwise separable 1D conv blocks (default
    ``dilations=(1, 2, 4)``, covering a 15-frame / ~300 ms receptive field at
    50 Hz to span full dialect syllables and tone contours) while preserving
    pretrained acoustic representations via an input-to-output residual path.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.residual: nn.Module = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.temporal_blocks = nn.ModuleList(
            [
                DepthwiseSeparableConvBlock(
                    output_dim,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        valid = frame_mask.bool().unsqueeze(-1)
        safe_frames = frames.masked_fill(~valid, 0.0)
        hidden = self.project(self.norm(safe_frames)) + self.residual(safe_frames)
        hidden = hidden.masked_fill(~valid, 0.0)
        for block in self.temporal_blocks:
            hidden = block(hidden, frame_mask)
        return hidden


TemporalDialectAdapter = LinearDialectAdapter

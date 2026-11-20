from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def masked_mean(features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Padding-safe mean over time dimension (dim=1)."""
    valid = frame_mask.bool().unsqueeze(-1)
    safe_features = features.masked_fill(~valid, 0.0)
    weights = valid.to(features.dtype)
    return safe_features.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@dataclass
class AggregatorOutput:
    pooled: torch.Tensor
    sequence: torch.Tensor


class MeanPooling(nn.Module):
    def forward(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> AggregatorOutput:
        return AggregatorOutput(masked_mean(sequence, frame_mask), sequence)


class TemporalROIPooling(nn.Module):
    """Vectorized ROI pooling over arbitrary (possibly non-contiguous) valid frames."""

    def __init__(self, bins: int = 4):
        super().__init__()
        if bins <= 0:
            raise ValueError("bins must be positive")
        self.bins = bins

    def forward(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> AggregatorOutput:
        batch, _, channels = sequence.shape
        valid = frame_mask.bool()
        safe_sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0.0)

        lengths = valid.sum(dim=1)
        valid_pos = valid.long().cumsum(dim=1) - 1

        bin_idx = torch.arange(self.bins, device=sequence.device, dtype=torch.long)
        safe_lengths = lengths.clamp_min(1).unsqueeze(1)
        starts = (bin_idx.unsqueeze(0) * safe_lengths) // self.bins
        ends = ((bin_idx.unsqueeze(0) + 1) * safe_lengths + self.bins - 1) // self.bins

        in_bin = (
            valid.unsqueeze(1)
            & (valid_pos.unsqueeze(1) >= starts.unsqueeze(2))
            & (valid_pos.unsqueeze(1) < ends.unsqueeze(2))
        )
        weights = in_bin.to(sequence.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_weights = weights / denom
        pooled = torch.bmm(norm_weights, safe_sequence)
        return AggregatorOutput(pooled.reshape(batch, self.bins * channels), sequence)


class MultiStatisticPNAPooling(nn.Module):
    """Order-preserving multi-scale PNA + positional ROI pooling with clean empty-mask handling."""

    def __init__(
        self,
        input_dim: int,
        bins: int = 4,
        output_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim * 4
        self.raw_dim = input_dim * 4
        self.eps = float(eps)
        self.project = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, self.output_dim),
            nn.GELU(),
        )
        self.roi = TemporalROIPooling(bins)
        self.bin_pos = nn.Parameter(torch.randn(1, bins, input_dim) * 0.02)
        self.regional_project = nn.Sequential(
            nn.LayerNorm(bins * input_dim),
            nn.Linear(bins * input_dim, self.raw_dim),
        )

    @staticmethod
    def _statistics(
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if mask is None:
            mean = values.mean(dim=1)
            variance = ((values - mean.unsqueeze(1)) ** 2).mean(dim=1)
            standard_deviation = (variance.clamp_min(0.0) + eps).sqrt()
            maximum = values.max(dim=1).values
            minimum = values.min(dim=1).values
        else:
            valid = mask.bool().unsqueeze(-1)
            safe_values = values.masked_fill(~valid, 0.0)
            weights = valid.to(values.dtype)
            count = weights.sum(dim=1)
            has_valid = count > 0
            safe_count = count.clamp_min(1.0)

            mean = safe_values.sum(dim=1) / safe_count
            diff = (safe_values - mean.unsqueeze(1)).masked_fill(~valid, 0.0)
            variance = (diff ** 2).sum(dim=1) / safe_count
            standard_deviation = (variance.clamp_min(0.0) + eps).sqrt()
            # Zero out std on all-False masks so no sqrt(eps)=1e-3 residual leaks.
            standard_deviation = torch.where(
                has_valid, standard_deviation, torch.zeros_like(standard_deviation)
            )

            neg_inf = torch.finfo(values.dtype).min
            pos_inf = torch.finfo(values.dtype).max
            maximum = values.masked_fill(~valid, neg_inf).max(dim=1).values
            minimum = values.masked_fill(~valid, pos_inf).min(dim=1).values
            maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))
            minimum = torch.where(has_valid, minimum, torch.zeros_like(minimum))
        return torch.cat((mean, standard_deviation, maximum, minimum), dim=-1)

    def forward(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> AggregatorOutput:
        batch, _, channels = sequence.shape
        dense = self._statistics(sequence, frame_mask, eps=self.eps)
        regions = self.roi(sequence, frame_mask).pooled.reshape(batch, self.roi.bins, channels)
        has_valid = frame_mask.bool().any(dim=1).view(batch, 1, 1)
        ordered_regions = torch.where(has_valid, regions + self.bin_pos, torch.zeros_like(regions))
        regional = self.regional_project(ordered_regions.reshape(batch, self.roi.bins * channels))
        combined = dense + regional
        return AggregatorOutput(self.project(combined), sequence)


class ProjectedROIPooling(nn.Module):
    """Fixed-bin ROI pooling with an explicit linear projection (no LazyLinear)."""

    def __init__(self, model_dim: int, bins: int = 4, input_dim: int | None = None):
        super().__init__()
        in_channels = input_dim or model_dim
        self.pooling = TemporalROIPooling(bins)
        self.project = nn.Linear(bins * in_channels, model_dim)

    def forward(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> AggregatorOutput:
        pooled = self.pooling(sequence, frame_mask).pooled
        return AggregatorOutput(self.project(pooled), sequence)


class TextCrossAttention(nn.Module):
    """Padding-safe and empty-mask-safe cross-attention over an acoustic sequence."""

    def __init__(self, model_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        text: torch.Tensor,
        frame_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, model_dim = sequence.shape
        valid_frames = frame_mask.bool()
        valid_text = text_mask.bool()
        has_valid_frame = valid_frames.any(dim=1, keepdim=True)

        safe_sequence = sequence.masked_fill(~valid_frames.unsqueeze(-1), 0.0)
        safe_text = text.masked_fill(~valid_text.unsqueeze(-1), 0.0)

        queries = self.query(safe_text).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = self.key(safe_sequence).reshape(batch, frames, self.heads, self.head_dim).transpose(1, 2)
        values = self.value(safe_sequence).reshape(batch, frames, self.heads, self.head_dim).transpose(1, 2)

        scores = queries @ keys.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~valid_frames[:, None, None, :], torch.finfo(scores.dtype).min)
        probabilities = scores.softmax(dim=-1)
        probabilities = probabilities * valid_frames[:, None, None, :].to(probabilities.dtype)
        probabilities = self.dropout(probabilities)

        attended = (probabilities @ values).transpose(1, 2).reshape(batch, -1, model_dim)
        out_mask = (valid_text & has_valid_frame).unsqueeze(-1).to(attended.dtype)
        return self.output(attended) * out_mask

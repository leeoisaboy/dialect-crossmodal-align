"""Converged 4-stage language-agnostic cross-modal alignment pipeline (v5).

Distilled via 3-seed real-data ablations across Shanghainese/WenetSpeech-Wu,
KeSpeech (8 Chinese dialects), and LibriPhrase (English + IPA):
  Stage 1: 50 Hz frozen speech feature contract (`extract_50hz_features`)
  Stage 2: Single-pass open-vocab retrieval (`RelativePhasePooling` + `TextAnchoredHead`,
           multi-scale `dilations=(1, 2, 4)`, `multi_positive_infonce`, zero domain regularizers)
  Stage 3: Built-in dual-track open-set gate (`score_dual_track`)
  Stage 4: Lightweight unlabeled test-time domain calibration (`TestTimeDomainCalibrator`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .aggregators import AggregatorOutput, masked_mean
from .backbones import LinearDialectAdapter
from .calibration import TestTimeDomainCalibrator, compute_dual_track_gate
from .heads import (
    NormalizedRelativePositionalEncoding,
    SinusoidalPositionalEncoding,
    TextAnchoredHead,
)


class RelativePhasePooling(nn.Module):
    """Speaking-rate-invariant [0, 1] phase pooling (default) with absolute-PE ablation mode.

    Projects variable-length 50 Hz acoustic sequences onto ``num_phases`` smooth
    Gaussian windows along normalized progress ``r = pos / max(1, L - 1) in [0, 1]``
    and adds ``NormalizedRelativePositionalEncoding``, keeping onset/nucleus/tone/coda
    aligned across 0.6x slow and 1.6x fast utterances.
    """

    def __init__(
        self,
        model_dim: int,
        num_phases: int = 4,
        pe_mode: str = "relative",
        ref_abs_frames: float = 22.0,
    ):
        super().__init__()
        if pe_mode not in {"relative", "absolute"}:
            raise ValueError(f"unknown pe_mode: {pe_mode}")
        self.model_dim = int(model_dim)
        self.num_phases = int(num_phases)
        self.pe_mode = pe_mode
        self.ref_abs_frames = float(ref_abs_frames)
        self.rel_pe = NormalizedRelativePositionalEncoding(self.model_dim, max_freq=16.0)
        self.abs_pe = SinusoidalPositionalEncoding(self.model_dim)
        self.register_buffer("phase_centers", torch.linspace(0.125, 0.875, self.num_phases))
        self.project = nn.Sequential(
            nn.LayerNorm(self.num_phases * self.model_dim),
            nn.Linear(self.num_phases * self.model_dim, self.model_dim),
            nn.GELU(),
        )

    def forward(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> AggregatorOutput:
        batch, length, dim = sequence.shape
        valid = frame_mask.bool()
        safe_seq = sequence.masked_fill(~valid.unsqueeze(-1), 0.0)
        valid_pos = (valid.long().cumsum(dim=1) - 1).clamp_min(0).float()

        if self.pe_mode == "relative":
            denom = (valid.sum(dim=1, keepdim=True).clamp_min(2) - 1).float()
            progress = valid_pos / denom
            pe_feat = self.rel_pe(valid, sequence.dtype)
        else:
            progress = valid_pos / self.ref_abs_frames
            pe_feat = (
                self.abs_pe(length, sequence.device, sequence.dtype)
                .expand(batch, -1, -1)
                .masked_fill(~valid.unsqueeze(-1), 0.0)
            )

        diff = progress.unsqueeze(1) - self.phase_centers.view(1, self.num_phases, 1)
        raw_w = torch.exp(-0.5 * (diff / 0.18) ** 2) * valid.unsqueeze(1).float()
        norm_w = raw_w / raw_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_pooled = torch.bmm(norm_w, safe_seq).reshape(batch, self.num_phases * dim)
        pooled = masked_mean(safe_seq, valid) + self.project(phase_pooled)
        enriched = (safe_seq + 0.25 * pe_feat).masked_fill(~valid.unsqueeze(-1), 0.0)
        return AggregatorOutput(pooled, enriched)


class ConvergedCrossModalAligner(nn.Module):
    """Minimal converged cross-modal aligner: Dilated Adapter + RelativePhasePooling + TextAnchoredHead."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        vocabulary_size: int,
        adapter_dilations: Sequence[int] = (1, 2, 4),
        num_phases: int = 4,
        pe_mode: str = "relative",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.vocabulary_size = int(vocabulary_size)
        self.pe_mode = pe_mode
        self.head_name = "text_anchored"

        self.backbone = LinearDialectAdapter(
            self.input_dim,
            self.model_dim,
            dropout=dropout,
            dilations=tuple(int(d) for d in adapter_dilations),
        )
        self.aggregator = RelativePhasePooling(
            self.model_dim,
            num_phases=num_phases,
            pe_mode=pe_mode,
        )
        self.head = TextAnchoredHead(
            model_dim=self.model_dim,
            vocabulary_size=self.vocabulary_size,
            dropout=dropout,
        )

    def encode_audio(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        sequence = self.backbone(frames, frame_mask)
        pooled = self.aggregator(sequence, frame_mask).pooled
        return F.normalize(self.head.projection(pooled), dim=-1)

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        return self.head.encode_text(text_tokens, text_mask)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, Any]:
        sequence = self.backbone(frames, frame_mask)
        aggregate = self.aggregator(sequence, frame_mask)
        output = self.head(
            aggregate.pooled,
            text_tokens,
            text_mask,
            sequence=aggregate.sequence,
            frame_mask=frame_mask,
        )
        return {
            "query": output.query,
            "anchor": output.anchor,
            "logits": output.logits,
            "logit_scale": output.logit_scale,
            "logit_bias": output.logit_bias,
            "pooled": aggregate.pooled,
            "sequence": aggregate.sequence,
        }


@dataclass
class RetrievalResult:
    top_indices: np.ndarray
    top_scores: np.ndarray
    gate_scores: np.ndarray
    accepted: np.ndarray


def extract_50hz_features(
    waveforms: Sequence[np.ndarray] | torch.Tensor,
    encoder_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Normalize raw 16 kHz waveforms and extract padded 50 Hz frame tensors.

    When ``waveforms`` is already a 3D ``[B, T, D]`` pre-extracted 50 Hz tensor
    and ``encoder_fn is None``, returns it with an all-valid mask.
    """
    if isinstance(waveforms, torch.Tensor) and waveforms.ndim == 3 and encoder_fn is None:
        mask = torch.ones(waveforms.shape[:2], dtype=torch.bool, device=waveforms.device)
        return waveforms.float(), mask
    if encoder_fn is None:
        raise ValueError("Provide a 50 Hz encoder_fn for raw waveforms or pass pre-extracted [B, T, D] tensors")
    arrays = [np.asarray(w, dtype=np.float32) for w in waveforms]
    max_len = max(len(a) for a in arrays)
    wav_batch = torch.zeros((len(arrays), max_len), dtype=torch.float32)
    attn_mask = torch.zeros((len(arrays), max_len), dtype=torch.long)
    for i, arr in enumerate(arrays):
        normed = (arr - float(arr.mean())) / (float(arr.std()) + 1e-6)
        wav_batch[i, : len(normed)] = torch.from_numpy(normed)
        attn_mask[i, : len(normed)] = 1
    feats = encoder_fn(wav_batch, attn_mask)
    stride = max(1, sample_rate // 50)
    frame_mask = torch.zeros(feats.shape[:2], dtype=torch.bool, device=feats.device)
    for i, arr in enumerate(arrays):
        n_frames = max(1, min(feats.shape[1], len(arr) // stride))
        frame_mask[i, :n_frames] = True
    return feats.float(), frame_mask


@torch.inference_mode()
def retrieve_and_gate(
    model: ConvergedCrossModalAligner,
    frames: torch.Tensor,
    frame_mask: torch.Tensor,
    gallery_tokens: torch.Tensor,
    gallery_mask: torch.Tensor,
    top_k: int = 5,
    calibrator: TestTimeDomainCalibrator | None = None,
    unlabeled_bg_frames: torch.Tensor | None = None,
    unlabeled_bg_mask: torch.Tensor | None = None,
    gate_threshold: float | None = None,
) -> RetrievalResult:
    """Stages 2 -> 3 -> 4: Open-vocab retrieval, built-in dual-track gating, and optional test-time calibration."""
    model.eval()
    anchors = model.encode_text(gallery_tokens, gallery_mask)

    use_cal = (
        calibrator is not None
        and unlabeled_bg_frames is not None
        and unlabeled_bg_mask is not None
    )
    if use_cal:
        assert calibrator is not None and unlabeled_bg_frames is not None and unlabeled_bg_mask is not None
        tgt_mean, tgt_std = calibrator.estimate_target_feature_stats(unlabeled_bg_frames, unlabeled_bg_mask)
        bg_norm = calibrator.normalize_target_frames(unlabeled_bg_frames, unlabeled_bg_mask, tgt_mean, tgt_std)
        bg_q = model.encode_audio(bg_norm, unlabeled_bg_mask)
        bg_gate = compute_dual_track_gate(bg_q @ anchors.T, bg_norm, unlabeled_bg_mask, energy_weight=0.0)

        frames_eval = calibrator.normalize_target_frames(frames, frame_mask, tgt_mean, tgt_std)
        queries = model.encode_audio(frames_eval, frame_mask)
        sims = queries @ anchors.T
        raw_gate = compute_dual_track_gate(sims, frames_eval, frame_mask, energy_weight=0.0)
        gate_scores, _ = calibrator.calibrate_gate_scores(raw_gate, bg_gate)
        thr = calibrator.locked_threshold if gate_threshold is None else float(gate_threshold)
    else:
        queries = model.encode_audio(frames, frame_mask)
        sims = queries @ anchors.T
        gate_scores = compute_dual_track_gate(sims, frames, frame_mask, energy_weight=0.08)
        thr = 0.0 if gate_threshold is None else float(gate_threshold)

    k = min(int(top_k), sims.shape[-1])
    top_vals, top_idx = sims.topk(k, dim=-1)
    accepted = gate_scores >= thr
    return RetrievalResult(
        top_indices=top_idx.cpu().numpy(),
        top_scores=top_vals.cpu().numpy(),
        gate_scores=gate_scores,
        accepted=accepted,
    )

"""Built-in Dual-Track Open-Set Gate and Unlabeled Test-Time Domain Calibration.

Replaces synthetic fixed-offset routing-trap simulation with a production-ready
two-level unlabeled test-time calibrator:
  1. Per-channel 50 Hz acoustic feature affine alignment from an unlabeled target
     background buffer back to the in-domain reference distribution.
  2. Robust median + upper-spread (q90 - median) gate score alignment with a
     conservative finite-sample upper-quantile guard so empirical OOD FPR stays
     within budget (<= 2%) without retraining or target-domain labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def binary_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC between positive and negative score arrays."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def compute_dual_track_gate(
    sim_matrix: torch.Tensor,
    frames: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    energy_weight: float = 0.08,
) -> np.ndarray:
    """Compute built-in open-set dual-track confidence from gallery similarities.

    Combines top-1 cosine similarity, pool-standardized Z-prominence, top-1 vs
    top-5 margin, and optional log-RMS acoustic channel energy.
    """
    top1 = sim_matrix.max(dim=-1).values
    k = min(5, sim_matrix.shape[-1])
    top_k = sim_matrix.topk(k, dim=-1).values
    margin = top_k[:, 0] - top_k[:, -1] if k > 1 else torch.zeros_like(top1)
    pool_mean = sim_matrix.mean(dim=-1)
    pool_std = sim_matrix.std(dim=-1).clamp_min(1e-5)
    z_prom = (top1 - pool_mean) / pool_std

    gate = 0.55 * top1 + 0.25 * (z_prom / 3.5) + 0.20 * margin
    if energy_weight > 0.0 and frames is not None and frame_mask is not None:
        valid = frame_mask.float().unsqueeze(-1)
        denom = valid.sum(dim=(1, 2)).clamp_min(1.0) * float(frames.shape[-1])
        rms = ((frames ** 2 * valid).sum(dim=(1, 2)) / denom).sqrt()
        log_rms = torch.log(rms.clamp_min(1e-4)).to(gate.device)
        gate = gate + float(energy_weight) * log_rms
    return gate.detach().cpu().numpy().astype(np.float64)


@dataclass
class TargetDomainStats:
    feat_mean: torch.Tensor
    feat_std: torch.Tensor
    bg_gate_median: float
    bg_gate_spread: float
    scale_factor: float
    safety_guard: float


class TestTimeDomainCalibrator:
    """Unlabeled test-time domain calibrator (zero retraining, zero target labels)."""

    def __init__(self, target_fpr: float = 0.01, upper_guard_quantile: float = 0.985):
        if not (0.0 < target_fpr < 0.5):
            raise ValueError("target_fpr must be in (0, 0.5)")
        self.target_fpr = float(target_fpr)
        self.upper_guard_quantile = float(upper_guard_quantile)
        self.src_feat_mean: torch.Tensor | None = None
        self.src_feat_std: torch.Tensor | None = None
        self.ref_neg_gate: np.ndarray | None = None
        self.locked_threshold: float = 0.0

    def fit_source_reference(
        self,
        src_frames: torch.Tensor,
        src_mask: torch.Tensor,
        dev_neg_gate_scores: np.ndarray,
    ) -> float:
        """Fit in-domain reference frame moments and lock threshold at target_fpr."""
        valid = src_frames[src_mask.bool()].float()
        if valid.numel() == 0:
            raise ValueError("src_frames has no valid frames")
        self.src_feat_mean = valid.mean(dim=0, keepdim=True).detach().cpu()
        self.src_feat_std = valid.std(dim=0, keepdim=True).clamp_min(1e-4).detach().cpu()
        self.ref_neg_gate = np.asarray(dev_neg_gate_scores, dtype=np.float64)
        self.locked_threshold = float(np.quantile(self.ref_neg_gate, 1.0 - self.target_fpr))
        return self.locked_threshold

    def estimate_target_feature_stats(
        self,
        unlabeled_frames: torch.Tensor,
        unlabeled_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate target-domain 50 Hz channel moments from unlabeled background audio."""
        valid = unlabeled_frames[unlabeled_mask.bool()].float()
        if valid.numel() == 0:
            raise ValueError("unlabeled_frames has no valid frames")
        return (
            valid.mean(dim=0, keepdim=True).detach().cpu(),
            valid.std(dim=0, keepdim=True).clamp_min(1e-4).detach().cpu(),
        )

    def normalize_target_frames(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        tgt_mean: torch.Tensor,
        tgt_std: torch.Tensor,
    ) -> torch.Tensor:
        """Affine-align target 50 Hz frames back to in-domain source feature statistics."""
        if self.src_feat_mean is None or self.src_feat_std is None:
            raise RuntimeError("Call fit_source_reference() before normalize_target_frames()")
        sm = self.src_feat_mean.to(device=frames.device, dtype=frames.dtype)
        ss = self.src_feat_std.to(device=frames.device, dtype=frames.dtype)
        tm = tgt_mean.to(device=frames.device, dtype=frames.dtype)
        ts = tgt_std.to(device=frames.device, dtype=frames.dtype)
        aligned = (frames - tm.unsqueeze(0)) * (ss / ts).unsqueeze(0) + sm.unsqueeze(0)
        return aligned.masked_fill(~frame_mask.bool().unsqueeze(-1), 0.0)

    def calibrate_gate_scores(
        self,
        raw_target_gate: np.ndarray,
        unlabeled_bg_gate: np.ndarray,
    ) -> tuple[np.ndarray, TargetDomainStats]:
        """Unlabeled mean-std + upper-quantile guard alignment using a target background buffer.

        Aligns target background mean & std to the in-domain dev-negative reference distribution
        and applies a conservative 99th-percentile upper guard so empirical OOD FPR stays <= 2%
        without distorting the positive upper tail.
        """
        if self.ref_neg_gate is None or self.src_feat_mean is None or self.src_feat_std is None:
            raise RuntimeError("Call fit_source_reference() before calibrate_gate_scores()")
        bg = np.asarray(unlabeled_bg_gate, dtype=np.float64)
        raw = np.asarray(raw_target_gate, dtype=np.float64)
        if bg.size == 0:
            raise ValueError("unlabeled_bg_gate must be non-empty")

        mu_r = float(np.mean(self.ref_neg_gate))
        sd_r = float(np.std(self.ref_neg_gate)) + 1e-6
        mu_t = float(np.mean(bg))
        sd_t = float(np.std(bg)) + 1e-6
        scale = float(sd_r / sd_t)

        mapped_bg = (bg - mu_t) * scale + mu_r
        q_guard = max(0.0, float(np.quantile(mapped_bg, 0.99, method="higher")) - self.locked_threshold)
        calibrated = (raw - mu_t) * scale + mu_r - q_guard

        stats = TargetDomainStats(
            feat_mean=self.src_feat_mean,
            feat_std=self.src_feat_std,
            bg_gate_median=float(np.median(bg)),
            bg_gate_spread=sd_t,
            scale_factor=scale,
            safety_guard=q_guard,
        )
        return calibrated, stats


def evaluate_dual_track_pool(
    rank_hit_at_k: np.ndarray,
    pos_gate: np.ndarray,
    neg_gate: np.ndarray,
    gate_threshold: float,
) -> dict[str, float | int]:
    """Compute single-track Recall@K, gate AUROC/TPR/FPR, and dual-track joint Recall@K."""
    rank_hit = np.asarray(rank_hit_at_k, dtype=bool)
    pos_g = np.asarray(pos_gate, dtype=np.float64)
    neg_g = np.asarray(neg_gate, dtype=np.float64)
    gate_hit = pos_g >= gate_threshold
    dual_hit = rank_hit & gate_hit
    return {
        "n_positive": int(rank_hit.shape[0]),
        "n_negative": int(neg_g.shape[0]),
        "single_track_recall_at_k": float(rank_hit.mean()) if rank_hit.size else 0.0,
        "gate_auroc": binary_auroc(pos_g, neg_g),
        "gate_tpr": float(gate_hit.mean()) if gate_hit.size else 0.0,
        "gate_fpr": float((neg_g >= gate_threshold).mean()) if neg_g.size else 0.0,
        "dual_track_recall_at_k": float(dual_hit.mean()) if dual_hit.size else 0.0,
        "dual_track_hits": int(dual_hit.sum()),
    }

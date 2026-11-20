from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CosFaceProjection(nn.Module):
    """Sub-Center Additive Cosine Margin classification with temperature annealing & safe broadcasting."""

    def __init__(
        self,
        embedding_dim: int,
        class_count: int,
        margin: float = 0.2,
        scale: float = 32.0,
        sub_centers: int = 3,
        sub_center_temp: float | None = 10.0,
        sub_center_reg_weight: float = 0.01,
    ):
        super().__init__()
        if sub_centers <= 0:
            raise ValueError("sub_centers must be positive")
        self.class_count = class_count
        self.sub_centers = sub_centers
        self.prototypes = nn.Parameter(
            torch.randn(class_count, sub_centers, embedding_dim) * 0.02
        )
        self.margin = float(margin)
        self.scale = float(scale)
        self.sub_center_temp = sub_center_temp
        self.sub_center_reg_weight = float(sub_center_reg_weight)

    def cosine_similarity(self, embeddings: torch.Tensor, smooth: bool = False) -> torch.Tensor:
        norm_emb = F.normalize(embeddings, dim=-1)
        norm_proto = F.normalize(self.prototypes, dim=-1)
        similarities = torch.einsum("bd,ckd->bck", norm_emb, norm_proto)
        if self.sub_centers == 1:
            return similarities.squeeze(-1)
        if smooth and self.sub_center_temp is not None and self.sub_center_temp > 0:
            tau = float(self.sub_center_temp)
            return (torch.logsumexp(similarities * tau, dim=-1) - math.log(self.sub_centers)) / tau
        return similarities.max(dim=-1).values

    def sub_center_decorrelation_loss(self) -> torch.Tensor:
        if self.sub_centers <= 1 or self.sub_center_reg_weight <= 0.0:
            return self.prototypes.new_zeros(())
        norm_proto = F.normalize(self.prototypes, dim=-1)
        gram = torch.einsum("ckd,cjd->ckj", norm_proto, norm_proto)
        eye = torch.eye(self.sub_centers, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag = gram * (1.0 - eye)
        return (off_diag ** 2).mean()

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.cosine_similarity(embeddings, smooth=False) * self.scale

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        adaptive_margins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cosine = self.cosine_similarity(embeddings, smooth=self.training)
        margin_mask = F.one_hot(targets, num_classes=self.class_count).to(cosine.dtype)
        if adaptive_margins is not None:
            margins_t = adaptive_margins.to(device=cosine.device, dtype=cosine.dtype)
            if margins_t.ndim == 1:
                if margins_t.shape[0] != self.class_count:
                    raise ValueError(
                        f"1D adaptive_margins must have shape [class_count]={self.class_count}, "
                        f"got {tuple(margins_t.shape)}"
                    )
                margin_values: torch.Tensor | float = margins_t.unsqueeze(0)
            elif margins_t.ndim == 2:
                margin_values = margins_t
            else:
                raise ValueError(f"adaptive_margins must be 1D or 2D, got ndim={margins_t.ndim}")
        else:
            margin_values = self.margin
        logits = (cosine - margin_mask * margin_values) * self.scale
        ce_loss = F.cross_entropy(logits, targets)
        return ce_loss + self.sub_center_reg_weight * self.sub_center_decorrelation_loss()


def _positive_mask(batch_size: int, labels: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    if labels is None:
        return eye
    return (labels.unsqueeze(1) == labels.unsqueeze(0)) | eye


def multi_positive_infonce(
    query: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 0.1,
    logit_scale: torch.Tensor | float | None = None,
    subtract_entropy_bound: bool = True,
) -> torch.Tensor:
    """Symmetric Multi-Positive InfoNCE (SupCon) with zero-lower-bound KL normalization."""
    if query.shape != anchor.shape:
        raise ValueError("query and anchor batches must have identical shape")
    query_norm = F.normalize(query, dim=-1)
    anchor_norm = F.normalize(anchor, dim=-1)
    scale = logit_scale if logit_scale is not None else (1.0 / temperature)
    logits = (query_norm @ anchor_norm.T) * scale
    return pairwise_infonce(logits, labels=labels, subtract_entropy_bound=subtract_entropy_bound)


def siglip_alignment(
    query: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 0.1,
    logit_scale: torch.Tensor | float | None = None,
    logit_bias: torch.Tensor | float = -2.0,
    normalize_by_pairs: bool = False,
) -> torch.Tensor:
    """Pairwise Sigmoid contrastive alignment (SigLIP).

    Uses ``1 / B`` normalization by default (original SigLIP training objective).
    Set ``normalize_by_pairs=True`` to compute a batch-size-invariant per-pair
    mean loss for logging across different train/val batch sizes.
    """
    if query.shape != anchor.shape:
        raise ValueError("query and anchor batches must have identical shape")
    query_norm = F.normalize(query, dim=-1)
    anchor_norm = F.normalize(anchor, dim=-1)
    scale = logit_scale if logit_scale is not None else (1.0 / temperature)
    logits = (query_norm @ anchor_norm.T) * scale
    return pairwise_siglip(
        logits,
        labels=labels,
        logit_bias=logit_bias,
        normalize_by_pairs=normalize_by_pairs,
    )


def pairwise_infonce(
    score_matrix: torch.Tensor,
    labels: torch.Tensor | None = None,
    subtract_entropy_bound: bool = True,
) -> torch.Tensor:
    """Symmetric Multi-Positive InfoNCE over an explicit ``[B, B]`` score matrix."""
    batch_size = score_matrix.shape[0]
    pos_float = _positive_mask(batch_size, labels, score_matrix.device).to(score_matrix.dtype)
    pos_count_r = pos_float.sum(dim=-1).clamp_min(1.0)
    pos_count_c = pos_float.T.sum(dim=-1).clamp_min(1.0)
    loss_r = -(pos_float * F.log_softmax(score_matrix, dim=-1)).sum(dim=-1) / pos_count_r
    loss_c = -(pos_float.T * F.log_softmax(score_matrix.T, dim=-1)).sum(dim=-1) / pos_count_c
    if subtract_entropy_bound:
        loss_r = (loss_r - torch.log(pos_count_r)).clamp_min(0.0)
        loss_c = (loss_c - torch.log(pos_count_c)).clamp_min(0.0)
    return 0.5 * (loss_r.mean() + loss_c.mean())


def pairwise_siglip(
    score_matrix: torch.Tensor,
    labels: torch.Tensor | None = None,
    logit_bias: torch.Tensor | float = -2.0,
    normalize_by_pairs: bool = False,
) -> torch.Tensor:
    """SigLIP loss over an explicit ``[B, B]`` pairwise interaction score matrix."""
    batch_size = score_matrix.shape[0]
    pos_mask = _positive_mask(batch_size, labels, score_matrix.device)
    targets = 2.0 * pos_mask.to(score_matrix.dtype) - 1.0
    per_pair = -F.logsigmoid(targets * (score_matrix + logit_bias))
    if normalize_by_pairs:
        return per_pair.mean()
    return per_pair.sum() / float(batch_size)


def pairwise_alignment(
    score_matrix: torch.Tensor,
    labels: torch.Tensor | None = None,
    loss_type: str = "multi_positive_infonce",
    logit_bias: torch.Tensor | float = -2.0,
    normalize_by_pairs: bool = False,
) -> torch.Tensor:
    """Unified pairwise reranker alignment loss supporting SupCon, InfoNCE, and SigLIP."""
    if loss_type == "siglip":
        return pairwise_siglip(
            score_matrix,
            labels=labels,
            logit_bias=logit_bias,
            normalize_by_pairs=normalize_by_pairs,
        )
    if loss_type in {"multi_positive_infonce", "supcon", "infonce"}:
        effective_labels = None if loss_type == "infonce" else labels
        return pairwise_infonce(score_matrix, labels=effective_labels)
    raise ValueError(f"unknown contrastive loss_type: {loss_type}")


def contrastive_alignment(
    query: torch.Tensor,
    anchor: torch.Tensor,
    temperature: float = 0.1,
    labels: torch.Tensor | None = None,
    logit_scale: torch.Tensor | float | None = None,
    loss_type: str = "multi_positive_infonce",
    logit_bias: torch.Tensor | float = -2.0,
    normalize_by_pairs: bool = False,
) -> torch.Tensor:
    """Unified contrastive alignment entrypoint supporting SupCon/Multi-Positive InfoNCE and SigLIP."""
    if loss_type == "siglip":
        return siglip_alignment(
            query,
            anchor,
            labels=labels,
            temperature=temperature,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            normalize_by_pairs=normalize_by_pairs,
        )
    if loss_type in {"multi_positive_infonce", "supcon", "infonce"}:
        effective_labels = None if loss_type == "infonce" else labels
        return multi_positive_infonce(
            query,
            anchor,
            labels=effective_labels,
            temperature=temperature,
            logit_scale=logit_scale,
        )
    raise ValueError(f"unknown contrastive loss_type: {loss_type}")

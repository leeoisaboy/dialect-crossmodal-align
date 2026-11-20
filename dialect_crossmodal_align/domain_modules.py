from __future__ import annotations

import math

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return features.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.alpha * gradient, None


def dann_alpha_schedule(progress: float, gamma: float = 10.0) -> float:
    """Standard DANN smooth sigmoid schedule: 2 / (1 + exp(-gamma * p)) - 1."""
    clamped = max(0.0, min(1.0, float(progress)))
    return float(2.0 / (1.0 + math.exp(-gamma * clamped)) - 1.0)


class DomainAdversarialHead(nn.Module):
    """Domain discriminator with Gradient Reversal Layer (GRL)."""

    def __init__(self, input_dim: int, domain_count: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, domain_count),
        )

    def forward(self, pooled: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self.classifier(GradientReversal.apply(pooled, alpha))


def _marginal_domain_alignment(
    features: torch.Tensor,
    domains: torch.Tensor,
    cov_weight: float,
    shrinkage: float,
) -> torch.Tensor:
    unique_domains = domains.unique()
    if unique_domains.numel() < 2:
        return features.new_zeros(())

    means: list[torch.Tensor] = []
    covs: list[torch.Tensor | None] = []
    dim = features.shape[-1]
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for domain_id in unique_domains:
        subset = features[domains == domain_id]
        if subset.shape[0] < 1:
            continue
        mean = subset.mean(dim=0)
        means.append(mean)
        if subset.shape[0] >= 2 and cov_weight > 0.0:
            centered = subset - mean.unsqueeze(0)
            cov = (centered.T @ centered) / float(subset.shape[0] - 1)
            if shrinkage > 0.0:
                trace_target = torch.trace(cov) / float(dim)
                cov = (1.0 - shrinkage) * cov + shrinkage * trace_target * eye
            covs.append(cov)
        else:
            covs.append(None)

    if len(means) < 2:
        return features.new_zeros(())

    total = features.new_zeros(())
    pairs = 0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            mean_diff = means[i] - means[j]
            pair_loss = (mean_diff * mean_diff).sum()
            if cov_weight > 0.0 and covs[i] is not None and covs[j] is not None:
                cov_diff = covs[i] - covs[j]
                pair_loss = pair_loss + cov_weight * (cov_diff * cov_diff).sum()
            total = total + pair_loss
            pairs += 1
    return total / float(max(1, pairs))


def domain_alignment_loss(
    features: torch.Tensor,
    domains: torch.Tensor,
    labels: torch.Tensor | None = None,
    cov_weight: float = 1.0,
    shrinkage: float = 0.1,
) -> torch.Tensor:
    """Class-conditional (when ``labels`` provided) unit-sphere domain distribution alignment.

    When ``labels`` is supplied, aligns first-order means and shrinkage-stabilized
    second-order covariances **within each class** across domains, preventing
    small-batch class-imbalance from pulling different words together across domains.
    Falls back to marginal alignment only if ``labels is None``.
    """
    if labels is None:
        return _marginal_domain_alignment(features, domains, cov_weight=cov_weight, shrinkage=shrinkage)

    unique_classes = labels.unique()
    class_losses: list[torch.Tensor] = []
    for cls_id in unique_classes:
        cls_mask = labels == cls_id
        cls_domains = domains[cls_mask]
        if cls_domains.unique().numel() < 2:
            continue
        cls_loss = _marginal_domain_alignment(
            features[cls_mask],
            cls_domains,
            cov_weight=cov_weight,
            shrinkage=shrinkage,
        )
        class_losses.append(cls_loss)

    if not class_losses:
        return features.new_zeros(())
    return torch.stack(class_losses).mean()


def coral_loss(
    features: torch.Tensor,
    domains: torch.Tensor,
    labels: torch.Tensor | None = None,
    cov_weight: float = 1.0,
    shrinkage: float = 0.1,
) -> torch.Tensor:
    """Backward-compatible alias for ``domain_alignment_loss``."""
    return domain_alignment_loss(
        features,
        domains,
        labels=labels,
        cov_weight=cov_weight,
        shrinkage=shrinkage,
    )


class FSQBottleneck(nn.Module):
    """Finite Scalar Quantization (Mentzer et al.) with pre-norm, mask-aware perplexity, and int64 overflow protection."""

    def __init__(
        self,
        input_dim: int,
        levels: tuple[int, ...] = (5, 5, 5, 5, 5, 5),
        output_dim: int | None = None,
    ):
        super().__init__()
        if not levels or any(level < 2 for level in levels):
            raise ValueError("each FSQ level must be an integer >= 2")
        self.levels = tuple(int(level) for level in levels)
        self.codebook_size = int(math.prod(self.levels))
        self._overflow_safe_hash = self.codebook_size >= (1 << 62)

        levels_tensor = torch.tensor(self.levels, dtype=torch.float32)
        half_l = (levels_tensor - 1.0) / 2.0
        offset = torch.where((levels_tensor.long() % 2) == 0, 0.5, 0.0)
        self.register_buffer("half_l", half_l)
        self.register_buffer("offset", offset)

        if not self._overflow_safe_hash:
            basis = torch.ones(len(self.levels), dtype=torch.long)
            running = 1
            for idx, level in enumerate(self.levels):
                basis[idx] = running
                running *= level
            self.register_buffer("basis", basis)
        else:
            self.register_buffer("basis", torch.zeros(len(self.levels), dtype=torch.long))

        self.pre_norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, len(self.levels))
        self.out_project: nn.Module | None = (
            nn.Sequential(
                nn.Linear(len(self.levels), output_dim),
                nn.LayerNorm(output_dim),
            )
            if output_dim is not None
            else None
        )
        self.last_perplexity: float = 0.0
        self.last_unique_ratio: float = 0.0

    def quantize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Quantize unconstrained latent ``[..., D_fsq]`` to exact discrete codes in ``[-1, 1]``."""
        scaled = torch.tanh(latent) * self.half_l
        quantized = torch.round(scaled - self.offset) + self.offset
        straight_through = scaled + (quantized - scaled).detach()
        return straight_through / self.half_l

    def codes_to_zero_based_ints(self, codes: torch.Tensor) -> torch.Tensor:
        scaled = codes.detach() * self.half_l
        int_codes = torch.round(scaled - self.offset).long()
        min_int = torch.round(-self.half_l - self.offset).long()
        return (int_codes - min_int).clamp_min(0)

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert normalized discrete codes ``[..., D_fsq]`` to flat int64 codebook indices."""
        if self._overflow_safe_hash:
            raise OverflowError(
                f"FSQ codebook_size={self.codebook_size} exceeds int64 range; "
                "use codebook_metrics() tuple counting instead of flat int64 indices."
            )
        zero_based = self.codes_to_zero_based_ints(codes)
        return (zero_based * self.basis).sum(dim=-1)

    def codebook_metrics(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Return ``(perplexity, unique_code_ratio)`` strictly over valid (unmasked) elements."""
        if mask is not None:
            valid_codes = codes.detach()[mask.bool()]
        else:
            valid_codes = codes.detach().reshape(-1, codes.shape[-1])
        total = int(valid_codes.shape[0])
        if total == 0:
            return 0.0, 0.0

        if self._overflow_safe_hash:
            zero_based = self.codes_to_zero_based_ints(valid_codes).cpu().tolist()
            counts_map: dict[tuple[int, ...], int] = {}
            for row in zero_based:
                key = tuple(row)
                counts_map[key] = counts_map.get(key, 0) + 1
            counts = torch.tensor(list(counts_map.values()), dtype=torch.float32)
            num_unique = len(counts_map)
        else:
            flat_indices = self.codes_to_indices(valid_codes)
            unique_ids, counts = torch.unique(flat_indices, return_counts=True)
            counts = counts.to(torch.float32)
            num_unique = int(unique_ids.numel())

        probs = counts / float(total)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        perplexity = float(torch.exp(entropy).item())
        unique_ratio = float(num_unique) / float(total)
        return perplexity, unique_ratio

    def quantize(self, features: torch.Tensor) -> torch.Tensor:
        """Return the low-dimensional discrete FSQ codes in ``[-1, 1]``."""
        return self.quantize_latent(self.project(self.pre_norm(features)))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.quantize(features)
        perplexity, unique_ratio = self.codebook_metrics(codes, mask=mask)
        self.last_perplexity = perplexity
        self.last_unique_ratio = unique_ratio
        output = self.out_project(codes) if self.out_project is not None else codes
        commitment = features.new_zeros(())
        return output, commitment

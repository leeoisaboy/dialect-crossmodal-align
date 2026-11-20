from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .aggregators import TextCrossAttention, masked_mean


@dataclass
class AlignmentOutput:
    query: torch.Tensor
    anchor: torch.Tensor
    logits: torch.Tensor
    pooled: torch.Tensor
    logit_scale: torch.Tensor
    logit_bias: torch.Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Parameter-free sinusoidal positional encoding supporting arbitrary sequence lengths."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        half_dim = (self.model_dim + 1) // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, half_dim))
        )
        angles = positions * div_term.unsqueeze(0)
        pe = torch.zeros(length, self.model_dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles[:, : pe[:, 0::2].shape[1]])
        if self.model_dim > 1:
            pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0).to(dtype)


class NormalizedRelativePositionalEncoding(nn.Module):
    """Speaking-rate-invariant relative [0, 1] sinusoidal positional encoding for acoustic frames.

    Maps each valid frame's ordinal position ``p in [0, L-1]`` to normalized progress
    ``r = p / max(1, L - 1) in [0, 1]`` so Q-Former latent queries distinguish onset,
    nucleus, tone inflection, and coda regardless of utterance speaking rate.
    """

    def __init__(self, model_dim: int, max_freq: float = 100.0):
        super().__init__()
        self.model_dim = model_dim
        self.max_freq = float(max_freq)

    def forward(self, frame_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        valid = frame_mask.bool()
        lengths = valid.sum(dim=1, keepdim=True).clamp_min(1)  # [B, 1]
        valid_pos = (valid.long().cumsum(dim=1) - 1).clamp_min(0)  # [B, T]
        denom = (lengths - 1).clamp_min(1).to(torch.float32)
        rel_pos = valid_pos.to(torch.float32) / denom  # [B, T] in [0, 1]

        half_dim = (self.model_dim + 1) // 2
        freqs = torch.exp(
            torch.linspace(0.0, math.log(self.max_freq), half_dim, device=frame_mask.device, dtype=torch.float32)
        )
        angles = rel_pos.unsqueeze(-1) * freqs.view(1, 1, -1) * math.pi  # [B, T, half_dim]
        pe = torch.zeros((*frame_mask.shape, self.model_dim), device=frame_mask.device, dtype=torch.float32)
        pe[..., 0::2] = torch.sin(angles[..., : pe[..., 0::2].shape[-1]])
        if self.model_dim > 1:
            pe[..., 1::2] = torch.cos(angles[..., : pe[..., 1::2].shape[-1]])
        return pe.masked_fill(~valid.unsqueeze(-1), 0.0).to(dtype)


class TextSequenceEncoder(nn.Module):
    """Order-sensitive, padding-safe text encoder (sinusoidal PE + depthwise Conv1D + attention pool)."""

    def __init__(self, vocabulary_size: int, model_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, model_dim, padding_idx=0)
        self.positional = SinusoidalPositionalEncoding(model_dim)
        self.depthwise_conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=model_dim,
        )
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.pool_scorer = nn.Linear(model_dim, 1)

    def encode_sequence(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        valid = text_mask.bool()
        safe_tokens = text_tokens.masked_fill(~valid, 0)
        embedded = self.embedding(safe_tokens)
        pe = self.positional(text_tokens.shape[1], embedded.device, embedded.dtype)
        seq = (embedded + pe).masked_fill(~valid.unsqueeze(-1), 0.0)
        conv_out = self.depthwise_conv(seq.transpose(1, 2)).transpose(1, 2)
        conv_out = conv_out.masked_fill(~valid.unsqueeze(-1), 0.0)
        contextual = seq + self.ffn(self.norm(conv_out))
        return contextual.masked_fill(~valid.unsqueeze(-1), 0.0)

    def pool(self, sequence: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        valid = text_mask.bool()
        scores = self.pool_scorer(sequence).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).masked_fill(~valid, 0.0)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.encode_sequence(text_tokens, text_mask)
        pooled = self.pool(seq, text_mask)
        return seq, pooled


class _LearnableScaleMixin(nn.Module):
    """Provides learnable log-temperature scale in [1, 100] and learnable SigLIP logit bias."""

    def __init__(self, init_scale: float = 10.0, init_bias: float = -2.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(float(init_scale))))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_bias)))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(1.0, 100.0)


class FixedSlotHead(_LearnableScaleMixin):
    """A closed vocabulary head with one learned acoustic-to-class anchor."""

    def __init__(self, model_dim: int, class_count: int):
        super().__init__(init_scale=10.0)
        self.projection = nn.Linear(model_dim, model_dim)
        self.anchors = nn.Parameter(torch.randn(class_count, model_dim) * 0.02)

    def forward(
        self,
        pooled: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        sequence: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> AlignmentOutput:
        query = F.normalize(self.projection(pooled), dim=-1)
        anchors = F.normalize(self.anchors, dim=-1)
        scale = self.scale
        return AlignmentOutput(query, anchors, query @ anchors.T * scale, pooled, scale, self.logit_bias)


class TextAnchoredHead(_LearnableScaleMixin):
    """Map audio into the same dense space as an order-sensitive variable-length text query."""

    def __init__(self, model_dim: int, vocabulary_size: int, dropout: float = 0.1):
        super().__init__(init_scale=10.0)
        self.text_encoder = TextSequenceEncoder(vocabulary_size, model_dim, dropout=dropout)
        self.projection = nn.Linear(model_dim, model_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    @property
    def embedding(self) -> nn.Embedding:
        return self.text_encoder.embedding

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        _, text_pooled = self.text_encoder(text_tokens, text_mask)
        return F.normalize(self.text_projection(text_pooled), dim=-1)

    def forward(
        self,
        pooled: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        sequence: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> AlignmentOutput:
        query = F.normalize(self.projection(pooled), dim=-1)
        anchor = self.encode_text(text_tokens, text_mask)
        scale = self.scale
        logits = (query * anchor).sum(dim=-1, keepdim=True) * scale
        return AlignmentOutput(query, anchor, logits, pooled, scale, self.logit_bias)


class AudioToTextSpaceHead(_LearnableScaleMixin):
    """Project acoustic states toward order-sensitive learned or pretrained text anchors."""

    def __init__(
        self,
        model_dim: int,
        vocabulary_size: int,
        text_embedding_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(init_scale=10.0)
        self.audio_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        text_dim = text_embedding_dim or model_dim
        self.text_encoder = TextSequenceEncoder(vocabulary_size, text_dim, dropout=dropout)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim, elementwise_affine=False),
        )
        self.background = nn.Parameter(torch.zeros(model_dim))

    @property
    def raw_text(self) -> nn.Embedding:
        return self.text_encoder.embedding

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        _, text_pooled = self.text_encoder(text_tokens, text_mask)
        return F.normalize(self.text_projection(text_pooled) + self.background, dim=-1)

    def forward(
        self,
        pooled: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        sequence: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> AlignmentOutput:
        query = F.normalize(self.audio_projection(pooled), dim=-1)
        anchor = self.encode_text(text_tokens, text_mask)
        scale = self.scale
        logits = (query * anchor).sum(dim=-1, keepdim=True) * scale
        return AlignmentOutput(query, anchor, logits, pooled, scale, self.logit_bias)


class LMBridgedHead(_LearnableScaleMixin):
    """Leak-free Q-Former dual-tower head or zero-redundancy pairwise reranker.

    - Injects speaking-rate-invariant normalized relative positional encoding
      into the acoustic ``sequence`` before cross-attention so queries distinguish
      global temporal order.
    - Only registers ``latent_queries`` in ``qformer`` mode (zero unused DDP
      parameters in ``pairwise`` mode).
    - Reuses ``text_seq`` and ``anchors`` in ``pairwise`` mode without redundant
      text encoding.
    """

    def __init__(
        self,
        model_dim: int,
        vocabulary_size: int,
        heads: int = 4,
        dropout: float = 0.1,
        num_latents: int = 4,
        interaction_mode: str = "qformer",
    ):
        super().__init__(init_scale=10.0)
        if interaction_mode not in {"qformer", "pairwise"}:
            raise ValueError(f"unknown interaction_mode: {interaction_mode}")
        self.interaction_mode = interaction_mode
        self.num_latents = int(num_latents)
        self.text_encoder = TextSequenceEncoder(vocabulary_size, model_dim, dropout=dropout)
        self.audio_pos = NormalizedRelativePositionalEncoding(model_dim)
        if self.interaction_mode == "qformer":
            self.latent_queries: nn.Parameter | None = nn.Parameter(
                torch.randn(1, self.num_latents, model_dim) * 0.02
            )
        else:
            self.latent_queries = None
        self.attention = TextCrossAttention(model_dim, heads=heads, dropout=dropout)
        self.bridger = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    @property
    def text_embedding(self) -> nn.Embedding:
        return self.text_encoder.embedding

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        _, text_pooled = self.text_encoder(text_tokens, text_mask)
        return F.normalize(self.text_projection(text_pooled), dim=-1)

    def _add_audio_position(self, sequence: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        pos = self.audio_pos(frame_mask, sequence.dtype)
        return (sequence + pos).masked_fill(~frame_mask.bool().unsqueeze(-1), 0.0)

    def encode_audio_qformer(
        self,
        pooled: torch.Tensor,
        sequence: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.latent_queries is None:
            raise RuntimeError("encode_audio_qformer is only available in interaction_mode='qformer'")
        batch = sequence.shape[0]
        pos_seq = self._add_audio_position(sequence, frame_mask)
        latents = self.latent_queries.expand(batch, -1, -1)
        latent_mask = torch.ones((batch, self.num_latents), dtype=torch.bool, device=sequence.device)
        attended = self.attention(pos_seq, latents, frame_mask, latent_mask)
        attended_audio = attended.mean(dim=1)
        return F.normalize(self.bridger(torch.cat((attended_audio, pooled), dim=-1)), dim=-1)

    def _score_pairs_from_encoded(
        self,
        pooled: torch.Tensor,
        sequence: torch.Tensor,
        frame_mask: torch.Tensor,
        text_seq: torch.Tensor,
        text_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        batch_a, frames, dim = sequence.shape
        batch_t, tok_len = text_mask.shape
        pos_seq = self._add_audio_position(sequence, frame_mask)

        seq_exp = pos_seq.unsqueeze(1).expand(batch_a, batch_t, frames, dim).reshape(batch_a * batch_t, frames, dim)
        fmask_exp = frame_mask.unsqueeze(1).expand(batch_a, batch_t, frames).reshape(batch_a * batch_t, frames)
        pooled_exp = pooled.unsqueeze(1).expand(batch_a, batch_t, dim).reshape(batch_a * batch_t, dim)
        tseq_exp = text_seq.unsqueeze(0).expand(batch_a, batch_t, tok_len, dim).reshape(batch_a * batch_t, tok_len, dim)
        tmask_exp = text_mask.unsqueeze(0).expand(batch_a, batch_t, tok_len).reshape(batch_a * batch_t, tok_len)

        attended = self.attention(seq_exp, tseq_exp, fmask_exp, tmask_exp)
        attended_audio = masked_mean(attended, tmask_exp)
        pair_query = F.normalize(self.bridger(torch.cat((attended_audio, pooled_exp), dim=-1)), dim=-1)
        pair_query = pair_query.reshape(batch_a, batch_t, dim)
        return torch.einsum("ijd,jd->ij", pair_query, anchors) * self.scale

    def score_pairs(
        self,
        pooled: torch.Tensor,
        sequence: torch.Tensor,
        frame_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_seq, text_pooled = self.text_encoder(text_tokens, text_mask)
        anchors = F.normalize(self.text_projection(text_pooled), dim=-1)
        return self._score_pairs_from_encoded(pooled, sequence, frame_mask, text_seq, text_mask, anchors)

    def forward(
        self,
        pooled: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        sequence: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> AlignmentOutput:
        if sequence is None or frame_mask is None:
            raise ValueError("LMBridgedHead requires frame-level acoustic `sequence` and `frame_mask`.")
        scale = self.scale
        if self.interaction_mode == "pairwise":
            # Single text pass; no dead latent_queries; query exposes normalized audio state for domain alignment.
            text_seq, text_pooled = self.text_encoder(text_tokens, text_mask)
            anchor = F.normalize(self.text_projection(text_pooled), dim=-1)
            logits = self._score_pairs_from_encoded(pooled, sequence, frame_mask, text_seq, text_mask, anchor)
            query = F.normalize(pooled, dim=-1)
            return AlignmentOutput(query, anchor, logits, pooled, scale, self.logit_bias)

        query = self.encode_audio_qformer(pooled, sequence, frame_mask)
        anchor = self.encode_text(text_tokens, text_mask)
        logits = (query * anchor).sum(dim=-1, keepdim=True) * scale
        return AlignmentOutput(query, anchor, logits, pooled, scale, self.logit_bias)


HEADS = {
    "fixed_slot": FixedSlotHead,
    "text_anchored": TextAnchoredHead,
    "audio_to_text_space": AudioToTextSpaceHead,
    "lm_bridged": LMBridgedHead,
}

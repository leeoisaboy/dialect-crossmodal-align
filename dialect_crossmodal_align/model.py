from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .aggregators import MeanPooling, MultiStatisticPNAPooling, ProjectedROIPooling
from .backbones import LinearDialectAdapter
from .domain_modules import DomainAdversarialHead, FSQBottleneck
from .heads import HEADS, AlignmentOutput


DEFAULT_FSQ_LEVELS: tuple[int, ...] = (5, 5, 5, 5, 5, 5)


class AlignmentModel(nn.Module):
    """A configurable adapter, temporal aggregator, and dense alignment head."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        vocabulary_size: int,
        class_count: int,
        head: str = "text_anchored",
        aggregator: str = "relative_phase",
        domain_count: int = 0,
        use_domain_adversarial: bool = False,
        use_fsq: bool = False,
        fsq_levels: Sequence[int] | None = None,
        domain_target: str = "query",
        lm_interaction_mode: str = "qformer",
        adapter_dilations: Sequence[int] = (1, 2, 4),
        pe_mode: str = "relative",
        dropout: float = 0.1,
    ):
        super().__init__()
        if domain_target not in {"query", "pooled"}:
            raise ValueError(f"unknown domain_target: {domain_target}")
        self.domain_target = domain_target
        self.head_name = head
        self.lm_interaction_mode = lm_interaction_mode
        self.backbone = LinearDialectAdapter(
            input_dim,
            model_dim,
            dropout=dropout,
            dilations=adapter_dilations,
        )
        if aggregator == "relative_phase":
            from .pipeline import RelativePhasePooling

            self.aggregator: nn.Module = RelativePhasePooling(model_dim, num_phases=4, pe_mode=pe_mode)
        elif aggregator == "mean":
            self.aggregator = MeanPooling()
        elif aggregator == "roi":
            self.aggregator = ProjectedROIPooling(model_dim, bins=4, input_dim=model_dim)
        elif aggregator == "pna_roi":
            self.aggregator = MultiStatisticPNAPooling(
                model_dim,
                bins=4,
                output_dim=model_dim,
            )
        else:
            raise ValueError(f"unknown aggregator: {aggregator}")
        if head not in HEADS:
            raise ValueError(f"unknown head: {head}")
        kwargs: dict[str, Any] = {"model_dim": model_dim}
        if head == "fixed_slot":
            kwargs["class_count"] = class_count
        else:
            kwargs["vocabulary_size"] = vocabulary_size
            kwargs["dropout"] = dropout
            if head == "lm_bridged":
                kwargs["interaction_mode"] = lm_interaction_mode
        self.head = HEADS[head](**kwargs)

        if use_fsq:
            resolved_levels = tuple(int(x) for x in (fsq_levels or DEFAULT_FSQ_LEVELS))
            self.fsq: FSQBottleneck | None = FSQBottleneck(
                model_dim,
                levels=resolved_levels,
                output_dim=model_dim,
            )
            # Dedicated sequence-level FSQ prevents distribution conflict & padding pollution.
            self.sequence_fsq: FSQBottleneck | None = (
                FSQBottleneck(model_dim, levels=resolved_levels, output_dim=model_dim)
                if head == "lm_bridged"
                else None
            )
        else:
            self.fsq = None
            self.sequence_fsq = None

        self.domain_head = (
            DomainAdversarialHead(model_dim, domain_count)
            if (use_domain_adversarial and domain_count > 1)
            else None
        )

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.head, "encode_text"):
            raise TypeError(f"head {self.head_name!r} does not support open-vocabulary encode_text")
        return self.head.encode_text(text_tokens, text_mask)

    def forward(
        self,
        frames: torch.Tensor,
        frame_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        domain_alpha: float = 1.0,
    ) -> dict[str, Any]:
        sequence = self.backbone(frames, frame_mask)
        aggregate = self.aggregator(sequence, frame_mask)
        pooled = aggregate.pooled
        head_sequence = aggregate.sequence
        commitment = pooled.new_zeros(())
        fsq_perplexity = 0.0
        fsq_unique_ratio = 0.0
        sequence_fsq_perplexity = 0.0

        if self.fsq is not None:
            pooled, commitment = self.fsq(pooled)
            fsq_perplexity = self.fsq.last_perplexity
            fsq_unique_ratio = self.fsq.last_unique_ratio

        if self.sequence_fsq is not None:
            valid_mask = frame_mask.bool()
            seq_quant, _ = self.sequence_fsq(head_sequence, mask=valid_mask)
            head_sequence = seq_quant.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            sequence_fsq_perplexity = self.sequence_fsq.last_perplexity

        output: AlignmentOutput = self.head(
            pooled,
            text_tokens,
            text_mask,
            sequence=head_sequence,
            frame_mask=frame_mask,
        )
        result: dict[str, Any] = {
            "query": output.query,
            "anchor": output.anchor,
            "logits": output.logits,
            "logit_scale": output.logit_scale,
            "logit_bias": output.logit_bias,
            "pooled": pooled,
            "sequence": head_sequence,
            "fsq_commitment": commitment,
            "fsq_perplexity": fsq_perplexity,
            "fsq_unique_ratio": fsq_unique_ratio,
            "sequence_fsq_perplexity": sequence_fsq_perplexity,
        }
        if self.domain_head is not None:
            domain_input = output.query if self.domain_target == "query" else pooled
            result["domain_logits"] = self.domain_head(domain_input, domain_alpha)
        return result


def build_alignment_model(config: dict[str, Any]) -> AlignmentModel:
    raw_fsq_levels = config.get("fsq_levels")
    fsq_levels = tuple(int(v) for v in raw_fsq_levels) if raw_fsq_levels is not None else None
    raw_dilations = config.get("adapter_dilations")
    adapter_dilations = tuple(int(v) for v in raw_dilations) if raw_dilations is not None else (1, 2, 4)
    domain_count = int(config.get("domain_count", 0))
    use_domain_adversarial = bool(
        config.get("use_domain_adversarial", float(config.get("domain_alpha_max", 0.0)) > 0.0)
    )
    return AlignmentModel(
        input_dim=int(config["input_dim"]),
        model_dim=int(config["model_dim"]),
        vocabulary_size=int(config["vocabulary_size"]),
        class_count=int(config["class_count"]),
        head=str(config.get("head", "text_anchored")),
        aggregator=str(config.get("aggregator", "relative_phase")),
        domain_count=domain_count,
        use_domain_adversarial=use_domain_adversarial,
        use_fsq=bool(config.get("use_fsq", False)),
        fsq_levels=fsq_levels,
        domain_target=str(config.get("domain_target", "query")),
        lm_interaction_mode=str(config.get("lm_interaction_mode", "qformer")),
        adapter_dilations=adapter_dilations,
        pe_mode=str(config.get("pe_mode", "relative")),
        dropout=float(config.get("dropout", 0.1)),
    )

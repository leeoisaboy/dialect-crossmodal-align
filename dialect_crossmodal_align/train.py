from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .criteria import CosFaceProjection, contrastive_alignment, pairwise_alignment
from .data import SyntheticDialectWorld
from .domain_modules import dann_alpha_schedule, domain_alignment_loss
from .model import AlignmentModel, build_alignment_model


NO_DECAY_KEYWORDS = (
    "bias",
    "norm",
    "log_scale",
    "logit_bias",
    "bin_pos",
    "latent_queries",
    "background",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_batch(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {key: value[:count] for key, value in batch.items()}


def build_optimizer_param_groups(
    modules: Iterable[nn.Module],
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Separate 2D+ weight tensors (decayed) from 1D/bias/norm/temperature/position params (no_decay)."""
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            lowered = name.lower()
            if param.ndim < 2 or any(token in lowered for token in NO_DECAY_KEYWORDS):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def _warmup_cosine_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.05,
) -> float:
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _multi_positive_ranking_metrics(
    scores: torch.Tensor,
    row_classes: torch.Tensor,
    col_classes: torch.Tensor,
) -> dict[str, float]:
    """Compute exact Recall@1, Recall@3, and MRR for both deduplicated galleries and multi-positive batches."""
    num_candidates = scores.shape[-1]
    sorted_indices = scores.argsort(dim=-1, descending=True)
    ranked_classes = col_classes[sorted_indices]  # [B, M]
    matches = ranked_classes == row_classes.unsqueeze(1)  # [B, M]
    # Best (smallest 1-indexed) rank among all matching positive columns.
    has_match = matches.any(dim=-1)
    best_ranks = matches.float().argmax(dim=-1) + 1
    best_ranks = torch.where(has_match, best_ranks, torch.full_like(best_ranks, num_candidates + 1))

    recall_at_1 = (best_ranks <= 1).float().mean()
    recall_at_3 = (best_ranks <= min(3, num_candidates)).float().mean()
    mrr = torch.where(has_match, 1.0 / best_ranks.float(), torch.zeros_like(best_ranks, dtype=torch.float32)).mean()
    return {
        "recall_at_1": float(recall_at_1),
        "recall_at_3": float(recall_at_3),
        "mrr": float(mrr),
    }


def evaluate(
    model: AlignmentModel,
    batch: dict[str, torch.Tensor],
    criterion: CosFaceProjection | None,
    gallery: dict[str, torch.Tensor] | None = None,
    contrastive_loss_type: str = "multi_positive_infonce",
    temperature: float = 0.1,
) -> dict[str, float]:
    model.eval()
    if criterion is not None:
        criterion.eval()
    with torch.inference_mode():
        output = model(
            batch["frames"],
            batch["frame_mask"],
            batch["text_tokens"],
            batch["text_mask"],
        )
        cosface_loss = torch.zeros((), device=batch["frames"].device)
        val_alignment_loss = torch.zeros((), device=batch["frames"].device)
        val_siglip_pair_loss = 0.0

        if criterion is not None:
            logits = criterion.logits(output["query"])
            cosface_loss = criterion(output["query"], batch["classes"])
            val_alignment_loss = cosface_loss
            col_classes = torch.arange(logits.shape[-1], device=logits.device)
            ranking = _multi_positive_ranking_metrics(logits, batch["classes"], col_classes)
            accuracy = ranking["recall_at_1"]
        elif model.head_name == "fixed_slot":
            logits = output["logits"]
            val_alignment_loss = F.cross_entropy(logits, batch["classes"])
            col_classes = torch.arange(logits.shape[-1], device=logits.device)
            ranking = _multi_positive_ranking_metrics(logits, batch["classes"], col_classes)
            accuracy = ranking["recall_at_1"]
        elif model.head_name == "lm_bridged" and model.lm_interaction_mode == "pairwise":
            val_alignment_loss = pairwise_alignment(
                output["logits"],
                labels=batch["classes"],
                loss_type=contrastive_loss_type,
                logit_bias=output["logit_bias"],
            )
            if contrastive_loss_type == "siglip":
                val_siglip_pair_loss = float(
                    pairwise_alignment(
                        output["logits"],
                        labels=batch["classes"],
                        loss_type="siglip",
                        logit_bias=output["logit_bias"],
                        normalize_by_pairs=True,
                    )
                )
            if gallery is not None:
                gallery_scores = model.head.score_pairs(
                    output["pooled"],
                    output["sequence"],
                    batch["frame_mask"],
                    gallery["text_tokens"],
                    gallery["text_mask"],
                )
                ranking = _multi_positive_ranking_metrics(
                    gallery_scores, batch["classes"], gallery["classes"]
                )
            else:
                ranking = _multi_positive_ranking_metrics(
                    output["logits"], batch["classes"], batch["classes"]
                )
            accuracy = ranking["recall_at_1"]
        else:
            val_alignment_loss = contrastive_alignment(
                output["query"],
                output["anchor"],
                labels=batch["classes"],
                temperature=temperature,
                logit_scale=output["logit_scale"],
                loss_type=contrastive_loss_type,
                logit_bias=output["logit_bias"],
            )
            if contrastive_loss_type == "siglip":
                val_siglip_pair_loss = float(
                    contrastive_alignment(
                        output["query"],
                        output["anchor"],
                        labels=batch["classes"],
                        temperature=temperature,
                        logit_scale=output["logit_scale"],
                        loss_type="siglip",
                        logit_bias=output["logit_bias"],
                        normalize_by_pairs=True,
                    )
                )
            if gallery is not None:
                gallery_anchors = model.encode_text(gallery["text_tokens"], gallery["text_mask"])
                gallery_scores = F.normalize(output["query"], dim=-1) @ gallery_anchors.T
                ranking = _multi_positive_ranking_metrics(
                    gallery_scores, batch["classes"], gallery["classes"]
                )
            else:
                sim = F.normalize(output["query"], dim=-1) @ F.normalize(output["anchor"], dim=-1).T
                ranking = _multi_positive_ranking_metrics(sim, batch["classes"], batch["classes"])
            accuracy = ranking["recall_at_1"]

        return {
            "accuracy": float(accuracy),
            "val_alignment_loss": float(val_alignment_loss),
            "val_siglip_pair_loss": float(val_siglip_pair_loss),
            "cosface_loss": float(cosface_loss),
            "fsq_perplexity": float(output["fsq_perplexity"]),
            "fsq_unique_ratio": float(output["fsq_unique_ratio"]),
            "sequence_fsq_perplexity": float(output["sequence_fsq_perplexity"]),
            **ranking,
        }


def train_model(
    config: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    seed = int(config.get("seed", 17))
    torch.manual_seed(seed)
    model = build_alignment_model(config).to(device)
    domain_count = int(config.get("domain_count", 0))
    epochs = int(config.get("epochs", 3))
    steps_per_epoch = int(config.get("steps_per_epoch", 1))
    total_steps = max(1, epochs * steps_per_epoch)

    sub_center_temp_start = float(config.get("cosface_sub_center_temp_start", 10.0))
    sub_center_temp_end = float(config.get("cosface_sub_center_temp_end", 40.0))
    criterion = (
        CosFaceProjection(
            int(config["model_dim"]),
            int(config["class_count"]),
            margin=float(config.get("cosface_margin", 0.2)),
            scale=float(config.get("cosface_scale", 32.0)),
            sub_centers=int(config.get("cosface_sub_centers", 3)),
            sub_center_temp=sub_center_temp_start,
        ).to(device)
        if config.get("criterion") == "cosface"
        else None
    )

    modules: list[nn.Module] = [model, *([criterion] if criterion is not None else [])]
    param_groups = build_optimizer_param_groups(
        modules, weight_decay=float(config.get("weight_decay", 0.01))
    )
    all_params = [param for group in param_groups for param in group["params"]]
    base_lr = float(config.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if total_steps > 2 else 0
    min_lr_ratio = float(config.get("min_lr_ratio", 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_factor(step, total_steps, warmup_steps, min_lr_ratio),
    )

    raw_syllables = config.get("syllable_vocab_size")
    world = SyntheticDialectWorld(
        input_dim=int(config["input_dim"]),
        vocabulary_size=int(config["vocabulary_size"]),
        class_count=int(config["class_count"]),
        domain_count=max(1, domain_count),
        num_segments=int(config.get("num_segments", 3)),
        seed=seed,
        compositional=bool(config.get("compositional", True)),
        syllable_vocab_size=int(raw_syllables) if raw_syllables is not None else None,
    )
    train_domains = config.get("train_domains")
    val_domains = config.get("val_domains")
    train_classes = config.get("train_classes")
    val_classes = config.get("val_classes")
    if world.compositional and train_classes is not None and val_classes is not None:
        if not world.verify_oov_syllable_coverage(train_classes, val_classes, per_position=False):
            raise ValueError(
                "Zero-shot OOV split contains syllables never observed in train_classes; "
                "reduce syllable_vocab_size or expand train_classes."
            )
    gallery = world.canonical_gallery(device=device, allowed_classes=val_classes)
    validation = world.sample_batch(
        batch_size=int(config.get("val_batch_size", 64)),
        seed=seed + 991,
        device=device,
        allowed_domains=val_domains,
        allowed_classes=val_classes,
    )

    contrastive_loss_type = str(config.get("contrastive_loss", "multi_positive_infonce"))
    temperature = float(config.get("temperature", 0.1))
    dann_gamma = float(config.get("dann_gamma", 10.0))
    use_domain_adv = model.domain_head is not None
    domain_alpha_max = float(config.get("domain_alpha_max", 1.0 if use_domain_adv else 0.0))
    coral_weight = float(config.get("coral_weight", 0.0))

    history = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        if criterion is not None:
            criterion.train()
        last_alignment_loss = 0.0
        last_domain_loss = 0.0
        last_coral_loss = 0.0
        last_fsq_perplexity = 0.0
        domain_alpha = 0.0

        for step in range(steps_per_epoch):
            progress = (global_step + 1) / total_steps
            if criterion is not None and criterion.sub_centers > 1:
                # Anneal smooth LSE temperature toward sharp max to close train/eval gap.
                criterion.sub_center_temp = sub_center_temp_start + progress * (
                    sub_center_temp_end - sub_center_temp_start
                )

            domain_alpha = (
                domain_alpha_max * dann_alpha_schedule(progress, gamma=dann_gamma)
                if use_domain_adv
                else 0.0
            )
            batch = world.sample_batch(
                batch_size=int(config.get("batch_size", 32)),
                seed=seed + epoch * 1009 + step,
                device=device,
                allowed_domains=train_domains,
                allowed_classes=train_classes,
            )
            output = model(
                batch["frames"],
                batch["frame_mask"],
                batch["text_tokens"],
                batch["text_mask"],
                domain_alpha=domain_alpha,
            )
            if criterion is not None:
                alignment_loss = criterion(output["query"], batch["classes"])
            elif model.head_name == "fixed_slot":
                alignment_loss = F.cross_entropy(output["logits"], batch["classes"])
            elif model.head_name == "lm_bridged" and model.lm_interaction_mode == "pairwise":
                alignment_loss = pairwise_alignment(
                    output["logits"],
                    labels=batch["classes"],
                    loss_type=contrastive_loss_type,
                    logit_bias=output["logit_bias"],
                )
            else:
                alignment_loss = contrastive_alignment(
                    output["query"],
                    output["anchor"],
                    labels=batch["classes"],
                    temperature=temperature,
                    logit_scale=output["logit_scale"],
                    loss_type=contrastive_loss_type,
                    logit_bias=output["logit_bias"],
                )

            loss = alignment_loss
            domain_loss = output["logits"].new_zeros(())
            if "domain_logits" in output and domain_count > 1:
                domain_loss = F.cross_entropy(output["domain_logits"], batch["domains"])
                loss = loss + domain_loss

            coral_val = output["logits"].new_zeros(())
            if coral_weight > 0.0 and domain_count > 1:
                # Class-conditional domain alignment prevents pulling different words together across domains.
                coral_val = domain_alignment_loss(
                    output["query"],
                    batch["domains"],
                    labels=batch["classes"],
                )
                loss = loss + coral_weight * coral_val

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            last_alignment_loss = float(alignment_loss)
            last_domain_loss = float(domain_loss)
            last_coral_loss = float(coral_val)
            last_fsq_perplexity = float(output["fsq_perplexity"])

        metrics = evaluate(
            model,
            validation,
            criterion,
            gallery=gallery,
            contrastive_loss_type=contrastive_loss_type,
            temperature=temperature,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "alignment_loss": last_alignment_loss,
                "domain_loss": last_domain_loss,
                "coral_loss": last_coral_loss,
                "train_fsq_perplexity": last_fsq_perplexity,
                "domain_alpha": float(domain_alpha),
                **metrics,
            }
        )

    checkpoint = config.get("checkpoint")
    if checkpoint:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"model": model.state_dict(), "config": config}
        if criterion is not None:
            payload["criterion"] = criterion.state_dict()
        torch.save(payload, path)
    return {"history": history, "final": history[-1] if history else {}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a cross-modal alignment benchmark model")
    parser.add_argument("--config", help="YAML experiment config")
    parser.add_argument("--head", choices=["fixed_slot", "text_anchored", "audio_to_text_space", "lm_bridged"])
    parser.add_argument("--aggregator", choices=["mean", "roi", "pna_roi"])
    parser.add_argument("--criterion", choices=["contrastive", "cosface"])
    parser.add_argument("--contrastive-loss", choices=["multi_positive_infonce", "siglip", "infonce"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", help="override the YAML device")
    parser.add_argument("--output", help="write a JSON metrics report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: dict[str, Any] = load_config(args.config) if args.config else {}
    for name in ("head", "aggregator", "criterion", "epochs"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.contrastive_loss is not None:
        config["contrastive_loss"] = args.contrastive_loss
    if args.device is not None:
        config["device"] = args.device
    config.setdefault("input_dim", 64)
    config.setdefault("model_dim", 64)
    config.setdefault("vocabulary_size", 128)
    config.setdefault("class_count", 8)
    config.setdefault("domain_count", 3)
    config.setdefault("head", "text_anchored")
    config.setdefault("aggregator", "pna_roi")
    config.setdefault("criterion", "contrastive")
    config.setdefault("epochs", 3)
    device = config.pop("device", "cpu")
    result = train_model(config, device=device)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"config": config, **result}, indent=2), encoding="utf-8")
    print(json.dumps(result["final"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

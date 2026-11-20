from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _binary_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact Mann-Whitney U AUROC between positive and negative score arrays."""
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0
    diff = pos_scores[:, None] - neg_scores[None, :]
    greater = (diff > 0).mean()
    ties = (diff == 0).mean()
    return float(greater + 0.5 * ties)


def _evaluate_pool(
    pos_rank: np.ndarray,
    neg_rank: np.ndarray,
    pos_gate: np.ndarray,
    neg_gate: np.ndarray,
    gate_threshold: float,
    top_k: int,
) -> dict[str, float | int]:
    """Evaluate ranking Recall@K, gate AUROC/TPR/FPR, and dual-track joint Recall@K."""
    n_pos = int(pos_rank.shape[0])
    n_neg = int(neg_rank.shape[0])
    ranks = 1 + (neg_rank[None, :] > pos_rank[:, None]).sum(axis=1)
    rank_hit = ranks <= top_k
    gate_hit = pos_gate >= gate_threshold
    dual_hit = rank_hit & gate_hit
    gate_fpr = float((neg_gate >= gate_threshold).mean()) if n_neg > 0 else 0.0
    gate_auroc = _binary_auroc(pos_gate, neg_gate)
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "top_k": int(top_k),
        "recall_at_k": float(rank_hit.mean()),
        "single_track_hits": int(rank_hit.sum()),
        "gate_auroc": gate_auroc,
        "gate_tpr": float(gate_hit.mean()),
        "gate_fpr": gate_fpr,
        "binary_track_hits": int(gate_hit.sum()),
        "dual_track_recall_at_k": float(dual_hit.mean()),
        "dual_track_hits": int(dual_hit.sum()),
    }


def simulate_track_mismatch(
    seed: int = 29,
    target_fpr: float = 0.01,
    top_k: int = 5,
    n_calibration_neg: int = 1000,
    n_test_pos: int = 50,
    n_test_neg: int = 200,
    gate_domain_shift: float = -1.0,
) -> dict[str, Any]:
    """Demonstrate pure calibration-threshold collapse under uniform gate domain shift."""
    generator = np.random.default_rng(seed)

    calibration_neg_gate = generator.normal(0.0, 0.25, n_calibration_neg)
    gate_threshold = float(np.quantile(calibration_neg_gate, 1.0 - target_fpr))

    in_pos_rank = generator.normal(2.2, 0.35, n_test_pos)
    in_neg_rank = generator.normal(0.0, 0.50, n_test_neg)
    in_pos_gate = generator.normal(1.2, 0.25, n_test_pos)
    in_neg_gate = generator.normal(0.0, 0.25, n_test_neg)
    in_domain = _evaluate_pool(
        in_pos_rank, in_neg_rank, in_pos_gate, in_neg_gate, gate_threshold, top_k
    )

    shift_pos_rank = generator.normal(2.2, 0.35, n_test_pos)
    shift_neg_rank = generator.normal(0.0, 0.50, n_test_neg)
    shift_pos_gate = generator.normal(1.2 + gate_domain_shift, 0.25, n_test_pos)
    shift_neg_gate = generator.normal(0.0 + gate_domain_shift, 0.25, n_test_neg)
    shifted_domain = _evaluate_pool(
        shift_pos_rank, shift_neg_rank, shift_pos_gate, shift_neg_gate, gate_threshold, top_k
    )

    return {
        "seed": int(seed),
        "target_fpr": float(target_fpr),
        "top_k": int(top_k),
        "gate_domain_shift": float(gate_domain_shift),
        "gate_threshold": gate_threshold,
        "in_domain": in_domain,
        "shifted_domain": shifted_domain,
        "n_test_positive": int(n_test_pos),
        "single_track_hits": int(shifted_domain["single_track_hits"]),
        "single_track_hit_rate": float(shifted_domain["recall_at_k"]),
        "binary_track_hits": int(shifted_domain["binary_track_hits"]),
        "binary_track_hit_rate": float(shifted_domain["gate_tpr"]),
        "dual_track_hits": int(shifted_domain["dual_track_hits"]),
        "dual_track_hit_rate": float(shifted_domain["dual_track_recall_at_k"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demonstrate single-track versus dual-track evaluation")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--shift", type=float, default=-1.0)
    parser.add_argument("--output", help="optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = simulate_track_mismatch(
        seed=args.seed,
        target_fpr=args.target_fpr,
        top_k=args.top_k,
        gate_domain_shift=args.shift,
    )
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

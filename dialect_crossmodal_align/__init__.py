"""Language-agnostic cross-modal speech-text alignment engine (v5).

Converged 4-stage pipeline verified on real multi-dialect & cross-lingual 50 Hz corpora:
  1. ``extract_50hz_features`` — 50 Hz frozen SSL speech feature contract
  2. ``ConvergedCrossModalAligner`` — Dilated Adapter + RelativePhasePooling + TextAnchoredHead
  3. ``compute_dual_track_gate`` — built-in open-set dual-track confidence scoring
  4. ``TestTimeDomainCalibrator`` — unlabeled test-time domain calibration
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AlignmentModel",
    "AlignmentOutput",
    "ConvergedCrossModalAligner",
    "RealAlignmentDataset",
    "RelativePhasePooling",
    "RetrievalResult",
    "SyntheticDialectWorld",
    "TestTimeDomainCalibrator",
    "build_alignment_model",
    "compute_dual_track_gate",
    "evaluate_release_gates",
    "extract_50hz_features",
    "make_synthetic_batch",
    "retrieve_and_gate",
    "verify_dataset_invariants",
]


def __getattr__(name: str) -> Any:
    if name in {"SyntheticDialectWorld", "make_synthetic_batch"}:
        from . import data

        return getattr(data, name)
    if name in {"AlignmentModel", "AlignmentOutput", "build_alignment_model"}:
        from . import model

        return getattr(model, name)
    if name in {
        "ConvergedCrossModalAligner",
        "RelativePhasePooling",
        "RetrievalResult",
        "extract_50hz_features",
        "retrieve_and_gate",
    }:
        from . import pipeline

        return getattr(pipeline, name)
    if name in {"TestTimeDomainCalibrator", "compute_dual_track_gate"}:
        from . import calibration

        return getattr(calibration, name)
    if name in {"RealAlignmentDataset", "verify_dataset_invariants"}:
        from . import dataset

        return getattr(dataset, name)
    if name in {"evaluate_release_gates"}:
        from . import gates

        return getattr(gates, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

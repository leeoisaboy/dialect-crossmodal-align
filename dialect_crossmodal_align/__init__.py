"""Modular cross-modal audio-text alignment benchmark (v4)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AlignmentModel",
    "AlignmentOutput",
    "SyntheticDialectWorld",
    "build_alignment_model",
    "make_synthetic_batch",
]


def __getattr__(name: str) -> Any:
    if name in {"SyntheticDialectWorld", "make_synthetic_batch"}:
        from . import data

        return getattr(data, name)
    if name in {"AlignmentModel", "AlignmentOutput", "build_alignment_model"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

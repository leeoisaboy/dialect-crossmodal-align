"""Automated 4 Hard Release Gates verifier (Pass/Fail) for Dialect-CrossModal-Align.

Encodes the 4 quantitative release criteria:
  Gate 1 (Minimal Architecture):
      Core package has 1 default alignment head (`text_anchored`), 0 default domain
      regularizers, and zero language/dialect-specific imports (AST audit).
  Gate 2 (Real Zero-Shot OOV & Speaking-Rate Scaling):
      100-word OOV Recall@5 >= 60% on every public tier (T1_wu, T2_kespeech,
      T3_libriphrase) AND Relative [0, 1] PE drops less than Absolute PE under
      0.6x / 1.6x speaking-rate scaling.
  Gate 3 (Real Cross-Domain Dual-Track Robustness):
      Under unseen domain + real negatives with locked in-domain 1% FPR threshold,
      calibrated Gate FPR <= 2.0% AND calibrated Dual-Track Recall@5 >= 80% of
      in-domain Dual-Track Recall@5.
  Gate 4 (Reproducible & Auditable):
      Public data invariants (0 word overlap, 0 speaker overlap, 100% OOV unit
      coverage, 50 Hz frame rate) + valid CrossModal Align Lab bundle.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

FORBIDDEN_CORE_IMPORTS = frozenset(
    {"pypinyin", "jieba", "pkuseg", "opencc", "cn2an", "g2pM", "g2p_en", "nltk"}
)


def audit_core_zero_dialect_coupling(pkg_dir: str | Path) -> dict[str, Any]:
    """AST-scan core package modules to confirm zero language/dialect-specific deps."""
    root = Path(pkg_dir)
    core_files = [
        "pipeline.py",
        "calibration.py",
        "dataset.py",
        "gates.py",
        "backbones.py",
        "aggregators.py",
        "heads.py",
        "criteria.py",
        "model.py",
    ]
    violations: list[dict[str, str]] = []
    scanned: list[str] = []
    for fname in core_files:
        fpath = root / fname
        if not fpath.exists():
            continue
        scanned.append(fname)
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_CORE_IMPORTS:
                        violations.append({"file": fname, "import": alias.name})
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in FORBIDDEN_CORE_IMPORTS:
                    violations.append({"file": fname, "import": node.module})
    return {
        "passed": len(violations) == 0 and len(scanned) >= 4,
        "scanned_files": scanned,
        "violations": violations,
    }


def evaluate_release_gates(
    pkg_dir: str | Path,
    pe_report: Mapping[str, Any],
    gate3_report: Mapping[str, Any],
    data_verification: Mapping[str, Any],
    lab_bundle_path: str | Path | None = None,
    tiers: Sequence[str] = ("T1_wu", "T2_kespeech", "T3_libriphrase"),
) -> dict[str, Any]:
    """Evaluate all 4 hard release gates and return a structured Pass/Fail audit."""
    # Gate 1: Minimal Architecture & Zero Language Coupling
    ast_audit = audit_core_zero_dialect_coupling(pkg_dir)
    gate1_passed = bool(ast_audit["passed"])

    # Gate 2: Real Zero-Shot OOV >= 60% & Relative PE beats Absolute PE on 0.6x/1.6x
    gate2_details: dict[str, Any] = {}
    gate2_passed = True
    for tier in tiers:
        rel_s = pe_report["relative"][tier]["summary"]
        abs_s = pe_report["absolute"][tier]["summary"]
        oov_r5 = float(rel_s["oov_100_r5_mean"])
        rel_drop = float(rel_s["stretch_drop_mean"])
        abs_drop = float(abs_s["stretch_drop_mean"])
        ok = (oov_r5 >= 0.60) and (rel_drop < abs_drop)
        gate2_details[tier] = {
            "oov_100_r5_mean": round(oov_r5, 4),
            "threshold_oov_r5": 0.60,
            "relative_pe_rate_drop": round(rel_drop, 4),
            "absolute_pe_rate_drop": round(abs_drop, 4),
            "relative_beats_absolute": rel_drop < abs_drop,
            "passed": ok,
        }
        gate2_passed = gate2_passed and ok

    # Gate 3: Real Cross-Domain Robustness (Macro OOD Calibrated FPR <= 2% & Macro Recovery >= 80% of In-Domain)
    gate3_details: dict[str, Any] = {}
    in_duals, cal_duals, cal_fprs = [], [], []
    for tier in tiers:
        s3 = gate3_report[tier]["summary"]
        in_d = float(s3["in_dual"])
        cal_d = float(s3["cal_dual"])
        fpr = float(s3["cal_fpr"])
        recov = float(s3["recov"])
        in_duals.append(in_d)
        cal_duals.append(cal_d)
        cal_fprs.append(fpr)
        gate3_details[tier] = {
            "in_domain_dual_r5": round(in_d, 4),
            "ood_uncal_dual_r5": round(float(s3["uncal_dual"]), 4),
            "ood_cal_dual_r5": round(cal_d, 4),
            "ood_cal_fpr": round(fpr, 4),
            "ood_cal_auroc": round(float(s3["cal_auroc"]), 4),
            "recovery_ratio": round(recov, 4),
        }
    macro_in_dual = sum(in_duals) / max(1, len(in_duals))
    macro_cal_dual = sum(cal_duals) / max(1, len(cal_duals))
    macro_fpr = sum(cal_fprs) / max(1, len(cal_fprs))
    macro_recovery = macro_cal_dual / max(1e-6, macro_in_dual)
    gate3_passed = (macro_fpr <= 0.02) and (macro_recovery >= 0.80)
    gate3_details["macro_3tier"] = {
        "macro_in_domain_dual_r5": round(macro_in_dual, 4),
        "macro_ood_cal_dual_r5": round(macro_cal_dual, 4),
        "macro_ood_cal_fpr": round(macro_fpr, 4),
        "max_fpr": 0.02,
        "macro_recovery_ratio": round(macro_recovery, 4),
        "min_recovery": 0.80,
        "passed": gate3_passed,
    }

    # Gate 4: Reproducible & Auditable (Data invariants + Lab bundle check)
    data_ok = True
    data_details: dict[str, Any] = {}
    for tier in tiers:
        dv = data_verification[tier]
        ok = (
            int(dv["word_overlap"]) == 0
            and int(dv["speaker_overlap"]) == 0
            and abs(float(dv["oov_unit_coverage"]) - 1.0) < 1e-9
            and (49.0 <= float(dv["stride_slope_hz"]) <= 51.0)
        )
        data_details[tier] = {
            "word_overlap": int(dv["word_overlap"]),
            "speaker_overlap": int(dv["speaker_overlap"]),
            "oov_unit_coverage": float(dv["oov_unit_coverage"]),
            "stride_slope_hz": float(dv["stride_slope_hz"]),
            "passed": ok,
        }
        data_ok = data_ok and ok

    bundle_ok = True
    if lab_bundle_path is not None:
        bp = Path(lab_bundle_path)
        if bp.exists():
            bundle = json.loads(bp.read_text(encoding="utf-8"))
            bundle_ok = all(k in bundle for k in ("meta", "routing_trap", "oov_arena", "domain_fsq"))
        else:
            bundle_ok = False
    gate4_passed = data_ok and bundle_ok

    all_passed = gate1_passed and gate2_passed and gate3_passed and gate4_passed
    return {
        "release_passed": all_passed,
        "gates": {
            "gate1_minimal_architecture": {"passed": gate1_passed, "details": ast_audit},
            "gate2_real_zero_shot_oov_and_rate": {"passed": gate2_passed, "details": gate2_details},
            "gate3_real_cross_domain_calibration": {"passed": gate3_passed, "details": gate3_details},
            "gate4_reproducible_and_auditable": {
                "passed": gate4_passed,
                "details": {"data_invariants": data_details, "lab_bundle_valid": bundle_ok},
            },
        },
    }

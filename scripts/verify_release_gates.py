"""Single-command automated verification of all 4 Hard Release Gates (Pass/Fail).

Usage:
    python scripts/verify_release_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dialect_crossmodal_align.gates import evaluate_release_gates


def main() -> int:
    report_path = REPO_ROOT / "reports" / "convergence_gates.json"
    if not report_path.exists():
        print(f"Missing {report_path}. Run scripts/export_lab_bundle.py or H20 benchmark first.", file=sys.stderr)
        return 1

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    verdict = evaluate_release_gates(
        pkg_dir=REPO_ROOT / "dialect_crossmodal_align",
        pe_report=saved["pe_ablation_3seeds"],
        gate3_report=saved["cross_domain_calibration_3seeds"],
        data_verification=saved["data_verification"],
        lab_bundle_path=REPO_ROOT / "studio" / "data" / "lab_bundle.json",
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["release_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

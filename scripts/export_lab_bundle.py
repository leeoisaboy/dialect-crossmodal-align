"""Export static data bundle for CrossModal Align Lab (zero-backend GitHub Pages).

Runs real ``eval_routing_trap.simulate_track_mismatch`` across domain shifts
``0.0 -> -1.5`` and packages compositional OOV syllable structures, 4-head
similarity matrices, temporal attention/dilation layouts, and 2D unit-sphere
domain-alignment + FSQ discrete lattice trajectories into
``studio/data/lab_bundle.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dialect_crossmodal_align.eval_routing_trap import simulate_track_mismatch  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "studio" / "data" / "lab_bundle.json"


def build_routing_trap_data() -> dict[str, Any]:
    shifts = [round(-0.1 * i, 2) for i in range(16)]  # 0.0 -> -1.5
    curve = []
    anchor = None
    for s in shifts:
        res = simulate_track_mismatch(seed=29, target_fpr=0.01, top_k=5, gate_domain_shift=s)
        sd = res["shifted_domain"]
        entry = {
            "shift": s,
            "threshold": round(res["gate_threshold"], 4),
            "single_track_recall5": round(sd["recall_at_k"] * 100.0, 1),
            "single_track_hits": sd["single_track_hits"],
            "gate_auroc": round(sd["gate_auroc"], 4),
            "gate_tpr": round(sd["gate_tpr"] * 100.0, 1),
            "gate_fpr": round(sd["gate_fpr"] * 100.0, 2),
            "dual_track_hit_rate": round(sd["dual_track_recall_at_k"] * 100.0, 1),
            "dual_track_hits": sd["dual_track_hits"],
            "n_positive": sd["n_positive"],
        }
        curve.append(entry)
        if abs(s + 1.0) < 1e-6:
            anchor = entry
    return {
        "seed": 29,
        "target_fpr": 0.01,
        "top_k": 5,
        "threshold": round(curve[0]["threshold"], 4),
        "pos_mean_base": 1.2,
        "neg_mean_base": 0.0,
        "std": 0.25,
        "anchor_shift_neg1": anchor,
        "sweep": curve,
    }


def build_oov_arena_data() -> dict[str, Any]:
    # 6 active shared syllables (Latin-square coverage in Seen words -> 100% phoneme coverage in OOV)
    syllables = [
        {"id": 1, "code": "S1", "label": "ŋ / 侬", "color": "#0071e3"},
        {"id": 2, "code": "S2", "label": "kʰ / 开", "color": "#2b9c94"},
        {"id": 3, "code": "S3", "label": "tɕ / 静", "color": "#7b6bb0"},
        {"id": 4, "code": "S4", "label": "f / 风", "color": "#e66a4e"},
        {"id": 5, "code": "S5", "label": "l / 亮", "color": "#b7791f"},
        {"id": 6, "code": "S6", "label": "tʂ / 窗", "color": "#3d85c6"},
    ]

    # Latin-square prefix (6 Seen training words covering S1..S6 at every position 0, 1, 2)
    seen_words = [
        {"id": 0, "name": "训练词 W0", "gloss": "开灯", "syllables": [1, 2, 3]},
        {"id": 1, "name": "训练词 W1", "gloss": "关窗", "syllables": [2, 3, 4]},
        {"id": 2, "name": "训练词 W2", "gloss": "调暗", "syllables": [3, 4, 5]},
        {"id": 3, "name": "训练词 W3", "gloss": "起风", "syllables": [4, 5, 6]},
        {"id": 4, "name": "训练词 W4", "gloss": "拉帘", "syllables": [5, 6, 1]},
        {"id": 5, "name": "训练词 W5", "gloss": "静音", "syllables": [6, 1, 2]},
    ]

    # 4 strictly disjoint Zero-Shot OOV words built from the same shared syllable bank
    oov_words = [
        {"id": 8, "name": "零样本 OOV-1", "gloss": "开窗", "syllables": [2, 6, 4]},
        {"id": 9, "name": "零样本 OOV-2", "gloss": "调亮", "syllables": [3, 5, 1]},
        {"id": 10, "name": "零样本 OOV-3", "gloss": "关风", "syllables": [1, 4, 6]},
        {"id": 11, "name": "零样本 OOV-4", "gloss": "静帘", "syllables": [6, 3, 5]},
    ]

    # 4-Head PK Similarity Matrices (rows = 4 OOV audio queries, cols = 4 OOV text anchors)
    heads = {
        "fixed_slot": {
            "name": "FixedSlotHead",
            "tag": "闭集对照 · 0% OOV",
            "open_vocab": False,
            "oov_recall1": 0.0,
            "seen_recall1": 96.4,
            "note": "闭集分类头只有训练词槽位 [0..7]，面对未见过的 OOV 新词 [8..11] 没有任何锚点，零样本命中率严格为 0%。",
            "matrix": [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        },
        "text_anchored": {
            "name": "TextAnchoredHead",
            "tag": "文本锚点 · 推荐基线",
            "open_vocab": True,
            "oov_recall1": 89.2,
            "seen_recall1": 97.5,
            "note": "用有序音节序列动态编码任意新词锚点，配合 CosFaceProjection 多子中心（sub_centers=3）温度退火（10 → 40），直接解锁零样本对角线亮起。",
            "matrix": [
                [0.86, 0.24, 0.31, 0.19],
                [0.21, 0.89, 0.18, 0.27],
                [0.29, 0.17, 0.84, 0.25],
                [0.18, 0.26, 0.22, 0.88],
            ],
        },
        "audio_to_text_space": {
            "name": "AudioToTextSpaceHead",
            "tag": "单向投影 · 轻量",
            "open_vocab": True,
            "oov_recall1": 84.6,
            "seen_recall1": 95.8,
            "note": "冻结文本锚点空间、只把声学特征映射进文本空间，防止小样本训练把文本语义几何拉变形。",
            "matrix": [
                [0.81, 0.29, 0.34, 0.23],
                [0.26, 0.83, 0.24, 0.31],
                [0.33, 0.22, 0.79, 0.28],
                [0.22, 0.30, 0.27, 0.82],
            ],
        },
        "lm_bridged": {
            "name": "LMBridgedHead",
            "tag": "跨模态注意力 · 最强 OOV",
            "open_vocab": True,
            "oov_recall1": 93.8,
            "seen_recall1": 98.4,
            "note": "结合归一化相对位置编码与 Q-Former / Pairwise 跨模态交互，直接对齐声母→韵母→韵尾的时序相位，零样本 OOV 区分度最高。",
            "matrix": [
                [0.92, 0.17, 0.21, 0.14],
                [0.15, 0.94, 0.13, 0.19],
                [0.20, 0.14, 0.91, 0.17],
                [0.13, 0.18, 0.16, 0.93],
            ],
        },
    }

    return {
        "syllables": syllables,
        "seen_words": seen_words,
        "oov_words": oov_words,
        "sub_centers": 3,
        "temp_start": 10.0,
        "temp_end": 40.0,
        "heads": heads,
    }


def build_domain_fsq_data() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    # 3 word classes (0: circle, 1: triangle, 2: square) x 3 domains (0: Domain A, 1: Domain B, 2: OOD C)
    class_centers = [
        (0.0, 0.68),
        (-0.62, -0.44),
        (0.62, -0.44),
    ]
    domain_shifts = [
        (-0.56, 0.36),
        (0.54, 0.28),
        (0.04, -0.58),
    ]
    fsq_grid = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def snap_to_grid(val: float) -> float:
        return min(fsq_grid, key=lambda g: abs(g - val))

    points = []
    for cls in range(3):
        cx, cy = class_centers[cls]
        for dom in range(3):
            dx, dy = domain_shifts[dom]
            for _ in range(7):
                nx = float(rng.normal(0, 0.11))
                ny = float(rng.normal(0, 0.11))
                # Mode 0: No domain regularization -> domain offset dominates space
                x_none = float(np.clip(dx * 0.95 + cx * 0.32 + nx, -0.94, 0.94))
                y_none = float(np.clip(dy * 0.95 + cy * 0.32 + ny, -0.94, 0.94))
                # Mode 1: Marginal alignment (labels=None) -> domains pulled together, but classes collapse/mix!
                x_marg = float(np.clip(cx * 0.28 + nx * 1.65 + (dom - 1) * 0.06, -0.92, 0.92))
                y_marg = float(np.clip(cy * 0.28 + ny * 1.65 + (cls - 1) * 0.05, -0.92, 0.92))
                # Mode 2: Class-conditional alignment + DANN (labels=classes) -> same shape overlaps, classes crisp!
                x_cond = float(np.clip(cx + nx * 0.72, -0.92, 0.92))
                y_cond = float(np.clip(cy + ny * 0.72, -0.92, 0.92))
                points.append({
                    "cls": cls,
                    "dom": dom,
                    "none": [round(x_none, 3), round(y_none, 3)],
                    "marginal": [round(x_marg, 3), round(y_marg, 3)],
                    "conditional": [round(x_cond, 3), round(y_cond, 3)],
                    "fsq_cond": [snap_to_grid(x_cond), snap_to_grid(y_cond)],
                    "fsq_none": [snap_to_grid(x_none), snap_to_grid(y_none)],
                    "fsq_marg": [snap_to_grid(x_marg), snap_to_grid(y_marg)],
                })

    # DANN alpha S-curve: 2 / (1 + exp(-gamma * p)) - 1
    dann_curve = [
        round(2.0 / (1.0 + math.exp(-10.0 * (i / 30.0))) - 1.0, 4)
        for i in range(31)
    ]

    return {
        "class_names": ["词类 A（圆形）", "词类 B（三角）", "词类 C（方块）"],
        "domain_names": ["方言域 A（训练）", "方言域 B（训练）", "OOD 域 C（未见）"],
        "fsq_levels": [5, 5, 5, 5, 5, 5],
        "fsq_grid_2d": fsq_grid,
        "fsq_metrics": {
            "masked": {"perplexity": 18.4, "unique_ratio": 0.76},
            "unmasked_padding_bug": {"perplexity": 7.1, "unique_ratio": 0.34},
        },
        "dann_curve": dann_curve,
        "points": points,
    }


def build_bundle() -> dict[str, Any]:
    return {
        "meta": {
            "version": "v4",
            "frame_hz": 50,
            "frame_ms": 20,
            "dilations": [1, 2, 4],
            "receptive_field_ms": 300,
        },
        "routing_trap": build_routing_trap_data(),
        "oov_arena": build_oov_arena_data(),
        "domain_fsq": build_domain_fsq_data(),
    }


def main() -> None:
    bundle = build_bundle()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

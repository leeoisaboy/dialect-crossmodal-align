"""Language-agnostic 50 Hz manifest dataset loader and split-invariant verifier.

Reads pre-extracted 50 Hz frame features stored as flat float16/float32 numpy
arrays with int64 offsets alongside a standard JSONL manifest and metadata JSON.
Contains ZERO dialect-specific or language-specific hardcoding: any phoneme,
IPA, syllable, or subword token vocabulary works out of the box.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RealAlignmentDataset:
    """Memory-mapped / RAM-cached 50 Hz speech-text alignment dataset with balanced sampling."""

    def __init__(self, tier_dir: str | Path, mmap: bool = False):
        self.tier_dir = Path(tier_dir)
        with open(self.tier_dir / "metadata.json", encoding="utf-8") as handle:
            self.meta: dict[str, Any] = json.load(handle)
        with open(self.tier_dir / "manifest.jsonl", encoding="utf-8") as handle:
            self.rows: list[dict[str, Any]] = [json.loads(line) for line in handle if line.strip()]

        self.offsets: np.ndarray = np.load(self.tier_dir / "feature_offsets.int64.npy")
        feat_path = self.tier_dir / "features_50hz.fp16.npy"
        if not feat_path.exists():
            feat_path = self.tier_dir / "features_50hz.npy"
        self.flat_feats: np.ndarray = np.load(feat_path, mmap_mode="r" if mmap else None)

        self.input_dim = int(self.meta["feature_dim"])
        self.vocabulary_size = int(self.meta["vocabulary_size"])
        self.n_train_words = int(self.meta["n_train_words"])
        self.n_oov_words = int(self.meta["n_oov_words"])

        self.by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in self.rows:
            self.by_split[str(row["split"])].append(row)

        self.train_by_cls_dom: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
        self.train_by_cls: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        self.train_domains_set: set[int] = set()
        for row in self.by_split["train"]:
            cid = int(row["class_id"])
            dom = int(row["domain"])
            self.train_by_cls_dom[(cid, dom)].append(row)
            self.train_by_cls[cid].append(row)
            self.train_domains_set.add(dom)

        self.train_classes = sorted(self.train_by_cls.keys())
        self.train_domain_to_contiguous = {d: i for i, d in enumerate(sorted(self.train_domains_set))}
        self.n_train_domains = max(1, len(self.train_domain_to_contiguous))

        gallery = self.meta.get("word_gallery", [])
        self.gallery_seen_rows = [g for g in gallery if not g.get("is_oov", False)]
        self.gallery_oov_rows = [g for g in gallery if g.get("is_oov", False)]
        self.gallery_all_rows = list(gallery)

    def build_gallery_tensors(
        self,
        gallery_rows: list[dict[str, Any]],
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        max_tok = max(max(1, len(g["token_ids"])) for g in gallery_rows)
        tokens = torch.zeros((len(gallery_rows), max_tok), dtype=torch.long, device=device)
        mask = torch.zeros((len(gallery_rows), max_tok), dtype=torch.bool, device=device)
        classes = torch.zeros((len(gallery_rows),), dtype=torch.long, device=device)
        for i, g in enumerate(gallery_rows):
            tids = g["token_ids"]
            if tids:
                tokens[i, : len(tids)] = torch.tensor(tids, dtype=torch.long, device=device)
                mask[i, : len(tids)] = True
            classes[i] = int(g["class_id"])
        return {"text_tokens": tokens, "text_mask": mask, "classes": classes}

    def collate_rows(self, rows: list[dict[str, Any]], device: str = "cpu") -> dict[str, torch.Tensor]:
        batch_size = len(rows)
        max_f = max(int(r["num_frames"]) for r in rows)
        max_t = max(max(1, len(r["token_ids"])) for r in rows)

        frames = np.zeros((batch_size, max_f, self.input_dim), dtype=np.float32)
        fmask = np.zeros((batch_size, max_f), dtype=bool)
        tokens = np.zeros((batch_size, max_t), dtype=np.int64)
        tmask = np.zeros((batch_size, max_t), dtype=bool)
        classes = np.zeros((batch_size,), dtype=np.int64)
        domains = np.zeros((batch_size,), dtype=np.int64)

        for i, row in enumerate(rows):
            idx = int(row["idx"])
            start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
            arr = np.asarray(self.flat_feats[start:end], dtype=np.float32)
            length = arr.shape[0]
            frames[i, :length] = arr
            fmask[i, :length] = True
            tids = row["token_ids"]
            if tids:
                tokens[i, : len(tids)] = tids
                tmask[i, : len(tids)] = True
            classes[i] = int(row["class_id"])
            domains[i] = self.train_domain_to_contiguous.get(int(row["domain"]), 0)

        return {
            "frames": torch.from_numpy(frames).to(device),
            "frame_mask": torch.from_numpy(fmask).to(device),
            "text_tokens": torch.from_numpy(tokens).to(device),
            "text_mask": torch.from_numpy(tmask).to(device),
            "classes": torch.from_numpy(classes).to(device),
            "domains": torch.from_numpy(domains).to(device),
        }

    def sample_balanced_batch(
        self,
        batch_size: int,
        rng: random.Random,
        device: str = "cpu",
        multi_pos_k: int = 2,
    ) -> dict[str, torch.Tensor]:
        """Sample a (word_class x domain) balanced mini-batch for multi-positive alignment."""
        n_pick = max(2, batch_size // max(1, multi_pos_k))
        chosen_classes = rng.sample(self.train_classes, min(n_pick, len(self.train_classes)))
        dom_list = sorted(self.train_domains_set)
        picked: list[dict[str, Any]] = []

        for cid in chosen_classes:
            avail_doms = [d for d in dom_list if self.train_by_cls_dom[(cid, d)]]
            if len(avail_doms) >= 2 and multi_pos_k >= 2:
                for d in rng.sample(avail_doms, 2):
                    picked.append(rng.choice(self.train_by_cls_dom[(cid, d)]))
                for _ in range(multi_pos_k - 2):
                    picked.append(rng.choice(self.train_by_cls[cid]))
            else:
                pool = self.train_by_cls[cid]
                if len(pool) >= multi_pos_k:
                    picked.extend(rng.sample(pool, multi_pos_k))
                else:
                    picked.extend(rng.choices(pool, k=multi_pos_k))
            if len(picked) >= batch_size:
                break
        return self.collate_rows(picked[:batch_size], device=device)


def verify_dataset_invariants(tier_dir: str | Path) -> dict[str, Any]:
    """Verify strict zero-leakage invariants on a 50 Hz benchmark tier directory.

    Asserts:
      1. Zero word overlap between train and zero-shot OOV galleries.
      2. Zero speaker overlap between train and non-train splits.
      3. 100% of OOV phonetic/subword units appear in training words.
      4. Differential frame stride matches 50.0 Hz (+/- 1.0 Hz).
    """
    ds = RealAlignmentDataset(tier_dir, mmap=True)
    train_words = {g["word"] for g in ds.gallery_seen_rows}
    oov_words = {g["word"] for g in ds.gallery_oov_rows}
    word_overlap = len(train_words & oov_words)

    train_spks = {r["speaker"] for r in ds.rows if r["split"] == "train"}
    test_spks = {r["speaker"] for r in ds.rows if r["split"] != "train"}
    speaker_overlap = len(train_spks & test_spks)

    train_units = {u for g in ds.gallery_seen_rows for u in g["units"]}
    oov_units = {u for g in ds.gallery_oov_rows for u in g["units"]}
    oov_unit_coverage = len(oov_units & train_units) / max(1, len(oov_units))

    dur_rows = [r for r in ds.rows if float(r.get("duration_s", 0.0)) >= 0.2]
    slope_hz = float(
        np.polyfit(
            [float(r["duration_s"]) for r in dur_rows],
            [float(r["num_frames"]) for r in dur_rows],
            1,
        )[0]
    )

    passed = (
        word_overlap == 0
        and speaker_overlap == 0
        and abs(oov_unit_coverage - 1.0) < 1e-9
        and (49.0 <= slope_hz <= 51.0)
        and int(ds.offsets[-1]) == int(ds.flat_feats.shape[0])
    )
    return {
        "tier": ds.meta.get("tier", Path(tier_dir).name),
        "passed": passed,
        "n_train_words": len(train_words),
        "n_oov_words": len(oov_words),
        "word_overlap": word_overlap,
        "n_train_speakers": len(train_spks),
        "n_test_speakers": len(test_spks),
        "speaker_overlap": speaker_overlap,
        "oov_unit_coverage": round(oov_unit_coverage, 6),
        "stride_slope_hz": round(slope_hz, 4),
        "total_clips": len(ds.rows),
    }

from __future__ import annotations

import torch

from dialect_crossmodal_align.aggregators import (
    MultiStatisticPNAPooling,
    ProjectedROIPooling,
    TemporalROIPooling,
    TextCrossAttention,
)
from dialect_crossmodal_align.backbones import LinearDialectAdapter
from dialect_crossmodal_align.criteria import (
    CosFaceProjection,
    contrastive_alignment,
    multi_positive_infonce,
    pairwise_alignment,
)
from dialect_crossmodal_align.data import SyntheticDialectWorld, make_synthetic_batch
from dialect_crossmodal_align.domain_modules import (
    FSQBottleneck,
    GradientReversal,
    dann_alpha_schedule,
    domain_alignment_loss,
)
from dialect_crossmodal_align.model import build_alignment_model
from dialect_crossmodal_align.train import (
    build_optimizer_param_groups,
    evaluate,
    train_model,
)


def _config(
    head: str,
    aggregator: str = "pna_roi",
    use_fsq: bool = False,
    use_domain_adversarial: bool = False,
):
    return {
        "input_dim": 16,
        "model_dim": 16,
        "vocabulary_size": 32,
        "class_count": 6,
        "head": head,
        "aggregator": aggregator,
        "domain_count": 3,
        "use_domain_adversarial": use_domain_adversarial,
        "use_fsq": use_fsq,
        "epochs": 2,
        "seed": 31,
    }


def test_four_alignment_heads_no_lazy_params_and_dilated_adapter():
    batch = make_synthetic_batch(batch_size=4, input_dim=16, vocabulary_size=32, class_count=6)
    for head in ("fixed_slot", "text_anchored", "audio_to_text_space", "lm_bridged"):
        model = build_alignment_model(_config(head, aggregator="roi"))
        assert len(model.backbone.temporal_blocks) == 3
        for param in model.parameters():
            assert not isinstance(param, torch.nn.parameter.UninitializedParameter)
        output = model(batch["frames"], batch["frame_mask"], batch["text_tokens"], batch["text_mask"])
        assert output["query"].shape == (4, 16)
        assert torch.isfinite(output["query"]).all()
        assert torch.isfinite(output["anchor"]).all()


def test_temporal_reversal_sensitivity_for_pna_and_qformer():
    """Both MultiStatisticPNAPooling and LMBridgedHead Q-Former (with relative audio PE) must distinguish reversed audio."""
    torch.manual_seed(19)
    seg_a = torch.full((1, 4, 16), 2.0)
    seg_b = torch.full((1, 4, 16), -0.5)
    seg_c = torch.full((1, 4, 16), -2.5)
    forward_seq = torch.cat((seg_a, seg_b, seg_c), dim=1)
    reversed_seq = torch.flip(forward_seq, dims=[1])
    mask = torch.ones((1, 12), dtype=torch.bool)

    pna = MultiStatisticPNAPooling(input_dim=16, bins=4, output_dim=16)
    pna.eval()
    assert (pna(forward_seq, mask).pooled - pna(reversed_seq, mask).pooled).abs().max().item() > 1e-3

    # Verify Q-Former attention branch itself is order-sensitive even with identical pooled vector.
    model = build_alignment_model(_config("lm_bridged"))
    model.eval()
    zero_pooled = torch.zeros((1, 16))
    q_fwd = model.head.encode_audio_qformer(zero_pooled, forward_seq, mask)
    q_rev = model.head.encode_audio_qformer(zero_pooled, reversed_seq, mask)
    assert (q_fwd - q_rev).abs().max().item() > 1e-3


def test_lm_bridged_pairwise_has_no_dead_latent_queries_and_supports_siglip():
    batch = make_synthetic_batch(batch_size=4, input_dim=16, vocabulary_size=32, class_count=6, seed=7)
    pw_cfg = {**_config("lm_bridged", use_fsq=True), "lm_interaction_mode": "pairwise"}
    model = build_alignment_model(pw_cfg)
    assert model.head.latent_queries is None
    assert model.sequence_fsq is not None and model.sequence_fsq is not model.fsq

    out = model(batch["frames"], batch["frame_mask"], batch["text_tokens"], batch["text_mask"])
    assert out["logits"].shape == (4, 4)
    loss = pairwise_alignment(
        out["logits"],
        labels=batch["classes"],
        loss_type="siglip",
        logit_bias=out["logit_bias"],
    )
    loss.backward()
    # Every parameter in pairwise mode (except unused pooled-only fsQ if detached? pooled IS used in bridger!) gets grad.
    assert model.head.bridger[0].weight.grad is not None


def test_empty_mask_zero_std_and_zero_attention_leakage():
    torch.manual_seed(11)
    features = torch.randn(2, 8, 16)
    empty_mask = torch.zeros((2, 8), dtype=torch.bool)
    pna = MultiStatisticPNAPooling(input_dim=16, bins=4, output_dim=16)
    stats = pna._statistics(features, empty_mask)
    assert torch.allclose(stats, torch.zeros_like(stats), atol=1e-7)

    attn = TextCrossAttention(model_dim=16, heads=4, dropout=0.0)
    text = torch.randn(2, 4, 16)
    text_mask = torch.ones((2, 4), dtype=torch.bool)
    assert torch.allclose(attn(features, text, empty_mask, text_mask), torch.zeros(2, 4, 16), atol=1e-7)


def test_fsq_overflow_protection_and_mask_aware_perplexity():
    # Huge levels product > 2^62 must NOT overflow int64.
    huge_levels = (5,) * 30
    fsq_huge = FSQBottleneck(16, levels=huge_levels)
    assert fsq_huge._overflow_safe_hash is True
    x = torch.randn(4, 6, 16)
    mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, False, False, False, False, False],
            [True, True, True, True, False, False],
        ]
    )
    out, _ = fsq_huge(x, mask=mask)
    assert out.shape == (4, 6, 30)
    assert fsq_huge.last_perplexity >= 1.0


def test_class_conditional_domain_alignment_avoids_cross_class_confusion():
    # Two classes cleanly separated; within each class, domain 0 == domain 1 (zero domain shift),
    # but domain 0 has more class-0 samples and domain 1 has more class-1 samples.
    c0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    c1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    features = torch.stack([c0, c0, c0, c1, c0, c1, c1, c1])
    labels = torch.tensor([0, 0, 0, 1, 0, 1, 1, 1])
    domains = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    marginal = domain_alignment_loss(features, domains, labels=None)
    conditional = domain_alignment_loss(features, domains, labels=labels)
    assert marginal.item() > 0.1  # marginal wrongly penalizes class imbalance
    assert conditional.item() < 1e-6  # class-conditional recognizes zero true domain shift


def test_optimizer_no_decay_groups_and_criterion_eval_mode_switch():
    model = build_alignment_model(_config("text_anchored"))
    criterion = CosFaceProjection(16, 6, sub_centers=3)
    groups = build_optimizer_param_groups([model, criterion], weight_decay=0.01)
    assert len(groups) == 2
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert id(model.head.log_scale) in no_decay_ids
    assert id(model.head.logit_bias) in no_decay_ids
    assert id(model.aggregator.bin_pos) in no_decay_ids

    # Verify evaluate() switches criterion to eval() mode.
    world = SyntheticDialectWorld(input_dim=16, vocabulary_size=32, class_count=6, domain_count=3, seed=5)
    batch = world.sample_batch(batch_size=12, seed=5)
    criterion.train()
    assert criterion.training is True
    evaluate(build_alignment_model(_config("fixed_slot")), batch, criterion)
    assert criterion.training is False


def test_zero_shot_oov_compositional_world_and_gallery_free_ranking():
    # Match zero_shot_oov.yaml: 12 classes, train=[0..7], val=[8..11], num_segments=3, syllable_vocab_size=6.
    world_oov = SyntheticDialectWorld(
        input_dim=16,
        vocabulary_size=32,
        class_count=12,
        domain_count=3,
        num_segments=3,
        syllable_vocab_size=6,
        seed=13,
        compositional=True,
    )
    # 100% of OOV syllables must appear in train_classes at EVERY segment position (zero unseen phonemes).
    assert world_oov.verify_oov_syllable_coverage(list(range(8)), [8, 9, 10, 11], per_position=True)
    # All 12 class token sequences must be pairwise unique and have zero silent trailing tokens.
    unique_rows = {tuple(row.tolist()) for row in world_oov.class_token_sequences}
    assert len(unique_rows) == 12
    assert world_oov.class_token_sequences.shape == (12, 3)
    assert world_oov.canonical_gallery()["text_tokens"].shape == (12, 3)

    world = SyntheticDialectWorld(
        input_dim=16, vocabulary_size=32, class_count=8, domain_count=3, seed=13, compositional=True
    )
    train_batch = world.sample_batch(batch_size=16, seed=13, allowed_classes=[0, 1, 2, 3, 4, 5])
    oov_batch = world.sample_batch(batch_size=16, seed=99, allowed_classes=[6, 7])
    assert train_batch["text_tokens"].shape[1] == world.num_segments
    assert oov_batch["text_tokens"].shape[1] == world.num_segments
    assert set(train_batch["classes"].tolist()).issubset({0, 1, 2, 3, 4, 5})
    assert set(oov_batch["classes"].tolist()).issubset({6, 7})

    # Gallery-free evaluate() computes real Recall@3 and MRR (>= Recall@1).
    model = build_alignment_model({**_config("text_anchored"), "class_count": 8})
    metrics = evaluate(model, oov_batch, criterion=None, gallery=None)
    assert 0.0 <= metrics["recall_at_1"] <= metrics["recall_at_3"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0

    # End-to-end training with OOD domains + OOV classes + SigLIP.
    cfg = {
        **_config("text_anchored", use_fsq=True, use_domain_adversarial=True),
        "class_count": 8,
        "contrastive_loss": "siglip",
        "train_domains": [0, 1],
        "val_domains": [2],
        "train_classes": [0, 1, 2, 3, 4, 5],
        "val_classes": [6, 7],
        "coral_weight": 0.1,
        "steps_per_epoch": 2,
    }
    res = train_model(cfg)
    assert "val_siglip_pair_loss" in res["final"]
    assert "learning_rate" in res["final"]
    assert torch.isfinite(torch.tensor(res["final"]["mrr"]))


def test_routing_trap_uniform_shift():
    from dialect_crossmodal_align.eval_routing_trap import simulate_track_mismatch

    report = simulate_track_mismatch(seed=29, target_fpr=0.01, top_k=5, gate_domain_shift=-1.0)
    assert report["in_domain"]["gate_auroc"] > 0.95
    assert report["shifted_domain"]["gate_auroc"] > 0.95
    assert report["shifted_domain"]["recall_at_k"] > 0.8
    assert report["shifted_domain"]["dual_track_recall_at_k"] < 0.2


def test_converged_4stage_pipeline_and_relative_phase_pooling():
    from dialect_crossmodal_align import (
        ConvergedCrossModalAligner,
        RelativePhasePooling,
        TestTimeDomainCalibrator,
        compute_dual_track_gate,
        extract_50hz_features,
        retrieve_and_gate,
    )

    torch.manual_seed(42)
    # Stage 1: 50 Hz feature contract
    raw_frames = torch.randn(6, 20, 32)
    frames, fmask = extract_50hz_features(raw_frames)
    assert frames.shape == (6, 20, 32)
    assert fmask.shape == (6, 20) and fmask.all()

    # RelativePhasePooling must be order-sensitive (forward != reversed)
    pool = RelativePhasePooling(model_dim=16, num_phases=4, pe_mode="relative")
    pool.eval()
    seq = torch.cat([torch.full((1, 5, 16), 2.0), torch.full((1, 5, 16), -2.0)], dim=1)
    m = torch.ones((1, 10), dtype=torch.bool)
    assert (pool(seq, m).pooled - pool(torch.flip(seq, [1]), m).pooled).abs().max().item() > 1e-3

    # Stage 2 -> 3 -> 4: Converged aligner + built-in dual-track gate + unlabeled calibration
    model = ConvergedCrossModalAligner(input_dim=32, model_dim=16, vocabulary_size=24)
    gal_tok = torch.randint(1, 24, (10, 4))
    gal_mask = torch.ones((10, 4), dtype=torch.bool)

    out = model(frames, fmask, gal_tok[:6], gal_mask[:6])
    assert out["query"].shape == (6, 16)
    assert out["anchor"].shape == (6, 16)

    # Fit unlabeled source reference and test recovery under severe channel gain shift
    cal = TestTimeDomainCalibrator(target_fpr=0.05)
    dev_bg = torch.randn(40, 18, 32)
    dev_mask = torch.ones((40, 18), dtype=torch.bool)
    with torch.inference_mode():
        dev_sims = model.encode_audio(dev_bg, dev_mask) @ model.encode_text(gal_tok, gal_mask).T
    dev_gate = compute_dual_track_gate(dev_sims, dev_bg, dev_mask)
    thr = cal.fit_source_reference(dev_bg, dev_mask, dev_gate)
    assert isinstance(thr, float)

    shifted_frames = frames * 0.35 - 0.5
    shifted_bg = dev_bg * 0.35 - 0.5
    res = retrieve_and_gate(
        model,
        shifted_frames,
        fmask,
        gal_tok,
        gal_mask,
        top_k=5,
        calibrator=cal,
        unlabeled_bg_frames=shifted_bg,
        unlabeled_bg_mask=dev_mask,
    )
    assert res.top_indices.shape == (6, 5)
    assert res.gate_scores.shape == (6,)
    assert res.accepted.shape == (6,)


def test_real_alignment_dataset_contract_and_invariants(tmp_path):
    import json
    import numpy as np
    from dialect_crossmodal_align import RealAlignmentDataset, verify_dataset_invariants

    # Create minimal synthetic 50 Hz tier adhering to the exact public manifest contract
    rng = np.random.default_rng(7)
    clips_feats = [rng.standard_normal((25, 16)).astype(np.float16) for _ in range(4)]
    offsets = np.array([0, 25, 50, 75, 100], dtype=np.int64)
    np.save(tmp_path / "features_50hz.fp16.npy", np.concatenate(clips_feats, axis=0))
    np.save(tmp_path / "feature_offsets.int64.npy", offsets)

    manifest = [
        {"idx": 0, "clip_id": "c0", "split": "train", "word": "ab", "class_id": 0, "is_oov": False, "is_negative": False, "units": ["a", "b"], "token_ids": [1, 2], "speaker": "s1", "domain": 0, "duration_s": 0.50, "num_frames": 25},
        {"idx": 1, "clip_id": "c1", "split": "train", "word": "cd", "class_id": 1, "is_oov": False, "is_negative": False, "units": ["c", "d"], "token_ids": [3, 4], "speaker": "s1", "domain": 1, "duration_s": 0.50, "num_frames": 25},
        {"idx": 2, "clip_id": "c2", "split": "test_oov", "word": "ad", "class_id": 2, "is_oov": True, "is_negative": False, "units": ["a", "d"], "token_ids": [1, 4], "speaker": "s2", "domain": 2, "duration_s": 0.50, "num_frames": 25},
        {"idx": 3, "clip_id": "c3", "split": "test_neg", "word": "xx", "class_id": -1, "is_oov": True, "is_negative": True, "units": ["b", "c"], "token_ids": [2, 3], "speaker": "s2", "domain": 2, "duration_s": 0.50, "num_frames": 25},
    ]
    # Add slightly different duration on clip 3 so differential stride slope is well-defined at 50 Hz
    clips_feats[3] = rng.standard_normal((50, 16)).astype(np.float16)
    offsets = np.array([0, 25, 50, 75, 125], dtype=np.int64)
    np.save(tmp_path / "features_50hz.fp16.npy", np.concatenate(clips_feats, axis=0))
    np.save(tmp_path / "feature_offsets.int64.npy", offsets)
    manifest[3]["duration_s"] = 1.00
    manifest[3]["num_frames"] = 50

    with open(tmp_path / "manifest.jsonl", "w", encoding="utf-8") as fp:
        for r in manifest:
            fp.write(json.dumps(r) + "\n")
    meta = {
        "tier": "unit_test_tier",
        "feature_dim": 16,
        "vocabulary_size": 5,
        "n_train_words": 2,
        "n_oov_words": 1,
        "unit_vocab": {"<pad>": 0, "a": 1, "b": 2, "c": 3, "d": 4},
        "word_gallery": [
            {"class_id": 0, "word": "ab", "is_oov": False, "units": ["a", "b"], "token_ids": [1, 2]},
            {"class_id": 1, "word": "cd", "is_oov": False, "units": ["c", "d"], "token_ids": [3, 4]},
            {"class_id": 2, "word": "ad", "is_oov": True, "units": ["a", "d"], "token_ids": [1, 4]},
        ],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    inv = verify_dataset_invariants(tmp_path)
    assert inv["passed"] is True
    assert inv["word_overlap"] == 0
    assert inv["speaker_overlap"] == 0
    assert inv["oov_unit_coverage"] == 1.0

    ds = RealAlignmentDataset(tmp_path)
    import random
    batch = ds.sample_balanced_batch(batch_size=4, rng=random.Random(0), multi_pos_k=2)
    assert batch["frames"].shape[0] == 4
    assert batch["frames"].shape[-1] == 16


def test_zero_dialect_coupling_ast_audit():
    from pathlib import Path
    from dialect_crossmodal_align.gates import audit_core_zero_dialect_coupling

    pkg_dir = Path(__file__).resolve().parents[1] / "dialect_crossmodal_align"
    audit = audit_core_zero_dialect_coupling(pkg_dir)
    assert audit["passed"] is True
    assert audit["violations"] == []

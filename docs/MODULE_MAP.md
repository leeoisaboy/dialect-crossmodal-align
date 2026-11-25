# Modular Source Map (v5 Converged)

## Core 4-Stage Converged Pipeline (Default Production Path)

| Public Module | Converged Role (v5) | Empirical Justification |
| --- | --- | --- |
| `pipeline.extract_50hz_features` | Stage 1: 50 Hz frozen SSL speech feature interface | Decouples model from any single backbone (`WavLM-Large`, `Chinese-HuBERT-Large`, etc.) |
| `backbones.LinearDialectAdapter` | Multi-scale dilated Conv1D (`dilations=(1, 2, 4)`) | Spans ~300 ms at 50 Hz; beats 0-conv ablation by +1.6–1.8 pp on dialect OOV |
| `pipeline.RelativePhasePooling` | `[0, 1]` speaking-rate-invariant phase pooling | Consistently reduces `0.6x / 1.6x` speaking-rate drop vs absolute PE across all 3 tiers |
| `heads.TextAnchoredHead` + `criteria.multi_positive_infonce` | Stage 2: Single default open-vocab dual-tower head & loss | Matches or beats `LMBridgedHead` (3.1× faster than `pairwise`) and `AudioToTextSpace` across 3 seeds |
| `calibration.compute_dual_track_gate` | Stage 3: Built-in open-set dual-track confidence scorer | Combines top-1 cosine, pool Z-prominence, and top-1/top-5 margin |
| `calibration.TestTimeDomainCalibrator` | Stage 4: Unlabeled test-time domain calibration | Aligns 50 Hz channel stats & conformal gate quantiles on unlabeled target background (FPR ≤ 2%) |
| `dataset.RealAlignmentDataset` | Language-agnostic 50 Hz manifest loader & balanced sampler | Zero dialect/pinyin hardcoding; enforces 0 word/speaker overlap & 100% OOV unit coverage |
| `gates.evaluate_release_gates` | Automated 4 Hard Release Gates (`Pass/Fail`) + AST audit | Verifies minimal core, real OOV/rate robustness, OOD dual-track calibration, and auditability |

## Archived / Ablation-Only Modules (See `experiments/ARCHIVED.md`)

- `heads.FixedSlotHead`, `heads.AudioToTextSpaceHead`, `heads.LMBridgedHead`
- `criteria.CosFaceProjection`, `criteria.siglip_alignment`
- `aggregators.MultiStatisticPNAPooling`
- `domain_modules.DomainAdversarialHead`, `domain_modules.domain_alignment_loss`, `domain_modules.FSQBottleneck`
- `data.SyntheticDialectWorld` (CPU unit-test fixture only)
- `eval_routing_trap.simulate_track_mismatch` (diagnostic simulator)

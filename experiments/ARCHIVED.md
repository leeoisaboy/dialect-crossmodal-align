# Archived Experimental Branches & Falsified Regularizers

Following the white-box empirical principle ("never hide negative results"), every
prototype branch that failed to deliver a statistically significant gain across
3 random seeds on real 50 Hz multi-dialect & cross-lingual benchmarks
(`T1_wu` Shanghainese + WenetSpeech-Wu, `T2_kespeech` 8 Chinese dialects,
`T3_libriphrase` English + IPA) has been retired from the default production path
(`ConvergedCrossModalAligner`) and kept solely for ablation reproducibility and
regression tests.

## 1. First Subtraction — Pruned Heads, Losses, and Aggregators (3 Seeds)

| Retired Component | 3-Seed Macro OOV R@5 (100-word pool) | Converged Default (`TextAnchored` + `RelativePhase`) | Reason for Retirement |
| --- | ---: | ---: | --- |
| `FixedSlotHead` + `CosFaceProjection` | **0.00%** | **86.6%** (`75.6% / 91.2% / 93.1%`) | Closed-set slots have zero anchors for unseen OOV words (kept only as leakage-check control). |
| `AudioToTextSpaceHead` | 83.41% | 85.61% (same PNA base) | Consistently trails bidirectional `TextAnchoredHead` by -2.20 pp across all 3 tiers. |
| `LMBridgedHead (qformer)` | 84.81% | 85.61% (same PNA base) | Extra cross-attention + 4 latent queries add latency/parameters without beating `TextAnchoredHead`. |
| `LMBridgedHead (pairwise)` | 85.51% (`0.398 ms/clip`) | 85.61% (`0.114–0.130 ms/clip`) | Within 0.10 pp of `TextAnchoredHead`, but **3.1× slower** at inference due to `O(G)` pairwise cross-attention. |
| `CosFaceProjection` (3 sub-centers + temp annealing) | 83.97% | 85.61% (`multi_positive_infonce`) | Sub-center margin annealing hurt zero-shot OOV generalization by -1.64 pp vs `multi_positive_infonce`. |
| `SigLIP` | 85.08% | 85.61% (`multi_positive_infonce`) | Slightly lower macro OOV R@5 (-0.53 pp) and weaker on English IPA (`88.1%` vs `92.5%`). |
| `MultiStatisticPNAPooling` (`4D` stats concat) | 85.61% | **89.15%** (`Mean`) / **86.6%** (`RelativePhase` + rate invariance) | High-dimensional `4D` mean/std/max/min projection overfit on real 1024-D WavLM features compared to lightweight pooling. |
| `SinusoidalPositionalEncoding` (absolute frame grid) | 0.6×/1.6× drop: `34.6 / 38.0 / 35.3 pp` | **Relative `[0, 1]` PE drop: `33.2 / 36.5 / 32.7 pp`** | Absolute frame indices drift out of phase when speakers drag (`0.6×`) or rush (`1.6×`). |

## 2. Second Subtraction — Pruned Domain Regularizers & FSQ (3 Seeds)

Starting from **No Domain Regularizer** on frozen 50 Hz `WavLM-Large` features across
unseen Wu subdialects, unseen KeSpeech dialects (`Southwestern`, `Lan-Yin`, `Northeastern`),
and far-field noisy OOD splits:

| Domain Regularizer Branch | Macro OOV R@5 | Unseen-Dialect OOD R@5 | Far-Field Noisy OOD R@5 | FSQ Codebook Health | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| **None (Converged Default)** | **84.81%** | **81.56%** | **45.27%** | — | **Kept** — frozen 50 Hz SSL features are already domain-invariant. |
| + Class-Conditional CORAL (`labels=classes`) | 84.63% (-0.18 pp) | 80.95% (-0.61 pp) | 44.55% (-0.72 pp) | — | **Removed** — no statistically significant OOD gain. |
| + Marginal CORAL (`labels=None`) | 84.95% (+0.14 pp) | 82.28% (+0.72 pp) | 44.91% (-0.36 pp) | — | **Removed** — within noise & risks small-batch class collapse. |
| + DANN (Gradient Reversal) | 84.89% (+0.08 pp) | 81.75% (+0.19 pp) | 45.28% (+0.01 pp) | — | **Removed** — negligible delta (+0.08 pp) with extra adversarial head. |
| + FSQ Discrete Lattice (`[5,5,5,5,5,5]`) | **77.51% (-7.30 pp)** | **78.13% (-3.43 pp)** | **38.90% (-6.37 pp)** | Perp **90.2**, Unique **95.8%** | **Removed** — even with healthy mask-safe codebook utilization, 6-dim scalar quantization bottlenecks fine phonetic cues. |
| + Class-Cond CORAL + DANN | 84.70% (-0.11 pp) | 81.27% (-0.29 pp) | 44.40% (-0.87 pp) | — | **Removed** — no gain. |
| + All Jointly (CORAL + DANN + FSQ) | **77.60% (-7.21 pp)** | **78.65% (-2.91 pp)** | **38.88% (-6.39 pp)** | Perp **83.7**, Unique **90.3%** | **Removed** — severe OOV degradation + 2.5× slower training. |

## 3. Synthetic Scaffolding Retired to Test Fixtures

- `SyntheticDialectWorld` (`dialect_crossmodal_align/data.py`): decoupled from the converged
  training/inference core and retained strictly as a fast CPU unit-test fixture.
- `simulate_track_mismatch` (`dialect_crossmodal_align/eval_routing_trap.py`): replaced in
  production by real built-in dual-track gating + `TestTimeDomainCalibrator` (`calibration.py`).

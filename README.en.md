<p align="center">
  <a href="https://github.com/leeoisaboy/dialect-crossmodal-align">
    <img src="docs/assets/title_banner.en.png" alt="Dialect-CrossModal-Align — Converged 4-stage cross-modal alignment & unlabeled test-time calibration" />
  </a>
</p>

<p align="center">
  <a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-d73a49?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/"><img alt="Live Lab" src="https://img.shields.io/badge/Live_Lab-CrossModal_Align_Lab-0071e3?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Trilogy 1" src="https://img.shields.io/badge/Trilogy_1-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Trilogy 2" src="https://img.shields.io/badge/Trilogy_2-Latent--KWS_Audit_Studio-7b6bb0?style=flat-square"></a>
  <img alt="Release Gates" src="https://img.shields.io/badge/release_gates-4%2F4%20PASS-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Dialect-CrossModal-Align</h1>

<p align="center"><strong>Subtracting redundant heads and fragile regularizers from cross-modal speech-text alignment on real 50 Hz public corpora—and building unlabeled test-time domain calibration directly into inference.</strong></p>

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="Figure 1: Dialect-CrossModal-Align v5 converged 4-stage open-vocab pipeline and unlabeled test-time calibration" width="1000" />
  <br />
  <sub>Figure 1: Dialect-CrossModal-Align (v5) converged 4-stage open-vocabulary pipeline, 3-seed subtractive pruning, and unlabeled test-time domain calibration</sub>
</p>

Our earlier v4 prototype served as an ablation testbed with four alignment heads, three losses, three domain regularizers, and a synthetic syllable world. In **v5**, we follow the same empirical convergence discipline as [AudioLLM-Latent-KWS](https://github.com/leeoisaboy/audiollm-latent-kws): **benchmark the full prototype on real 50 Hz public multi-dialect & cross-lingual speech first, prune every redundant or negative-gain module across 3 random seeds, and upgrade the routing-trap simulator into built-in open-set dual-track gating with unlabeled test-time domain calibration**.

## 1. Three-Tier Public 50 Hz Real Speech Benchmark

All tiers use frozen **50 Hz** (`20 ms/frame`) `WavLM-Large` representations paired with language-agnostic phonetic/IPA unit sequences, verified strictly by [`verify_dataset_invariants`](dialect_crossmodal_align/dataset.py):

| Tier | Public Corpora & Domain Split | Train / Zero-Shot OOV Words | Word Overlap | Train / Test Speakers | Speaker Overlap | OOV Unit Coverage | Frame Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **T1 Wu** | MagicHub Shanghainese (clean indoor) + WenetSpeech-Wu (wild multi-subdialect OOD) | 600 / **100** | **0** | 349 / 870 | **0** | **100.0%** | 49.96 Hz |
| **T2 Chinese Dialects** | KeSpeech 8 dialects (Mandarin/Ji-Lu/Jiao-Liao/Zhongyuan/Jiang-Huai → **unseen Southwestern/Lan-Yin/Northeastern OOD**) | 500 / **100** | **0** | 287 / 247 | **0** | **100.0%** | 49.97 Hz |
| **T3 English IPA** | LibriPhrase 1–2 word phrases + exact IPA phones (real hard/easy negatives + channel shift) | 180 / **100** | **0** | 630 / 404 | **0** | **100.0%** | 50.05 Hz |

## 2. Two-Stage 3-Seed Subtractive Convergence (H20 GPU 0)

Full raw tables are committed in [reports/ablation_summary.json](reports/ablation_summary.json) and falsified branches are documented in [experiments/ARCHIVED.md](experiments/ARCHIVED.md):

- **First Subtraction (Heads, Losses, Receptive Field, PE)**:
  - `FixedSlotHead` achieves **0.0%** on every 100-word OOV pool (confirming zero leakage).
  - Bidirectional **`TextAnchoredHead` + `Multi-Positive InfoNCE`** achieves **85.61%** macro 3-tier OOV R@5 (`0.130 ms/clip`), matching `LMBridgedHead (pairwise)` (**85.51%**, **3.1× slower** at `0.398 ms/clip`) and outperforming `LMBridgedHead (qformer)` (**84.81%**), `AudioToTextSpaceHead` (**83.41%**), `SigLIP` (**85.08%**), and `CosFace Hybrid` (**83.97%**).
  - Replacing heavy `MultiStatisticPNAPooling` with **`RelativePhasePooling`** (`[0, 1]` `NormalizedRelativePositionalEncoding`) + `dilations=(1, 2, 4)` lifts 3-seed 100-word OOV R@5 to **75.6% (T1) / 91.2% (T2) / 93.1% (T3)** and cuts `0.6×/1.6×` speaking-rate drop below absolute PE on all 3 tiers (`33.2 vs 34.6 pp`, `36.5 vs 38.0 pp`, `32.7 vs 35.3 pp`).
- **Second Subtraction (Domain Regularizers & FSQ)**:
  - On frozen 50 Hz `WavLM-Large` features, **No Domain Regularizer** (**84.81%** OOV / **81.56%** unseen-dialect OOD) matches or beats Class-Conditional CORAL (**84.63% / 80.95%**) and DANN (**84.89% / 81.75%**).
  - Enabling **FSQ (`[5,5,5,5,5,5]`)** shows healthy mask-safe codebook utilization (**Perplexity 90.2, Unique Ratio 95.8%**) yet degrades OOV R@5 by **-7.30 pp** (**77.51%**) due to the 6-dim scalar bottleneck. All three regularizers are stripped from the default training path.

## 3. Built-In Dual-Track Gate & Unlabeled Test-Time Domain Calibration

[`calibration.py`](dialect_crossmodal_align/calibration.py) locks the open-set gate (`Top-1 Cosine + Pool Z-Prominence + Top1–Top5 Margin + Channel Energy`) at `target_fpr = 1%` on in-domain dev negatives. On unseen dialects/channels, `TestTimeDomainCalibrator` aligns 50 Hz per-channel stats and background gate moments using **only an unlabeled target background buffer** (zero labels, zero retraining):

| Tier | In-Domain Dual-Track R@5 (`FPR`) | Uncalibrated OOD Dual-Track R@5 (`AUROC`) | **Calibrated OOD Dual-Track R@5** | **Calibrated OOD FPR** | **Recovery vs In-Domain** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **T1 Wu** (Indoor → Unseen Wild Wu + Channel) | 43.7% (`0.97%`) | **10.1%** (`0.897`) | **54.2%** (`0.943`) | **0.00%** | **125.0%** |
| **T2 Dialects** (Seen → Unseen SW/LY/NE + Channel) | 53.2% (`0.62%`) | **2.9%** (`0.850`) | **35.0%** (`0.892`) | **0.79%** | **66.8%** (12.1× lift) |
| **T3 English** (LibriPhrase + Channel) | 98.9% (`2.92%`) | **34.6%** (`0.974`) | **96.2%** (`0.993`) | 2.31% | **97.3%** |
| **3-Tier Macro (Gate 3)** | **65.3%** | **15.9%** (collapse) | **61.8%** (`AUROC 0.943`) | **1.03% (≤ 2%)** | **94.7% (≥ 80%)** |

## 4. Automated 4-Gate Release Verification & Live Lab

```bash
python -m pip install -e . pytest
pytest -q                                  # Unit tests + AST zero-dialect audit
python scripts/verify_release_gates.py     # All 4 Hard Release Gates (PASS)
python scripts/export_lab_bundle.py        # Export studio/data/lab_bundle.json
python -m http.server 8767 --directory studio
```

<p align="center"><strong><a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">▶ Open CrossModal Align Lab (Zero-Backend Live Demo)</a></strong></p>

## White-Box Speech Retrieval Trilogy

| Repository | Focus | Stack | Live Interactive Studio |
| --- | --- | --- | --- |
| **[disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print)** | Acoustic geometry disentanglement & 4-layer fingerprint oscilloscope | Pure NumPy · Training-free · 60 FPS Live Mic | [Acoustic Print Studio ↗](https://leeoisaboy.github.io/disentangle-acoustic-print/) |
| **[audiollm-latent-kws](https://github.com/leeoisaboy/audiollm-latent-kws)** | Frozen Audio-LLM two-stage retrieval & zero-leak audit studio | Frozen LLM · 0.00% Leak · 7 Archived Ablations | [Latent-KWS Audit Studio ↗](https://leeoisaboy.github.io/audiollm-latent-kws/) |
| **[dialect-crossmodal-align](https://github.com/leeoisaboy/dialect-crossmodal-align)** | Minimal 4-stage cross-modal alignment & unlabeled test-time calibration | PyTorch · 50 Hz Public Bench · 4/4 Gates PASS | [CrossModal Align Lab ↗](https://leeoisaboy.github.io/dialect-crossmodal-align/) |

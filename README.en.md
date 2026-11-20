<p align="center">
  <a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README-d73a49?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/"><img alt="Live Lab" src="https://img.shields.io/badge/Live_Lab-CrossModal_Align_Lab-0071e3?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Trilogy 1" src="https://img.shields.io/badge/Trilogy_1-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Trilogy 2" src="https://img.shields.io/badge/Trilogy_2-Latent--KWS_Audit_Studio-7b6bb0?style=flat-square"></a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Dialect-CrossModal-Align</h1>

<p align="center"><strong>A trainable, ablatable cross-modal dialect alignment lab: interactive routing-trap collapse, compositional zero-shot OOV, speed-invariant attention, and FSQ discrete lattices.</strong></p>

## The Live Interactive Lab

<p align="center">
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">
    <img src="docs/assets/lab-demo.gif" alt="CrossModal Align Lab four-module live interactive showcase" width="960" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">▶ Open CrossModal Align Lab</a></strong></p>

Completing the white-box speech retrieval trilogy alongside [disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print) and [audiollm-latent-kws](https://github.com/leeoisaboy/audiollm-latent-kws), this benchmark explores how to train lightweight cross-modal alignment heads that generalize to unseen dialect words (Zero-Shot OOV), stay phase-locked across speaking speeds, align across domains without class collapse, and expose the **dual-track routing trap** (`AUROC 0.999` yet `4%` joint hit rate under domain shift).

| Module | What you explore live | Code |
| --- | --- | --- |
| 01 Routing Trap Simulator | Drag `gate_domain_shift (0.0 → -1.5)` and watch Recall@5 stay `100%` and AUROC stay `0.999` while joint hit rate collapses `100% → 4%` | `eval_routing_trap.py`, `reports/routing_trap.json` |
| 02 Syllable LEGO × OOV Arena | 3-segment compositional syllable provenance + 4-head OOV similarity PK (`FixedSlot` `0%` vs `LMBridged` `93.8%`) + CosFace temperature annealing | `data.py`, `heads.py`, `criteria.py` |
| 03 Temporal Microscope | Stretch speaking rate `0.5x ↔ 2.0x`: `dilations=(1,2,4)` (`300 ms`), Absolute PE drift vs Normalized Relative PE `[0,1]`, Q-Former vs Pairwise | `backbones.py`, `heads.py` |
| 04 Geometry & FSQ Lattice | 2D unit-sphere alignment (`None` vs `Marginal` collapse vs `Class-Conditional + DANN`) + FSQ `[-1, 1]` discrete snapping & mask-safe Perplexity | `domain_modules.py` |

## Quick Start

```bash
# Preview the zero-backend interactive lab locally
python -m http.server 8767 --directory studio

# Run PyTorch tests, OOV training, and routing-trap benchmark
pytest
python -m dialect_crossmodal_align.train --config configs/zero_shot_oov.yaml
python -m dialect_crossmodal_align.train --config configs/domain_regularized.yaml
python -m dialect_crossmodal_align.eval_routing_trap --output reports/routing_trap.json
```

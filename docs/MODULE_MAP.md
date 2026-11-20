# Modular source map (v4)

| Public module | Consolidated idea (v4) | Integration rule |
| --- | --- | --- |
| `backbones.LinearDialectAdapter` | residual multi-scale dilated Conv1D (`dilations=(1,2,4)`) | spans ~300 ms at 50 Hz while preserving pretrained features |
| `aggregators.MultiStatisticPNAPooling` | order-preserving PNA + positional ROI with zero empty-mask std | all-False masks yield exact `0.0` across mean/std/max/min |
| `aggregators.ProjectedROIPooling` | vectorized ROI + explicit `nn.Linear` | zero `LazyLinear` hazards |
| `aggregators.TextCrossAttention` | padding-safe and empty-mask-safe cross-attention | post-softmax frame mask prevents `value.bias` leakage |
| `heads.NormalizedRelativePositionalEncoding` | speaking-rate-invariant `[0, 1]` relative sinusoidal PE | gives Q-Former & pairwise cross-attention global order awareness |
| `heads.LMBridgedHead` | leak-free Q-Former dual tower or zero-redundancy `pairwise` reranker | `latent_queries` only registered in `qformer` mode; single text pass in `pairwise` |
| `domain_modules` | class-conditional spherical domain alignment, DANN sigmoid, overflow-safe mask-aware FSQ | avoids cross-class confusion in small batches & int64 overflow |
| `criteria` | annealed Sub-Center CosFace, KL-zero SupCon, `1/B` & per-pair SigLIP, `pairwise_alignment` | honors `--contrastive-loss siglip` in both dual-tower & pairwise modes |
| `data.SyntheticDialectWorld` | compositional `token_acoustic_bank`, OOD `allowed_domains` & OOV `allowed_classes` | enables genuine Zero-Shot OOV + OOD evaluation |
| `train` | `decay`/`no_decay` AdamW, warmup cosine LR, `criterion.eval/train` sync, real multi-positive Recall@3/MRR | clean ablation configs + `configs/zero_shot_oov.yaml` |

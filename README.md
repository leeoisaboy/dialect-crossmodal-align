<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/"><img alt="Live Lab" src="https://img.shields.io/badge/Live_Lab-CrossModal_Align_Lab-0071e3?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Trilogy 1" src="https://img.shields.io/badge/三部曲_1-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Trilogy 2" src="https://img.shields.io/badge/三部曲_2-Latent--KWS_Audit_Studio-7b6bb0?style=flat-square"></a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Dialect-CrossModal-Align</h1>

<p align="center"><strong>可训练、可消融的方言开放词表跨模态对齐实验室：把路由陷阱塌方、组合式零样本 OOV、语速不变注意力与 FSQ 离散晶格摊开在浏览器里。</strong></p>

## 正在运行的交互实验室

<p align="center">
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">
    <img src="docs/assets/lab-demo.gif" alt="CrossModal Align Lab 四模块交互演示动图" width="960" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">▶ 进入在线 CrossModal Align Lab</a></strong></p>

作为「白盒语音检索三部曲」的第三块拼图（前两部为纯几何声学示波 [disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print) 与冻结大模型审计台 [audiollm-latent-kws](https://github.com/leeoisaboy/audiollm-latent-kws)），这个项目专门回答：**当我们要从头训练一个轻量跨模态对齐头，如何在低资源方言上扛住跨域偏移、认出训练从未见过的新词（Zero-Shot OOV），并避免踩进「AUROC 0.999 却实战暴跌到 4%」的评测陷阱？**

在线实验室采用无边框浅色长卷轴设计，零后端、纯本地运行，四个核心模块随滚动自动演进并随时可手动调节：

| 模块 | 页面里你能直观看到什么 | 对应代码实现 |
| --- | --- | --- |
| 01 路由陷阱模拟器 | 拖动 `gate_domain_shift (0.0 → -1.5)`，亲眼看 AUROC 稳在 0.999、Recall@5 稳在 100%，双轨命中却撞上红线墙从 100% 塌方到 4% | `eval_routing_trap.py`、`reports/routing_trap.json` |
| 02 音节乐高 × OOV 对决 | 声母/韵母/韵尾三段乐高连线溯源 + 四类对齐头（`FixedSlot` / `TextAnchored` / `AudioToTextSpace` / `LMBridged`）零样本矩阵 PK | `data.py`、`heads.py`、`criteria.py` |
| 03 时序白盒显微镜 | 语速拉杆 `0.5x ↔ 2.0x`：1/2/4 膨胀感受野（300 ms）、绝对 PE 跑偏 vs 归一化相对 PE `[0, 1]`、Q-Former vs Pairwise 切换 | `backbones.py`、`heads.py` |
| 04 域对齐 × FSQ 晶格 | 2D 单位球面（形状=词义，颜色=方言域）：无域正则 vs 边缘对齐坍缩 vs 类条件对齐 + DANN，以及 FSQ 离散吸附与掩码安全 Perplexity | `domain_modules.py` |

## 为什么这套实验值得关注

1. **戳破「假高分真崩溃」的双轨路由陷阱**
   在很多语音唤醒/检索系统里，大家习惯把排序 Recall@5 和门控 AUROC 分开报分。`eval_routing_trap.py` 用 50 个正样本 + 200 个负样本证明：只要换了一个方言或信道导致门控分整体左移 `-1.0`，**AUROC 依然高达 0.999、单轨 Recall@5 依然 100%，但被 `target_fpr=0.01` 锁死的双轨系统真实命中率却从 50/50 断崖暴跌到 2/50（4%）**。
2. **用「音节乐高」做真正公平的零样本 OOV 评测**
   `SyntheticDialectWorld` 维护共享音节声学库 `token_acoustic_bank`，用拉丁方前缀保证训练词覆盖全部底层音节，再把训练词 `[0..7]` 与验证新词 `[8..11]` 严格互斥切分——让闭集 `FixedSlotHead` 当场现形（0% OOV），而开放词表头（`TextAnchoredHead` / `LMBridgedHead`）真正拼出未见新词。
3. **不怕语速快慢的 300 ms 膨胀感受野 + 相对相位编码**
   `LinearDialectAdapter(dilations=(1, 2, 4))` 在 50 Hz 下覆盖 ~300 ms 完整方言声调；`NormalizedRelativePositionalEncoding` 把任意长度有效帧映射到 `[0, 1]` 归一化进度，拖长音或快读连音时，注意力相位依然严丝合缝咬住声母、韵母与韵尾。
4. **类条件域对齐 + 掩码安全的 FSQ 离散晶格**
   小批次类别不均衡时，普通边缘对齐（`labels=None`）会把不同词挤成一团；`domain_alignment_loss(..., labels=classes)` 按类内对齐均值与收缩协方差，让同词跨方言精准重合。配合 `FSQBottleneck` 离散量化时，只统计有效帧（过滤 padding）才能避免码本困惑度与利用率被虚假拉低。

## 快速开始

### 1. 直接打开本地实验室页面（零依赖）
```bash
python -m http.server 8767 --directory studio
```

### 2. 运行 PyTorch 训练、消融与路由陷阱复现
```bash
git clone https://github.com/leeoisaboy/dialect-crossmodal-align.git
cd dialect-crossmodal-align
python -m venv .venv
python -m pip install -e . pytest

pytest
python -m dialect_crossmodal_align.train --config configs/zero_shot_oov.yaml
python -m dialect_crossmodal_align.train --config configs/domain_regularized.yaml
python -m dialect_crossmodal_align.eval_routing_trap --output reports/routing_trap.json
```

<details>
<summary>展开 v4 工程精修细节与代码目录</summary>

### v4 核心工程修复
- **`CosFaceProjection` 训练/评估口径统一**：`train.py` 显式同步 `criterion.train()` / `criterion.eval()`，训练期 `sub_center_temp` 从 `10.0` 平滑退火至 `40.0`。
- **`LMBridgedHead` 零死参数与单次复用**：仅在 `qformer` 模式注册 `latent_queries`；`pairwise` 模式单次编码复用 `text_seq`/`anchors`，完整支持 `siglip` / `infonce` / `multi_positive_infonce`。
- **独立 `sequence_fsq` 与溢出保护**：`FSQBottleneck.codebook_metrics(codes, mask=valid_mask)` 仅统计有效帧，并加入 $\prod L_i \ge 2^{62}$ 的 `int64` 溢出元组哈希保护。
- **优化器 `decay / no_decay` 分组 + Warmup Cosine 调度**：将 `log_scale`、`logit_bias`、`bin_pos`、`latent_queries`、`LayerNorm` 及所有 1D/bias 参数移出权重衰减。

### 目录结构
```text
dialect-crossmodal-align/
├── dialect_crossmodal_align/
│   ├── data.py                  组合式零样本 OOV 合成方言世界
│   ├── backbones.py             1/2/4 膨胀卷积方言声学适配器
│   ├── aggregators.py           掩码安全 MultiStatisticPNAPooling
│   ├── heads.py                 FixedSlot / TextAnchored / AudioToTextSpace / LMBridged
│   ├── criteria.py              CosFace 多子中心退火、SigLIP、Multi-Positive InfoNCE
│   ├── domain_modules.py        类条件域对齐、DANN 调度、FSQ 离散晶格与码本统计
│   ├── eval_routing_trap.py     双轨路由陷阱塌方模拟器
│   └── train.py                 统一训练与评测入口
├── configs/                     6 套即跑即用的消融 YAML 配置
├── studio/                      零后端 CrossModal Align Lab 静态前端（GitHub Pages）
├── scripts/                     Lab 数据打包与演示 GIF 生成脚本
├── reports/routing_trap.json    路由陷阱实测报告
└── tests/test_align.py          完整单元测试
```

</details>

## 白盒语音检索三部曲

- **[disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print)** — 纯 NumPy 声学几何解耦与 60 FPS 实时麦克风示波（[在线体验](https://leeoisaboy.github.io/disentangle-acoustic-print/)）
- **[audiollm-latent-kws](https://github.com/leeoisaboy/audiollm-latent-kws)** — 冻结 Audio-LLM 两阶段方言检索与双网格防泄漏审计台（[在线体验](https://leeoisaboy.github.io/audiollm-latent-kws/)）
- **[dialect-crossmodal-align](https://github.com/leeoisaboy/dialect-crossmodal-align)** — 可训练跨模态对齐、组合式零样本 OOV 与路由陷阱实验室（[在线体验](https://leeoisaboy.github.io/dialect-crossmodal-align/)）

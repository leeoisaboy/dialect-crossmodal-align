<p align="center">
  <a href="https://github.com/leeoisaboy/dialect-crossmodal-align">
    <img src="docs/assets/title_banner.zh.png" alt="Dialect-CrossModal-Align 给跨模态方言对齐做减法，把测试时域校准装进推理" />
  </a>
</p>

<p align="center">
  <a href="README.en.md"><img alt="English" src="https://img.shields.io/badge/English-README-2f81f7?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/"><img alt="Live Lab" src="https://img.shields.io/badge/Live_Lab-CrossModal_Align_Lab-0071e3?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/Leeoisaboy/dialect-crossmodal-align-bench"><img alt="HF Dataset" src="https://img.shields.io/badge/HuggingFace-50Hz_3--Tier_Bench-ff9d00?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/disentangle-acoustic-print/"><img alt="Trilogy 1" src="https://img.shields.io/badge/三部曲_1-Acoustic_Print_Studio-2b9c94?style=flat-square"></a>
  <a href="https://leeoisaboy.github.io/audiollm-latent-kws/"><img alt="Trilogy 2" src="https://img.shields.io/badge/三部曲_2-Latent--KWS_Audit_Studio-7b6bb0?style=flat-square"></a>
  <img alt="Release Gates" src="https://img.shields.io/badge/release_gates-4%2F4%20PASS-2ea44f?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-555555?style=flat-square">
</p>

<h1 align="center">Dialect-CrossModal-Align</h1>

<p align="center"><strong>给跨模态方言对齐做减法：用三套公开 50 Hz 语料实测剪掉多余对齐头与脆弱域正则，再把无标签测试时域校准装进默认推理链路。</strong></p>

<p align="center">
  <strong>🤗 公开三级 50 Hz 评测数据集（34,138 条切片 · T1 沪语/吴语 · T2 八大方言 · T3 英文 IPA）：<a href="https://huggingface.co/datasets/Leeoisaboy/dialect-crossmodal-align-bench">Leeoisaboy/dialect-crossmodal-align-bench ↗</a></strong>
</p>

<p align="center">
  <img src="docs/assets/pipeline_architecture.jpg" alt="图 1：Dialect-CrossModal-Align v5 四阶段最简收敛流水线与无标签测试时域校准架构" width="1000" />
  <br />
  <sub>图 1：Dialect-CrossModal-Align (v5) 四阶段最简开放词表流水线、三随机种子减法收敛与无标签测试时域校准架构</sub>
</p>

上一版原型把方言跨模态对齐做成了“4 种对齐头 + 3 套损失 + 3 套域正则 + 合成音节乐高世界”的实验台。机制在合成数据上跑得很漂亮，却始终悬着四个工程真问题：换成真实预训练 50 Hz 语音特征以后，重型交叉注意力到底比最简双向文本锚点强多少？类条件 CORAL、DANN 和 FSQ 离散晶格在真实未见方言上真有正向增益吗？绝对位置编码与相对位置编码在真实快慢语速缩放下差距有多大？以及，不靠人工加固定偏移，怎么在真实跨域负样本上把「AUROC 很高、双轨命中却塌方」的路由陷阱真正修好？

这一版更新（v5），我按照和姊妹仓库 [AudioLLM-Latent-KWS](https://github.com/leeoisaboy/audiollm-latent-kws) 一致的路线推进：**先接入三套公开多方言与跨语种 50 Hz 真实语料摸清原版底牌，再两刀做三随机种子减法消融，最后把路由陷阱诊断升级为推理内置的双轨门控与无标签测试时域校准**。

## 一、脱离合成世界，搭建三级公开 50 Hz 真实考卷

为了彻底避免合成数据带来的失真乐观，新管线统一认标准 `50 Hz`（`20 ms/帧`）冻结语音特征（默认 `WavLM-Large` 16–20 层均值）和语言无关音素/IPA 序列，不写死任何单一语种逻辑。三套公开评测集全部通过 [`verify_dataset_invariants`](dialect_crossmodal_align/dataset.py) 自动断言校验：

| 评测层级 | 公开语料与域划分 | 训练词 / 零样本 OOV 词 | 词表交集 | 训练 / 测试说话人 | 说话人交集 | OOV 构词单元覆盖率 | 实测帧率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **T1 吴语** | MagicHub 沪语室内域（训练）+ WenetSpeech-Wu 野生多子方言（含未见 OOD 子域） | 600 / **100** | **0** | 349 / 870 | **0** | **100.0%** | 49.96 Hz |
| **T2 跨汉语方言** | KeSpeech 八大方言（官话/冀鲁/胶辽/中原/江淮训练 → **西南/兰银/东北未见 OOD**） | 500 / **100** | **0** | 287 / 247 | **0** | **100.0%** | 49.97 Hz |
| **T3 英文开放集** | LibriPhrase 英文短语 + 精确 IPA 音素（含难/易真实负例与远场信道偏移） | 180 / **100** | **0** | 630 / 404 | **0** | **100.0%** | 50.05 Hz |

每套测试集还同步生成了经声学变速拉伸（`0.6×` 慢速拖音与 `1.6×` 快速连读）后重新提取 50 Hz 特征的测试切片，以及严格的开放集非关键词负例流。

## 二、原版 6 套配置摸底：真数据一眼照出虚胖模块

不动原版 `AlignmentModel` 结构，直接把仓库原有的 6 套 YAML 配置拉到 H20（GPU 0, fp32）上做全量摸底（完整原始数据见 [reports/ablation_summary.json](reports/ablation_summary.json)）：

| 原版配置 | T1 吴语 100词 OOV R@5 | T2 八大方言 100词 OOV R@5 | T3 英文 IPA 100词 OOV R@5 | T1 单次训练 / 推理耗时 | 摸底诊断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fixed_slot` (闭集 CosFace) | **0.0%** | **0.0%** | **0.0%** | 17.96s / 0.102ms | 闭集命中 50.4%–97.5%，OOV 严格为 0%（零泄漏对照） |
| `text_anchored` | **72.8%** | **92.2%** | **91.5%** | 19.33s / 0.129ms | 最轻双向文本锚点，三集全部大幅越过随机基线 |
| `audio_to_text_space` | 72.0% | 88.9% | 88.6% | 19.53s / 0.130ms | 略逊于双向文本锚点 |
| `lm_bridged` (`qformer`) | 74.9% | 91.5% | 87.3% | 21.86s / 0.137ms | 与最简头互有胜负，优势微弱 |
| `domain_regularized` (CORAL+DANN+FSQ) | **55.2%** | **66.9%** | **57.8%** | 51.25s / 0.135ms | **全线掉点 18–34 个百分点，训练慢近 3 倍** |
| `zero_shot_oov` (含全套域正则) | **54.8%** | **66.1%** | **66.7%** | 38.53s / 0.113ms | 同样被重型正则拖垮 |

## 三、两刀架构减法：三随机种子消融砍掉所有负收益模块

### 第一刀：收敛对齐头、损失函数、池化与位置编码（3 种子均值）

| 消融维度 | 候选方案 | 三集宏平均 OOV R@5 | T1 吴语 | T2 八大方言 | T3 英文 | 单条推理 | 减法决策 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 对齐头 PK | **`TextAnchoredHead`** | **85.61%** | 73.0±0.2% | 91.4% | **92.5%** | **0.130 ms** | **保留为唯一默认头** |
| 对齐头 PK | `LMBridgedHead (pairwise)` | 85.51% | 75.5±0.2% | 91.6% | 89.4% | 0.398 ms | 差 0.10 pp 但慢 **3.1×** → 归档 |
| 对齐头 PK | `LMBridgedHead (qformer)` | 84.81% | 73.8±0.8% | 90.4% | 90.3% | 0.128 ms | 低于最简头 → 归档 |
| 对齐头 PK | `AudioToTextSpaceHead` | 83.41% | 72.5±0.7% | 88.2% | 89.5% | 0.130 ms | 落后 2.20 pp → 归档 |
| 损失函数 | **`Multi-Positive InfoNCE`** | **85.61%** | 73.0% | 91.4% | **92.5%** | 0.130 ms | **保留为唯一默认损失** |
| 损失函数 | `SigLIP` / `CosFace Hybrid` | 85.08% / 83.97% | 74.9% / 73.5% | 92.3% / 91.4% | 88.1% / 87.1% | 0.124 ms | 无显著优势 → 归档 |
| 膨胀感受野 | **`dilations=(1,2,4)` vs 无卷积** | 84.81% vs 84.02% | 73.8% vs 72.1% | 90.4% vs 88.6% | 90.3% vs 91.4% | 0.125 ms | 方言净增益 +1.7–1.8 pp → **保留** |
| 语速位置编码 | **`RelativePhasePooling [0,1]` vs 绝对 PE** | **86.65%** | **75.6%** (`-33.2pp` vs `-34.6pp`) | **91.2%** (`-36.5pp` vs `-38.0pp`) | **93.1%** (`-32.7pp` vs `-35.3pp`) | **0.12 ms** | 三集 `0.6×/1.6×` 掉点全部小于绝对 PE → **保留** |

### 第二刀：逐项剥离类条件 CORAL、DANN 与 FSQ（3 种子均值）

在真实未见子方言与远场加噪域上，以“无任何域正则”为起点逐项叠加（完整负收益归档见 [experiments/ARCHIVED.md](experiments/ARCHIVED.md)）：

| 域正则配置 | 三集宏平均 OOV R@5 | 未见方言 OOD R@5 | 远场加噪 OOD R@5 | FSQ 码本健康度 | 减法结论 |
| --- | ---: | ---: | ---: | --- | --- |
| **无域正则（v5 默认）** | **84.81%** | **81.56%** | **45.27%** | — | **保留**（冻结 50 Hz WavLM 特征自带域不变性） |
| + 类条件 CORAL (`labels=classes`) | 84.63% (-0.18 pp) | 80.95% (-0.61 pp) | 44.55% (-0.72 pp) | — | **移出默认流** |
| + DANN 梯度反转 | 84.89% (+0.08 pp) | 81.75% (+0.19 pp) | 45.28% (+0.01 pp) | — | **移出默认流**（增益可忽略） |
| + FSQ 离散晶格 (`[5,5,5,5,5,5]`) | **77.51% (-7.30 pp)** | **78.13% (-3.43 pp)** | **38.90% (-6.37 pp)** | Perp **90.2** / Unique **95.8%** | **移出默认流**（码本很健康，但 6 维量化瓶颈直接损伤细粒度音素） |
| + 全套叠加 (CORAL + DANN + FSQ) | **77.60% (-7.21 pp)** | **78.65% (-2.91 pp)** | **38.88% (-6.39 pp)** | Perp **83.7** / Unique **90.3%** | **移出默认流** |

## 四、把路由陷阱诊断升级为内置双轨门控与无标签测试时域校准

上一版 [`eval_routing_trap.py`](dialect_crossmodal_align/eval_routing_trap.py) 只用高斯随机数加固定偏移展示塌方现象。v5 直接将它升级为推理内置的 [`calibration.py`](dialect_crossmodal_align/calibration.py)：
1. **内置开放集双轨门控（`compute_dual_track_gate`）**：融合 Top-1 余弦相似度、池内 Z-Prominence、Top-1/Top-5 裕度与信道能量，在同域开发负例上按 `target_fpr = 1%` 锁死门控墙。
2. **轻量无标签测试时域校准（`TestTimeDomainCalibrator`）**：部署到未见方言或衰减信道时，只需收集一小段**无标签目标域背景音频**（零标注、零重训），先做 50 Hz 逐通道仿射对齐，再把目标域背景门控分对齐到同域负例均值方差并施加 99% 分位护栏。

三随机种子真实跨域实测结果（完整数据见 [reports/convergence_gates.json](reports/convergence_gates.json)）：

| 评测层级 | 同域双轨 R@5 (`FPR`) | 跨域未校准双轨 R@5 (`AUROC`) | **跨域无标签校准后双轨 R@5** | **校准后 OOD FPR** | **相对同域恢复率** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **T1 吴语**（室内沪语 → 未见野生吴语+信道漂移） | 43.7% (`0.97%`) | **10.1%** (`0.897`) | **54.2%** (`0.943`) | **0.00%** | **125.0%** |
| **T2 八大方言**（官话等 → 未见西南/兰银/东北+信道漂移） | 53.2% (`0.62%`) | **2.9%** (`0.850`) | **35.0%** (`0.892`) | **0.79%** | **66.8%** (12.1× 恢复) |
| **T3 英文开放集**（LibriPhrase 信道漂移） | 98.9% (`2.92%`) | **34.6%** (`0.974`) | **96.2%** (`0.993`) | 2.31% | **97.3%** |
| **三集宏平均（Gate 3 门禁口径）** | **65.3%** | **15.9%**（断崖塌方） | **61.8%**（`AUROC 0.943`） | **1.03% (≤ 2%)** | **94.7% (≥ 80%)** |

## 五、四项硬性发布门禁全绿与交互实验室

仓库内置 [`scripts/verify_release_gates.py`](scripts/verify_release_gates.py) 一键执行四门禁自动化验收（全部 **PASS**）：

| 发布门禁 | 量化验收标准 | v5 实测结果 | 状态 |
| --- | --- | --- | :---: |
| **门禁 1：最简架构与零语种绑定** | 单默认头、零默认域正则、核心包 AST 扫描 0 方言/拼音硬编码 | 扫描 9 个核心文件，0 违规导入 | **PASS** |
| **门禁 2：真实零样本 OOV 与语速弹性** | 三集 100 词 OOV R@5 ≥ 60%，且 0.6×/1.6× 相对 PE 掉点 < 绝对 PE | T1 **75.6%** / T2 **91.2%** / T3 **93.1%**；三集相对 PE 掉点全胜 | **PASS** |
| **门禁 3：真实跨域双轨鲁棒性** | 未见域+真实负例下，校准后 FPR ≤ 2% 且恢复 ≥ 80% 同域双轨命中 | 宏平均 **FPR = 1.03%**，双轨命中由 15.9% 恢复至 **61.8% (94.7%)** | **PASS** |
| **门禁 4：可复现与可审计** | 0 词表/说话人重叠、100% 单元覆盖、50 Hz 校验、Lab 数据包与单测全绿 | 34,138 切片全检通过 + 单元测试全绿 | **PASS** |

<p align="center">
  <a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">
    <img src="docs/assets/lab-demo.gif" alt="CrossModal Align Lab 四模块交互演示动图" width="960" />
  </a>
</p>

<p align="center"><strong><a href="https://leeoisaboy.github.io/dialect-crossmodal-align/">▶ 进入在线 CrossModal Align Lab（零后端纯本地）</a></strong></p>

## 快速开始

```bash
git clone https://github.com/leeoisaboy/dialect-crossmodal-align.git
cd dialect-crossmodal-align
python -m pip install -e . pytest

# 1. 运行全部单元测试（含 v5 四阶段主干、真实 50 Hz 契约校验与 AST 零语种依赖审计）
pytest -q

# 2. 一键验收四项硬性发布门禁
python scripts/verify_release_gates.py

# 3. 导出最新白盒数据并本地打开 CrossModal Align Lab
python scripts/export_lab_bundle.py
python -m http.server 8767 --directory studio
```

## 白盒语音检索三部曲

| 项目 | 核心定位 | 底层形态 | 在线交互界面 |
| --- | --- | --- | --- |
| **[disentangle-acoustic-print](https://github.com/leeoisaboy/disentangle-acoustic-print)** | 声学几何解耦与四层指纹示波器 | 纯 NumPy · 免训练 · 支持 60 FPS 实时麦克风 | [Acoustic Print Studio ↗](https://leeoisaboy.github.io/disentangle-acoustic-print/) |
| **[audiollm-latent-kws](https://github.com/leeoisaboy/audiollm-latent-kws)** | 冻结 Audio-LLM 两阶段检索与防泄漏审计台 | 冻结大模型 · 0.00% 泄题 · 7 项阴性归档 | [Latent-KWS Audit Studio ↗](https://leeoisaboy.github.io/audiollm-latent-kws/) |
| **[dialect-crossmodal-align](https://github.com/leeoisaboy/dialect-crossmodal-align)** | 最简四阶段跨模态对齐与无标签测试时域校准 | PyTorch · 50 Hz 公开实测 · 四门禁全绿 | [CrossModal Align Lab ↗](https://leeoisaboy.github.io/dialect-crossmodal-align/) |

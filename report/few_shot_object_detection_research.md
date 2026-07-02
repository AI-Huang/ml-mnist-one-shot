# Few-Shot Object Detection 领域的研究成果

## 1. 任务背景

Few-Shot Object Detection,简称 FSOD,研究的是在**基类拥有充足标注数据、新类只有极少标注框**的条件下,让检测模型快速识别并定位新类别目标。它比普通小样本分类更难,因为模型不仅要判断类别,还要完成候选区域生成、边界框回归、前景/背景区分和类别间细粒度区分。

典型设置包括:

| 设置 | 含义 |
| --- | --- |
| Base classes | 训练阶段拥有大量标注的类别 |
| Novel classes | 测试或微调阶段只有 K-shot 标注的类别 |
| K-shot | 每个新类别仅有 K 个标注实例,常见为 1/2/3/5/10/30-shot |
| 评价指标 | PASCAL VOC 上常用 nAP / mAP,COCO 上常用 AP / AP50 / AP75 |

FSOD 的核心目标不是单纯提高检测精度,而是在**标注稀缺、类别迁移、背景干扰和定位要求并存**的场景下获得稳定泛化能力。

---

## 2. 主要技术路线

### 2.1 基于微调的两阶段方法

早期且影响很大的路线是先在基类上训练检测器,再用少量新类样本微调。代表性方法包括 TFA,即 Two-Phase Fine-Tuning Approach。它的关键结论是:在强检测器和合理训练策略下,简单微调分类头和回归头就能取得很强基线。

这类方法的贡献在于建立了 FSOD 的强基准,也让后续研究意识到:复杂元学习模块不一定总是优于良好控制的预训练与微调流程。其局限是对样本选择敏感,新类微调容易过拟合,且基类与新类性能之间存在明显权衡。

### 2.2 元学习与支持集条件检测

元学习路线将少量标注样本视为 support set,查询图像视为 query image,通过特征重加权、注意力或类别原型来指导检测。代表思想包括:

- 用 support image 提取类别条件特征,再调制 query feature map。
- 学习类别原型,让检测头根据原型完成分类。
- 通过 episodic training 模拟测试时的 few-shot 场景。

这类方法的优势是任务形式更贴近小样本学习,能更自然地处理新类别扩展。但实际表现常受 support 样本质量、背景噪声和训练 episode 设计影响,工程复杂度也更高。

### 2.3 度量学习与原型分类

度量学习方法将检测区域特征映射到嵌入空间,通过与类别原型或支持样本的相似度完成分类。研究重点包括:

- 构造更稳定的新类原型。
- 减少基类对新类的偏置。
- 改善 RoI 特征在少样本条件下的类间可分性。

这类方法通常比普通分类头更适合低样本条件,但当新类内部差异较大或目标姿态变化明显时,单一原型可能表达不足。

### 2.4 数据增强与生成式方法

由于 FSOD 的根本瓶颈是新类标注实例稀缺,数据增强一直是重要方向。常见方法包括:

- 几何增强、颜色扰动、Copy-Paste 等传统增强。
- 基于上下文的目标粘贴,缓解目标与背景分布不匹配。
- 使用生成模型合成新类样本或 RoI 特征。

这类方法的价值在于直接扩充训练分布,但风险是生成样本质量不稳定,过量低质量样本可能引入标签噪声和定位偏差。

### 2.5 Transformer 与开放词表检测带来的新进展

近几年,DETR 系列、视觉语言模型和开放词表检测推动了 FSOD 的新方向。研究者开始利用大规模预训练模型的语义知识,降低对新类框标注的依赖。典型思路包括:

- 用 CLIP 等视觉语言模型提供类别文本语义。
- 将检测器训练成可迁移的 class-agnostic proposal generator。
- 结合 prompt、文本类别名和少量框标注进行快速适配。
- 利用大规模预训练检测器做参数高效微调。

这一方向的意义在于,FSOD 不再只依赖少量图片本身,而是可以借助外部语义和大规模预训练知识完成类别迁移。

---

## 3. 代表性研究成果

| 阶段 | 代表方向 | 主要成果 |
| --- | --- | --- |
| 早期探索 | 元学习、特征重加权 | 将 few-shot classification 的 episode 思想引入检测任务 |
| 强基线阶段 | 基类预训练 + 新类微调 | TFA 等方法证明简单两阶段微调可以成为强基准 |
| 稳定性改进 | 度量学习、对比学习、去偏置 | 缓解基类偏置,提升新类分类边界稳定性 |
| 数据效率提升 | Copy-Paste、生成式增强、伪标签 | 扩大新类训练分布,降低标注稀缺带来的过拟合 |
| 大模型阶段 | Vision-Language Pretraining、Open-Vocabulary Detection | 利用文本语义和大规模预训练知识提升新类迁移能力 |

从整体发展看,FSOD 的研究成果可以概括为三点:

1. **从复杂元学习走向强基线和可复现训练策略**:领域逐渐重视公平数据划分、多次采样评估和稳定微调流程。
2. **从只看新类性能走向基类-新类平衡**:实际应用要求模型不能为了适配新类而严重遗忘基类。
3. **从封闭类别迁移走向开放语义迁移**:视觉语言预训练让少样本检测与开放词表检测、开放世界检测发生交叉。

### 3.1 SOTA / 代表性论文清单

严格来说,FSOD 的“SOTA”依赖数据集、novel split、shot 数、是否使用额外数据、是否使用视觉语言预训练等设置,不能只用一个绝对排名概括。下面按研究脉络列出最值得优先阅读和对比的论文。

| 类别 | 论文 | 会议/年份 | 关键贡献 |
| --- | --- | --- | --- |
| 早期元学习 | **Few-Shot Object Detection via Feature Reweighting** | ICCV 2019 | 用 support 特征重加权 query 特征,是早期 FSOD 代表作之一 |
| 早期元学习 | **Meta R-CNN: Towards General Solver for Instance-Level Low-Shot Learning** | ICCV 2019 | 将元学习机制引入 Faster R-CNN,同时覆盖检测与分割 |
| 强基线 | **Frustratingly Simple Few-Shot Object Detection (TFA)** | ICML 2020 | 两阶段训练 + 微调检测头,建立了非常强且常用的 FSOD baseline |
| 多尺度鲁棒性 | **Multi-Scale Positive Sample Refinement for Few-Shot Object Detection (MPSR)** | ECCV 2020 | 针对少样本检测中的尺度变化问题改进正样本学习 |
| 对比学习 | **FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding** | CVPR 2021 | 在 proposal 表征上加入对比学习,提升类间可分性 |
| 去耦训练 | **DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection** | ICCV 2021 | 解耦 RPN、RoI 分类与回归,是传统封闭集 FSOD 中非常强的代表方法 |
| 注意力匹配 | **QA-FewDet: Query-Adaptive Few-Shot Object Detection with Heterogeneous Graph Convolutional Networks** | ICCV 2021 | 建模 query 与 support 的自适应关系,改善类别条件检测 |
| Transformer | **Few-Shot Object Detection via Fully Cross-Transformer (FCT)** | CVPR 2022 | 用 cross-transformer 强化 support-query 交互,代表 Transformer 化 FSOD 路线 |
| 开放词表 | **GLIP: Grounded Language-Image Pre-training** | CVPR 2022 | 将检测与短语 grounding 统一到视觉语言预训练中,显著提升开放类别迁移能力 |
| 开放词表 | **OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers** | ECCV 2022 | 基于图文预训练实现开放词表检测,常被视作 FSOD 邻近路线的重要基线 |
| 开放词表 | **Detic: Detecting Twenty-thousand Classes using Image-level Supervision** | ECCV 2022 | 用图像级监督扩展检测类别空间,推动长尾/开放词表检测 |
| 开放词表 | **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection** | 2023 | 强开放集/文本条件检测器,常作为大模型时代检测迁移的重要基础模型 |

如果只关注**传统封闭集 FSOD benchmark**(如 VOC / COCO 的 base-novel split),建议优先看 **TFA → FSCE → DeFRCN → FCT** 这一条线;如果关注**最新趋势和实际可迁移能力**,建议同时看 **GLIP / OWL-ViT / Detic / Grounding DINO** 这一类开放词表检测论文。

### 3.2 指标数据说明

本仓库当前实验是 **MNIST one-shot 分类**,只产生 classification accuracy,没有运行目标检测任务,因此仓库内**没有本地生成的 nAP / mAP / AP / AP50 / AP75 指标数据**。这些指标需要在 PASCAL VOC、MS COCO、LVIS 等检测数据集上训练和评估检测器后才能得到。

FSOD 论文中的指标通常这样解读:

| 指标 | 常见数据集 | 含义 |
| --- | --- | --- |
| nAP | VOC / COCO few-shot split | Novel classes AP,只统计新类检测性能 |
| bAP | VOC / COCO few-shot split | Base classes AP,统计基类保持能力 |
| mAP | VOC 常用 | 所有类别平均 AP,有时也指 AP50 下的均值 |
| AP | COCO 常用 | IoU 0.50:0.95 多阈值平均 AP,比 AP50 更严格 |
| AP50 | VOC / COCO 常用 | IoU=0.50 时的 AP |
| AP75 | COCO 常用 | IoU=0.75 时的 AP,更强调定位质量 |

如果要整理论文中的数值,建议按如下维度分表,否则不同论文结果不可直接比较:

| 维度 | 需要固定的设置 |
| --- | --- |
| 数据集 | PASCAL VOC / MS COCO / LVIS |
| 类别划分 | base / novel split,例如 VOC split 1/2/3 |
| shot 数 | 1/2/3/5/10/30-shot |
| 是否用额外数据 | ImageNet、Objects365、CLIP/视觉语言预训练等 |
| 检测器骨干 | Faster R-CNN、ResNet-50/101、ViT、DETR 系列等 |
| 指标口径 | nAP50、nAP75、COCO-style AP、base/novel/all classes |

因此,更严谨的写法不是单独给一个“SOTA AP”,而是写成“在某数据集、某 split、某 shot、某训练协议下,某论文报告了某指标”。

### 3.3 论文声明的代表性指标

下面整理的是论文或官方仓库公开表格中的**代表性数值**,主要用于快速了解量级。由于不同论文可能使用 single run / multiple runs、不同 few-shot sample、不同 backbone,这些数值不应脱离协议直接做绝对排名。

#### PASCAL VOC novel AP50

VOC FSOD 通常报告 novel classes 上的 AP50。下表摘取各论文在 **10-shot** 下的三个 novel split 结果,数值越高越好。

| 方法 | 年份 | Backbone | Split 1 | Split 2 | Split 3 |
| --- | --- | --- | --- | --- | --- |
| FSRW | ICCV 2019 | YOLOv2 | 47.2 | 40.5 | 45.9 |
| Meta R-CNN | ICCV 2019 | ResNet-101 | 51.5 | 45.4 | 48.1 |
| TFA w/ cos | ICML 2020 | ResNet-101 | 56.0 | 39.1 | 49.8 |
| MPSR | ECCV 2020 | ResNet-101 | 61.8 | 47.8 | 49.7 |
| FSCE | CVPR 2021 | ResNet-101 | 63.4 | 50.2 | 58.5 |
| QA-FewDet | ICCV 2021 | ResNet-101 | 63.4 | 51.1 | 53.5 |
| DeFRCN | ICCV 2021 | ResNet-101 | 60.8 | 47.9 | 57.4 |
| FCT | CVPR 2022 | PVTv2-B2-Li | 67.1 | 51.2 | 58.7 |

FCT 论文还报告了 multiple runs 平均结果。在同一表格中,FCT 的 10-shot VOC novel AP50 平均为 **64.3 / 47.4 / 56.3**(split 1/2/3),DeFRCN 为 **66.5 / 52.8 / 60.8**。

#### MS COCO novel AP

COCO FSOD 通常报告 novel classes 上的 COCO-style AP,即 IoU 0.50:0.95 平均 AP。下表列出常见 shot 设置下的代表性结果。

| 方法 | 1-shot | 2-shot | 3-shot | 5-shot | 10-shot | 30-shot |
| --- | --- | --- | --- | --- | --- | --- |
| FSRW | - | - | - | - | 5.6 | 9.1 |
| Meta R-CNN | - | - | - | - | 8.7 | 12.4 |
| TFA | 4.4 | 5.4 | 6.0 | 7.7 | 10.0 | 13.7 |
| MPSR | 5.1 | 6.7 | 7.4 | 8.7 | 9.8 | 14.1 |
| FSCE | - | - | - | - | 11.9 | 16.4 |
| QA-FewDet | 4.9 | 7.6 | 8.4 | 9.7 | 11.6 | 16.5 |
| DeFRCN | 9.3 | 12.9 | 14.8 | 16.1 | 18.5 | 22.6 |
| FCT | 5.6 | 7.9 | 11.1 | 14.0 | 17.1 | 21.4 |

如果采用 multiple runs 平均协议,FCT 论文表格中 COCO novel AP 为 **5.1 / 7.2 / 9.8 / 12.0 / 15.3 / 20.2**,DeFRCN 为 **4.8 / 8.5 / 10.7 / 13.6 / 16.8 / 21.2**。

#### AP / AP50 / AP75 示例

很多 FSOD 主表只给 VOC AP50 或 COCO AP,不一定同时给 AP50/AP75。FCT 论文的 MS COCO 消融表明确给出了 AP、AP50、AP75,可作为指标口径示例:

| 方法 | Shot | AP | AP50 | AP75 |
| --- | --- | --- | --- | --- |
| FCT | 2-shot | 7.9 | 14.2 | 7.9 |
| FCT | 10-shot | 17.1 | 30.2 | 17.0 |
| FCT | 30-shot | 21.4 | 35.5 | 22.1 |

以上数值说明:AP50 通常显著高于 AP,因为 AP 是多个 IoU 阈值的平均;AP75 更关注定位精度,在小样本检测中通常比 AP50 更难提升。

---

## 4. 常用数据集与评价协议

### 4.1 PASCAL VOC

PASCAL VOC 是 FSOD 最常用的早期基准。通常将 20 个类别划分为 base / novel,使用多个 split 进行评估。它的优点是规模适中、对比充分;缺点是类别数量较少,难以反映复杂场景。

### 4.2 MS COCO

COCO 更接近真实场景,类别更多、目标尺度变化更大、背景更复杂。FSOD 在 COCO 上的 AP 通常更低,因此更能检验方法的泛化和定位能力。

### 4.3 LVIS 与开放词表基准

LVIS 具有长尾类别分布,适合研究少样本、长尾和开放词表检测之间的关系。随着大规模视觉语言模型发展,LVIS、COCO open-vocabulary split 等基准的重要性持续上升。

---

## 5. 当前挑战

### 5.1 新类过拟合

K-shot 标注框极少,模型很容易记住少数实例的外观,而不能学习类别层面的稳定概念。

### 5.2 基类遗忘与类别偏置

微调新类时,模型可能损害基类性能;不微调或微调不足时,分类器又容易偏向基类。

### 5.3 定位泛化困难

小样本分类只需识别类别,FSOD 还要回归边界框。新类样本少会导致尺度、姿态、遮挡和上下文变化覆盖不足。

### 5.4 评价协议不统一

不同论文可能使用不同 split、shot 采样、随机种子和微调策略,导致结果难以直接比较。现代研究越来越强调多次采样平均和公开配置。

### 5.5 真实应用仍有差距

真实场景常包含类别持续增加、标注噪声、长尾分布、开放集干扰和域迁移,比标准 FSOD benchmark 更复杂。

---

## 6. 发展趋势

1. **与开放词表检测融合**:用文本类别名、属性描述和视觉语言对齐减少对框标注的依赖。
2. **参数高效适配**:通过 adapter、prompt tuning、LoRA 等方式快速适配新类,降低训练成本和遗忘风险。
3. **更强的类别无关定位能力**:先学习通用 proposal / objectness,再用少量样本或文本语义完成分类。
4. **生成式数据与合成标注**:扩散模型和可控生成有望提供更高质量的新类样本,但需要控制真实性和标签一致性。
5. **统一小样本、长尾和开放世界检测**:未来 FSOD 可能不再是孤立任务,而会成为开放环境视觉识别系统的一部分。

---

## 7. 对本项目的启发

本仓库研究的是 MNIST one-shot 分类,与 FSOD 不同,但二者共享“小样本泛化”的核心矛盾。结合 FSOD 研究成果,可以得到以下启发:

- **强基线很重要**:在引入复杂模块前,应先建立可靠的预训练 + 微调或增强基线。
- **数据增强需控制质量与比例**:本项目中 cGAN 过量会降低准确率,这与 FSOD 中低质量生成样本可能伤害检测性能的现象一致。
- **迁移学习是低样本任务的关键手段**:无论是 MNIST one-shot 还是 FSOD,预训练特征通常能显著提升小样本泛化。
- **多 seed 评估不可省略**:少样本任务对样本抽样高度敏感,单次结果不具备充分说服力。

---

## 8. 小结

Few-Shot Object Detection 的研究已经从早期元学习探索,发展到强基线微调、度量学习、数据增强、去偏置训练以及视觉语言预训练融合的多路线并进阶段。当前最重要的研究趋势是利用大规模预训练模型获得更强的通用视觉与语义表示,再通过少量标注实现快速、稳定、低遗忘的新类别检测。

总体而言,FSOD 的核心成果不只是提高 benchmark 分数,而是推动目标检测从“依赖大量框标注的封闭类别识别”走向“能快速适配新类别的开放环境检测”。
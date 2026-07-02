# LeNet 复现实验报告

## 1. 数据来源

本报告从 `data/experiments/` 下已保存的实验产物凝练而来。每个实验目录包含:

- `training_config.json`: 训练配置
- `results.json`: 测试准确率与训练损失轨迹
- `model.pt`: 模型权重
- `origin-<seed>.png` / `data_augmentation.png`: 样本可视化

主结果仅统计 `num_ep=301` 的正式实验。`num_ep=1` 的目录是流程验证运行,不纳入均值计算。

## 2. 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | `rmnist` |
| 模型 | `lenet` |
| 优化器 | Adam |
| 学习率 | `3e-3` |
| epoch | `301` |
| GAN 增强比例 | `0` |
| seeds | `31`, `317`, `31731` |

对比两组设置:

- **LeNet with Dropout**: `--no-data-augmentation`
- **LeNet with data_augmentation (1024)**: `--data-augmentation`

复现脚本为 [`../task.sh`](../task.sh)。

## 3. 正式实验结果

### 3.1 LeNet with Dropout

| seed | accuracy | final_running_loss | 实验目录 |
| --- | --- | --- | --- |
| 31 | 0.4246 | 0.029357 | `data/experiments/20260702-170321-rmnist-lenet-noaug-gan0p0-seed31/` |
| 317 | 0.4758 | 0.026139 | `data/experiments/20260702-170327-rmnist-lenet-noaug-gan0p0-seed317/` |
| 31731 | 0.4375 | 0.027715 | `data/experiments/20260702-170333-rmnist-lenet-noaug-gan0p0-seed31731/` |
| **平均** | **0.4460** | **0.027737** | - |

### 3.2 LeNet with data_augmentation (1024)

| seed | accuracy | final_running_loss | 实验目录 |
| --- | --- | --- | --- |
| 31 | 0.5670 | 0.027134 | `data/experiments/20260702-170339-rmnist-lenet-aug-gan0p0-seed31/` |
| 317 | 0.5721 | 0.024319 | `data/experiments/20260702-170357-rmnist-lenet-aug-gan0p0-seed317/` |
| 31731 | 0.5192 | 0.019537 | `data/experiments/20260702-170414-rmnist-lenet-aug-gan0p0-seed31731/` |
| **平均** | **0.5528** | **0.023663** | - |

## 4. 汇总对比

| 方法 | 平均 accuracy | 最低 | 最高 | 极差 |
| --- | --- | --- | --- | --- |
| LeNet with Dropout | 0.4460 | 0.4246 | 0.4758 | 0.0512 |
| LeNet with data_augmentation (1024) | 0.5528 | 0.5192 | 0.5721 | 0.0529 |

数据增强带来的平均准确率提升为:

| 对比 | 增益 |
| --- | --- |
| 0.5528 - 0.4460 | **+0.1068** |

## 5. 与仓库作者 README 结果对照

仓库作者 README 中给出了同样 3 个 seed 的 LeNet 结果。本次复现与 README 的总体趋势一致:无增强 LeNet 约在 0.43-0.45 区间,加入 1024 张几何增强样本后提升到 0.55 左右。

### 5.1 LeNet with Dropout

| seed | README | 本次复现 | 差值 |
| --- | --- | --- | --- |
| 31 | 0.4213 | 0.4246 | +0.0033 |
| 317 | 0.4315 | 0.4758 | +0.0443 |
| 31731 | 0.4354 | 0.4375 | +0.0021 |
| **平均** | **0.4294** | **0.4460** | **+0.0166** |

主要差异来自 `seed=317`:本次复现比 README 高 0.0443。其余两个 seed 与 README 非常接近,差异都小于 0.004。

### 5.2 LeNet with data_augmentation (1024)

| seed | README | 本次复现 | 差值 |
| --- | --- | --- | --- |
| 31 | 0.5490 | 0.5670 | +0.0180 |
| 317 | 0.5612 | 0.5721 | +0.0109 |
| 31731 | 0.5301 | 0.5192 | -0.0109 |
| **平均** | **0.5467** | **0.5528** | **+0.0061** |

增强组的平均值与 README 很接近,本次复现高 0.0061。逐 seed 看,`31` 和 `317` 略高于 README,`31731` 略低于 README。

### 5.3 数据增强收益对照

| 来源 | 无增强平均 | 增强平均 | 增强收益 |
| --- | --- | --- | --- |
| README | 0.4294 | 0.5467 | +0.1173 |
| 本次复现 | 0.4460 | 0.5528 | +0.1068 |
| 差值 | +0.0166 | +0.0061 | -0.0105 |

本次复现的增强收益比 README 低 0.0105,原因不是增强组明显退化,而是无增强 baseline 在本次复现中更高。总体结论不变:几何数据增强仍然带来约 10-12 个百分点的稳定提升。

### 5.4 差异来源判断

这些差异可能来自当前依赖版本、训练实现细节和随机性控制方式的变化。尤其是本仓库当前实现已显式区分 `--no-data-augmentation` 与 `--data-augmentation`,并修正了 10-shot baseline 每个 epoch 至少训练 1 step 的问题,因此本次结果更贴近当前代码路径的可复现实验产物。

## 6. 观察与结论

1. **无增强 LeNet 仍能达到约 44.6% 平均准确率**,略高于 README 中的 42.94%。说明在 10-shot 条件下,LeNet + Dropout 有一定泛化能力,但仍明显受限于样本数量。
2. **几何数据增强带来约 10.7 个百分点提升**,是本组实验中最主要的收益来源。
3. **不同 seed 的波动仍然明显**:两组实验的极差都约为 5 个百分点,说明 one-shot 任务对抽到哪 10 张样本较敏感。
4. **当前实现可追踪性更好**:每个实验都有独立目录、配置、权重和结果文件,后续可以直接基于这些产物追加统计显著性分析或绘制曲线。

## 7. 附:流程验证运行

以下结果来自 `NUM_EP=1 ./task.sh`,仅用于确认脚本、目录创建、权重保存和结果保存流程能跑通,不用于实验结论。

| 方法 | seed | accuracy |
| --- | --- | --- |
| LeNet with Dropout | 31 | 0.1022 |
| LeNet with Dropout | 317 | 0.1235 |
| LeNet with Dropout | 31731 | 0.1212 |
| LeNet with data_augmentation (1024) | 31 | 0.2161 |
| LeNet with data_augmentation (1024) | 317 | 0.2707 |
| LeNet with data_augmentation (1024) | 31731 | 0.2319 |

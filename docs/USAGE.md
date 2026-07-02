# 使用说明

项目命令统一通过 `uv run oneshot` 执行。

## 查看可用命令

```bash
uv run oneshot --help
```

## 下载数据集

下载 MNIST 数据集到默认 `DATASET_DIR/mnist`。未设置 `DATA_DIR` 时，实际路径为 `data/dataset/mnist`:

```bash
uv run oneshot download mnist
```

也可以指定下载根目录，文件仍会写入该目录下的 `mnist/` 子目录:

```bash
uv run oneshot download mnist --data-dir data
```

## 训练模型

训练 one-shot MNIST 模型:

```bash
uv run oneshot train --seed 31
```

常用训练参数:

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dataset` | `rmnist` | 训练数据集，目前支持 `rmnist` |
| `--model` | `resnet` | 深度学习模型，可选 `resnet` 或 `lenet` |
| `--seed` | `31` | 控制 one-shot 样本抽取、增强和训练随机性 |
| `--lr` | `3e-3` | 深度学习训练学习率 |
| `--num_ep` | `301` | 训练 epoch 数 |
| `--gan_ratio` | `0` | cGAN 额外生成样本比例 |

例如使用 LeNet 训练:

```bash
uv run oneshot train --model lenet --seed 31
```

每次训练都会在 `data/experiments/` 下创建独立实验目录，例如 `data/experiments/<timestamp>-rmnist-lenet-seed31/`。该目录会保存本次实验的 `training_config.json`、模型权重 `model.pt`、评估结果 `results.json`，以及训练过程生成的样本可视化图片。

## 可视化训练样本

可视化当前 seed 对应的 10 张 one-shot 训练样本:

```bash
uv run oneshot visualize --seed 31 --output data/outputs/training-samples.png
```

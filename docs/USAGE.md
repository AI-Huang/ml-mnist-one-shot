# 使用说明

项目命令统一通过 `uv run oneshot` 执行。

## 查看可用命令

```bash
uv run oneshot --help
```

## 下载数据集

下载 MNIST 数据集到默认 `DATA_DIR/mnist`:

```bash
uv run oneshot download mnist
```

也可以指定下载根目录,文件仍会写入该目录下的 `mnist/` 子目录:

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
| `--seed` | `31` | 控制 one-shot 样本抽取、增强和训练随机性 |
| `--lr` | `3e-3` | 深度学习训练学习率 |
| `--num_ep` | `301` | 训练 epoch 数 |
| `--gan_ratio` | `0` | cGAN 额外生成样本比例 |

## 可视化训练样本

可视化当前 seed 对应的 10 张 one-shot 训练样本:

```bash
uv run oneshot visualize --seed 31 --output data/outputs/training-samples.png
```

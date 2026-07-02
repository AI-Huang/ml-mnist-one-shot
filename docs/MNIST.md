# MNIST 数据集加载说明

本文档说明项目中基础 MNIST 数据集的下载、缓存、解析和输出接口。RMNIST 的 one-shot 抽样逻辑见 [RMNIST.md](RMNIST.md),本文只关注 `src/datasets/mnist.py` 提供的通用 MNIST 加载能力。

## 数据来源

当前实现使用 CVDF 公共镜像提供的 MNIST IDX 文件。文件内容与 Yann LeCun 官方 MNIST 文件一致,并继续通过 MD5 校验确认完整性:

```text
https://storage.googleapis.com/cvdf-datasets/mnist/
```

下载文件如下:

| 逻辑名称 | 文件名 | 内容 |
| --- | --- | --- |
| `train_images` | `train-images-idx3-ubyte.gz` | 官方训练集图像,60,000 张 |
| `train_labels` | `train-labels-idx1-ubyte.gz` | 官方训练集标签,60,000 个 |
| `test_images` | `t10k-images-idx3-ubyte.gz` | 官方测试集图像,10,000 张 |
| `test_labels` | `t10k-labels-idx1-ubyte.gz` | 官方测试集标签,10,000 个 |

每个文件都会通过 `download_url()` 做 MD5 校验。本地已有文件且校验通过时会跳过重复下载。

## 本地缓存

MNIST 原始 IDX gzip 文件默认缓存到 `settings.DATA_DIR / "mnist"`。默认情况下:

```text
DATA_DIR=./data
MNIST_DIR=./data/mnist
```

仓库中的 `data` 是运行数据入口,应按数据分离原则指向外部运行数据目录,并且不会进入 Git 提交。

可以通过环境变量覆盖数据目录:

```bash
DATA_DIR=/path/to/runtime-data uv run oneshot download mnist
```

也可以在 CLI 中临时指定下载目录:

```bash
uv run oneshot download mnist --data-dir data
```

上述命令都会把四个 MNIST gzip 文件放在指定数据目录下的 `mnist/` 子目录中,例如 `data/mnist/train-images-idx3-ubyte.gz`。

## 下载命令

推荐先显式下载 MNIST:

```bash
uv run oneshot download mnist
```

训练和可视化流程也会通过 `mnist(DATA_DIR)` 触发下载检查。如果文件已存在并通过校验,后续运行只会读取本地缓存。

## 解析流程

`src/datasets/mnist.py` 解析 gzip 压缩的 IDX 文件:

| 函数 | 职责 |
| --- | --- |
| `_download_mnist_files(data_dir)` | 下载四个 IDX gzip 文件并返回本地路径 |
| `_read_images(path)` | 校验图像 magic number `2051`,读取图像并展平 |
| `_read_labels(path)` | 校验标签 magic number `2049`,读取标签 |
| `mnist(data_dir, one_hot=False)` | 组合下载、解析和 train/valid/test 切分 |

图像会被转换为 `float32`,展平成 `784` 维向量,并归一化到 `[0, 1]`。标签默认是 `int64` 类别编号。

## 输出接口

`mnist(data_dir, one_hot=False)` 返回三个二元组:

```python
train_set, valid_set, test_set = mnist(DATA_DIR)
```

输出形状如下:

| 子集 | 图像形状 | 标签形状 | 来源 |
| --- | --- | --- | --- |
| `train_set` | `(50000, 784)` | `(50000,)` | 官方训练集前 50,000 张 |
| `valid_set` | `(10000, 784)` | `(10000,)` | 官方训练集后 10,000 张 |
| `test_set` | `(10000, 784)` | `(10000,)` | 官方测试集 |

当 `one_hot=True` 时,标签会通过 `get_one_hot(labels, 10)` 转为 one-hot 编码。

## 与 RMNIST 的关系

`datasets.mnist.mnist()` 只负责标准 MNIST 的下载、解析和切分。

`datasets.rmnist.make_dataset()` 在此基础上从 `train_set` 中每个类别随机抽 1 张样本,构造 one-shot 训练集,并保留完整 `test_set` 用于评估。

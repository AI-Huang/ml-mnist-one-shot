import gzip
import struct
import sys
from pathlib import Path

import numpy as np

from utils import download_url, get_one_hot

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": (
        "train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    "train_labels": (
        "train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    "test_images": (
        "t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    "test_labels": (
        "t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
}


def _download_mnist_files(data_dir):
    data_dir = Path(data_dir)
    paths = {}
    for key, (filename, checksum) in MNIST_FILES.items():
        url = f"{MNIST_BASE_URL}/{filename}"
        save_path = data_dir / filename
        download_url(url, str(save_path), checksum)
        paths[key] = save_path
    return paths


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, columns = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid MNIST image file: %s" % path)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows * columns).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid MNIST label file: %s" % path)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count).astype(np.int64)


def mnist(data_dir, one_hot=False):
    """download mnist file and split the dataset into train/val/test set"""
    print("Preparing MNIST dataset ...")
    try:
        paths = _download_mnist_files(data_dir)
    except Exception as e:
        print("Error downloading dataset: %s" % str(e))
        sys.exit(1)

    train_x = _read_images(paths["train_images"])
    train_y = _read_labels(paths["train_labels"])
    test_x = _read_images(paths["test_images"])
    test_y = _read_labels(paths["test_labels"])

    train_set = (train_x[:50000], train_y[:50000])
    valid_set = (train_x[50000:], train_y[50000:])
    test_set = (test_x, test_y)

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set

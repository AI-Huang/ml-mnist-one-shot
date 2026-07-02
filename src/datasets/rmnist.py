import numpy as np

from datasets.mnist import mnist
from settings import DATASET_DIR


def make_dataset():
    """get train_x/train_y/test_x/test_y from the dataset"""
    train_set, _, test_set = mnist(DATASET_DIR)
    test_x, test_y = test_set
    full_train_x, full_train_y = train_set

    train_idx = []
    for label in range(10):
        idx_list = np.where(full_train_y == label)[0]
        train_idx.append(np.random.choice(idx_list))
    train_x, train_y = full_train_x[train_idx], full_train_y[train_idx]
    print("train sample idx: ", train_idx)

    return train_x, train_y, test_x, test_y

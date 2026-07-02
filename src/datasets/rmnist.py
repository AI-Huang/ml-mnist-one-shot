from datasets.mnist import mnist
from pipelines.random_choose_one import random_choose_one
from settings import DATASET_DIR


def make_dataset():
    """get train_x/train_y/test_x/test_y from the dataset"""
    train_set, _, test_set = mnist(DATASET_DIR)
    test_x, test_y = test_set
    full_train_x, full_train_y = train_set

    train_x, train_y, train_idx = random_choose_one(full_train_x, full_train_y)
    print("train sample idx: ", train_idx)

    return train_x, train_y, test_x, test_y

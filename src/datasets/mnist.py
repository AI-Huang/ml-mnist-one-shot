import gzip
import pickle
import sys
from pathlib import Path

from utils import download_url, get_one_hot


def mnist(data_dir, one_hot=False):
    """download mnist file and split the dataset into train/val/test set"""
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    checksum = "a02cd19f81d51c426d7ca14024243ce9"

    save_path = Path(data_dir) / url.split("/")[-1]
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, str(save_path), checksum)
    except Exception as e:
        print("Error downloading dataset: %s" % str(e))
        sys.exit(1)

    with gzip.open(save_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set

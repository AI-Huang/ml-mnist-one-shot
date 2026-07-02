import numpy as np


def random_choose_one(samples, labels, num_classes=10, shuffle=True):
    """Choose one random sample for each class from a labeled dataset."""
    chosen_idx = []

    for label in range(num_classes):
        label_idx = np.where(labels == label)[0]
        if len(label_idx) == 0:
            raise ValueError(f"No samples found for label {label}")
        chosen_idx.append(np.random.choice(label_idx))

    chosen_idx = np.array(chosen_idx)
    if shuffle:
        chosen_idx = chosen_idx[np.random.permutation(len(chosen_idx))]

    return samples[chosen_idx], labels[chosen_idx], chosen_idx

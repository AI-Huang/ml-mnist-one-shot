import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets.rmnist import make_dataset
from settings import OUTPUT_DIR


def plot_samples(
    samples, path, labels=None, columns=10, image_shape=(28, 28), cmap="gray"
):
    """Save image samples as a compact grid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(samples)
    if samples.ndim == 4 and samples.shape[-1] == 1:
        samples = samples[..., 0]
    samples = samples.reshape((len(samples), *image_shape))

    columns = min(columns, len(samples))
    rows = int(np.ceil(len(samples) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows))
    axes = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(samples):
            continue
        axis.imshow(samples[index], cmap=cmap)
        if labels is not None:
            axis.set_title(str(labels[index]), fontsize=8)

    fig.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def visualize_training_samples(output_path=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    train_x, train_y, _, _ = make_dataset()
    output_path = output_path or OUTPUT_DIR / "training-samples.png"
    return plot_samples(train_x, output_path, labels=train_y)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize one-shot MNIST training samples"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "training-samples.png"
    )
    parser.add_argument("--seed", type=int, default=31)
    return parser


def main(parsed_args=None):
    if parsed_args is None:
        parsed_args = build_parser().parse_args()

    output_path = visualize_training_samples(parsed_args.output, parsed_args.seed)
    print(f"Saved training sample visualization to {output_path}")


if __name__ == "__main__":
    main()

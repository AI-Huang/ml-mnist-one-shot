import argparse
from pathlib import Path


def build_parser():
    from train import build_parser as build_train_parser
    from visualization import build_parser as build_visualize_parser

    parser = argparse.ArgumentParser(prog="oneshot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        parents=[build_train_parser()],
        add_help=False,
        help="Train the MNIST one-shot model",
    )
    train_parser.set_defaults(handler=run_train)

    visualize_parser = subparsers.add_parser(
        "visualize",
        parents=[build_visualize_parser()],
        add_help=False,
        help="Visualize one-shot MNIST training samples",
    )
    visualize_parser.set_defaults(handler=run_visualize)

    download_parser = subparsers.add_parser(
        "download",
        help="Download datasets",
    )
    download_subparsers = download_parser.add_subparsers(dest="dataset", required=True)

    mnist_parser = download_subparsers.add_parser(
        "mnist",
        help="Download the MNIST dataset",
    )
    mnist_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory for downloaded datasets; defaults to settings.DATASET_DIR",
    )
    mnist_parser.set_defaults(handler=run_download_mnist)

    return parser


def run_train(args):
    from train import main as train_main

    train_main(args)


def run_visualize(args):
    from visualization import main as visualize_main

    visualize_main(args)


def run_download_mnist(args):
    from datasets.mnist import mnist
    from settings import DATASET_DIR

    data_dir = args.data_dir or DATASET_DIR
    mnist(data_dir)
    print(f"MNIST dataset ready in {data_dir / 'mnist'}")


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)

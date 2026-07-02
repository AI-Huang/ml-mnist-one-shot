import argparse


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

    return parser


def run_train(args):
    from train import main as train_main

    train_main(args)


def run_visualize(args):
    from visualization import main as visualize_main

    visualize_main(args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)

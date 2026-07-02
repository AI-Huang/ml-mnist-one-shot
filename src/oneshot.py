import argparse


def build_parser():
    from train import build_parser as build_train_parser

    parser = argparse.ArgumentParser(prog="oneshot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        parents=[build_train_parser()],
        add_help=False,
        help="Train the MNIST one-shot model",
    )
    train_parser.set_defaults(handler=run_train)

    return parser


def run_train(args):
    from train import main as train_main

    train_main(args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)

import os

from docopt import docopt

import mayo
from mayo.config import Config
from mayo.train import Train


usage = """
{__mayo__} {__version__} {__date__}
{__description__}
{__author__}

Usage:
    {__executable__} train <net-yaml> <dataset-yaml> [--train=<train-yaml>]
    {__executable__} (-h | --help)
""".format(**vars(mayo))


_mayo_root = os.path.dirname(mayo.__file__)
DEFAULT_TRAIN_YAML = os.path.join(_mayo_root, 'train.yaml')


def train(args):
    train_yaml = args['--train'] or DEFAULT_TRAIN_YAML
    config = Config(
        net=args['<net-yaml>'], dataset=args['<dataset-yaml>'],
        train=train_yaml)
    train = Train(config)
    return train.train()


def main():
    args = docopt(usage, version=mayo.__version__)
    commands = {
        'train': train,
    }
    for name, func in commands.items():
        if not args[name]:
            continue
        return func(args)

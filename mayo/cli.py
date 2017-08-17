import os
import sys

import yaml
from docopt import docopt

__root__ = os.path.dirname(__file__)

_DOC = """
{__mayo__} {__version__} ({__date__})
{__description__}
{__author__}
"""
_USAGE = """
Usage:
    {__executable__} train <net-yaml> <dataset-yaml> [--train=<train-yaml>]
    {__executable__} (-h | --help)
"""


def meta():
    meta_file = os.path.join(__root__, 'meta.yaml')
    meta_dict = yaml.load(open(meta_file, 'r'))
    meta_dict['__executable__'] = os.path.basename(sys.argv[0])
    return meta_dict


def doc():
    return _DOC.format(**meta())


def usage():
    return doc() + _USAGE.format(**meta())


def train(args):
    from mayo.config import Config
    from mayo.train import Train

    default_train_yaml = os.path.join(__root__, 'train.yaml')
    train_yaml = args['--train'] or default_train_yaml
    config = Config(
        net=args['<net-yaml>'], dataset=args['<dataset-yaml>'],
        train=train_yaml)
    return Train(config).train()


def validate(args):
    from mayo.config import Config
    from mayo.evaluate import Evaluate

    config = Config(net=args['<net-yaml>'], dataset=args['<dataset-yaml>'])
    return Evaluate(config).evaluate()


def main():
    args = docopt(usage(), version=meta()['__version__'])
    commands = {
        'train': train,
        'validate': validate,
    }
    for name, func in commands.items():
        if not args.get(name, None):
            continue
        return func(args)

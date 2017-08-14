from docopt import docopt

import mayo
from mayo.config import Config
from mayo.train import Train


usage = """
{__mayo__} {__version__} {__date__}
{__description__}
{__author__}

Usage:
    {__executable__} train <yaml>
    {__executable__} (-h | --help)
""".format(**vars(mayo))


def train(args):
    yaml_file = args['<yaml>']
    config = Config(path=yaml_file)
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

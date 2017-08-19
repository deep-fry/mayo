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
    {__executable__} train <yaml>... [options]
    {__executable__} export <yaml>... [options]
    {__executable__} (-h | --help)

Options:
    --overrides=<overrides>     Specify hyper-parameters to override.
                                Example: --overrides="a.b = c; d = e"
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


def _config(args):
    from importlib.util import spec_from_file_location, module_from_spec
    path = os.path.join(__root__, 'config.py')
    spec = spec_from_file_location('mayo.config', path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Config(args['<yaml>'], overrides=args['--overrides'])


def train(args):
    from mayo.train import Train
    return Train(_config(args)).train()


def validate(args):
    from mayo.evaluate import Evaluate
    return Evaluate(_config(args)).evaluate()


def export(args):
    print(_config(args).to_yaml())


def main():
    args = docopt(usage(), version=meta()['__version__'])
    commands = [train, validate, export]
    for func in commands:
        if not args.get(func.__name__, None):
            continue
        return func(args)
    raise NotImplementedError('Command not found')

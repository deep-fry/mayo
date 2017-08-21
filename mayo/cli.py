import os
import sys
import base64

import yaml
from docopt import docopt

_root = os.path.dirname(__file__)

_DOC = """
{__mayo__} {__version__} ({__date__})
{__description__}
{__credits__}
"""
_USAGE = """
Usage:
    {__executable__} train <yaml>... [options]
    {__executable__} export <yaml>... [options]
    {__executable__} (-h | --help)

Options:
    --overrides=<overrides>     Specify hyper-parameters to override.
                                Example: --overrides="a.b = c; d = [e, f]"
"""


def _vigenere(key, string, decode=False):
    if decode:
        string = base64.b64decode(string.encode('utf-8')).decode('utf-8')
    encoded_chars = []
    for i in range(len(string)):
        key_c = ord(key[i % len(key)]) % 256
        encoded_c = ord(string[i])
        if decode:
            encoded_c -= key_c
        else:
            encoded_c += key_c
        encoded_chars.append(chr(encoded_c))
    encoded_str = "".join(encoded_chars)
    if decode:
        return encoded_str
    return base64.b64encode(encoded_str.encode('utf-8')).decode('utf-8')


def meta():
    meta_file = os.path.join(_root, 'meta.yaml')
    meta_dict = yaml.load(open(meta_file, 'r'))
    meta_dict['__root__'] = _root
    meta_dict['__executable__'] = os.path.basename(sys.argv[0])
    email = '__email__'
    encrypted_email = meta_dict[email].replace('\n', '').replace(' ', '')
    meta_dict[email] = _vigenere(email, encrypted_email, decode=True)
    authors_emails = zip(
        meta_dict['__author__'].split(', '), meta_dict[email].split(', '))
    credits = ', '.join('{} ({})'.format(a, e) for a, e in authors_emails)
    meta_dict['__credits__'] = credits
    return meta_dict


def doc():
    return _DOC.format(**meta())


def usage():
    return doc() + _USAGE.format(**meta())


def _config(args):
    from importlib.util import spec_from_file_location, module_from_spec
    path = os.path.join(_root, 'config.py')
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

import os
import sys
import base64

import yaml
import tensorflow as tf
from docopt import docopt

from mayo.log import log
from mayo.config import Config
from mayo.eval import Evaluate
from mayo.net import Net
from mayo.train import Train


_root = os.path.dirname(__file__)


def _vigenere(key, string, decode=False):
    if decode:
        string = base64.b64decode(string.encode('utf-8')).decode('utf-8')
    encoded_chars = []
    for i in range(len(string)):
        key_c = ord(key[i % len(key)]) % 256
        encoded_c = ord(string[i])
        encoded_c += -key_c if decode else key_c
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


class CLI(object):
    _DOC = """
{__mayo__} {__version__} ({__date__})
{__description__}
{__credits__}
"""
    _USAGE = """
Usage:
    {__executable__} <anything>...
    {__executable__} (-h | --help)

Arguments:
  <anything> can be one of the following:
     * action, one of:
         {commands};
     * a YAML file; or
     * an overrider argument, formatted as <dot_key_path>=<yaml_value>, e.g.:
         "system.num_gpus=2".
"""

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.session = None

    def doc(self):
        return self._DOC.format(**meta())

    def commands(self):
        prefix = 'cli_'
        commands = {}
        for method in dir(self):
            if not method.startswith(prefix):
                continue
            name = method[len(prefix):].replace('_', '-')
            commands[name] = getattr(self, method)
        return commands

    def usage(self):
        usage_meta = meta()
        commands = (c for c in self.commands() if 'checkpoint' not in c)
        usage_meta['commands'] = ', '.join(commands)
        return self.doc() + self._USAGE.format(**usage_meta)

    def cli_train(self):
        return Train(self.config).train()

    def cli_eval(self):
        return Evaluate(self.config).eval()

    def cli_eval_all(self):
        print(Evaluate(self.config).eval_all())

    def cli_export(self):
        print(self.config.to_yaml())

    def cli_info(self):
        config = self.config
        batch_size = config.system.get('batch_size', None)
        images_shape = (batch_size, ) + config.image_shape()
        labels_shape = (batch_size, config.dataset.num_classes)
        images = tf.placeholder(tf.float32, images_shape, 'images')
        labels = tf.placeholder(tf.int32, labels_shape, 'labels')
        print(Net(config, images, labels, False).info())

    def main(self, args=None):
        if args is None:
            args = docopt(self.usage(), version=meta()['__version__'])
        anything = args['<anything>']
        commands = self.commands()
        for each in anything:
            if any(each.endswith(suffix) for suffix in ('.yaml', '.yml')):
                self.config.yaml_update(each)
                log.key('Using config yaml {!r}...'.format(each))
            elif '=' in each:
                self.config.override_update(*each.split('='))
                log.key('Overriding config with {!r}...'.format(each))
            elif each in commands:
                log.key('Executing command {!r}...'.format(each))
                commands[each]()
            else:
                with log.use_pause_level('off'):
                    log.error(
                        'We don\'t know what you mean by {!r}'.format(each))
                return

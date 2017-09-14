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
  <anything> can be one of the following given in sequence:
     * A YAML file with a `.yaml` or `.yml` suffix.  If a YAML file is given,
       it will attempt to load the YAML file to update the config.
     * An overrider argument to update the config, formatted as
       "<dot_key_path>=<yaml_value>", e.g., "system.num_gpus=2".
     * An action to execute, one of:
         {commands}.
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

    def _error_exit(self, error_msg):
        with log.use_pause_level('off'):
            log.error(error_msg)
        sys.exit(-1)

    def _validate_config(self, keys, action):
        for k in keys:
            if k in self.config:
                continue
            self._error_exit(
                'Please ensure config content {!r} is imported before '
                'executing {!r}.'.format(k, action))

    _model_keys = [
        'model.name',
        'model.net',
        'model.logits',
        'dataset.num_classes',
        'dataset.preprocess.shape',
        'dataset.background_class.use',
    ]
    _dataset_keys = [
        'dataset.name',
        'dataset.background_class.has',
    ]
    _validate_keys = [
        'dataset.preprocess.validate',
        'dataset.preprocess.final',
        'dataset.path.validate',
        'dataset.num_examples_per_epoch.validate',
    ]
    _train_keys = [
        'dataset.preprocess.train',
        'dataset.preprocess.final',
        'dataset.path.train',
        'dataset.num_examples_per_epoch.train',
        'train.learning_rate',
        'train.optimizer',
    ]

    def _get_session(self, action):
        keys = self._model_keys + self._dataset_keys
        if action == 'train':
            cls = Train
            keys += self._train_keys
        elif action == 'validate':
            cls = Evaluate
            keys += self._validate_keys
        else:
            raise TypeError('Action {!r} not recognized.'.format(action))
        self._validate_config(keys, action)
        if not isinstance(self.session, cls):
            self.session = cls(self.config)
        return self.session

    def cli_train(self):
        return self._get_session('train').train()

    def cli_eval(self):
        return self._get_session('validate').eval()

    def cli_eval_all(self):
        print(self._get_session('validate').eval_all())

    def cli_export(self):
        print(self.config.to_yaml())

    def cli_info(self):
        keys = self._model_keys
        self._validate_config(keys, 'info')
        config = self.config
        batch_size = config.get('system.batch_size', None)
        images_shape = (batch_size, ) + config.image_shape()
        labels_shape = (batch_size, config.dataset.num_classes)
        with tf.Graph().as_default():
            images = tf.placeholder(tf.float32, images_shape, 'images')
            labels = tf.placeholder(tf.int32, labels_shape, 'labels')
            print(Net(config, images, labels, False).info())

    def cli_override(self):
        self._get_session('train').update_overriders()

    def cli_save(self):
        if not self.session:
            raise ValueError(
                'Session not initialized, please train or eval first.')
        self.session.checkpoint.save('latest')

    def _invalidate_session(self):
        if not self.session:
            return
        log.debug('Invalidating session because config is updated.')
        self.session = None

    def main(self, args=None):
        if args is None:
            args = docopt(self.usage(), version=meta()['__version__'])
        anything = args['<anything>']
        commands = self.commands()
        for each in anything:
            if any(each.endswith(suffix) for suffix in ('.yaml', '.yml')):
                self.config.yaml_update(each)
                log.key('Using config yaml {!r}...'.format(each))
                self._invalidate_session()
            elif '=' in each:
                self.config.override_update(*each.split('='))
                log.key('Overriding config with {!r}...'.format(each))
                self._invalidate_session()
            elif each in commands:
                log.key('Executing command {!r}...'.format(each))
                commands[each]()
            else:
                with log.use_pause_level('off'):
                    log.error(
                        'We don\'t know what you mean by {!r}'.format(each))
                return

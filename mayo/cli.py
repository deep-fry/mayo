import os
import sys
import base64

import yaml
from docopt import docopt

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
    {commands}
    {__executable__} checkpoint-info <ckpt>
    {__executable__} checkpoint-rename <ckpt> <to_ckpt> [<match_ckpt>] \
 --rules=<yaml> [--dry-run]
    {__executable__} (-h | --help)

Arguments:
    <anything> can either be a YAML file or an override command
    formatted as <dot_key_path>=<yaml_value>

Options:
    --dry-run           Performs a dry run, not actually changing anything
                        but shows things to be changed.
    --rules=<yaml>      Replaces keys with new keys given in the specified YAML
                        file using `re.sub`.  The YAML file should be written
                        as ordered mappings with `<pattern>: <replacement>`,
                        which we will apply the substitution in the order of
                        mapping.
"""

    def _commands(self):
        commands = {}
        for method in dir(self):
            if not method.startswith('cli_'):
                continue
            commands[method[4:]] = getattr(self, method)
        return commands

    def doc(self):
        return self._DOC.format(**meta())

    def usage(self):
        usage_meta = meta()
        commands = []
        for k in self._commands().keys():
            command = '{__executable__} {command} [<anything>...]'
            commands.append(command.format(command=k, **usage_meta))
        usage_meta['commands'] = '\n    '.join(commands)
        return self.doc() + self._USAGE.format(**usage_meta)

    def _config(self, args):
        from importlib.util import spec_from_file_location, module_from_spec
        path = os.path.join(_root, 'config.py')
        spec = spec_from_file_location('mayo.config', path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        anything = args['<anything>']
        yamls, overrides = [], []
        for each in anything:
            (overrides if '=' in each else yamls).append(each)
        return mod.Config(yamls, overrides=overrides)

    def cli_train(self, args):
        from mayo.train import Train
        return Train(self._config(args)).train()

    def cli_eval(self, args):
        from mayo.eval import Evaluate
        return Evaluate(self._config(args)).eval()

    def cli_export(self, args):
        print(self._config(args).to_yaml())

    def cli_info(self, args):
        import tensorflow as tf
        from mayo.net import Net
        config = self._config(args)
        batch_size = config.system.get('batch_size', None)
        images_shape = (batch_size, ) + config.image_shape()
        labels_shape = (batch_size, config.dataset.num_classes)
        images = tf.placeholder(tf.float32, images_shape, 'images')
        labels = tf.placeholder(tf.int32, labels_shape, 'labels')
        print(Net(config, images, labels, False).info())

    def _disable_tensorflow_logger(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def cli_checkpoint_rename(self, args):
        from mayo.checkpoint import CheckpointSurgeon
        self._disable_tensorflow_logger()
        from_ckpt = args['<ckpt>']
        to_ckpt = args['<to_ckpt>']
        match_ckpt = args['<match_ckpt>']
        dry = ['--dry-run']
        rules = args['--rules']
        with open(rules, 'r') as f:
            rules = yaml.load(f)
        surgeon = CheckpointSurgeon(from_ckpt)
        return surgeon.rename(to_ckpt, match_ckpt, rules, dry)

    def cli_checkpoint_info(self, args):
        from mayo.checkpoint import CheckpointSurgeon
        self._disable_tensorflow_logger()
        surgeon = CheckpointSurgeon(args['<ckpt>'])
        print(yaml.dump(surgeon.var_to_shape_map()))

    def main(self, args=None):
        prefix = 'cli_'
        if args is None:
            args = docopt(self.usage(), version=meta()['__version__'])
        for name in dir(self):
            if not name.startswith(prefix):
                continue
            command_name = name[len(prefix):].replace('_', '-')
            if not args[command_name]:
                continue
            func = getattr(self, name)
            return func(args)
        raise NotImplementedError('Command not found')

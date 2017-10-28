import os
import sys
import base64
import time

import yaml
import tensorflow as tf
import numpy as np

from docopt import docopt

from mayo.log import log
from mayo.config import Config
from mayo.net import Net
from mayo.session import (
    Evaluate, FastEvaluate, Train, LayerwiseRetrain, GlobalRetrain)

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


class SessionNotInitializedError(Exception):
    pass


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
{commands}
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
        commands = self.commands()
        name_len = max(len(name) for name in commands)
        descriptions = []
        for name, func in commands.items():
            doc = func.__doc__ or ''
            doc = '{}{:{l}} {}'.format(' ' * 9, name, doc.strip(), l=name_len)
            descriptions.append(doc)
        usage_meta['commands'] = '\n'.join(descriptions)
        return self.doc() + self._USAGE.format(**usage_meta)

    def _validate_config(self, keys, action):
        for k in keys:
            if k in self.config:
                continue
            log.error_exit(
                'Please ensure config content {!r} is imported before '
                'executing {!r}.'.format(k, action))

    _model_keys = [
        'model.name',
        'model.layers',
        'model.graph',
        'dataset.num_classes',
        'dataset.preprocess.shape',
        'dataset.background_class.use',
    ]
    _dataset_keys = [
        'dataset.name',
        'dataset.background_class.has',
    ]
    _validate_keys = [
        'dataset.path.validate',
        'dataset.num_examples_per_epoch.validate',
    ]
    _train_keys = [
        'dataset.path.train',
        'dataset.num_examples_per_epoch.train',
        'train.learning_rate',
        'train.optimizer',
    ]

    def _get_session(self, action=None):
        if not action:
            if not self.session:
                raise SessionNotInitializedError(
                    'Session not initialized, please train or eval first.')
            return self.session
        keys = self._model_keys + self._dataset_keys
        if action == 'train':
            cls = Train
            keys += self._train_keys
        elif action == 'retrain-layer':
            cls = LayerwiseRetrain
            keys += self._train_keys
        elif action == 'retrain-global':
            cls = GlobalRetrain
            keys += self._train_keys
        elif action == 'validate':
            cls = Evaluate
            keys += self._validate_keys
        elif action == 'fast-validate':
            cls = FastEvaluate
            keys += self._validate_keys
        else:
            raise TypeError('Action {!r} not recognized.'.format(action))
        self._validate_config(keys, action)
        if not isinstance(self.session, cls):
            log.info('Starting a {} session...'.format(action))
            self.session = cls(self.config)
        return self.session

    def cli_test(self):
        # TEMPORARY check speed with cpu
        PROFILE = True
        # define a dummy graph
        session = self._get_session('train')
        imgs = []
        with session.as_default():
            for i, (images, labels) in enumerate(session._preprocess()):
                imgs.append(images)
            add_op = tf.assign_add(session.imgs_seen, session.batch_size)
        epoch_cnt = 0
        start = time.time()
        from tensorflow.python.client import timeline
        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        while (epoch_cnt < 0.2):
            tasks = [imgs, add_op, session.num_epochs]
            img_o, total_imgs, epoch_cnt = session.run(tasks)
            # fetch timeline after warmed up
            if (epoch_cnt > 0.1) and PROFILE:
                session.run(tasks,
                            options=options,
                            run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline.json', 'w') as f:
                    f.write(chrome_trace)
                break
            interval = total_imgs / float(time.time() - start)
            info = 'avg: {:.2f}p/s'.format(interval)
            log.info(info, update=True)

    def cli_profile(self):
        """Performs training profiling to produce timeline.json.  """
        from tensorflow.python.client import timeline
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        session = self._get_session('train')
        # run 100 iterations to warm up
        max_iterations = 100
        for i in range(max_iterations):
            log.info(
                'Running {}/{} iterations to warm up...'
                .format(i, max_iterations), update=True)
            session.run(session._train_op)
        log.info('Running the final iteration to generate timeline...')
        session.run(
            session._train_op, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(chrome_trace)

    def cli_visualize(self):
        # collect all info
        session = self._get_session('train')
        overriders = session.nets[0].overriders
        meta = {}
        picked = []
        for o in overriders:
            meta[o.name] = session.run(o.after)
            if 'conv1' in o.name or 'fc1' in o.name:
                picked.append(meta[o.name])
        import pandas as pd
        import ggplot
        for name, data in meta.items():
            df = pd.DataFrame({name: meta[name].flatten()})
            df = pd.melt(df)
            p = ggplot.ggplot(ggplot.aes(x='value', color='variable'), data=df)
            p += ggplot.geom_histogram(bins=1024)
            directory = './plots/'
            name = name.replace('/', '_')
            p.save(directory + "{}.png".format(name))


    def cli_train(self):
        """Performs training.  """
        return self._get_session('train').train()

    def cli_retrain_layer(self):
        """Performs retraining.  """
        return self._get_session('retrain-layer').retrain()

    def cli_retrain_global(self):
        """Performs retraining.  """
        return self._get_session('retrain-global').retrain()

    def cli_fast_eval(self):
        """
        Evaluates the approximate accuracy of a saved model with
        multiple threads.
        """
        return self._get_session('fast-validate').eval()

    def cli_eval(self):
        """Evaluates the accuracy of a saved model.  """
        return self._get_session('validate').eval()

    def cli_eval_all(self):
        """Evaluates all checkpoints for accuracy.  """
        print(self._get_session('validate').eval_all())

    def cli_export(self):
        """Exports the current config.  """
        print(self.config.to_yaml())

    def cli_info(self):
        """Prints parameter and layer info of the model.  """
        keys = self._model_keys
        self._validate_config(keys, 'info')
        config = self.config
        batch_size_per_gpu = config.get('system.batch_size_per_gpu', None)
        images_shape = (batch_size_per_gpu, ) + config.image_shape()
        labels_shape = (batch_size_per_gpu, config.num_classes())
        with tf.Graph().as_default():
            images = tf.placeholder(tf.float32, images_shape, 'images')
            labels = tf.placeholder(tf.int32, labels_shape, 'labels')
            info = Net(config, images, labels, False).info()
        print(info['variables'].format())
        print(info['layers'].format())
        if not isinstance(self.session, Train):
            return
        for overrider_cls, table in self.session.overrider_info().items():
            overrider_cls.finalize_info(table)
            print(table.format())

    def cli_reset_num_epochs(self):
        """Resets the number of training epochs.  """
        self._get_session('train').reset_num_epochs()

    def cli_overriders_update(self):
        """Updates variable overriders in the training session.  """
        self._get_session('train').overriders_update()

    def cli_overriders_assign(self):
        """Assign overridden values to original parameters.  """
        self._get_session('train').overriders_assign()

    def cli_overriders_reset(self):
        """Reset the internal state of overriders.  """
        self._get_session('train').overriders_reset()

    def cli_save(self):
        """Saves the latest checkpoint.  """
        self.session.checkpoint.save('latest')

    def cli_interact(self):
        """Interacts with the train/eval session using iPython.  """
        try:
            self._get_session().interact()
        except SessionNotInitializedError:
            log.warn('Session not initalized, interacting with "mayo.cli".')
            from IPython import embed
            embed()

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
                        'We don\'t know what you mean by {!r}.'.format(each))
                return

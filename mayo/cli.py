import os
import sys
import base64

import yaml
import tensorflow as tf

from docopt import docopt

from mayo.log import log
from mayo.config import Config
from mayo.session import (
    Evaluate, FastEvaluate, Train, LayerwiseRetrain, GlobalRetrain, Profile)

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

    def _validate_config(self, keys, action, test=False):
        for k in keys:
            if k in self.config:
                continue
            if test:
                return False
            log.error_exit(
                'Please ensure config content {!r} is imported before '
                'executing {!r}.'.format(k, action))
        return True

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

    _session_map = {
        'train': Train,
        'profile': Profile,
        'retrain-layer': LayerwiseRetrain,
        'retrain-global': GlobalRetrain,
        'validate': Evaluate,
        'fast-validate': FastEvaluate,
    }
    _keys_map = {
        'train': _train_keys,
        'profile': _train_keys,
        'retrain-layer': _train_keys,
        'retrain-global': _train_keys,
        'validate': _validate_keys,
        'fast-validate': _validate_keys,
    }

    def _get_session(self, action=None):
        if not action:
            if self.session:
                return self.session
            keys = self._train_keys
            if self._validate_config(keys, 'train', test=True):
                self.session = self._get_session('train')
            else:
                self.session = self._get_session('validate')
            return self.session
        keys = self._model_keys + self._dataset_keys

        try:
            cls = self._session_map[action]
            keys += self._keys_map[action]
        except KeyError:
            raise TypeError('Action {!r} not recognized.'.format(action))
        self._validate_config(keys, action)
        if not isinstance(self.session, cls):
            log.info('Starting a {} session...'.format(action))
            self.session = cls(self.config)
        return self.session

    def cli_profile(self):
        """Performs training profiling for run-time statistics.  """
        return self._get_session('profile').profile()

    def cli_profile_timeline(self):
        """Performs training profiling to produce timeline.json.  """
        # TODO integrate this into Profile.
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

    def cli_plot(self):
        """Plots activation maps as images and parameters as histograms."""
        return self._get_session('validate').plot()

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
        result = self._get_session('validate').eval_all()
        file_name = 'eval_all.csv'
        with open(file_name, 'w') as f:
            f.write(result.csv())
        log.info(
            'Evaluation results saved in {!r}.'.format(file_name))

    def cli_overriders_update(self):
        """Updates variable overriders in the training session.  """
        self._get_session('train').overriders_update()

    def cli_overriders_assign(self):
        """Assign overridden values to original parameters.  """
        self._get_session('train').overriders_assign()

    def cli_overriders_reset(self):
        """Reset the internal state of overriders.  """
        self._get_session('train').overriders_reset()

    def cli_reset_num_epochs(self):
        """Resets the number of training epochs.  """
        self._get_session('train').reset_num_epochs()

    def cli_export(self):
        """Exports the current config.  """
        name = 'export.yaml'
        with open(name, 'w') as f:
            f.write(self.config.to_yaml())
        log.info('Config successfully exported to {!r}.'.format(name))

    def cli_info(self):
        """Prints parameter and layer info of the model.  """
        info = self._get_session().info()
        for key in ('trainables', 'nontrainables', 'layers'):
            print(info[key].format())
        for table in info.get('overriders', {}).values():
            print(table.format())

    def cli_interact(self):
        """Interacts with the train/eval session using iPython.  """
        self._get_session().interact()

    def cli_save(self):
        """Saves the latest checkpoint.  """
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
            each = each.strip() # some problems with \r when running through Linux subsystem on Windows
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

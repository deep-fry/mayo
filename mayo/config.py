import os
import re
import sys
import glob
import subprocess

from mayo.log import log
from mayo.parse import ConfigBase


def _auto_select_gpus(num_gpus, memory_bound):
    try:
        info = subprocess.check_output(
            'nvidia-smi', shell=True, stderr=subprocess.STDOUT)
        info = re.findall('(\d+)MiB\s/', info.decode('utf-8'))
        log.debug('GPU memory usages (MB): {}'.format(', '.join(info)))
        info = [int(m) for m in info]
        gpus = [i for i in range(len(info)) if info[i] <= memory_bound]
    except subprocess.CalledProcessError:
        gpus = []
    if len(gpus) < num_gpus:
        log.warn(
            'Number of GPUs available {} is less than the number of '
            'GPUs requested {}.'.format(len(gpus), num_gpus))
    return ','.join(str(g) for g in gpus[:num_gpus])


def _init_gpus(system):
    """
    gpus: 'auto' -> auto select GPUs.
    """
    cuda_key = 'CUDA_VISIBLE_DEVICES'
    if os.environ.pop(cuda_key, None):
        log.warn(
            'Ignoring {!r}, as it is overridden '
            'by config "system.visible_gpus".'.format(cuda_key))
    gpus = system.visible_gpus
    if gpus != 'auto':
        if isinstance(gpus, list):
            gpus = ','.join(str(g) for g in gpus)
        else:
            gpus = str(gpus)
    else:
        gpus = _auto_select_gpus(
            system.num_gpus, system.gpu_memory_bound)
    if gpus:
        log.info('Using GPUs: {}'.format(gpus))
    else:
        log.info('No GPUs available, using one clone of the network.')
        # FIXME doesn't work. hacky way to make it instantiate only one tower
        system.num_gpus = 1
    # force ordering to match PCIE bus id, hopefully the same
    # ordering seen in nvidia-smi
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # sets the visible GPUs
    os.environ[cuda_key] = gpus


class Config(ConfigBase):
    def __init__(self):
        merge_hook = {
            'system.log': self._setup_log_level,
        }
        super().__init__(merge_hook)
        self._setup_excepthook()
        self._init_system_config()
        self._finalize()

    def _init_system_config(self):
        root = os.path.dirname(__file__)
        self.yaml_update(os.path.join(root, 'system.yaml'))

    def _finalize(self):
        _init_gpus(self)

    def data_files(self, mode):
        path = self.dataset.path
        try:
            path = path[mode]
        except KeyError:
            raise KeyError('Mode {!r} not recognized.'.format(mode))
        files = []
        search_path = self.system.search_path.dataset
        paths = [path]
        if not os.path.isabs(path):
            paths = [os.path.join(d, path) for d in search_path]
        for p in paths:
            files += glob.glob(p)
        if not files:
            msg = 'No files found for dataset {!r} with mode {!r} at {!r}'
            raise FileNotFoundError(msg.format(
                self.dataset.name, mode, ', '.join(paths)))
        return files

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb
        from mayo.util import import_from_string
        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in self.get('system.pdb.skip', []):
            exc = import_from_string(exc)
            if issubclass(etype, exc):
                sys.exit(-1)
        if self.get('system.pdb.use', True):
            import ipdb
            ipdb.post_mortem(etb)

    def _setup_excepthook(self):
        sys.excepthook = self._excepthook

    def _setup_log_level(self):
        log.level = self.get('system.log.level', 'info')
        log.frame = self.get('system.log.frame', False)
        tf_level = self.get('system.log.tensorflow', 0)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_level)

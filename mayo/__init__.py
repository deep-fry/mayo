import os
import sys

__mayo__ = 'Mayo'
__description__ = """
A deep learning framework with condiment preference as religion."""
__version__ = '0'
__date__ = '14 Aug 2017'
__author__ = 'Xitong Gao'
__executable__ = os.path.basename(sys.argv[0])
__doc__ = """
{__mayo__} {__version__} {__date__}
{__description__}
{__author__}
""".format(**locals())


from mayo.config import Config
from mayo.net import Net
from mayo.train import Train

__all__ = [Config, Net, Train]


def _excepthook(type, value, traceback):
    import IPython

    class Shell(IPython.terminal.embed.InteractiveShellEmbed):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            banner = '{name} {version} by {author}\n{desc}'.format(
                name=__mayo__, version=__version__,
                author=__author__, desc=__description__)
            self.banner1 += '\n'.join(['', banner, ''])

        def run_cell(
                self, raw_cell, store_history=False,
                silent=False, shell_futures=True):
            self.raw_cell = raw_cell
            super().run_cell(raw_cell, store_history, silent, shell_futures)

    rv = ultratb.VerboseTB(include_vars=False)(type, value, traceback)
    print('Launching Pdb...')
    shell = Shell()
    pdb = IPython.core.debugger.Pdb(shell.colors)
    pdb.interaction(None, traceback)
    return rv


try:
    from IPython.core import ultratb
except ImportError:
    pass
else:
    sys.excepthook = _excepthook

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

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


from mayo.config import Config
from mayo.net import Net
from mayo.train import Train

__all__ = [Config, Net, Train]

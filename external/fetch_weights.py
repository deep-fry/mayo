"""
unzip the following datafile
tar
ref:
1. https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
2. https://www.tensorflow.org/s/results/?q=freezegraph&p=%2F
Loading meta only might not work.
If the checkpoint contains:
model.ckpt.meta
model.ckpt.index
model.ckpt.data-XXXX
model_name = model.ckpt
If the checkpoint only contains model.ckpt
model_name = model.ckpt
Checked on inceptionv3 and mobilenet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import numpy as np
import pickle
np.set_printoptions(precision=32)

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

params = {}
# model_name = 'checkpoint-1765'
model_name = 'vgg_16.ckpt'

reader = pywrap_tensorflow.NewCheckpointReader(model_name)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in sorted(var_to_shape_map):
    params[key] = reader.get_tensor(key)

with open('vgg_16.pkl','wb') as f:
    pickle.dump(params,f)

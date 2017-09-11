import os
import io
import re
import glob
import subprocess


net = 'mobilenet_v1'
net_yaml = 'models/mobilenet.yaml'
dataset = 'imagenet'
dataset_yaml = 'datasets/imagenet.yaml'
#  net = 'lenet5'
#  net_yaml = 'models/lenet5.yaml'
#  dataset = 'mnist'
#  dataset_yaml = 'datasets/mnist.yaml'

path = os.path.join('checkpoints', net, dataset)
files = glob.glob(os.path.join(path, 'checkpoint-*.index'))
steps = []
for f in files:
    step = int(re.findall(r'checkpoint-(\d+).index', f)[0])
    steps.append(step)

command = './my eval {} {} system.checkpoint.load={} system.use_pdb=false'
for step in sorted(steps):
    proc = subprocess.Popen(
        command.format(net_yaml, dataset_yaml, step).split(' '),
        stdout=subprocess.PIPE)
    out = list(io.TextIOWrapper(proc.stdout, encoding="utf-8"))
    top1 = float(re.findall(r'top1: (\d+.\d+)%', out[-1])[0])
    top5 = float(re.findall(r'top5: (\d+.\d+)%', out[-1])[0])
    print(step, top1, top5)

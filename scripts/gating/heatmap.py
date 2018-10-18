import yaml
import subprocess


model = 'models/gate/cifarnet.yaml'
dataset = 'datasets/cifar10.yaml'
ckpt_base = 'gate'
files = ['checkpoint-{}'.format(i) for i in range(0,13)]
#  files = ['checkpoint-20']

cmd_base = './my {} {} system.info.plumbing=true'
cmd_base = cmd_base.format(model, dataset)
results = []
for i, f in enumerate(files):
    print(f)
    cmd = '{} system.checkpoint.load={} _gate.density=0.5 eval plot'
    cmd = cmd.format(cmd_base, f)
    subprocess.call(cmd, cwd='../..', shell=True)
    subprocess.call('cp active.npy active-{}.npy'.format(i), cwd='../..', shell=True)
    print(results)

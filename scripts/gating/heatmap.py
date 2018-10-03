import yaml
import subprocess


model = 'models/gate/cifarnet.yaml'
dataset = 'datasets/cifar10.yaml'
ckpt_base = 'gate'
files = ['checkpoint-{}'.format(i * 10) for i in range(16)]

cmd_base = './my {} {} system.info.plumbing=true'
cmd_base = cmd_base.format(model, dataset)
results = []
for f in files:
    print(f)
    cmd = '{} system.checkpoint.load={} _gate.density=0.5 eval plot'
    cmd = cmd.format(cmd_base, f)
    subprocess.call(cmd, cwd='../..', shell=True)
    subprocess.call('cp actives.npy actives-{}.npy'.format(f), cwd='../..', shell=True)
    print(results)

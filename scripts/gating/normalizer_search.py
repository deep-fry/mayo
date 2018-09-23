import yaml
import os
import subprocess

meta_yaml = yaml.load(open('cifarnet.yaml', 'rb'))
mayo_dir = "../../"
target_key = 'l1'
gpus = [0, 1]
cmd_formatter = './my datasets/cifar10.yaml {} trainers/cifarnet.yaml system.checkpoint.load=gate100 train.learning_rate._initial=0.1 train.decay_steps=50 system.max_epochs=180 system.checkpoint.save.interval=10 system.num_gpus=2 system.visible_gpus=[{},{}] reset-num-epochs train'
eval_cmd_formatter = './my datasets/cifar10.yaml {} trainers/cifarnet.yaml system.checkpoint.load=gate100 train.learning_rate._initial=0.1 train.decay_steps=50 system.max_epochs=180 system.checkpoint.save.interval=10 system.num_gpus=2 system.visible_gpus=[{},{}] eval-all'


regularizer = {
    'moe': 0.0,
    'l1': 0.0,
}

# sweep values
values = list(range(1,7))
values = [0.1 / (10 ** v) for v in values]
print(values)

config_dir = os.path.join(mayo_dir, 'gate_configs')
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

for index, value in enumerate(values):
    meta_yaml['_gate']['regularizer'].update(regularizer)
    regularizer[target_key] = value
    with open(os.path.join(config_dir, 'gate'+str(index)+'.yaml'), 'w') as f:
        yaml.dump(meta_yaml, f, default_flow_style=False)
    cmd = cmd_formatter.format(
        os.path.join('gate_configs', 'gate'+str(index)+'.yaml'),
        gpus[0],
        gpus[1]
    )
    eval_cmd = eval_cmd_formatter.format(
        os.path.join('gate_configs', 'gate'+str(index)+'.yaml'),
        gpus[0],
        gpus[1]
    )
    subprocess.call(cmd, cwd=mayo_dir, shell=True)
    subprocess.call(eval_cmd, cwd=mayo_dir, shell=True)
    subprocess.call("cp eval_all.csv eval_all{}.csv".format(index), cwd=mayo_dir, shell=True)
    subprocess.call("cp models/cifarnet/cifar10/latest.index models/cifarnet/cifar10/latest{}.index".format(index), cwd=mayo_dir, shell=True)
    subprocess.call("cp models/cifarnet/cifar10/latest.data-00000-of-00001 models/cifarnet/cifar10/latest{}.data-00000-of-00001".format(index), cwd=mayo_dir, shell=True)
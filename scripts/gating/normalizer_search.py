import subprocess

epochs = 300
density = 1.0
target_key = 'l1'
mayo_dir = "../../"
cmd_formatter = './my models/gate/cifarnet.yaml _gate.density={} _gate.regularizer.{}={} trainers/cifarnet.yaml datasets/cifar10.yaml system.checkpoint.load=gate100 train.learning_rate._initial=0.001 train.learning_rate.decay_steps=80 system.max_epochs={} system.checkpoint.save.interval=10 system.num_gpus=1 system.visible_gpus="2" system.pdb.use=false reset-num-epochs train'
eval_cmd_formatter = './my models/gate/cifarnet.yaml _gate.density={} datasets/cifar10.yaml trainers/cifarnet.yaml eval-all'


regularizer = {
    'moe': 0.0,
    'l1': 0.0,
}

# sweep values
exps = list(reversed(range(2, 10, 2)))
print(exps)
for exp in exps:
    value = 10 ** (-exp)
    print(value)
    cmd = cmd_formatter.format(density, target_key, value, epochs)
    eval_cmd = eval_cmd_formatter.format(density)
    subprocess.call(cmd, cwd=mayo_dir, shell=True)
    subprocess.call(eval_cmd, cwd=mayo_dir, shell=True)
    subprocess.call("cp eval_all.csv eval_all{}.csv".format(exp), cwd=mayo_dir, shell=True)
    subprocess.call("cp checkpoints/cifarnet/cifar10/checkpoint-{}.index checkpoints/cifarnet/cifar10/latest{}.index".format(epochs, exp), cwd=mayo_dir, shell=True)
    subprocess.call("cp checkpoints/cifarnet/cifar10/checkpoint-{}.data-00000-of-00001 checkpoints/cifarnet/cifar10/latest{}.data-00000-of-00001".format(epochs, exp), cwd=mayo_dir, shell=True)

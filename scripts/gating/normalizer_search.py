import subprocess

epochs = 150
density = 0.5
target_key = 'moe'
mayo_dir = "../../"
cmd_formatter = './my models/gate/cifarnet.yaml _gate.density={} _gate.regularizer.{}={} trainers/cifarnet.yaml datasets/cifar10.yaml system.checkpoint.load=pretrained train.learning_rate._initial=0.01 train.learning_rate.decay_steps=50 system.max_epochs={} system.checkpoint.save.interval=10 system.num_gpus=1 system.visible_gpus="2" system.pdb.use=false reset-num-epochs train'
eval_cmd_formatter = './my models/gate/cifarnet.yaml _gate.density={} datasets/cifar10.yaml trainers/cifarnet.yaml eval-all'

# sweep values
exps = [8, 6, 4, 2]
print(exps)
for exp in exps:
    value = 10 ** (-exp)
    print(value)
    cmd = cmd_formatter.format(density, target_key, value, epochs)
    eval_cmd = eval_cmd_formatter.format(density)
    subprocess.call(cmd, cwd=mayo_dir, shell=True)
    subprocess.call(eval_cmd, cwd=mayo_dir, shell=True)
    subprocess.call("cp eval_all.csv checkpoints/cifarnet/cifar10/moe/{}.csv".format(exp), cwd=mayo_dir, shell=True)
    subprocess.call("cp checkpoints/cifarnet/cifar10/checkpoint-{}.index checkpoints/cifarnet/cifar10/moe/{}.index".format(epochs, exp), cwd=mayo_dir, shell=True)
    subprocess.call("cp checkpoints/cifarnet/cifar10/checkpoint-{}.data-00000-of-00001 checkpoints/cifarnet/cifar10/moe/{}.data-00000-of-00001".format(epochs, exp), cwd=mayo_dir, shell=True)

import yaml
import os
import subprocess

mayo_dir = "../../"
gpus = [0, 1]
model='alexnet_bn'
cmd_formatter = './my datasets/imagenet.yaml models/override/{}.yaml models/override/quantize/fixed.yaml trainers/cifarnet.yaml system.checkpoint.load=pretrained train.learning_rate._initial=0.01 train.learning_rate.decay_steps=10 system.max_epochs=30 system.checkpoint.save.interval=1 system.num_gpus=2 system.visible_gpus=[{},{}] reset-num-epochs train'
eval_cmd_formatter = './my datasets/imagenet.yaml models/override/{}.yaml models/override/quantize/fixed.yaml trainers/cifarnet.yaml system.checkpoint.load=pretrained train.learning_rate._initial=0.01 train.learning_rate.decay_steps=10 system.max_epochs=30 system.checkpoint.save.interval=1 system.num_gpus=2 system.visible_gpus=[{},{}] eval-all'

cmd = cmd_formatter.format(model, gpus[0], gpus[1])
subprocess.call(cmd, cwd=mayo_dir, shell=True)
subprocess.call(eval_cmd, cwd=mayo_dir, shell=True)
subprocess.call("cp eval_all.csv eval_all{}.csv".format(model), cwd=mayo_dir, shell=True)
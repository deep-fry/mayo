import yaml
import os
import subprocess

mayo_dir = "../../"
gpus = [0, 1]
model='resnet18'
cmd_formatter = './my datasets/imagenet.yaml models/override/{}.yaml models/override/quantize/fixed_incremental_sparse.yaml trainers/cifarnet.yaml system.checkpoint.load={} train.learning_rate._initial=0.01 train.learning_rate.decay_steps=3 system.max_epochs={} system.checkpoint.save.interval=1 system.num_gpus=2 system.visible_gpus=[{},{}] train.learning_rate._default_batch_size=256 system.batch_size_per_gpu=128 _overrider.weights.incremental.interval={} _overrider.activation.incremental.interval={} reset-num-epochs train'
eval_cmd_formatter = './my datasets/imagenet.yaml models/override/{}.yaml models/override/quantize/fixed_incremental_sparse.yaml trainers/cifarnet.yaml system.checkpoint.load=latest train.learning_rate._initial=0.01 train.learning_rate.decay_steps=3 system.max_epochs=10 system.checkpoint.save.interval=1 system.num_gpus=2 system.visible_gpus=[{},{}] train.learning_rate._default_batch_size=256 system.batch_size_per_gpu=128 _overrider.weights.incremental.interval={} _overrider.activation.incremental.interval={} reset-num-epochs eval-all'
# eval_cmd_formatter = './my datasets/imagenet.yaml models/override/{}.yaml models/override/quantize/fixed_incremetnal_sparse.yaml trainers/cifarnet.yaml system.checkpoint.load=pretrained train.learning_rate._initial=0.01 train.learning_rate.decay_steps=10 system.max_epochs=30 system.checkpoint.save.interval=1 system.num_gpus=2 system.visible_gpus=[{},{}] eval-all'

# interval training
for percentage in [0.0, 0.25, 0.5, 0.75, 1.0]:
    if percentage == 0.0:
        # At percentage 0.0, train to warmup, because of activation quantization
        # model name, ckpt, max_epochs, gpu, gpu, interval, act interval
        cmd = cmd_formatter.format(model, 'pruned', 10, gpus[0], gpus[1], percentage, percentage)
    elif percentage == 1.0:
        cmd = cmd_formatter.format(model, 'latest', 15, gpus[0], gpus[1], percentage, percentage)
    else:
        cmd = cmd_formatter.format(model, 'latest', 10, gpus[0], gpus[1], percentage, percentage)
    subprocess.call(cmd, cwd=mayo_dir, shell=True)

subprocess.call(eval_cmd, cwd=mayo_dir, shell=True)
# subprocess.call("cp eval_all.csv eval_all{}.csv".format(model), cwd=mayo_dir, shell=True)
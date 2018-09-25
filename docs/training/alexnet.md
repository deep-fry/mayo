## Train AlexNet-BN 

AlexNet training serves as a tutorial on deploying Mayo on
training large models.
The target AlexNet here is added with batchnormlization on each layer
and can be found at
`models/alexnet_bn.yaml`.
Note this is the alexnet described in 
["One weird trick..."](https://arxiv.org/abs/1404.5997>) paper.

## Command
```Bash
./my \
datasets/imagenet.yaml models/alexnet_bn.yaml trainers/cifarnet.yaml \
system.num_gpus=2 system.visible_gpus=[0,1] \
system.checkpoint.load=null \
system.batch_size_per_gpu=1024 \
system.max_epochs=100 \
train.learning_rate._initial=0.001 \
train.learning_rate.decay_steps=30 \
train._default_batch_size=256 \
reset-num-epochs train
```

## Command Breakdown
* `system.checkpoint.load=null`:
By default Mayo will load the latest ckpt, here we ask it to train from scratch

* `train._default_batch_size`:
Mayo uses a square root learning rate 
scheduling, this sets the default batch size
for the scheduling.
Details can be found at `trainers/exponential.yaml`

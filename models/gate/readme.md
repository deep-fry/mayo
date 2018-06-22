## Gating

All the files in this directory should be able to perform gating, but you need
to be carefull with the `_import`.

For example, `_import: ../lenet5_bn.yaml` means this is taking the original
lenet5_bn as a base.
In contrast, `_import: ../cifarnet_slim50.yaml` takes a slimmed model as
its base.

There are a bunch of hyper-parameters in gating, for example:

```Yaml
_gate:
    enable: true
    policy: parametric
    density: 1.0
    pool: avg
    regularizer:
        # l1: 0.000001
        # moe: 0.01
        null
    threshold: online
    trainable: true
```

* A number of regularizers are specified, and if not invoked, they are simply 
not applied.

* The policy we use now is called `parametric`.

* We first train with `trainable` set to `False` to warm up the gating network,
we then switch `trainable` to `True` to make the base network co-adapt.

* `threshold`, we randomly support three modes, but we use `online` most of 
the times.

```Bash
./my datasets/imagenet.yaml \
    models/gate/resnet18.yaml \
    trainers/cifarnet.yaml \
    system.num_gpus=2 system.visible_gpus=[0,1] \
    system.checkpoint.load=pretrained \
    train
```

The command above should run this gating on resnet18.

```Bash
./my datasets/imagenet.yaml \
    models/gate/resnet18.yaml \
    trainers/cifarnet.yaml \
    system.num_gpus=2 system.visible_gpus=[0,1] \
    system.checkpoint.load=pretrained \
    system.info.plmubing=True \
    info 
```

Command `info` would generate the number of MACs for this gating network and 
also the number of parameters of each layer.

`plumbing=True` would provide you an info.yaml as output -- makes life easier
if you want to parse the info results.

## Slimming

In the paper, we would like to compare gating to slimming and also consider
gating on top of a slimmed network to show compositability.

In `models/override/netslim.yaml`, you can find the description of network slimming.

```Yaml
---
_overrider:
    conv: &slim
        activation:
            netslim:
                type: mayo.override.NetworkSlimmer
                density: 1.0
                weight: 0.00001
                should_update: true
                _priority: 100
    pointwise: {<<: *slim}
    fc: null
    logits: null
```

The important hyperparameter here is the `weight`.
The slimming procedure is to train with this `weight` value set to a small constant and then put that to zero and continue training.

To execute network slimming, first run with a feasible `weight` value but **without** issuing overriders-update:

```Bash
./my datasets/cifar10.yaml \
    models/override/cifarnet.yaml \
    models/override/prune/netslim.yaml \
    trainers/cifarnet.yaml \
    reset-num-epochs train
```

Then you should set `weight=0` and issue an update:

```Bash
./my datasets/cifar10.yaml \
    models/override/cifarnet.yaml \
    models/override/prune/netslim.yaml \
    trainers/cifarnet.yaml \
    reset-num-epochs overriders-update train
```

Apparently, what we are doing here is nothing more than a pseudo-network slimming, because we are masking out channels instead actually taking them away.

What I did to generate a slimmed model is a little hacky.
We first enter an `interactive` session in mayo, remmebert to load the
gated checkpoint

```Bash
./my datasets/cifar10.yaml \
    models/override/cifarnet.yaml \
    models/override/prune/netslim.yaml \
    trainers/cifarnet.yaml \
    system.checkpoint.load=saved_gating
    train interact
```

Press `ctrl+c` to escape from training but enter the interactie session. 
Load a script:

```Bash
%load scripts/export_slim_model.py
```

Actually, since we've changed overriders data strucutre in Mayo, this script might somehow become out-of-date...

Anyway, the idea is to reform a checkpoint from this pseudo-network slimming, then you would have to write another network description yaml with the correct number of channels.
I might put something to automate this in the future.
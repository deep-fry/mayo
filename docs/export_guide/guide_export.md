## How to export values

* Train for 200 epochs usign 2 gpus.

```Bash
./my \
    datasets/cifar10.yaml \ 
    models/cifarnet.yaml \
    trainers/cifarnet.yaml \
    system.checkpoint.load=null \
    system.num_gpus=2 \
    system.visible_gpus=[0,1] \
    system.max_epochs=200 \
    system.checkpoint.save.interval=10 \
    reset-num-epochs train
```
I tuned the `decay_steps` in `trainer.yaml` to be 80 to have three decays.

* Evaluate the results.

```Bash
./my \
    datasets/cifar10.yaml \ 
    models/cifarnet.yaml \
    trainers/cifarnet.yaml \
    system.checkpoint.load=latest \
    system.num_gpus=2 \
    system.visible_gpus=[0,1] \
    system.max_epochs=200 \
    system.checkpoint.save.interval=10 \
    eval
```

The accuracy I achieved is `top1: 90.14%, top5: 99.68%`


* Apply quantization and export the parameters in an interactive session

```Bash
./my \
    datasets/cifar10.yaml \
    models/override/cifarnet.yaml \
    models/override/quantize/dfp.yaml \
    trainers/cifarnet.yaml \
    system.checkpoint.load=latest system.num_gpus=2 \
    system.batch_size_per_gpu=1 \
    system.visible_gpus=[0,1] system.max_epochs=200 \
    system.checkpoint.save.interval=10 train overriders-update interact
```

Overriders-update automatically works out the point for dynamic fixed-point
based on current numerical values (current batch of inputs).
Making batch size to be 1 is easier, becasue updates traverses smaller tensors.
This might be very inaccurate for dynamic values (activations, gradients and erros).
 
Each model layer has to take a definition of overriders (`models/override/cifarnet.yaml`)

```YAML
model.layers:
    conv0: &conv {overrider: $(_overrider.conv)}
    conv1: {<<: *conv}
    fc1: {overrider: $(_overrider.fc)}
    logits: {overrider: $(_overrider.logits)}
```
Also make sure the correct components are picked (`models/override/quantize/dfp.yaml`)

```YAML
_import: ../_global.yaml
_overrider:
    weights: &quantizer
         dynamic:
            type: mayo.override.DGQuantizer
            width: 16
            overflow_rate: 0.0
            should_update: true
            stochastic: false
            _priority: 90
#    biases: {<<: *quantizer}
    activation: {<<: *quantizer}
    gradient:
        weights: *quantizer
        error: *quantizer
```

In the interactive mode (pdb enabled), load the python script and call the
loaded export function to obtain a pickle file.
The pickle file will have components' values before/after quantizations.
```Bash
%load guide_export.py
tmp = export(self, 'tmp.pickle')
```
Optionally, you can save that tf checkpoint:
```Bash
self.task.nets[0].checkpoint.save_checkpoint('tmp')
```

You can also find more information about the quantization by issuing:
```Bash
./my \
    datasets/cifar10.yaml \
    models/override/cifarnet.yaml \
    models/override/quantize/dfp.yaml \
    trainers/cifarnet.yaml \
    system.checkpoint.load=latest system.num_gpus=2 \
    system.batch_size_per_gpu=1 \
    system.visible_gpus=[0,1] system.max_epochs=200 \
    system.checkpoint.save.interval=10 train overriders-update info
```

Using `system.info.plumbing`, you can direct out the infomation in a yaml file
which is the info.yaml under this directory.
# Mayo 101

## Training
Make sure you are at ```whatever/mayo-dev```, then run:

```Shell
./my datasets/mnist.yaml trainers/lenet5.yaml models/lenet5.yaml train
```
The command line above executes training of a lenet5 network on mnist dataset.
In Mayo, we define **every** hyperparameter in YAML files and as you can see
from this command line, we fetch dataset yaml, trainer yaml and model yaml
from different files.
The final train is executed as an action.

You can also override some YAML definitions in the command line:
```Shell
./my datasets/mnist.yaml trainers/lenet5.yaml models/lenet5.yaml system.batch_size_per_gpu=1024 system.num_gpus=2 train
```

In this case, I overrided the system definitions in [system.yaml](../mayo/system.yaml) and changed
the batch size and number of gpus in the command line.
Notice system.yaml is read by default.

#### Datasets
Datasets are defined at ```whatever/mayo-dev/datasets```, only two datasets
come with the project (cifar10 and mnist).
However, datasets such as flowers and imagnet are declared in this directory,
you can create a symbolic link for your imagnet dataset and point them to this
directory, eg:
```Shell
ln -s where_your_imagenet_is whatever/mayo-dev/datasets
```

## Inference
```Shell
./my datasets/mnist.yaml trainers/lenet5.yaml models/lenet5.yaml system.checkpoint.load=pretrained eval
```

In this case, we specified a pretrained model that is under ```whatever/mayo-dev/checkpoints/lenet5/mnist/```.
We adopt this naming habbit ```checkpoints/$(model.name)/$(dataset.name)/```, as
declared in [system.yaml](../mayo/system.yaml).

## Instantiate your own layer

## Parameter Overriding
An important feature of Mayo is that it supports parameter overriding...

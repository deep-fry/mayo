# Mayo 101

Assuming you've cloned mayo and put it under the directory ```whatever```
Notice you might need git lfs to download all the datasets (mnist and cifa10).
Mayo now runs at tensorflow v1.4, which is the latest release.

## Training
Make sure you are at ```whatever/mayo-dev```, then run:

```Shell
./my datasets/mnist.yaml trainers/lenet5.yaml models/lenet5.yaml train
```
The command above executes training of a lenet5 network on a mnist dataset.
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

## Datasets
Datasets are defined at ```whatever/mayo-dev/datasets```, only two datasets
come with the project (cifar10 and mnist).

YOU NEED GIT LFS FOR CLONING THE DATASETS!

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

## Parameter Overriding
An important feature of Mayo is that it supports parameter overriding and thus opens opportunities
for a number of optimizations.

#### To prune
```Shell
./my datasets/mnist.yaml models/lenet5.yaml trainers/lenet5.yaml external/dns_pruning/lenet5_dns.yaml train
```
This command asks mayo to train lenet5 from scratch using [dynamic network surgery pruners](../mayo/override/prune.py)

```Shell
! Variables missing in checkpoint:
    lenet5/conv0/weights/DynamicNetworkSurgeryPruner.mask
    lenet5/conv0/weights/DynamicNetworkSurgeryPruner.alpha
    lenet5/conv1/weights/DynamicNetworkSurgeryPruner.mask
    lenet5/conv1/weights/DynamicNetworkSurgeryPruner.alpha
    lenet5/fc1/weights/DynamicNetworkSurgeryPruner.mask
    lenet5/fc1/weights/DynamicNetworkSurgeryPruner.alpha
    lenet5/logits/weights/DynamicNetworkSurgeryPruner.mask
    lenet5/logits/weights/DynamicNetworkSurgeryPruner.alpha
```
Part of the command line output is shown above, this indicates the variables are created for the pruner and these variables would be stored later in the checkpoints.

#### To quantize
```Shell
./my datasets/mnist.yaml models/lenet5.yaml trainers/lenet5.yaml external/fixed_point_quan/lenet5_fixedpoint.yaml train
```
More overriders can be found at ```mayo/overriders```

## Instantiate your own layer
Instantiate layers is a common practice and if it is a common layer that Mayo has already supported, you can instantiate them in the YAML file:
```Yaml
conv0: &conv
    <<: *init
    type: convolution
    kernel_size: 5
    padding: valid
    num_outputs: 20
```

Mayo also supports instantiate customized modules in YAML files, for example, SqueezeNet has used a fire module, and we declare it as:
```Yaml
_fire: &fire
    type: module
    kwargs: {squeeze_depth: null, expand_depth: null}
    layers:
        squeeze:
            <<: *conv
            num_outputs: ^squeeze_depth
        expand1: &expand1
            <<: *conv
            num_outputs: ^expand_depth
        expand3: {<<: *expand1, kernel_size: 3}
        concat:
            type: concat
            axis: 3
    graph:
        - {from: input, with: squeeze, to: squeezed}
        - {from: squeezed, with: expand1, to: expanded1}
        - {from: squeezed, with: expand3, to: expanded3}
        - {from: [expanded1, expanded3], with: concat, to: output}
```
This graph definition instantiate a special module called fire and many complex vision models (Resnet, inception ...) use this way to instantiate some parts of the network.

Another common practice is to write your own layer!
Here we show how to define a layer in Mayo and instantiate it in Yaml.
First, all layer definitions are in ```mayo/net/tf.py```.
Line 199 in that file defines a hadamard layer.
This layer is later instantiated in ```models/hadamard```.

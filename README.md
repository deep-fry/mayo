# Mayo

**Mayo** is a deep learning framework developed with hardware acceleration in mind. It enables rapid deep neural network model design, and common network compression work flows such as [fine][fine]- and coarse-grained pruning, [network slimming][slim], and [quantization methods with various arithmetics][quantize] can be easily used jointly by writing simple [YAML][yaml] model description files (exemplified by the links above).  Additionally, Mayo can also accelerate the hyperparameter exploration of these models with automated hyperparameter optimization.  With minimal manual intervention, our automated optimization results achieve state-of-the-art compression rates (refer to the results section).

[fine]: https://github.com/admk/mayo/blob/develop/models/override/prune/dns.yaml
[slim]: https://github.com/admk/mayo/blob/develop/models/override/prune/netslim.yaml


## Installation

### Prerequisites

Before setting up Mayo, you will need to have [Git][git], [Git-LFS][git-lfs], [Python 3.6.5 or above][python3] and [TensorFlow 1.8.0][tensorflow] installed.

[git]: https://git-scm.com
[git-lfs]: https://git-lfs.github.com
[python3]: https://www.python.org/downloads/
[tensorflow]: https://www.tensorflow.org/install/

### Setting up Mayo

Mayo can be installed by checking out our repository and install the necessary Python packages:
```bash
$ git clone https://github.com/admk/mayo.git
$ cd mayo
$ pip3 install -r requirements.txt
```

### Adding TFRecord dataset files

Mayo accepts standard TFRecord dataset files.  You can use instructions outlined [here][tfrecord] to generate TFRecord files for ImageNet, CIFAR-10 and MNIST datasets.  The newly generated TFRecord files can then be placed in `[mayo]/datasets/{imagenet,cifar10,mnist}` folders to be used by Mayo.  It may be necessary to ensure `dataset.path.{train,validate}` in `[mayo]/datasets/{imagenet,cifar10,mnist}.yaml` points to the correct location.

[tfrecord]: https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format


## Testing Mayo

Run a simple LeNet-5 validation with the MNIST dataset using:
```bash
$ ./my \
    models/lenet5.yaml \
    datasets/lenet5.yaml \
    system.checkpoint.load=pretrained \
    eval
```
You should expect the final top-1 and top-5 accuracies to be 99.57% and 100% respectively.


## How to write your neural network application in YAML

In Mayo, we decouple the description of each neural network application into three separate components: the dataset, model and trainer, each written in [YAML][yaml].  The reason for decoupling is to encourage reuse, for instance a ResNet-50 model can not only be used for [ImageNet][imagenet] classification, but also object detection with [COCO][coco], or even customized tasks.

Furthermore, in network compression, we can use many fine- and coarse-grained pruning techniques in conjunction with a large range of quantization methods, even optionally on top of low-rank approximation of weight tensors, on a wide variety of neural networks, each could use different datasets and could be trained differently.  We now encounter a vast number of possible combinations of all of these above options, so by decoupling compression techniques from the neural network, from the dataset, from training methodologies, all possible combinations can be achieved by importing the respective YAML descriptions, without having to write a monolithic description file for each combination.

[yaml]: http://yaml.org
[imagenet]: http://www.image-net.org
[coco]: http://cocodataset.org


### A dataset description

It is easy to write a dataset description for Mayo, here we will use [`datasets/mnist.yaml`][mnist-yaml] as an example:
```yaml
---
dataset:
    # Specify the name of the dataset
    name: mnist
    task:
        # This dataset is used to train for image classification,
        # so we specify the type of this task to be `mayo.task.image.Classify`.
        # In the future, we will support other tasks such as object detection # and NLP applications.
        type: mayo.task.image.Classify
        # Specify the number of class labels.
        num_classes: 10
        # Specify whether the class labels in the dataset contain a
        # background class, we assume that if a background class exists,
        # it is always the first label (zero-indexed).
        background_class: {has: false}
        # The shape of images in the dataset, `height` and `width`
        # are optional.
        shape:
            height: 28
            width: 28
            channels: 1
        # The preprocessing required by the training dataset as the standard
        # training pipeline for all neural networks using the same dataset.
        # Neural network models can further specify validation preprocessing
        # pipeline, as well as the final preprocessing stages regardless of
        # training/validation.
        preprocess:
            train: []
    # the paths pointing to the TFRecord files and the text file
    # containing class labels.
    path:
        train: mnist/train.tfrecord
        validate: mnist/test.tfrecord
        labels: mnist/labels.txt
    # number of examples in the training/validation datasets.
    num_examples_per_epoch:
        train: 60000
        validate: 10000
```

[mnist-yaml]: https://github.com/admk/mayo/blob/develop/datasets/mnist.yaml


### A neural network model

Similarly, neural networks are also written in YAML, here we showcase features in YAML known as [anchors and mapping inheritance][yaml-anchors] which make it easy to reuse and substitute neural network layer definitions.  Here we use `models/lenet5.yaml` as an example:
```yaml
---
dataset:
    task:
        # We additionally specify that our LeNet-5 does not use a
        # background class.
        background_class: {use: false}
        preprocess:
            # We specify the shape of the input to our neural network.
            shape:
                height: 28
                width: 28
                channels: 1
            # We do not use additional preprocessing for validation.
            validate: null
            # For both validation or training, we add a final stage of
            # image preprocessing to transform the range of values from
            # [0, 1] to [-1, 1].
            final_cpu: {type: linear_map, scale: 2.0, shift: -1.0}
model:
    # Name of the model.
    name: lenet5
    # Layer definitions.
    layers:
        # `_init` is a partial layer definition, here we use `&init` as a
        # reference to the mapping, which contains an initializer for weights.
        _init: &init
            weights_initializer:
                type: tensorflow.truncated_normal_initializer
                stddev: 0.09
        # Definition for layer `conv0`
        conv0: &conv
            # It inherits the mapping referenced by `&init` above.
            <<: *init
            # The type of the layer is a convolution.
            type: convolution
            # We specify the size of the kernel, padding and number of
            # output channels.
            kernel_size: 5
            padding: valid
            num_outputs: 20
            # And in addition, a regularizer for the weights.
            weights_regularizer:
                type: tensorflow.contrib.layers.l2_regularizer
                scale: 0.004
        pool0: &pool
            # This defines a max pool layer with a 2x2 kernel, stride size 2,
            # and a valid padding.
            type: max_pool
            kernel_size: 2
            stride: 2
            padding: valid
        # `conv1` inherits all definitions in `conv0` as referenced by the
        # anchor `&conv`, and modifies the number of output channels to 50.
        conv1: {<<: *conv, num_outputs: 50}
        # `pool1` simply reuses the mapping referenced in `&pool`.
        pool1: *pool
        # all other layers should be straightforward.
        flatten: {type: flatten}
        dropout: {type: dropout, keep_prob: 0.5}
        fc1: &fc {<<: *init, type: fully_connected, num_outputs: 500}
        logits:
            <<: *fc
            # no activation function
            activation_fn: null
            # In Mayo, we allow `$(key_dot_path)` to access any value
            # pointed by the key path, here it substitutes the value with
            # the one in `dataset.task.num_classes`, which is 10 if we decide
            # to use the MNIST dataset.
            num_outputs: $(dataset.task.num_classes)
    graph:
        # A graph definition, stringing the layers specified above to form a
        # complete neural network.  Graph definition can also be a list of
        # such mappings, supporting diverging and converging paths.
        from: input
        with: [conv0, pool0, conv1, pool1, flatten, dropout, fc1, logits]
        to: output
```

For more complex examples, please check out the [`models`][models] folder.

[yaml-anchors]: https://gist.github.com/ddlsmurf/1590434#file-output-txt-L82
[models]: https://github.com/admk/mayo/tree/develop/models


### A trainer description

A trainer YAML describes the policy used to train the neural network.  Here is a simple example `mayo/trainers/lenet5.yaml`:
```yaml
---
# Mayo supports description importing using `_import`.
# Here it merges the trainer YAML with the contents of `exponential.yaml`
# in the same directory.
_import: exponential.yaml
train:
    learning_rate:
        # Here we specify the initial learning rate.
        _initial: 0.01
        # And the number of epochs required before we decay the learning rate.
        decay_steps: 300
    optimizer:
        # The type of optimizer used.
        type: tensorflow.train.GradientDescentOptimizer
```

The imported `exponential.yaml` specifies how the actual learning rate is computed, note that Mayo additionally supports `!arith` tag to express arithmetic expressions in YAML to be evaluated on-the-fly:
```yaml
---
train:
    learning_rate:
        # Here we use exponential decay as the policy to decay learning rate
        # after a certain number of epochs.
        type: tensorflow.train.exponential_decay
        # The default initial learning rate.
        _initial: 0.1
        # The default batch size which corresponds to the `_initial`
        # learning rate.
        _default_batch_size: 128
        # The factor used to decay the learning rate.
        decay_rate: 0.16
        # The actual initial learning rate, which scales correspondingly
        # to the current batch size used.
        learning_rate: !arith >
            $(train.learning_rate._initial) * math.sqrt(
                $(system.batch_size_per_gpu) * $(system.num_gpus) /
                $(train.learning_rate._default_batch_size))
        decay_steps: 30
        staircase: true
```


## The Mayo command line interface

The Mayo command line interface works in a different way from most deep learning framework and tools, because we specifically decouple models from datasets from trainers from compression techniques for maximum flexibility and reuseability.

The Mayo command line interface accepts a sequence of actions separated by space.  Each actions can either be a YAML file, a key-value pair or a command.  Each instance of Mayo maintains a global configuration and perform actions in sequence as specified by the actions.  If it encounters a YAML file, it would open the YAML file and import its configurations and merges the configuration with the existing configuration.  A key-value pair updates a specific key-path in the configuration with a new value.  Finally, a command would use the global configuration to, e.g., instantiate a neural network and start training on it with the given dataset.

The installation instruction catches a glimpse of how it works, here we further break down the execution:
```bash
$ ./my \                                # Invocation of Mayo
    models/lenet5.yaml \                # Imports LeNet-5 network description
    datasets/lenet5.yaml \              # Imports the dataset description
    system.checkpoint.load=pretrained \ # Specify that we load the pretrained checkpoint
    eval                                # Starts model evaluation
```
Each YAML import and key-value pair update recursively merges all mappings of the YAML file with the global configuration.  So right before `eval`, we would have a complete application description specifying the model, dataset and checkpoint used.


## Results

**TODO** complete this section.

# Mayo

**Mayo** is a deep learning framework developed with hardware acceleration in mind. It enables rapid deep neural network model design, and common network compression work flows such as [fine][fine]- and coarse-grained pruning, [network slimming][slim], and [quantization methods with various arithmetics][quantize] can be easily used jointly by writing simple [YAML][yaml] model description files (exemplified by the links above).  Additionally, Mayo can also accelerate the hyperparameter exploration of these models with automated hyperparameter optimization.  With minimal manual intervention, our automated optimization results achieve state-of-the-art compression rates (refer to the results section).


## Installation

### Prerequisites

Before setting up Mayo, you will need to have [Git][git], [Git-LFS][git-lfs], [Python 3.6.5 or above][python3] and [TensorFlow 1.8.0][tensorflow] installed.


### Setting up Mayo

Mayo can be installed by checking out our repository and install the necessary Python packages:
```bash
$ git clone https://github.com/admk/mayo.git
$ cd mayo
$ pip3 install -r requirements.txt
```

### Adding TFRecord dataset files

Mayo accepts standard TFRecord dataset files.  You can use instructions outlined [here][tfrecord] to generate TFRecord files for ImageNet, CIFAR-10 and MNIST datasets.  The newly generated TFRecord files can then be placed in `[mayo]/datasets/{imagenet,cifar10,mnist}` folders to be used by Mayo.  It may be necessary to ensure `dataset.path.{train,validate}` in `[mayo]/datasets/{imagenet,cifar10,mnist}.yaml` points to the correct location.


## Testing Mayo

Run a simple LeNet-5 validation with the MNIST dataset using:
```bash
$ ./my \
    models/lenet5.yaml \
    datasets/mnist.yaml \
    system.checkpoint.load=pretrained \
    eval
```
You should expect the final top-1 and top-5 accuracies to be 99.57% and 100% respectively.


## How to start writing your neural network application in YAML

For starters, you can checkout [`models/lenet5.yaml`](models/lenet5.yaml) for an example of a neural network model description, [`datasets/mnist.yaml`](datasets/mnist.yaml) of the MNIST dataset, and [`trainers/lenet5.yaml`](trainers/lenet5.yaml) of a simple training configuration.  For more tutorials, please refer to [this Wiki page][mayo-yaml].


## The Mayo command line interface

The Mayo command line interface works in a different way from most deep learning framework and tools, because we specifically decouple models from datasets from trainers from compression techniques for maximum flexibility and reuseability.  The installation instruction catches a glimpse of how it works, here we further break down the execution:
```bash
$ ./my \                                # Invocation of Mayo
    models/lenet5.yaml \                # Imports LeNet-5 network description
    datasets/lenet5.yaml \              # Imports the dataset description
    system.checkpoint.load=pretrained \ # Specify that we load the pretrained checkpoint
    eval                                # Starts model evaluation
```
The Mayo command line interface accepts a sequence of actions separated by space to be evaluated sequentially.  Each YAML import and key-value pair update recursively merges all mappings in the YAML file with the global configuration.  So right before `eval`, we would have a complete application description specifying the model, dataset and checkpoint used.


## Why so many YAML files?

In Mayo, we decouple the description of each neural network application into three separate components: the dataset, model and trainer, each written in [YAML][yaml].  The reason for decoupling is to encourage reuse, for instance a ResNet-50 model can not only be used for [ImageNet][imagenet] classification, but also object detection with [COCO][coco], or even customized tasks.

Furthermore, in network compression, we can use many fine- and coarse-grained pruning techniques in conjunction with a large range of quantization methods, even optionally on top of low-rank approximation of weight tensors, on a wide variety of neural networks, each could use different datasets and could be trained differently.  We now encounter a vast number of possible combinations of all of these above options, so by decoupling compression techniques from the neural network, from the dataset, from training methodologies, all possible combinations can be achieved by importing the respective YAML descriptions, without having to write a monolithic description file for each combination.


## Results

**TODO** complete this section.


[fine]: models/override/prune/dns.yaml
[slim]: models/override/prune/netslim.yaml
[quantize]: models/override/quantize/
[git]: https://git-scm.com
[git-lfs]: https://git-lfs.github.com
[python3]: https://www.python.org/downloads/
[tensorflow]: https://www.tensorflow.org/install/
[tfrecord]: https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format
[yaml]: http://yaml.org
[imagenet]: http://www.image-net.org
[coco]: http://cocodataset.org
[mayo-yaml]: https://github.com/admk/mayo/wiki/Writing-YAMLs

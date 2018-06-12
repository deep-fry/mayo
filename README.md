# Mayo

**Mayo** is a deep learning framework developed with hardware acceleration in mind. It enables rapid deep neural network model design, and common network compression work flows such as [fine][fine]- and coarse-grained pruning, [network slimming][slim], and [quantization methods with various arithmetics][quantize] can be easily used jointly by writing simple YAML model description files (exemplified by the links above).  Additionally, Mayo can also accelerate the hyperparameter exploration of these models with automated hyperparameter optimization.  With minimal manual intervention, our automated optimization results achieve state-of-the-art compression rates (refer to the results section).

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


### Testing Mayo

Run a simple LeNet-5 validation with the MNIST dataset using:
```bash
$ ./my models/lenet5.yaml datasets/lenet5.yaml eval
```
You should expect the final top-1 and top-5 accuracies to be 99.57% and 100% respectively.


## Writing a neural network model in YAML


## Results

**TODO** complete this section.

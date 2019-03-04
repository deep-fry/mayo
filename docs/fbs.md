
# Feature Boosting and Suppression

## Description

Feature Boosting and Suppression (FBS) is a method
that exploits run-time dynamic information flow in CNNs to
dynamically prune channel-wise parameters.

![pdf not found](fbs.png)

Intuitively, we can imagine that the flow of information of
each output channel can be amplified or restricted under
the control of a “valve”.
This allows salient information to
flow freely while we stop all information
from unimportant channels and skip their computation.
Unlike pruning statically, the valves use features from the previous layer
to predict the saliency of output channels.
FBS introduces tiny auxiliary connections to existing convolutional layers.
The minimal overhead added to the existing model is thus negligible when
compared to the potential speed up provided by the dynamic sparsity.

## To Run

The major codebase for FBS resides in `mayo/tf/gate/layers.py`.

* Training with FBS in Mayo

  Please read on docs regarding the mayo command line interface before
  training your model.
  We provide a set of models in `models/gate`
  and trainers in `trainers`.
  These can serve as reference YAML files for you to implement your
  own models.

  ```bash
  $ ./my \
      your_model.yaml \
      datasets/imagenet.yaml \
      your_trainar.yaml \
      train
  ```

* Example 1: FBS on CifarNet (Imagenet)

  We show the effectiveness of FBS on CifarNet.

  First, we estimate the Multiply-Accumulates (MACs) in a dense pretrained
  model.

  ```bash
  $ ./my \
      models/cifarnet.yaml
      datasets/cifar10.yaml \
      system.checkpoint.load=pretrained \
      eval info
  ```

  This generates us a model with `top1 91.37%, top5: 99.68%` using
  174M MACs.

  ```bash
  $ ./my \
      models/gate/cifarnet.yaml
      datasets/cifar10.yaml \
      _gate.density=0.5 \
      system.checkpoint.load=gate50 \
      eval info
  ```

  We then estimate the FBS generated model with a gating probability of 0.4.
  The model achieves `top1: 90.54%, top5: 99.75%` using 57M MACs, giving
  us a 3x reduction in MACs and within 1% loss of accuracy.

* Example 2: FBS on slimed VGG16 (Imagenet)

  FBS can act on top of traditional channel-wise pruning, we demonstrate this
  by providing an FBSed Vgg16 model that has also been slimed using
  [Network Slimming](https://arxiv.org/abs/1708.06519).

  ```bash
  $ ./my \
      models/gate/vgg16_compact.yaml \
      datasets/imagenet.yaml \
      _gate.density=0.6 \
      system.checkpoint.load=slimed_gate_60 \
      eval info
  ```

  The model reports `top1: 69.59%, top5: 89.39%`, and an estimated
  mac count of 3,379,002,273.

  The example models can be downloaded [here](https://universityofcambridgecloud-my.sharepoint.com/:f:/g/personal/yaz21_cam_ac_uk/Es4FrWNmJe1ImBgR_T1PyoUB24rlUrSVtOrA1NAaDrYxXg?e=2uZEO6).

## Citation

```bibtex
@inproceedings{gao2018dynamic,
  title={Dynamic Channel Pruning: Feature Boosting and Suppression},
  author={Xitong Gao and Yiren Zhao and Łukasz Dudziak and Robert Mullins and Cheng-zhong Xu},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=BJxh2j0qYm},
}
```

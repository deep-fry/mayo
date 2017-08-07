from mayo.net import Net
from mayo.config import Config


config = """
name: mobilenet_v1
dataset:
    name: mnist
    num_classes: 10
    num_examples_per_epoch: {train: 58001, validation: 1000, test: 9001}
    batch_size: 256
    height: 28
    width: 28
    channels: 1
train:
    num_epochs_per_decay: 30.0
    initial_learning_rate: 0.1
    learning_rate_decay_factor: 0.16
    num_preprocess_threads: 1
    num_gpus: 4
    optimizer:
        type: RMSPropOptimizer
        decay: 0.9
        momentum: 0.9
        epsilon: 1.0
default:
    convolution:
        padding: same
        batch_norm: true
    depthwise_separable_convolution:
        padding: same
        batch_norm: true
        depth_multiplier: 1
logits: logits
net:
    - name: conv0
      type: convolution
      params: {kernel_size: [3, 3], stride: 2, num_outputs: 32}
    - name: conv1
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 1, num_outputs: 64}
    - name: conv2
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 2, num_outputs: 128}
    - name: conv3
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 1, num_outputs: 128}
    - name: conv4
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 2, num_outputs: 256}
    - name: conv5
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 1, num_outputs: 256}
    - name: conv6
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 2, num_outputs: 512}
    - name: conv7
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 1, num_outputs: 512}
    - name: conv8
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 2, num_outputs: 1024}
    - name: conv9
      type: depthwise_separable_convolution
      params: {kernel_size: [3, 3], stride: 1, num_outputs: 1024}
    - name: pool
      type: average_pool
      params: {kernel_size: [7, 7], stride: 2, padding: valid}
    - name: dropout
      type: dropout
      params: {keep_prob: 0.999}
    - name: fc
      type: convolution
      params:
          kernel_size: [1, 1]
          num_outputs: 10
          activation_fn: null
          normalizer_fn: null
    - name: logits
      type: squeeze
      params: {axis: [1, 2]}
"""
config = Config(config)
n = Net(config)
n.save_graph()

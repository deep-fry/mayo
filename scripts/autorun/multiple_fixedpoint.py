import os
import subprocess
import itertools


def main():
    configdir = 'configs'
    storedir = 'checkpoints/fixedpoint'
    max_epochs = 1
    bitwidths = reversed([4, 8, 16, 32])
    wa_width, gwidth = combinations(bitwidths)
    points = [2] * len(wa_width)
    if not os.path.exists(configdir):
        os.mkdir(configdir)
    if not os.path.exists(storedir):
        os.mkdir(storedir)
    for p, b, g in zip(points, wa_width, gwidth):
        print('Generate a custom yaml at {}'.format(configdir))
        name = generate_yaml(b, p, g, p, configdir)
        print('Training starts for {} bits'.format(b))
        lenet_command = './my datasets/mnist.yaml {} models/lenet5.yaml trainers/lenet5.yaml system.checkpoint.load=null system.max_epochs={} reset-num-epochs train'.format(
            name, max_epochs)
        cifar_command = ('./my datasets/cifar10.yaml {} models/cifarnet.yaml trainers/cifarnet.yaml system.checkpoint.load=null system.checkpoint.save.interval=10 system.num_gpus=4 system.batch_size_per_gpu=1024 system.max_epochs={} reset-num-epochs train'.format(
            name, max_epochs))
        subprocess.run(cifar_command, shell=True)
        print('Training done')
        storebit_dir = '{}/{}bit'.format(storedir, str(b))
        if not os.path.exists(storebit_dir):
            os.mkdir(storebit_dir)
        subprocess.run('mv checkpoints/cifarnet/cifar10/checkpoint-{}.* {}'.format(
            max_epochs, storebit_dir), shell=True)


def combinations(x):
    comb_list = list(itertools.combinations(bitwidths, 2))
    list1, list2 = zip(*comb_list)
    return (list(list1), list(list2))


def generate_yaml(bitwidth=8, point=2, gwidth=8, gpoint=2, filename='configs'):
    yaml_str = """\
---
model.layers:
    conv0: &overrider
        weights_overrider: &quantizer
            type: mayo.override.FixedPointQuantizer
            width: {0}
            point: {1}
            should_update: true
        biases_overrider: *quantizer
        activation_overrider: *quantizer
        gradient_overrider:
            type: mayo.override.FixedPointQuantizer
            width: {2}
            point: {3}
            should_update: true
    conv1: {{<<: *overrider}}
    conv2: {{<<: *overrider}}
    conv3: {{<<: *overrider}}
    conv4: {{<<: *overrider}}
    conv5: {{<<: *overrider}}
    conv6: {{<<: *overrider}}
    conv7: {{<<: *overrider}}
    logits: {{<<: *overrider}}""".format(bitwidth, point, gwidth, gpoint)
    name = filename + '/' + 'custom{}.yaml'.format(bitwidth)
    with open(name, 'w') as f:
        f.write(yaml_str)
    return name


if __name__ == "__main__":
    bitwidths = reversed([4, 8, 16, 32])
    main()

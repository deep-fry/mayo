import os
import subprocess


def main():
    filename = 'configs'
    storedir = 'checkpoints/fixedpoint'
    max_epochs = 1
    bitwidths = [4, 8, 16, 32]
    points = [2, 2, 2, 2]
    if not os.path.exists(filename):
        os.mkdir(filename)
    if not os.path.exists(storedir):
        os.mkdir(storedir)
    for p, b in zip(points, bitwidths):
        name = generate_yaml(b, p, filename)
        my_command = './my datasets/mnist.yaml {} models/lenet5.yaml trainers/lenet5.yaml system.checkpoint.load=null system.max_epochs={} reset-num-epochs train'.format(name, max_epochs)
        subprocess.run(my_command, shell=True)
        print('training done')
        subprocess.run('mv checkpoints/lenet5/mnist/checkpoint-{}.* {}'.format(max_epochs, storedir), shell=True)


def generate_yaml(bitwidth=8, point=2, filename='configs'):
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
        gradient_overrider: *quantizer
    conv1: {{<<: *overrider}}
    conv2: {{<<: *overrider}}
    conv3: {{<<: *overrider}}
    conv4: {{<<: *overrider}}
    conv5: {{<<: *overrider}}
    conv6: {{<<: *overrider}}
    conv7: {{<<: *overrider}}
    logits: {{<<: *overrider}}""".format(bitwidth, point)
    name = filename + '/' + 'custom{}.yaml'.format(bitwidth)
    with open(name, 'w') as f:
        f.write(yaml_str)
    return name


if __name__ == "__main__":
    main()

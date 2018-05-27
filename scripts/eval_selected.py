import os
import subprocess
import itertools
import pickle


def main():
    checkpoints = [
        'slimed/slimed_{}'.format(percent) for percent in range(20, 100, 10)]
    # for testing
    # checkpoints = [
    #     'checkpoint-{}'.format(percent) for percent in range(5, 8)]
    raw_accs = []
    print('Evaluation starts for {}'.format(checkpoints))
    for checkpoint in checkpoints:
        lenet_command = './my datasets/mnist.yaml models/lenet5.yaml trainers/lenet5.yaml system.checkpoint.load={} reset-num-epochs eval'.format(
            checkpoint)
        vgg_command = 'LD_PRELOAD=/usr/lib/libtcmalloc.so.4 ./my datasets/imagenet.yaml models/override/vgg16_bn.yaml models/override/prune/netslim.yaml system.num_gpus=2 system.visible_gpus=[0,1] trainers/vgg16.yaml system.checkpoint.load={} system.num_gpus=1 system.visible_gpus=[2,3] system.checkpoint.save.interval=1 system.batch_size_per_gpu=64 system.preprocess.num_threads=16 train.learning_rate._initial=0.01 train.learning_rate.decay_steps=10 reset-num-epochs eval'.format(checkpoint)
        p = subprocess.Popen(
            vgg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        output = str(output)
        raw_accs.append(output.split('Evaluation complete', 1)[-1])
    print('Evaluations done')
    with open('eval_results.pkl', 'wb') as f:
        pickle.dump(raw_accs, f)


if __name__ == "__main__":
    main()

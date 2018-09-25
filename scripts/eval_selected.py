import os
import subprocess
import itertools
import pickle


def main():
    raw_accs = []
    print('Evaluation starts')
    network = 'alexnet'
    network = 'mobilenet'
    dataset = 'imagenet'
    ckpt = 11
    extra_setup = 'system.num_gpus=2 system.visible_gpus=[2,3]'
    info_dir = '{}info/'.format(network)
    outputs = []
    for bitwidth in [2, 4, 8, 16]:
        command = './my datasets/{}.yaml models/override/{}.yaml models/override/quantize/shift_sparse.yaml trainers/cifarnet.yaml system.checkpoint.load={} system.info.plumbing=true {} eval'.format(dataset, network, ckpt, extra_setup)
        if network == 'mobilenet':
            subprocess.run(
                'cp checkpoints/{0}_incre_shift/{2}bit/checkpoint-{3}.* checkpoints/{0}_v1/{1}/'.format(network, dataset, bitwidth, ckpt), shell=True)
        else:
            subprocess.run(
                'cp checkpoints/{0}_incre_shift/{2}bit/checkpoint-{3}.* checkpoints/{0}/{1}/'.format(network, dataset, bitwidth, ckpt), shell=True)
        p = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        output = str(output)
        raw_accs.append(output.split('Evaluation complete', 1)[-1])
        if not os.path.exists(info_dir):
            os.mkdir(info_dir)
        info_bit_dir = info_dir + '{}bit/'.format(bitwidth)
        if not os.path.exists(info_bit_dir):
            os.mkdir(info_bit_dir)
        subprocess.run(
            'cp info.yaml {}'.format(info_bit_dir), shell=True)
        outputs.append(output)
    import pdb; pdb.set_trace()
    print('Evaluations done')


if __name__ == "__main__":
    main()

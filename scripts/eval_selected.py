import os
import subprocess
import itertools
import pickle


def main():
    raw_accs = []
    print('Evaluation starts')
    for bitwidth in [2, 4, 6, 8, 10, 12, 14, 16]:
        cifar_command = './my datasets/cifar10.yaml models/override/cifarnet.yaml models/override/quantize/shift_sparse.yaml trainers/cifarnet.yaml system.checkpoint.load=300 system.info.plumbing=true eval info'
        subprocess.run(
            'cp checkpoints/cifar_incre_shift/{}bit/checkpoint-300.* checkpoints/cifarnet/cifar10/'.format(bitwidth), shell=True)
        p = subprocess.Popen(
            cifar_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        output = str(output)
        raw_accs.append(output.split('Evaluation complete', 1)[-1])
    print('Evaluations done')
    import pdb; pdb.set_trace()
    with open('eval_results.pkl', 'wb') as f:
        pickle.dump(raw_accs, f)


if __name__ == "__main__":
    main()

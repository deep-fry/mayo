import tensorflow as tf

# mac_count(self.nets[0].overriders, self.nets[0])
def mac_count(overriders, net):
    """
    Now your overriders are numpy arrays
    """
    def density():
        pass

    def extract_macs(estimates):
        new_estimates = {}
        for key, item in estimates.items():
            if 'MACs' in item.keys():
                new_estimates[key.name] = item['MACs']
        return new_estimates

    def extract_densities(estimates):
        new_estimates = {}
        for key, item in estimates.items():
            print(key, item)
            if 'MACs' in item.keys():
                if 'density' in item.keys():
                    new_estimates[key.name] = item['density']
                else:
                    # indeed 100% if no density, right?
                    new_estimates[key.name] = 1.0
        return new_estimates
    import numpy as np
    layer_names = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4',
                   'conv5', 'conv6', 'conv7', 'logits']
    estimates = net.session.estimator.get_estimates(net)
    densities = extract_densities(estimates)
    slimed_macs = extract_macs(estimates)
    # lets assume things are ordered
    macs = {}
    total_macs = 0
    prev_name = None
    for index, name in enumerate(layer_names):
        if index == 0:
            macs[name] = slimed_macs[name]
            prev_name = name
        else:
            macs[name] = slimed_macs[name] * densities[prev_name]
            prev_name = name
        total_macs += macs[name]
    return (macs, total_macs)

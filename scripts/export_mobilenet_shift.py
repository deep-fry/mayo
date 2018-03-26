def mobilenet_shift(overriders, variables):
    before = {}
    after = {}
    useful_info = {}
    bn_avgs = {}
    for overrider in overriders:
        name = overrider.name
        if 'weights' in name or 'biases' in name:
            before[name] = overrider.before.eval()
            after[name] = overrider.after.eval()
            useful_info[name] = {}
            if hasattr(overrider, 'quantizer'):
                useful_info['exp_bias'] = \
                    overrider.quantizer.exponent_bias.eval()
    for variable in variables:
        if 'moving' in variable.name:
            bn_avgs[variable.name] = variable.eval()
    raw = (before, after, useful_info, bn_avgs)
    STORE = True
    if STORE:
        import pickle
        with open('mobilenet.pkl', 'wb') as f:
            pickle.dump(raw, f)
    return raw

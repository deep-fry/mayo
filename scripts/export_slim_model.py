import tensorflow as tf


def reform_weights(overriders):
    """
    Now your overriders are numpy arrays
    """
    def netslim_convert(overriders):
        overriders_dict = {}
        mask_dict = {}
        for overrider in overriders:
            value = overrider.after.eval()
            if 'conv5_3' in overrider.name:
                mask = overrider.mask.eval()
                mask = np.ones(mask.shape)
            else:
                mask = overrider.mask.eval()
            name = overrider.name
            overriders_dict[name] = [value, mask]
            mask_dict[name] = mask
        return (overriders_dict, mask_dict)

    def pick_weight(target, trainables):
        for name, trainable in trainables.items():
            if target in name and 'weights' in name:
                return trainable
        raise ValueError('trainable not found with name {}'.format(name))

    def pick_mask(target, masks):
        for name, mask in masks.items():
            if target in name:
                return mask
        raise ValueError('mask not found with name {}'.format(target))

    # compress_weight_test(test, test_mask, 'out')
    def compress_weight_test(weights, masks, index_type='in'):
        import numpy as np
        shape = weights.shape
        dummy_index = 0
        if index_type == 'out':
            new_weights = np.zeros((shape[0], shape[1], shape[2], sum(masks)))
            for i, mask in enumerate(masks):
                if mask:
                    new_weights[:, :, :, dummy_index] = weights[:, :, :, i]
                    dummy_index += 1
        if index_type == 'in':
            new_weights = np.zeros((shape[0], shape[1], sum(masks), shape[3]))
            for i, mask in enumerate(masks):
                if mask:
                    new_weights[:, :, dummy_index, :] = weights[:, :, i, :]
                    dummy_index += 1
        return new_weights

    import numpy as np
    model = 'VGG'
    if model == 'Cifar':
        layer_names = ['conv0', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
            'conv6', 'conv7', 'logits']
    elif model == 'VGG':
        layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
            'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1',
            'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    trainables = tf.trainable_variables()
    trainable_names = [trainable.name for trainable in trainables]
    slimed_weights = {}
    prev_mask = None
    np_trainables = {}
    _, mask_dict = netslim_convert(overriders)
    for name, trainable in zip(trainable_names, trainables):
        np_trainables[name] = trainable.eval()
    for name in layer_names:
        weight = pick_weight(name, np_trainables)
        if 'fc' in name:
            slimed_weights[name] = weight
            continue
        if name != 'logits':
            mask = pick_mask(name, mask_dict)
        if prev_mask is None:
            weight = compress_weight_test(weight, mask, 'out')
        else:
            if name == 'logits':
                weight = compress_weight_test(weight, prev_mask, 'in')
            else:
                weight = compress_weight_test(weight, prev_mask, 'in')
                if 'conv5_3' not in name:
                    weight = compress_weight_test(weight, mask, 'out')
        prev_mask = mask
        slimed_weights[name] = weight
    np_raw = {}
    for variable in trainables:
        if 'weights' in variable.name:
            for name, weights in slimed_weights.items():
                print(name, variable.name)
                if name in variable.name:
                    np_raw[variable.name] = weights
        elif 'biases' in variable.name:
            np_raw[variable.name] = variable.eval()
        else:
            for name, mask in mask_dict.items():
                name = name.split('/')
                if name[2] in variable.name:
                    np_raw[variable.name] = variable.eval()[mask == 1]
    for variable in tf.global_variables():
        if 'moving' in variable.name:
            for name, mask in mask_dict.items():
                name = name.split('/')
                if name[2] in variable.name:
                    np_raw[variable.name] = variable.eval()[mask == 1]
    STORE = True
    if STORE:
        import pickle
        with open('test2.pkl', 'wb') as f:
            pickle.dump((np_raw, np_trainables, mask_dict), f)
        with open('test.pkl', 'wb') as f:
            pickle.dump(np_raw, f)
    return (np_raw, np_trainables, mask_dict)
# reform_weights(self.nets[0].overriders)

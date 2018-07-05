def mobilenet_shift(base):
    save_dir = '../ShiftMobileNet/tf_src/model_data/'
    meta = {}
    overriders = base.task.nets[0].overriders
    meta['input'] = base.task.nets[0].inputs()['input']
    for node, overrider_dict in overriders.items():
        name = node.formatted_name()
        meta[name] = {}
        for target_name, overrider in overrider_dict.items():
            meta[name][target_name]={
                'before': overrider.before,
                'after': overrider.after,
            }
            print(target_name)
            if 'weight' in target_name or 'bias' in target_name:
                meta[name][target_name]['bias'] = \
                    overrider.quantizer.exponent_bias
                meta[name][target_name]['width'] = \
                    overrider.quantizer.width

            else:
                meta[name][target_name]['point'] = overrider.point
                meta[name][target_name]['width'] = overrider.width
        for variable in base.global_variables():
            if 'moving' in variable.name and name in variable.name:
                meta[name][variable.name] = variable
    meta = base.run(meta)
    STORE = True
    if STORE:
        import pickle
        save_dir += 'mobilenet.pkl'
        with open(save_dir, 'wb') as f:
            pickle.dump(meta, f)
    return meta

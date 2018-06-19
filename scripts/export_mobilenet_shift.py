def mobilenet_shift(overriders, global_vars):
    save_dir = '/local/scratch-3/yaz21/tmp/'
    meta = {}
    for node, overrider_dict in overriders.items():
        name = node.formatted_name()
        meta[name] = {}
        for target_name, overrider in overrider_dict.items():
            meta[name][target_name]={
                'before': overrider.before.eval(),
                'after': overrider.after.eval(),
            }
        for variable in global_vars:
            if 'moving' in variable.name:
                meta[name][variable.name] = variable.eval()
    STORE = True
    if STORE:
        import pickle
        save_dir += 'mobilenet.pkl'
        with open(save_dir, 'wb') as f:
            pickle.dump(meta, f)
    return meta 
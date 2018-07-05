def mobilenet_shift(base, global_vars):
    save_dir = '/local/scratch-3/yaz21/tmp/'
    save_dir = './'
    meta = {}
    overriders = base.task.nets[0].overriders
    meta['input'] = base.task.nets[0].inputs()['input'].eval()
    for node, overrider_dict in overriders.items():
        name = node.formatted_name()
        meta[name] = {}
        for target_name, overrider in overrider_dict.items():
            meta[name][target_name]={
                'before': overrider.before.eval(),
                'after': overrider.after.eval(),
            }
            print(target_name)
            if 'weight' in target_name or 'bias' in target_name:
                meta[name][target_name]['bias'] = \
                    overrider.quantizer.exponent_bias.eval()
                meta[name][target_name]['width'] = \
                    overrider.quantizer.width.eval()

            else:
                meta[name][target_name]['point'] = overrider.point.eval()
                meta[name][target_name]['width'] = overrider.width.eval()
        for variable in global_vars:
            if 'moving' in variable.name and name in variable.name:
                meta[name][variable.name] = variable.eval()
    STORE = True
    if STORE:
        import pickle
        save_dir += 'mobilenet.pkl'
        with open(save_dir, 'wb') as f:
            pickle.dump(meta, f)
    return meta 
# scp yaz21@heimdall.cl.cam.ac.uk:/local/scratch-3/yaz21/tmp/mobilenet.pkl .
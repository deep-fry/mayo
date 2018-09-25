def export(self, pickle_name=None):
    overriders = self.task.nets[0].overriders
    meta = {}
    for node, overrider in overriders.items():
        meta[node.name] = {}
        for component, suboverrider in overrider.items():
            meta[node.name][component] = {}
            # handel gradients
            if isinstance(suboverrider, dict):
                for subcomponent, suboverrider2 in suboverrider.items():
                    meta[node.name][component][subcomponent] = {}
                    meta[node.name][component][subcomponent]['before'] = suboverrider2.before.eval()
                    meta[node.name][component][subcomponent]['after'] = suboverrider2.after.eval()
                    meta[node.name][component][subcomponent]['point'] = suboverrider2.point.eval()
            else:
                meta[node.name][component]['before'] = suboverrider.before.eval()
                meta[node.name][component]['after'] = suboverrider.after.eval()
                meta[node.name][component]['point'] = suboverrider.point.eval()
    if pickle_name is not None:
        import pickle
        pickle.dump(meta, open(pickle_name, 'wb'))
    return meta


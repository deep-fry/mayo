import tensorflow as tf

from mayo.log import log
from mayo.session.train import Train
from mayo.util import Table


class Profile(Train):
    def profile(self):
        log.debug('Profiling starts.')
        try:
            self.profile_multi_epochs()
        except KeyboardInterrupt:
            log.info('Stopped.')
            save = self.config.system.checkpoint.get('save', {})
            if save:
                countdown = save.get('countdown', 0)
                if log.countdown('Saving checkpoint', countdown):
                    self.save_checkpoint('latest')

    def _run(self, max_epochs, reset=True):
        log.info('Start profiling ...')
        self.config.system.checkpoint.save = False
        # reset num_epochs and stop at 1 epoch
        if reset:
            self.reset_num_epochs()
        # start training
        self.train(max_epochs=max_epochs)

    def profile_multi_epochs(self):
        print('Profile progressing ...')
        config = self.config.profile
        overriders = self.task.nets[0].overriders
        name_to_rules = config.parameters.overriders

        export_ckpt = config.pop('export_ckpt', False)
        num_epochs = config.pop('num_epochs', 0.0)
        # macs = self.task.nets[0].estimate()
        # priority_ranks = [(key, macs[key]) for key, o in overriders.items()]
        # priority_ranks = sorted(
        #     priority_ranks, key=lambda x:x[1]['macs'], reverse=True)
        target_values = {}
        if not num_epochs:
            # for key, _ in priority_ranks:
            #     for o in overriders[key]:
            for o, key in self.generate_overriders(overriders, prod_key=True):                
                target_values[o] = {}
                o.update()
                for keyword, rules in name_to_rules.items():
                    if keyword == type(o).__name__:
                        for target in rules.targets:
                            target_values[o][target] = self.run(
                                getattr(o, target))
        else:
            target_values = self.profiled_search(overriders, name_to_rules)
        self.present(overriders, target_values, export_ckpt)
        return False

    def profiled_search(self, overriders, rules):
        # decide to train or not
        profile_params = self.config.profile.parameters
        self._run_train_ops = True

        # empty run to speed to warm up
        for o, key in self.generate_overriders(overriders, prod_key=True):
            o.enable = False
        self._run(max_epochs=profile_params.profile.start)
        # lets profile the values
        self.register_values(
            overriders, samples=profile_params.samples,
            rules=rules)
        self._run(max_epochs=profile_params.profile.end, reset=False)
        meta_params = {}
        targets = {}
        for variable, o, key in self.generate_overriders(
                overriders, prod_key=True, label_o=True):
            # construct after, overrde again
            params = {}
            avg = self.estimator.get_value('avg_' + o.name, node=key)
            max_val = self.estimator.get_value('max_' + o.name, node=key)
            params['max'] = max_val[0]
            params['avg'] = avg
            params['targets'] = \
                profile_params.overriders.get(type(o).__name__).targets
            meta_params[o.name] = params
            # find a target -> suggested value dict
            target = o.search(params)
            # map this dict accroding to overriders
            targets[o] = [target, variable]
        return targets

    def register_values(
            self, overriders, reg_avg=True, reg_max=True, samples=10,
            rules=None):
        for variable, o, key in self.generate_overriders(
                overriders, prod_key=True, label_o=True):
            name = type(o).__name__
            if reg_avg:
                self.estimator.register(
                    o.before, 'avg_' + o.name, node=key,
                    history='running_mean')
            if reg_max:
                p_dict = rules[name].percentile
                # if isinstance(percentile, dict):
                if isinstance(p_dict, (int, float)):
                    percentile = p_dict
                else:
                    default_percentile = p_dict.get('default', 99)
                    if 'gradient' in variable:
                        percentile = p_dict.get(
                            'gradients', default_percentile)
                    if 'weights' in variable:
                        percentile = p_dict.get('weights', default_percentile)
                    if 'biases' in variable:
                        percentile = p_dict.get('biases', default_percentile)
                    if 'activation' in variable:
                        percentile = p_dict.get(
                            'activations', default_percentile)
                    else:
                        percentile = default_percentile
                percentile = tf.contrib.distributions.percentile(
                    tf.abs(o.before), percentile)
                self.estimator.register(
                    percentile, 'max_' + o.name, node=key,
                    history='running_mean')
        return

    def present(self, overriders, target_values, export_ckpt):
        table = Table(['variable', 'name', 'suggested value'])
        for o in self.generate_overriders(overriders):
            name = o.name
            if len(name) > 4:
                name = o.name.split('/')
                name = '/'.join(name[-4:])
            table.add_row((
                name, target_values[o][1], target_values[o][0]))
        print(table.format())
        if export_ckpt:
            self._assign_targets(overriders, target_values)
            model_name = self.config.model.name
            model_name += '_profiled'
            self.save_checkpoint(model_name)
        return

    def _assign_targets(self, overriders, target_values):
        assignment_ops = []
        for node, name_to_overrider in overriders.items():
            for name, overrider in name_to_overrider.items():
                if name == 'gradient':
                    for g_name, g_overrider in overrider.items():
                        g_overrider.enable = True
                        target = target_values[g_overrider][0]
                        for target_name, target_value in target.items():
                            setattr(g_overrider, target_name, target_value)
                    continue
                overrider.enable = True
                target = target_values[overrider][0]
                for target_name, target_value in target.items():
                    setattr(overrider, target_name, target_value)
                    op = getattr(overrider, target_name)
                    assignment_ops.append(op)
        # load the values
        _ = self.run(assignment_ops)

    def generate_overriders(self, overriders, prod_key=False, label_o=False):
        for key, os in overriders.items():
            for variable, o in os.items():
                if isinstance(o, dict):
                    for grad_variable, grad_o in o.items():
                        if prod_key:
                            if label_o:
                                yield [grad_variable, grad_o, key]
                            else:
                                yield (grad_o, key)
                        else:
                            if label_o:
                                yield (grad_variable, grad_o)
                            else:
                                yield grad_o
                else:
                    if prod_key:
                        if label_o:
                            yield [variable, o, key]
                        else:
                            yield (o, key)
                    else:
                        if label_o:
                            yield (variable, o)
                        else:
                            yield o
import os
import pickle
import tensorflow as tf

from mayo.log import log
from mayo.session.train import Train


class ProfileBase(Train):
    modes = ['one_shot', 'one_epoch', 'fine_tune']

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

    def _init_profile(self):
        self.search_simple()
        search_func = getattr(self, 'search_simple')
        search_func(self.search_mode)
        return False

    def profile_multi_epochs(self):
        print('Search progressing ...')
        config = self.config.profile
        overriders = self.task.nets[0].overriders
        name_to_rules = config.parameters.overriders

        # training = config.pop('training', False)
        export_ckpt = config.pop('export_ckpt', False)
        num_epochs = config.pop('num_epochs', 0.0)
        macs = self.task.nets[0].estimate()
        priority_ranks = [(key, macs[key]) for key, o in overriders.items()]
        priority_ranks = sorted(
            priority_ranks, key=lambda x:x[1]['macs'], reverse=True)
        target_values = {}
        if not num_epochs:
            for key, _ in priority_ranks:
                for o in overriders[key]:
                    target_values[o] = {}
                    o.update()
                    for keyword, rules in name_to_rules.items():
                        if keyword == type(o).__name__:
                            for target in rules.targets:
                                target_values[o][target] = self.run(
                                    getattr(o, target))
        else:
            target_values = self.profiled_search(
                config.training, overriders, name_to_rules)
        self.present(overriders, target_values, export_ckpt)
        return False

    def profiled_search(self, training, overriders, rules):
        # decide to train or not
        search_params = self.config.search.parameters
        self._run_train_ops = training

        self.config.system.max_epochs = search_params.profile.start
        # empty run to speed to warm up
        for o, key in self.generate_overriders(overriders, prod_key=True):
            o.enable = False
            o.width = 8
        self.run()
        # lets profile the values
        self.register_values(
            overriders, samples=search_params.samples,
            rules=rules)
        self.config.system.max_epochs = search_params.profile.end
        self.run(reset=False)
        meta_params = {}
        targets = {}
        for o, key in self.generate_overriders(overriders, prod_key=True):
            # construct after, overrde again
            params = {}
            avg = self.estimator.get_value('avg_' + o.name, node=key)
            max_val = self.estimator.get_value('max_' + o.name, node=key)
            params['max'] = max_val[0]
            params['avg'] = avg
            params['targets'] = \
                search_params.overriders.get(type(o).__name__).targets
            params['samples'] = self.estimator.get_value(o.name, node=key)
            meta_params[o.name] = params
            # find a target -> suggested value dict
            target = o.search(params)
            # map this dict accroding to overriders
            targets[o] = target
        return targets

    def register_values(
            self, overriders, reg_avg=True, reg_max=True, samples=10,
            rules=None):
        for o, key in self.generate_overriders(overriders, prod_key=True):
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
                    if 'gradients' in o.name:
                        percentile = p_dict.get(
                            'gradients', default_percentile)
                    if 'weights' in o.name:
                        percentile = p_dict.get('weights', default_percentile)
                    if 'biases' in o.name:
                        percentile = p_dict.get('biases', default_percentile)
                    else:
                        percentile = default_percentile
                percentile = tf.contrib.distributions.percentile(
                    tf.abs(o.before), percentile)
                self.estimator.register(
                    percentile, 'max_' + o.name, node=key,
                    history='running_mean')
            self.estimator.register(
                o.before, o.name, node=key, history=samples)
        return

    def present(self, overriders, target_values, export_ckpt):
        table = Table(['variable', 'suggested value'])
        for o in self.generate_overriders(overriders):
            name = o.name
            if len(name) > 4:
                name = o.name.split('/')
                name = '/'.join(name[-4:])
            table.add_row((
                name, target_values[o]))
        print(table.format())
        if export_ckpt:
            model_name = self.config.model.name
            model_name += '_profile_' + self.config.search.search_mode
            self.save_checkpoint(model_name)
        return

    def run(self, reset=True):
        log.info('Start profiling ...')
        self.config.system.checkpoint.save = False
        # reset num_epochs and stop at 1 epoch
        if reset:
            self.reset_num_epochs()
        # start training
        self.train()
        # save profiling results
        self.info()
        self.save()

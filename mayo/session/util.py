from mayo.log import log

class target(object):
    def __init__(self, target_var, associate_var, session):
        

class Info(object):
    def __init__(self, meta_info, session, targeting_vars, associated_vars,
        run_status):
        self.scales = {}
        self.min_scales = {}
        self.scale_update_factors = {}
        self.start_ths = {}
        self.ths = {}
        self.max_ths = {}

        # now only supports retraining on one overrider
        self.meta = meta_info
        self.targeting_vars = targeting_vars
        self.associated_vars = associated_vars

        for target in targeting_vars:
            name = target.name
            self.scales[name] = meta_info.range['scale']
            self.min_scales[name] = meta_info.range['min_scale']
            self.scale_update_factors[name] = \
                meta_info.range['scale_update_factor']

            if run_status == 'continue':
                th = session.run(target)
                self.ths[name] = th
                self.start_ths[name] = th
                log.info('{} is continuing on {}.'.format(name, th))
            else:
                self.ths[name] = meta_info.range['from']
                self.start_ths[name] = meta_info.range['from']
            self.max_ths[name] = meta_info.range['to']
        if self.scales == {}:
            raise ValueError(
                '{} is not found in overrider definitions, '
                'but has been specified as a target.'.format(meta_info.type))

    def get(self, variable, info_type):
        name = variable.name
        if info_type == 'end_threshold':
            return self.max_ths[name]
        if info_type == 'end_scale':
            return self.min_scales[name]
        if info_type == 'threshold':
            return self.ths[name]
        if info_type == 'start_threshold':
            return self.start_ths[name]
        if info_type == 'scale':
            return self.scales[name]
        if info_type == 'scale_factor':
            return self.scale_update_factors[name]
        raise ValueError('{} is not a collected info key.'.format(info_type))

    def set(self, variable, info_type, value):
        name = variable.name
        if info_type == 'threshold':
            self.ths[name] = value
        elif info_type == 'scale':
            self.scales[name] = value
        else:
            raise ValueError(
                '{} is not a collected info key.'.format(info_type))

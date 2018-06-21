import collections

import tensorflow as tf

from mayo.log import log
from mayo.util import Table, object_from_params, unknown, format_shape
from mayo.override import ChainOverrider
from mayo.net.base import LayerNode, JoinNode, NetBase
from mayo.net.tf.transform import ParameterTransformer


class TFNetBase(NetBase):
    """Instantiates a TensorFlow network from the DAG graph.  """
    def __init__(self, session, model, inputs, reuse):
        self.session = session
        self.estimator = self.session.estimator
        self.is_training = session.is_training
        self._transformer = ParameterTransformer(session, reuse)
        super().__init__(model, inputs)
        self._verify_io()

    def _verify_io(self):
        nodes = list(self._graph.input_nodes())
        if len(nodes) != 1 and nodes[0].name != 'input':
            raise ValueError(
                'We expect the graph to have a unique data input named '
                '"input", found {!r}.'.format(nodes))
        nodes = list(self._graph.output_nodes())
        if len(nodes) != 1 or nodes[0].name != 'output':
            raise ValueError(
                'We expect the graph to have a unique prediction output named '
                '"output", found {!r}.'.format(nodes))

    @property
    def overriders(self):
        return self._transformer.overriders

    @property
    def variables(self):
        return self._transformer.variables

    def _layer_info(self):
        stats = self.estimate()
        keys = set()
        for node, stat in stats.items():
            if isinstance(stat, list):
                for each in stat:
                    keys |= set(each)
            elif isinstance(stat, dict):
                keys |= set(stat)
            else:
                raise TypeError('Unrecognized type.')
        keys = sorted(k for k in keys if not k.startswith('_'))
        layer_info = Table(['layer', 'shape'] + keys)
        for node, shape in self.shapes.items():
            if isinstance(node, LayerNode):
                values = stats.get(node, {})
                values = tuple(values.get(k, unknown) for k in keys)
            else:
                values = tuple([unknown] * len(keys))
            if isinstance(node, JoinNode):
                shape = ', '.join(format_shape(s) for s in shape)
            else:
                shape = format_shape(shape)
            layer_info.add_row((node.formatted_name(), shape) + values)
        layer_info.footer_sum('macs')
        return layer_info

    def _overrider_info(self):
        def flatten(overriders):
            for o in overriders:
                if isinstance(o, ChainOverrider):
                    yield from flatten(o)
                else:
                    yield o
        info_dict = {}
        overriders = []
        for each in self.overriders.values():
            overriders += list(each.values())
        for o in flatten(overriders):
            info = o.info()
            if not info:
                continue
            table = info_dict.setdefault(o.__class__, Table(info._fields))
            table.add_row(info)
        for cls, table in info_dict.items():
            cls.finalize_info(table)
        return info_dict

    def _variable_info(self):
        trainable_vars = tf.trainable_variables()
        # trainable table
        trainable = Table(['trainable', 'shape'])
        trainable.add_rows((v, v.shape) for v in trainable_vars)
        trainable.add_column(
            'count', lambda row: trainable[row, 'shape'].num_elements())
        trainable.footer_sum('count')
        # nontrainable table
        nontrainable = Table(['nontrainable', 'shape'])
        for var in tf.global_variables():
            if var not in trainable_vars:
                nontrainable.add_row((var, var.shape))
        return trainable, nontrainable

    def info(self, plumbing):
        trainable, nontrainable = self._variable_info()
        info = {
            'layers': self._layer_info(),
            'trainables': trainable,
            'nontrainables': nontrainable,
            'overriders': self._overrider_info(),
        }
        if plumbing:
            for k in ['layers', 'trainables', 'nontrainables']:
                info[k] = info[k].plumb()
            info['overriders'] = {
                k: t.plumb() for k, t in info['overriders'].items()}
        return info

    @property
    def shapes(self):
        unify = lambda t: tuple(s.value for s in t.shape)
        shapes = {}
        for node, tensors in self._tensors.items():
            if isinstance(tensors, collections.Sequence):
                shapes[node] = [unify(t) for t in tensors]
            else:
                shapes[node] = unify(tensors)
        return shapes

    def _params_to_text(self, params):
        arguments = []
        for k, v in params.items():
            try:
                v = '{}()'.format(v.__qualname__)
            except (KeyError, AttributeError):
                pass
            arguments.append('{}={}'.format(k, v))
        return '    ' + '\n    '.join(arguments)

    def _instantiate_layer(self, node, tensors):
        # transform parameters
        params, scope = self._transformer.transform(node, node.params)
        with scope:
            tensors = self.instantiate_numeric_padding(node, tensors, params)
            layer_type = params['type']
            layer_key = '{}/{}'.format(
                tf.get_variable_scope().name, params['scope'])
            layer_args = self._params_to_text(params)
            log.debug(
                'Instantiating {!r} of type {!r} with arguments:\n{}\n'
                '  for tensor(s) {}.'
                .format(layer_key, layer_type, layer_args, tensors))
            # get method by its name to instantiate a layer
            try:
                func, params = object_from_params(params, self, 'instantiate_')
            except NotImplementedError:
                func = self.generic_instantiate
            # instantiation
            layer = func(node, tensors, params)
        return layer

    def generic_instantiate(self, node, tensors, params):
        raise NotImplementedError(
            '{!r} does not know how to instantiate layer with type {!r}.'
            .format(self, params['type']))

    def instantiate_numeric_padding(self, node, tensors, params):
        pad = params.get('padding')
        if pad is None or isinstance(pad, str):
            return tensors
        # 4D tensor NxHxWxC, pad H and W
        if isinstance(pad, int):
            paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        elif isinstance(pad, collections.Sequence):
            pad_h, pad_w = pad
            paddings = [[0, 0], [pad_h] * 2, [pad_w] * 2, [0, 0]]
        else:
            raise ValueError(
                'We do not know what to do with a padding {!r}, we accept an '
                'integer, a string or a sequence of height and width paddings '
                '[pad_h, pad_w].'.format(pad))
        # disable pad for next layer
        params['padding'] = 'VALID'
        log.debug(
            'Instantiating padding {!r} for tensors(s) {}.'
            .format(paddings, tensors))
        if isinstance(tensors, collections.Sequence):
            return [tf.pad(t, paddings) for t in tensors]
        return tf.pad(tensors, paddings)

    def _estimate_layer(self, node, info):
        # info pass-through
        if node.params['type'] in ['identity', 'dropout']:
            return self.estimator._mask_passthrough(info, {})
        layer_info = super()._estimate_layer(node, info)
        log.debug(
            'Estimated statistics for {!r}: {}.'
            .format(node.formatted_name(), layer_info))
        for o in self.overriders.get(node, {}).values():
            layer_info = o.estimate(layer_info, info)
            log.debug(
                'Overrider {!r} modified statistics: {}.'
                .format(o, layer_info))
        return layer_info

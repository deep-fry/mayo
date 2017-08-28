class Surgery(object):
    def __init__(self):
        pass

    def prune(self, x, mask):
        return x * mask

    def quantize_fixed_point(self, x, n, f):
        '''
        1 bit sign, n bit int and f bit frac
        ref:
        https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/array_grad.py
        '''
        G = tf.get_default_graph()
        # shift left f bits
        x = x * (2**f)
        # quantize
        with G.gradient_override_map({"Round":"Identity"}):
            x = tf.round(x)
        # shift right f bits
        x = tf.div(x, 2 ** f)

        # cap int
        int_max = 2 ** n
        x = tf.clip_by_value(x, -int_max, int_max)
        # x = x * 0
        return x

    def quantize_dynamic_fixed_point(self, x, n, f, dr):
        '''
        1 bit sign, n bit int, f bit fraction, dr bit dynamic range
        Ref:
        https://arxiv.org/pdf/1604.03168
        '''
        G = tf.get_default_graph()
        # shift left f + dr bits
        x = x * (2**f) * (2**(-dr))
        # quantize
        with G.gradient_override_map({"Round":"Identity"}):
            x = tf.round(x)
        x = tf.div(x, 2 ** f)
        x_max = 2 ** (n - dr)
        x = tf.clip_by_value(x, -x_max, x_max)
        # put back dynmaic range
        x = x * (2 ** dr)
        # x = x * 0
        return x

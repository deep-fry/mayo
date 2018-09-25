import tensorflow as tf
from tensorflow.contrib import slim


def custom_batch_norm(
        tensor, decay=0.999, center=True, scale=False, epsilon=0.001,
        is_training=True, trainable=True, overriders=None, scope=None):
    if is_training:
        # training is not implemented yet
        return slim.batch_norm(
            tensor, decay=decay, center=center, scale=scale, epsilon=epsilon,
            is_training=is_training, trainable=trainable, scope=scope)

    channels = tensor.shape[-1]
    scope = scope or 'BatchNorm'

    def get_variable(name, one_init=False):
        if one_init:
            initializer = tf.ones_initializer()
        else:
            initializer = tf.zeros_initializer()
        with tf.variable_scope(scope):
            return tf.get_variable(
                name, [channels], initializer=initializer,
                trainable=trainable, dtype=tf.float32)

    mean = get_variable('moving_mean')
    var = get_variable('moving_variance', True)
    gamma = get_variable('gamma', True)
    beta = get_variable('beta')
    invstd = tf.rsqrt(var + epsilon)
    scale = gamma * invstd if scale else invstd
    if center:
        shift = beta - mean * scale
    else:
        shift = -mean * scale
    return tensor * scale + shift

import tensorflow as tf

def conv2d_trans(inputs, filter_shape, strides, output_shape, padding='SAME', scope='conv_trans'):
  with tf.variable_scope(scope):
    initializer = tf.truncated_normal_initializer(stddev=1e-3)
    filters = tf.get_variable('filters',
                              shape=filter_shape,
                              dtype='float',
                              initializer=initializer)
    conv_t = tf.nn.conv2d_transpose(inputs, filters, output_shape, strides, padding)
    return conv_t
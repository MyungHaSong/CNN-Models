import tensorflow as tf

def conv(inputs, filters):
    return tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size=(5,5))
def pool(inputs):
    return tf.layers.max_pooling2d(inputs = inputs, pool_size = (2,2), strides= (2,2))
def dense(inputs, units, activation = None):
    return tf.layers.dense(inputs= inputs, units = units, activation = activation)
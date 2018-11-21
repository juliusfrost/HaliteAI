import tensorflow as tf
from tensorflow import keras


def wrap_padding(tensor, padding=1, axes=(1, 2)):
    """
    Create a padding which wraps around the image, to emulate border-less matrices.

    :param tensor: the input tensor
    :param int padding: desired padding on the input tensor
    :param tuple axes: the axes to apply the wrapped padding
    """

    t = tensor
    # channels, rows, columns = tensor.shape
    for axis, dim in enumerate(tensor.shape):
        if axis not in axes:
            continue
        first_half, _, second_half = tf.split(t, [padding, int(dim) - 2 * padding, padding], axis=axis)
        t = tf.concat([second_half, t, first_half], axis=axis)

    return t


class WrapPadding(tf.keras.layers.Layer):
    def __init__(self, padding, axes):
        super(WrapPadding, self).__init__()
        self.padding = padding
        self.axes = axes

    def call(self, input):
        t = input
        # channels, rows, columns = tensor.shape
        for axis, dim in enumerate(input.shape):
            if axis not in self.axes:
                continue
            first_half, _, second_half = tf.split(t, [self.padding, int(dim) - 2 * self.padding, self.padding],
                                                  axis=axis)
            t = tf.concat([second_half, t, first_half], axis=axis)

        return t


class WrapConv2D(keras.Model):
    """
    A 2D convolution layer where the input is first padded in a wrapped fashion.
    Useful for kernels to learn information about the other side of the input tensor.
    """

    def __init__(self, filters, kernel_size, data_format='channels_first'):
        super(WrapConv2D, self).__init__(name='')
        self.filters = filters
        self.kernel_size = kernel_size
        # TODO: implement strides
        self.conv2d = keras.layers.Conv2D(filters, kernel_size, padding='valid',
                                          data_format=data_format)
        if data_format == 'channels_first':
            axes = (2, 3)
        else:
            axes = (1, 2)
        self.wrap = WrapPadding(padding=(kernel_size - 1) // 2, axes=axes)
        # self.wrap = keras.layers.Lambda(wrap_padding, arguments={'padding': (kernel_size - 1) // 2})

    def build(self, input_shape):
        pass

    def call(self, input_tensor, training=False):
        X = self.wrap(input_tensor)
        X = self.conv2d(X)
        return X


def model(input_tensor, gpu=False, training=True):
    if gpu:
        data_format = 'channels_first'
        channel_axis = 1
    else:
        data_format = 'channels_last'
        channel_axis = 3

    # 32x32, 40x40, 48x48, 56x56, 64x64
    tensor = input_tensor

    tensor3 = WrapConv2D(8, 3, data_format=data_format)(tensor)
    tensor5 = WrapConv2D(8, 5, data_format=data_format)(tensor)
    tensor7 = WrapConv2D(8, 7, data_format=data_format)(tensor)
    tensor9 = WrapConv2D(8, 9, data_format=data_format)(tensor)

    tensor = tf.concat([tensor3, tensor5, tensor7, tensor9], axis=channel_axis)

    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor3 = WrapConv2D(8, 3, data_format=data_format)(tensor)
    tensor5 = WrapConv2D(8, 5, data_format=data_format)(tensor)
    tensor7 = WrapConv2D(8, 7, data_format=data_format)(tensor)
    tensor9 = WrapConv2D(8, 9, data_format=data_format)(tensor)

    tensor = tf.concat([tensor3, tensor5, tensor7, tensor9], axis=channel_axis)

    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor = keras.layers.MaxPool2D(data_format=data_format)(tensor)

    tensor3 = WrapConv2D(16, 3, data_format=data_format)(tensor)
    tensor5 = WrapConv2D(16, 5, data_format=data_format)(tensor)
    tensor7 = WrapConv2D(16, 7, data_format=data_format)(tensor)
    tensor9 = WrapConv2D(16, 9, data_format=data_format)(tensor)

    tensor = tf.concat([tensor3, tensor5, tensor7, tensor9], axis=channel_axis)

    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor3 = WrapConv2D(16, 3, data_format=data_format)(tensor)
    tensor5 = WrapConv2D(16, 5, data_format=data_format)(tensor)
    tensor7 = WrapConv2D(16, 7, data_format=data_format)(tensor)
    tensor9 = WrapConv2D(16, 9, data_format=data_format)(tensor)

    tensor = tf.concat([tensor3, tensor5, tensor7, tensor9], axis=channel_axis)

    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor = keras.layers.MaxPool2D(data_format=data_format)(tensor)

    tensor3 = WrapConv2D(64, 3, data_format=data_format)(tensor)
    tensor5 = WrapConv2D(64, 5, data_format=data_format)(tensor)

    tensor = tf.concat([tensor3, tensor5], axis=channel_axis)

    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor3 = WrapConv2D(64, 3, data_format=data_format)(tensor)
    tensor5 = WrapConv2D(64, 5, data_format=data_format)(tensor)

    tensor = tf.concat([tensor3, tensor5], axis=channel_axis)

    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor = keras.layers.MaxPool2D(data_format=data_format)(tensor)

    tensor3 = WrapConv2D(256, 3, data_format=data_format)(tensor)

    tensor = keras.layers.Activation('relu')(tensor3)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    tensor3 = WrapConv2D(256, 3, data_format=data_format)(tensor)

    tensor = keras.layers.Activation('relu')(tensor3)

    tensor = keras.layers.BatchNormalization()(tensor, training=training)

    if gpu:
        _, _, h, w = tensor.shape
    else:
        _, h, w, _ = tensor.shape
    assert h == w
    #print(tensor.shape)

    tensor = keras.layers.MaxPool2D(pool_size=(h, w), strides=(h, w), data_format=data_format)(tensor)

    tensor = keras.layers.Flatten(data_format=data_format)(tensor)
    #print(tensor.shape)

    tensor = keras.layers.Dense(64)(tensor)
    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.Dense(16)(tensor)
    tensor = keras.layers.Activation('relu')(tensor)

    tensor = keras.layers.Dense(1)(tensor)
    tensor = keras.layers.Activation('sigmoid')(tensor)

    return tensor


if __name__ == '__main__':
    # cpu only supports 'channels_last'
    map_size = 64
    random = tf.random_uniform((64, map_size, map_size, 7))
    tensor = model(random)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(tensor))

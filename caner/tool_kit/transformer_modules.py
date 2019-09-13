import tensorflow as tf
import math
import numpy as np

"""
This part is the components of transformer
Thanks for https://github.com/IsaacChanghau/neural_sequence_labeling
       and https://github.com/Kyubyong/transformer
Original paper: https://arxiv.org/abs/1706.03762
"""


def layer_normalize(inputs, epsilon=1e-8):
    with tf.variable_scope("layer_norm"):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = tf.add(tf.multiply(gamma, normalized), beta)
        return outputs


def multi_head_attention(queries, keys, num_heads, attention_size, drop_keep_rate=0.0, reuse=None):
    # borrowed from: https://github.com/Kyubyong/transformer/blob/master/modules.py
    with tf.variable_scope("multi_head_attention", reuse=reuse):
        if attention_size is None:
            attention_size = queries.get_shape().as_list()[-1]
        # linear projections, shape=(batch_size, max_time, attention_size)
        query = tf.layers.dense(queries, attention_size, activation=tf.nn.relu, name="query_project",
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        key = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="key_project",
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        value = tf.layers.dense(keys, attention_size, activation=tf.nn.relu, name="value_project",
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        # split and concatenation, shape=(batch_size * num_heads, max_time, attention_size / num_heads)
        query_ = tf.concat(tf.split(query, num_heads, axis=2), axis=0)
        key_ = tf.concat(tf.split(key, num_heads, axis=2), axis=0)
        value_ = tf.concat(tf.split(value, num_heads, axis=2), axis=0)
        # multiplication
        attn_outs = tf.matmul(query_, tf.transpose(key_, [0, 2, 1]))
        # scale
        attn_outs = attn_outs / (key_.get_shape().as_list()[-1] ** 0.5)
        # key masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # shape=(batch_size, max_time)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # shape=(batch_size * num_heads, max_time)
        # shape=(batch_size * num_heads, max_time, max_time)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
        paddings = tf.ones_like(attn_outs) * (-2 ** 32 + 1)
        # shape=(batch_size, max_time, attention_size)
        attn_outs = tf.where(tf.equal(key_masks, 0), paddings, attn_outs)
        # activation
        attn_outs = tf.nn.softmax(attn_outs)
        # query masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        attn_outs *= query_masks
        # dropout
        attn_outs = tf.nn.dropout(attn_outs, drop_keep_rate)
        # weighted sum
        outputs = tf.matmul(attn_outs, value_)
        # restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # residual connection
        outputs += queries
        outputs = layer_normalize(outputs)
    return outputs


def positional_encoding(inputs,
                        zero_pad=False,
                        scale=False,
                        reuse=None):
    """Sinusoidal Positional_Encoding.
    Args:
      inputs: A 3d Tensor with shape of (batch_size, seq_length, embedding_size).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor'  rank equals to inputs's
    """
    batch_size, seq_length, embedding_size = inputs.get_shape().as_list()
    with tf.variable_scope("positional_encoding", reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/embedding_size) for i in range(embedding_size)]
            for pos in range(seq_length)], dtype='float32')

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, embedding_size]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * embedding_size**0.5

        pos_emb_outputs = inputs + outputs

        return pos_emb_outputs


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different frequency and phase.
    This allows attention to learn to use absolute and relative positions. Timing signals should be added to some
    precursors of both the query and the memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be expressed in terms of y,
    sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with min_timescale and ending with max_timescale.
    The number of different timescales is equal to channels / 2. For each timescale, we generate the two sinusoidal
    signals sin(timestep/timescale) and cos(timestep/timescale).  All of these sinusoids are concatenated in the
    channels dimension.
    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor the same shape as x.
    """
    with tf.name_scope("add_timing_signal"):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                                   (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

        scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def feedforward(inputs, num_units, reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope("multi_head_attention", reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = layer_normalize(outputs)

    return outputs



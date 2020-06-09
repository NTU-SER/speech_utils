"""
Most of the functions in this module are adapted from:
    https://github.com/30stomercury/Interaction-aware_Attention_Network/blob/master/ops.py
"""
import numpy as np
import tensorflow as tf


def additive_attention(inputs_1, inputs_2, attention_size=16,
                       reuse=tf.AUTO_REUSE):
    """Additive attention layer between multiple inputs. For now, the
    TensorFlow version only supports 2 inputs.

    Parameters
    ----------
    inputs_1 : tensor
        Tensor of shape (B, T, D), where B is batch size, T is the sequence
        length (along the time axis), and D is the features dimension.
    inputs_2 : tensor
        Same as inputs_1 (same shape as well).
    attention_size : int
        Attention size. (default: 16)
    reuse
        Value to be passed when constructing the variable scope.

    Returns
    -------
    tensor
        Tensor of shape (B, D). The time axis (T) has been squeezed, meaning
        that the tensor itself computes most relevant data along the time axis
        for every features dimension.

    """
    with tf.variable_scope("additive_attention", reuse=reuse):
        # Hidden size of the RNN layer
        hidden_size = int(inputs_1.shape[-1])

        # Trainable parameters
        W1 = tf.Variable(
            tf.random_normal([hidden_size, attention_size], stddev=0.1))
        W2 = tf.Variable(
            tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        # Weighting parameters
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # Compute attention scores
        v = tf.nn.tanh(
            tf.tensordot(inputs_1, W1, axes=1) +
            tf.tensordot(inputs_2, W2, axes=1) + b
        )  # (B, T, A)
        vu = tf.tensordot(v, u, axes=1)  # (B, T)
        alphas = tf.nn.softmax(vu)  # (B, T)
        # Output reduced with context vector
        outputs = tf.reduce_sum(inputs_1 * tf.expand_dims(alphas, -1), 1)

    return outputs  # (B, D)


def self_attention(inputs, attention_size=16, reuse=tf.AUTO_REUSE):
    """Attention layer mechanism.

    Parameters
    ----------
    inputs : tensor
        Tensor of shape (B, T, D), where B is batch size, T is the sequence
        length (along the time axis), and D is the features dimension.
    attention_size : int
        Attention size. (default: 16)
    reuse
        Value to be passed when constructing the variable scope.

    Returns
    -------
    tensor
        Tensor of shape (B, D). The time axis (T) has been squeezed, meaning
        that the tensor itself computes most relevant data along the time axis
        for every features dimension.

    """
    with tf.variable_scope("self_attention", reuse=reuse):
        # Hidden size of the RNN layer
        hidden_size = int(inputs.shape[-1])

        # Trainable parameters
        W = tf.Variable(
            tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        # Weighting parameters
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # Compute attention scores
        v = tf.nn.tanh(tf.tensordot(inputs, W, axes=1) + b)  # (B, T)
        vu = tf.tensordot(v, u, axes=1)  # (B, T)
        alphas = tf.nn.softmax(vu)  # (B, T)
        outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return outputs  # (B, D)


def scaled_dot_product_attention(inputs, attention_size=64,
                                 dropout_keep_prob=1, masking=True,
                                 is_training=True, reuse=tf.AUTO_REUSE):
    """Scaled dot-product attention as described in:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
        A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need.
        ArXiv:1706.03762 [Cs]. http://arxiv.org/abs/1706.03762


    Parameters
    ----------
    inputs : tensor
        Tensor of shape (B, T, D), where B is batch size, T is the sequence
        length (along the time axis), and D is the features dimension.
    attention_size : int
        Attention size.
    dropout_keep_prob : float
        Probability of keeping a connection in the dropout layer.
    masking : bool
        Whether to apply the masking to the intermediate tensors or not.
    is_training : bool
        Whether it is training stage or not.
    reuse
        Value to be passed when constructing the variable scope.

    Returns
    -------
    tensor
        Tensor of shape (B, T, A), where A is the attention size.

    """
    with tf.variable_scope("scaled_dot_product_attention", reuse=reuse):
        # Generate Q (queries), K (keys) and V (values) tensors
        Q = tf.layers.dense(inputs, attention_size, use_bias=True)  # (B, T, A)
        K = tf.layers.dense(inputs, attention_size, use_bias=True)  # (B, T, A)
        V = tf.layers.dense(inputs, attention_size, use_bias=True)  # (B, T, A)

        # Q x K_T
        matmul_QK = tf.matmul(Q, K, transpose_b=True)  # (B, T, T)
        # Scale: divide resulting tensor with square root of attention size
        matmul_QK = matmul_QK / (attention_size ** 0.5)  # (B, T, T)
        # Mask the resulting tensor
        if masking:
            k_mask = get_mask(K)  # (B, T, T)
            paddings = tf.ones_like(matmul_QK) * (-1e8)  # (B, T, T)
            matmul_QK = tf.where(
                tf.equal(k_mask, 0), paddings, matmul_QK)  # (B, T, T)

        # Softmax to compute attention weights
        att_weights = tf.nn.softmax(matmul_QK, axis=-1)  # (B, T, T)
        # Mask the resulting tensor
        if masking:
            q_mask = get_mask(Q)  # (B, T, T)
            att_weights = q_mask * att_weights  # (B, T, T)
        # Dropouts
        if is_training:
            att_weights = tf.nn.dropout(
                att_weights, keep_prob=dropout_keep_prob)  # (B, T, T)

        # Compute outputs
        outputs = tf.matmul(att_weights, V)  # (B, T, A)

        return outputs


def get_mask(inputs, dim=None):
    """
    Mask the input tensor:
    If `dim=None`:
        inputs: (B, C, D) -> outputs: (B, C, C)
    Otherwise:
        inputs: (B, C, D) -> outputs: (B, C, `dim`)
    """
    dim_ = tf.shape(inputs)[1] if dim is None else dim
    with tf.variable_scope("mask"):
        mask = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))  # (B, C)
        mask = tf.expand_dims(mask, 1)  # (B, 1, C)
        mask = tf.tile(mask, [1, dim_, 1])  # (B, `dim`, C)
        # Transpose if needed
        if dim is not None:
            mask = tf.transpose(mask, [0, 2, 1])  # (B, C, `dim`)
    return mask


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3,
                       reuse=tf.AUTO_REUSE):
    """
    Wraps the batch normalization layer.
    Copied from `speech_utils.ACRNN.tf.batch_norm_wrapper`

    """
    with tf.variable_scope("batch_norm", reuse=reuse):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), name="scale")
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), name="beta")
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),
                               name="mean", trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]),
                              name="var", trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(
            inputs, pop_mean, pop_var, beta, scale, epsilon)


def layer_norm_wrapper(x, dropout_keep_prob, reuse=tf.AUTO_REUSE):
    """
    Wraps the layer normalization layer.
    """
    with tf.variable_scope('norm', reuse=reuse):
        x = tf.nn.dropout(x, keep_prob=dropout_keep_prob)
        # Normalize
        x = tf.contrib.layers.layer_norm(x)
    return x


def label_smoothing(inputs):
    K = inputs.get_shape().as_list()[-1]
    label = (0.9 * inputs) + (0.1 / K)
    return label

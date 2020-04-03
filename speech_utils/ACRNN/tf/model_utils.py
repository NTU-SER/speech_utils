import os
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Adapted from https://github.com/xuanjihe/speech-emotion-recognition/blob/master/attention.py
    Note that the author has changed the activation function of the fully
    connected layer from tanh to sigmoid.

    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention
    vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical
    Attention Networks for Document Classification",
    2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of
                    shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple
                (outputs_fw, outputs_bw) containing the forward and the backward
                RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped
            `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped
            `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.
            However, most TensorFlow data is batch-major, so by default this
            function accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along
            with layer's output. Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of RNN layer

    # Trainable parameters
    with tf.variable_scope("attention"):
        W_omega = tf.Variable(
            tf.random_normal([hidden_size, attention_size], stddev=0.1),
            name="weight")
        b_omega = tf.Variable(
            tf.random_normal([attention_size], stddev=0.1), name="bias")
        u_omega = tf.Variable(
            tf.random_normal([attention_size], stddev=0.1), name="u")

    # Applying fully connected layer with non-linear activation to each of
    # the B*T timestamps; the shape of `v` is (B,T,D)*(D,A)=(B,T,A),
    # where A=attention_size
    v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with
    # `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    alphas = tf.nn.softmax(vu)              # (B,T) shape also

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D)
    # shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    with tf.variable_scope("batch_norm"):
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
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def acrnn(inputs, num_classes=4, is_training=True, dropout_keep_prob=1,
          num_filt_1=128, num_filt_2=256, mp_fsize=(2, 4), num_lstm_units=128,
          num_linear_units=768, num_fcn_units=64):
    """
    Reference: "3-D Convolutional Recurrent Neural Networks with Attention Model
    for Speech Emotion Recognition"
    Authors: Chen Mingyi and
             He Xuanji and
             Yang Jing and
             Zhang Han.

    Adapted from
    https://github.com/xuanjihe/speech-emotion-recognition/blob/master/acrnn1.py

    3D-CRNN model wrapper with Tensorflow. Take `inputs` as model input and
    return the model logits.

    Parameters
    ----------
    inputs : tensor
        Input tensor.
    num_classes : int
        Number of classes.
    is_training : bool
        Whether the model is built for training or not.
    dropout_keep_prob : float
        Probability of keeping connections in dropout.

    num_filt_1 : int
        Number of filters of the first convolutional layer.
    num_filt_2 : int
        Number of filters of all the rest convolutional layers.
    mp_fsize : tuple or list
        Max pooling filter size (tuple/list of height and width, respectively)

    num_lstm_units : int
        Number of neurons in LSTM cell.
    num_linear_units : int
        Number of units in the linear layer (the layer after the last
        convolutional layer).
    num_fcn_units : int
        Number of neurons in the last fully connected layer.

    Returns
    -------
    logits
        Logits of the model.

    """
    image_height, image_width = inputs.get_shape().as_list()[1:3]
    mp_fsize_h, mp_fsize_w = mp_fsize
    # `time_step` and `p` can be calculated when knowing max pooling filter size
    time_step = image_height // mp_fsize_h
    p = image_width // mp_fsize_w

    layer1_filter = tf.get_variable(
        'layer1_filter', shape=[5, 3, 3, num_filt_1], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer1_bias = tf.get_variable(
        'layer1_bias', shape=[num_filt_1], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    layer1_stride = [1, 1, 1, 1]
    layer2_filter = tf.get_variable(
        'layer2_filter', shape=[5, 3, num_filt_1, num_filt_2], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer2_bias = tf.get_variable(
        'layer2_bias', shape=[num_filt_2], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    layer2_stride = [1, 1, 1, 1]
    layer3_filter = tf.get_variable(
        'layer3_filter', shape=[5, 3, num_filt_2, num_filt_2], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer3_bias = tf.get_variable(
        'layer3_bias', shape=[num_filt_2], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    layer3_stride = [1, 1, 1, 1]
    layer4_filter = tf.get_variable(
        'layer4_filter', shape=[5, 3, num_filt_2, num_filt_2], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer4_bias = tf.get_variable(
        'layer4_bias', shape=[num_filt_2], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    layer4_stride = [1, 1, 1, 1]
    layer5_filter = tf.get_variable(
        'layer5_filter', shape=[5, 3, num_filt_2, num_filt_2], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer5_bias = tf.get_variable(
        'layer5_bias', shape=[num_filt_2], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    layer5_stride = [1, 1, 1, 1]
    layer6_filter = tf.get_variable(
        'layer6_filter', shape=[5, 3, num_filt_2, num_filt_2], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer6_bias = tf.get_variable(
        'layer6_bias', shape=[num_filt_2], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    layer6_stride = [1, 1, 1, 1]

    linear1_weight = tf.get_variable(
        'linear1_weight', shape=[p * num_filt_2, num_linear_units],
        dtype=tf.float32, initializer=tf.truncated_normal_initializer(
            stddev=0.1))
    linear1_bias = tf.get_variable(
        'linear1_bias', shape=[num_linear_units], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))

    fully1_weight = tf.get_variable(
        'fully1_weight', shape=[2 * num_lstm_units, num_fcn_units],
        dtype=tf.float32, initializer=tf.truncated_normal_initializer(
            stddev=0.1))
    fully1_bias = tf.get_variable(
        'fully1_bias', shape=[num_fcn_units], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))
    fully2_weight = tf.get_variable(
        'fully2_weight', shape=[num_fcn_units, num_classes], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully2_bias = tf.get_variable(
        'fully2_bias', shape=[num_classes], dtype=tf.float32,
        initializer=tf.constant_initializer(0.1))

    layer1 = tf.nn.conv2d(inputs, layer1_filter, layer1_stride, padding='SAME')
    layer1 = tf.nn.bias_add(layer1, layer1_bias)
    layer1 = leaky_relu(layer1, 0.01)
    layer1 = tf.nn.max_pool(
        layer1, ksize=[1, mp_fsize_h, mp_fsize_w, 1],
        strides=[1, mp_fsize_h, mp_fsize_w, 1], padding='VALID', name='max_pool')
    layer1 = tf.contrib.layers.dropout(
        layer1, keep_prob=dropout_keep_prob, is_training=is_training)

    layer2 = tf.nn.conv2d(layer1, layer2_filter, layer2_stride, padding='SAME')
    layer2 = tf.nn.bias_add(layer2, layer2_bias)
    layer2 = leaky_relu(layer2, 0.01)
    layer2 = tf.contrib.layers.dropout(
        layer2, keep_prob=dropout_keep_prob, is_training=is_training)

    layer3 = tf.nn.conv2d(layer2, layer3_filter, layer3_stride, padding='SAME')
    layer3 = tf.nn.bias_add(layer3, layer3_bias)
    layer3 = leaky_relu(layer3, 0.01)
    layer3 = tf.contrib.layers.dropout(
        layer3, keep_prob=dropout_keep_prob, is_training=is_training)

    layer4 = tf.nn.conv2d(layer3, layer4_filter, layer4_stride, padding='SAME')
    layer4 = tf.nn.bias_add(layer4, layer4_bias)
    layer4 = leaky_relu(layer4, 0.01)
    layer4 = tf.contrib.layers.dropout(
        layer4, keep_prob=dropout_keep_prob, is_training=is_training)

    layer5 = tf.nn.conv2d(layer4, layer5_filter, layer5_stride, padding='SAME')
    layer5 = tf.nn.bias_add(layer5, layer5_bias)
    layer5 = leaky_relu(layer5, 0.01)
    layer5 = tf.contrib.layers.dropout(
        layer5, keep_prob=dropout_keep_prob, is_training=is_training)

    layer6 = tf.nn.conv2d(layer5, layer6_filter, layer6_stride, padding='SAME')
    layer6 = tf.nn.bias_add(layer6, layer6_bias)
    layer6 = leaky_relu(layer6, 0.01)
    layer6 = tf.contrib.layers.dropout(
        layer6, keep_prob=dropout_keep_prob, is_training=is_training)

    layer6 = tf.reshape(layer6, [-1, time_step, num_filt_2 * p])
    layer6 = tf.reshape(layer6, [-1, p * num_filt_2])

    linear1 = tf.matmul(layer6, linear1_weight) + linear1_bias
    linear1 = batch_norm_wrapper(linear1, is_training)
    linear1 = leaky_relu(linear1, 0.01)
    linear1 = tf.reshape(linear1, [-1, time_step, num_linear_units])

    # Define lstm cells with tensorflow
    # Forward direction cell
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(num_lstm_units, forget_bias=1.0)
    # Backward direction cell
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(num_lstm_units, forget_bias=1.0)

    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN
    # output.
    outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=gru_fw_cell1, cell_bw=gru_bw_cell1, inputs=linear1,
        dtype=tf.float32, time_major=False, scope='LSTM')

    # Attention layer
    gru, alphas = attention(outputs1, 1, return_alphas=True)
    # Fully connected layers
    fully1 = tf.matmul(gru, fully1_weight) + fully1_bias
    fully1 = leaky_relu(fully1, 0.01)
    fully1 = tf.nn.dropout(fully1, dropout_keep_prob)

    logits = tf.add(
        tf.matmul(fully1, fully2_weight), fully2_bias, name="logits")
    return logits


def CB_loss_tf(labels, logits, samples_per_cls, beta=0.9999, is_training=True,
               name=None):
    """
    Reference: "Class-Balanced Loss Based on Effective Number of Samples"
    Authors: Yin Cui and
             Menglin Jia and
             Tsung Yi Lin and
             Yang Song and
             Serge J. Belongie
    https://arxiv.org/abs/1901.05555, CVPR'19.

    Adapted from the PyTorch version:
      https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py

    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1 - beta) / (1 - beta^n)) * Loss(labels, logits)
    where Loss is standard cross entropy loss.

    Parameters
    ----------
    labels : tensor
        A tensor of shape (batch_size, num_classes).
    logits : tensor
        A tensor of shape (batch_size, num_classes). Output probabilities of
        the model.
    samples_per_cls : list or ndarray
        A list or ndarray of length `num_classes`.
    beta : float
        Hyperparameter for Class Balanced Loss.
    is_training : bool

    Returns
    -------
    loss : tensor
        If training, return scalar reduced loss over non-zero weights.
        Otherwise, return un-reduced loss (whose shape is (batch_size,)).

    """
    num_classes = labels.get_shape().as_list()[1]

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    weights = np.expand_dims(weights, 0)

    weights = tf.multiply(weights, labels)
    weights = tf.math.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(weights, axis=-1)

    softmax = tf.nn.softmax(logits, axis=-1)

    if is_training:
        loss = tf.losses.log_loss(
            labels=labels, predictions=softmax, weights=weights)
        loss = tf.identity(loss, name=name)
    else:
        loss = tf.losses.log_loss(
            labels=labels, predictions=softmax, weights=weights,
            reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_sum(loss, axis=-1, name=name)

    return loss


def test(test_data, test_labels, test_segs_labels, test_segs, sess,
         num_classes=4, batch_size=64):
    """Short summary.

    Parameters
    ----------
    test_data: ndarray
    test_labels: ndarray
    test_segs_labels: ndarray
    test_segs: ndarray
        The first four arguments are the data in order obtained using the
        script `extract_mel.py`.
    num_classes : int
        Number of classes.
    batch_size : int

    Returns
    -------
    test_cost : float
        The cost value of the model on the test data.
    test_ua : float
        The unweighted accuracy of the model on the test data.
    test_conf : float
        The confusion matrix of the model on the test data.
    """
    # Parameters
    num_test = test_segs.shape[0]
    num_test_segs = test_data.shape[0]
    # Store predictions
    y_test_segs = np.empty(
        (num_test_segs, num_classes), dtype=np.float32)
    y_test = np.empty((num_test, num_classes), dtype=np.float32)

    test_cost = 0
    if num_test_segs < batch_size:
        loss, y_test_segs = sess.run(
            ["cross_entropy:0", "logits:0"],
            feed_dict={"X_input:0": test_data, "Y_input:0": test_segs_labels,
                       "is_training:0": False, "keep_prob:0": 1}
        )
        test_cost = test_cost + np.sum(loss)

    num_batches_test = divmod((num_test_segs), batch_size)[0]
    for batch_test in range(num_batches_test):
        v_begin = batch_test * batch_size
        v_end = (batch_test + 1) * batch_size
        if batch_test == num_batches_test - 1:
            if v_end < num_test_segs:
                v_end = num_test_segs
        loss, y_test_segs[v_begin:v_end] = sess.run(
            ["cross_entropy:0", "logits:0"],
            feed_dict={"X_input:0": test_data[v_begin:v_end],
                       "Y_input:0": test_segs_labels[v_begin:v_end],
                       "is_training:0": False, "keep_prob:0": 1})
        test_cost = test_cost + np.sum(loss)
    test_cost = test_cost / num_test_segs
    # Accumulate results, since each utterance might contain
    # more than one segments
    curr_i = 0
    for v in range(num_test):
        y_test[v, :] = np.max(
            y_test_segs[curr_i:curr_i + test_segs[v]], 0)
        curr_i = curr_i + test_segs[v]
    # Recall and confusion matrix
    test_ua = recall(
        np.argmax(test_labels, 1), np.argmax(y_test, 1), average='macro')
    test_conf = confusion(
        np.argmax(test_labels, 1), np.argmax(y_test, 1))
    # Make confusion matrix into data frame for readability
    test_conf = pd.DataFrame({"ang": test_conf[:, 0], "sad": test_conf[:, 1],
                              "hap": test_conf[:, 2], "neu": test_conf[:, 3]})
    test_conf = test_conf.to_string(index=False)
    return test_cost, test_ua, test_conf


def train(data, epochs, batch_size, learning_rate, validate_every=10,
          random_seed=123, num_classes=4, grad_clip=False, dropout_keep_prob=1,
          save_path=None, use_CBL=False, beta=0.9999, perform_test=False,
          **kwargs):
    """Short summary.

    Parameters
    ----------
    data : tuple
        Data extracted using speech_utils.ACRNN.data_utils.extract_mel
    epochs : int
        Number of epochs.
    batch_size : int
    learning_rate : float
    validate_every : int
        Number of training steps between each validation.
    random_seed : int
        Random seed for reproducibility.
    num_classes : int
        Number of classes.
    grad_clip : boolean
        Whether to clip gradients of Adam optimizer.
    dropout_keep_prob : float
        The probability of keeping a connection in dropout.
    save_path : str
        Path to save the best models.
    use_CBL : boolean
        Whether to use Class Balanced Loss.
    beta : float
        Hyperparameter for Class Balanced Loss. Default: 0.9999
    perform_test: bool
        Whether to test on test data at the end of training process.
    **kwargs
        Custom keyword arguments to pass to the ACRNN constructor.

    """
    # For reproducibility
    tf.random.set_random_seed(random_seed)
    np.random.seed(random_seed)
    # Load data
    train_data, train_labels = data[0:2]
    val_data, val_labels, val_segs_labels, val_segs = data[2:6]
    test_data, test_labels, test_segs_labels, test_segs = data[6:10]
    # One-hot encoding
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    val_segs_labels = to_categorical(val_segs_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)
    test_segs_labels = to_categorical(test_segs_labels, num_classes)
    # Parameters
    num_train, image_height, image_width, image_channel = train_data.shape
    best_val_ua = 0
    # Construct model
    X = tf.placeholder(
        tf.float32, name="X_input",
        shape=[None, image_height, image_width, image_channel])
    Y = tf.placeholder(tf.int32, name="Y_input", shape=[None, num_classes])
    is_training = tf.placeholder(tf.bool, name="is_training")
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    # Cost
    logits = acrnn(
        X, num_classes=num_classes, is_training=is_training,
        dropout_keep_prob=keep_prob, **kwargs)
    if not use_CBL:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=logits, name="cross_entropy")
        cost = tf.reduce_mean(cross_entropy)
    else:
        _, samples_per_cls = np.unique(
            np.argmax(train_labels, axis=-1), return_counts=True)
        cross_entropy = CB_loss_tf(
            labels=tf.cast(Y, "float64"), logits=logits, is_training=False,
            samples_per_cls=samples_per_cls, beta=beta, name="cross_entropy")
        cost = CB_loss_tf(labels=tf.cast(Y, "float64"), logits=logits,
                          samples_per_cls=samples_per_cls, beta=beta)

    var_trainable_op = tf.trainable_variables()
    # Optimizer
    if not grad_clip:
        # Do not apply gradient clipping
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    else:
        # Apply gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, var_trainable_op), 5)
        opti = tf.train.AdamOptimizer(lr)
        train_op = opti.apply_gradients(zip(grads, var_trainable_op))
    # Predictions and accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            start = (i * batch_size) % num_train
            end = min(start + batch_size, num_train)
            feed_dict = {X: train_data[start:end], Y: train_labels[start:end],
                         is_training: True, keep_prob: dropout_keep_prob,
                         lr: learning_rate}
            _, train_cost, train_acc = sess.run([train_op, cost, accuracy],
                                                feed_dict=feed_dict)
            # Test on validation set
            if i % validate_every == 0:
                val_cost, val_ua, val_conf = test(
                    val_data, val_labels, val_segs_labels, val_segs, sess,
                    num_classes=num_classes, batch_size=batch_size)
                # Update
                if val_ua > best_val_ua:
                    best_val_ua = val_ua
                    best_val_conf = val_conf
                    best_val_step = i + 1
                    if save_path is not None:
                        saver.save(sess, save_path, global_step=i + 1)

                print("Epoch: {}".format(i + 1))
                print("Training cost: {:.04f}".format(train_cost))
                print("Training UA: {:.02f}%".format(train_acc * 100))
                print()
                print("Valid cost: {:.04f}".format(val_cost))
                print("Valid UA: {:.02f}%".format(val_ua * 100))
                print('Valid confusion matrix:\n["ang","sad","hap","neu"]')
                print(val_conf)
                print()
                print("Best valid UA: {:.02f}%".format(best_val_ua * 100))
                print('Best valid confusion matrix:\n["ang","sad","hap","neu"]')
                print(best_val_conf)
                print("*" * 30)

        # Test on test set
        if perform_test:
            meta_path = "{}-{}.meta".format(save_path, best_val_step)
            save_dir, _ = os.path.split(save_path)
            # Load weights
            saver.restore(sess, tf.train.latest_checkpoint(save_dir))
            # Start testing
            test_cost, test_ua, test_conf = test(
                test_data, test_labels, test_segs_labels, test_segs, sess,
                num_classes=num_classes, batch_size=batch_size)
            print("*" * 30)
            print("RESULTS ON TEST SET:")
            print("Test cost: {:.04f}, test unweighted accuracy: "
                  "{:.2f}%".format(test_cost, test_ua * 100))
            print('Test confusion matrix:\n["ang","sad","hap","neu"]')
            print(best_val_conf)

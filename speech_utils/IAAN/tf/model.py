import os

import tensorflow as tf

from .ops import (self_attention, get_mask, layer_norm_wrapper,
                  label_smoothing, class_balanced_loss)


class IAAN:
    def __init__(self, save_dir, features_dim=45, num_gru_units=128,
                 attention_size=16, num_linear_units=64, lr=1e-4,
                 weight_decay=1e-3, dropout_keep_prob=1.0, num_classes=4,
                 loss_type="ce", samples_per_cls=None):
        # Cache data
        self.save_dir = save_dir
        self.features_dim = features_dim
        self.num_gru_units = num_gru_units
        self.attention_size = attention_size
        self.num_linear_units = num_linear_units
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

        # Loss
        if loss_type not in ["ce", "cbl"]:
            raise ValueError("Invalid loss type. Expect one of ['ce', 'cbl'], "
                             "got '{}' instead.".format(loss_type))
        if loss_type == "cbl" and samples_per_cls is None:
            raise RuntimeError("Class balanced loss is used but the number of "
                               "samples per class is not provided.")
        self.loss_type = loss_type
        self.samples_per_cls = samples_per_cls

        self._get_session()  # get session
        self._build_model()  # build model
        self._init()  # initialize

    def _build_model(self):

        # Input placeholders
        with tf.variable_scope("input"):
            # center: current utterance
            self.center_pl = tf.placeholder(
                dtype=tf.float32, shape=[None, None, self.features_dim])
            # target: previous utterance of the current speaker
            self.target_pl = tf.placeholder(
                dtype=tf.float32, shape=[None, None, self.features_dim])
            # opposite: previous utterance of the interlocutor
            self.opposite_pl = tf.placeholder(
                dtype=tf.float32, shape=[None, None, self.features_dim])
            # Labels
            self.gt_pl = tf.placeholder(dtype=tf.int64, shape=[None])

            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob")

        # Target features encoder
        with tf.variable_scope('target_encoder', reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.GRUCell(self.num_gru_units)
            outputs, _ = tf.nn.dynamic_rnn(
                cell, inputs=self.target_pl, dtype=tf.float32)
            self.target_att = self_attention(outputs)

        # Opposite features encoder
        with tf.variable_scope('opposite_encoder', reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.GRUCell(self.num_gru_units)
            outputs, _ = tf.nn.dynamic_rnn(
                cell, inputs=self.opposite_pl, dtype=tf.float32)
            self.opposite_att = self_attention(outputs)

        # Center features encoder
        with tf.variable_scope('center_encoder', reuse=tf.AUTO_REUSE):
            cell = tf.contrib.rnn.GRUCell(self.num_gru_units)
            # Dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, input_keep_prob=self.keep_prob_pl,
                output_keep_prob=self.keep_prob_pl)

            outputs, _ = tf.nn.dynamic_rnn(
                cell, inputs=self.center_pl, dtype=tf.float32,
                time_major=False)

            # Add masks
            seq_len = tf.shape(outputs)[1]

            target_mask = get_mask(self.target_pl, self.num_gru_units)
            target_att_masked = tf.tile(
                tf.expand_dims(self.target_att, 1), [1, seq_len, 1])
            target_att_masked = target_att_masked * target_mask

            opposite_mask = get_mask(self.opposite_pl, self.num_gru_units)
            opposite_att_masked = tf.tile(
                tf.expand_dims(self.opposite_att, 1), [1, seq_len, 1])
            opposite_att_masked = opposite_att_masked * opposite_mask

            center_mask = get_mask(self.center_pl, self.num_gru_units)
            self.out = outputs * center_mask

            # Compute context vector with Bahdanau attention
            with tf.variable_scope('interaction_aware_attention',
                                   reuse=tf.AUTO_REUSE):
                # Trainable parameters
                W_c = tf.Variable(tf.random_normal(
                    [self.num_gru_units, self.attention_size], stddev=0.1))
                W_p = tf.Variable(tf.random_normal(
                    [self.num_gru_units, self.attention_size], stddev=0.1))
                W_r = tf.Variable(tf.random_normal(
                    [self.num_gru_units, self.attention_size], stddev=0.1))

                b = tf.Variable(
                    tf.random_normal([self.attention_size], stddev=0.1))
                u = tf.Variable(
                    tf.random_normal([self.attention_size], stddev=0.1))

                # Compute attention scores
                v = tf.nn.tanh(
                    tf.tensordot(self.out, W_c, axes=1) +
                    tf.tensordot(target_att_masked, W_p, axes=1) +
                    tf.tensordot(opposite_att_masked, W_r, axes=1) + b
                )
                vu = tf.tensordot(v, u, axes=1)

                # Attention weights
                mask_att = tf.sign(
                    tf.abs(tf.reduce_sum(self.center_pl, axis=-1)))
                paddings = tf.ones_like(mask_att) * 1e-8

                vu = tf.where(tf.equal(mask_att, 0), paddings, vu)  # (B, T)
                alphas = tf.nn.softmax(vu)

                # Output reduced with context vector: (B, T)
                self.center_att = tf.reduce_sum(
                    self.out * tf.expand_dims(alphas, -1), 1)

        # Multi-layer perceptron
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            self.out = tf.concat(
                [self.center_att, self.target_att, self.opposite_att], 1)
            # FCN 1 parameters
            out_weight1 = tf.get_variable(
                'out_weight1', dtype=tf.float32,
                shape=[self.num_gru_units * 3, self.num_linear_units],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            out_bias1 = tf.get_variable(
                'out_bias1', shape=[self.num_linear_units], dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
            # FCN 2 parameters
            out_weight2 = tf.get_variable(
                'out_weight2', dtype=tf.float32,
                shape=[self.num_linear_units, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            out_bias2 = tf.get_variable(
                'out_bias2', shape=[self.num_classes], dtype=tf.float32,
                initializer=tf.constant_initializer(0.1))
            # FC ops
            dense = tf.matmul(self.out, out_weight1) + out_bias1
            dense = layer_norm_wrapper(
                dense, self.keep_prob_pl)
            dense = tf.nn.relu(dense)
            dense = tf.matmul(dense, out_weight2) + out_bias2

            self.logits = dense

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # Label smoothing
            self.gt = tf.one_hot(self.gt_pl, depth=self.num_classes)
            self.gt = label_smoothing(self.gt)
            # Classification loss
            if self.loss_type == "ce":
                self.loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=self.gt, logits=self.logits)
            else:
                self.loss = class_balanced_loss(
                    labels=self.gt, logits=self.logits,
                    samples_per_cls=self.samples_per_cls)
            # Total loss, including weight decay
            self.tot_loss = (self.loss + self.weight_decay *
                             (tf.nn.l2_loss(out_weight1) +
                              tf.nn.l2_loss(out_bias1) +
                              tf.nn.l2_loss(out_weight2) +
                              tf.nn.l2_loss(out_bias2)
                              ))
            self.optimizer = tf.train.AdamOptimizer(
                self.lr).minimize(self.tot_loss)

        with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
            self.prediction = tf.argmax(self.logits, axis=1)
            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.gt, axis=1),
                predictions=self.prediction)

    def _init(self):
        # Initialzation
        self.saver = tf.train.Saver(max_to_keep=2000)
        self.sess.run(tf.global_variables_initializer())

    def _get_session(self):
        self.sess = tf.Session()

    def train_one_batch(self, data):
        center_features, target_features, opposite_features, labels = data

        feed_dict = {
            self.center_pl: center_features,
            self.target_pl: target_features,
            self.opposite_pl: opposite_features,
            self.gt_pl: labels,
            self.is_training_pl: True,
            self.keep_prob_pl: self.dropout_keep_prob
        }
        out_tensors = [self.loss, self.optimizer, self.prediction]
        loss, _, preds = self.sess.run(
            out_tensors, feed_dict=feed_dict)

        return loss, preds

    def test_one_batch(self, data):
        center_features, target_features, opposite_features, labels = data

        feed_dict = {
            self.center_pl: center_features,
            self.target_pl: target_features,
            self.opposite_pl: opposite_features,
            self.gt_pl: labels,
            self.is_training_pl: False,
            self.keep_prob_pl: 1.0
        }
        out_tensors = [self.loss, self.prediction]
        loss, preds = self.sess.run(out_tensors, feed_dict=feed_dict)

        return loss, preds

    def save(self, idx):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, "iaan_{}.ckpt".format(idx))
        self.saver.save(self.sess, save_path)

    def restore(self, idx):
        save_path = os.path.join(self.save_dir, "iaan_{}.ckpt".format(idx))
        self.saver.restore(self.sess, save_path)

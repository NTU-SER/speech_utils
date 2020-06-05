import sys
import argparse
import pickle

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf
from keras.utils import to_categorical

from speech_utils.ACRNN.tf.model_utils import test

config = ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config).as_default()


def main(args):
    # Load test data
    with open(args.data_path, "rb") as fin:
        data = pickle.load(fin)
    if args.swap:
        test_data, test_labels, test_segs_labels, test_segs = data[2:6]
    else:
        test_data, test_labels, test_segs_labels, test_segs = data[6:10]
    # One-hot encoding
    test_labels = to_categorical(test_labels, args.num_classes)
    test_segs_labels = to_categorical(test_segs_labels, args.num_classes)
    # Checkpoint
    ckpt_path = tf.train.latest_checkpoint(args.ckpt_dir)
    if args.global_step is not None:
        ckpt_path = ckpt_path.split("-")[0] + "-" + str(args.global_step)
    meta_path = ckpt_path + ".meta"
    # Test
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)
        test_cost, test_ua, test_conf = test(
            test_data, test_labels, test_segs_labels, test_segs, sess=sess,
            num_classes=args.num_classes, batch_size=args.batch_size)
        # Print
        print("*" * 30)
        print("RESULTS ON TEST SET:")
        print("Test cost: {:.04f}, test unweighted accuracy: "
              "{:.2f}%".format(test_cost, test_ua * 100))
        print('Test confusion matrix:')
        print(test_conf)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'data_path', type=str,
        help='Path to the features extracted from `extract_mel.py`.')
    parser.add_argument(
        'ckpt_dir', type=str, help='Path to the checkpoints directory.')

    parser.add_argument(
        '--global_step', type=int, default=None,
        help='Global step of the checkpoint to be tested. If `None`, use the '
             'latest checkpoint')
    parser.add_argument(
        '--batch_size', type=int, default=60, help='Mini batch size.')
    parser.add_argument(
        '--num_classes', type=int, default=4, help='Number of classes.')
    parser.add_argument(
        '--swap', action='store_true',
        help='By default, the female recordings of a chosen session is set to '
             'validation data, and the male recordings of that session is set '
             'to test data. Set this to true to swap the validation set with '
             'the test set.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

import sys
import argparse
import pickle

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf

from speech_utils.ACRNN.tf.model_utils import train

config = ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config).as_default()


def main(args):
    # Verify
    if args.save_path is None and args.perform_test:
        raise ValueError("Cannot test when `save_path` is set to `None`.")
    # Load data
    with open(args.data_path, "rb") as fin:
        data = pickle.load(fin)
    # If swap
    if args.swap:
        train_data = data[0:2]
        test_data = data[2:6]
        val_data = data[6:10]
        data = (*train_data, *val_data, *test_data)
    # Train
    train(data, args.num_epochs, args.batch_size, args.lr,
          validate_every=args.validate_every, random_seed=args.seed,
          num_classes=args.num_classes, grad_clip=args.grad_clip,
          dropout_keep_prob=1 - args.dropout, save_path=args.save_path,
          use_CBL=args.use_cbl, beta=args.beta, perform_test=args.perform_test)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str,
        help='Path to the features extracted from `extract_mel.py`.')
    parser.add_argument('num_epochs', type=int, help='Number of epochs.')

    parser.add_argument('--batch_size', type=int, default=60,
        help='Mini batch size.')
    parser.add_argument('--num_classes', type=int, default=4,
        help='Number of classes.')
    parser.add_argument('--lr', type=int, default=1e-5, help='Learning rate.')

    parser.add_argument('--dropout', type=float, default=0.0,
        help='Probability of a connection being set to 0 (i.e., disconnected).')
    parser.add_argument('--use_cbl', action="store_true",
        help='Whether to use Class Balanced Loss.')
    parser.add_argument('--beta', type=float, default=0.9999,
        help='Hyperparameter for Class Balanced Loss. Used when '
             '`use_cbl==True`.')
    parser.add_argument('--grad_clip', action='store_true',
        help='Whether to clip gradients of Adam optimizer.')
    parser.add_argument('--save_path', type=str, default=None,
        help='Path to save the best models with `.ckpt` as extension (e.g., '
             '`save_path=./model.ckpt`, then the model at global step 500 will '
             'be saved as `./model.ckpt-500.data-00000-of-00001`, '
             '`./model.ckpt-500.index` and `./model.ckpt-500.meta`).')

    parser.add_argument('--swap', action='store_true',
        help='By default, the female recordings of a chosen session is set to '
             'validation data, and the male recordings of that session is set '
             'to test data. Set this to true to swap the validation set with '
             'the test set.')
    parser.add_argument('--perform_test', action='store_true',
        help='Whether to test on test data at the end of training process.')
    parser.add_argument('--validate_every', type=int, default=10,
        help='Number of batches between each test.')
    parser.add_argument('--seed', type=int, default=123,
        help='Random seed for reproducibility.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

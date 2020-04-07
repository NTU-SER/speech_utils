import sys
import argparse
import pickle

from speech_utils.ACRNN.torch.model_utils import train


def main(args):
    # Verify
    if args.save_path is None and args.perform_test:
        raise ValueError("Cannot test when `save_path` is set to `None`.")
    if args.loss_type not in ["ce", "sigmoid", "softmax", "focal"]:
        raise ValueError("Invalid loss type. Expected one of "
                         "[\"ce\", \"sigmoid\", \"softmax\", \"focal\"]. Got {}"
                         " instead.".format(args.loss_type))
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
          random_seed=args.seed, num_classes=args.num_classes,
          dropout_keep_prob=1 - args.dropout, save_path=args.save_path,
          loss_type=args.loss_type, beta=args.beta,
          perform_test=args.perform_test)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train a 3DCRNN model in an iterative-based manner with "
                    "Tensorflow.")

    parser.add_argument('data_path', type=str,
        help='Path to the features extracted from `extract_mel.py`.')
    parser.add_argument('num_epochs', type=int,
        help='Number of training epochs.')

    parser.add_argument('--batch_size', type=int, default=60,
        help='Mini batch size.')
    parser.add_argument('--num_classes', type=int, default=4,
        help='Number of classes.')
    parser.add_argument('--lr', type=int, default=1e-5, help='Learning rate.')

    parser.add_argument('--dropout', type=float, default=0.0,
        help='Probability of a connection being set to 0 (i.e., disconnected).')
    parser.add_argument('--loss_type', type=str, default="softmax",
        help='One of ["ce", "sigmoid", "softmax", "focal"].')
    parser.add_argument('--beta', type=float, default=0.9999,
        help='Hyperparameter for Class Balanced Loss. Used when '
             '`use_cbl==True`.')
    parser.add_argument('--save_path', type=str, default=None,
        help='Path to save the best model.')

    parser.add_argument('--swap', action='store_true',
        help='By default, the female recordings of a chosen session is set to '
             'validation data, and the male recordings of that session is set '
             'to test data. Set this to true to swap the validation set with '
             'the test set.')
    parser.add_argument('--perform_test', action='store_true',
        help='Whether to test on test data at the end of training process.')
    parser.add_argument('--seed', type=int, default=None,
        help='Random seed for reproducibility.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

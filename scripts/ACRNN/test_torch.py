import sys
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn

from speech_utils.ACRNN.torch.data_utils import DatasetLoader
from speech_utils.ACRNN.torch.model_utils import ACRNN, ClassBalancedLoss, test


def main(args):
    # Verify
    if args.loss_type not in ["ce", "sigmoid", "softmax", "focal"]:
        raise ValueError("Invalid loss type. Expected one of "
                         "[\"ce\", \"sigmoid\", \"softmax\", \"focal\"]. Got {}"
                         " instead.".format(args.loss_type))
    # Load data
    with open(args.data_path, "rb") as fin:
        data = pickle.load(fin)
    # Handle data
    if args.swap:
        train_data = data[0:2]
        test_data = data[2:6]
        val_data = data[6:10]
        data = (*train_data, *val_data, *test_data)
    # Unpack data
    pre_process = lambda x: np.moveaxis(x, 3, 1).astype("float32")
    dataloader = DatasetLoader(
        data, num_classes=args.num_classes, pre_process=pre_process)
    train_dataset = dataloader.get_train_dataset()
    test_dataset = dataloader.get_test_dataset()
    # Construct model
    device = torch.device("cuda:0")
    model = ACRNN()
    model = model.to(device=device)
    # Load pre-trained model
    model.load_state_dict(torch.load(args.model_path))
    # Loss
    _, samples_per_cls = np.unique(
        train_dataset.target, return_counts=True)
    if args.loss_type == "ce": # cross-entropy:
        loss_function = nn.CrossEntropyLoss(weight=samples_per_cls)
    else:
        loss_function = ClassBalancedLoss(
            samples_per_cls=samples_per_cls, loss_type=args.loss_type,
            beta=args.beta)
    # Test
    with torch.no_grad():
        test_loss, test_ua, test_ur, test_conf = test(
            model, loss_function, test_dataset, args.batch_size, device,
            return_matrix=True)

    print("RESULTS ON TEST SET:")
    print("Loss:{:.4f}\tUnweighted Accuracy: {:.2f}\tUnweighted Recall: "
          "{:.2f}".format(test_loss, test_ua, test_ur))
    print("Confusion matrix:\n{}".format(test_conf))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str,
        help='Path to the features extracted from `extract_mel.py`.')
    parser.add_argument('model_path', type=str,
        help='Path to the trained model.')

    parser.add_argument('--batch_size', type=int, default=60,
        help='Mini batch size.')
    parser.add_argument('--num_classes', type=int, default=4,
        help='Number of classes.')
    parser.add_argument('--loss_type', type=str, default="softmax",
        help='One of ["ce", "sigmoid", "softmax", "focal"].')
    parser.add_argument('--beta', type=float, default=0.9999,
        help='Hyperparameter for Class Balanced Loss. Used when '
             '`use_cbl==True`.')
    parser.add_argument('--swap', action='store_true',
        help='By default, the female recordings of a chosen session is set to '
             'validation data, and the male recordings of that session is set '
             'to test data. Set this to true to swap the validation set with '
             'the test set.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

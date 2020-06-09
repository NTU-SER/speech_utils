import os
import sys
import argparse

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

from speech_utils.IAAN.tf.model import IAAN
from speech_utils.IAAN.data_utils import (
    get_dialog_order, get_label, generate_interaction_data,
    InteractionDataGenerator)


def evaluate(model, dataloader, epoch_idx, phase="val"):
    tot_loss = 0
    tot_labels = list()
    tot_preds = list()

    for batch in dataloader:
        loss, preds = model.test_one_batch(batch)
        tot_loss += loss * len(batch[0])
        tot_labels.append(batch[-1])
        tot_preds.append(preds)

    # Compute accuracy, recall and confusion matrix
    tot_labels = np.concatenate(tot_labels)
    tot_preds = np.concatenate(tot_preds)

    tot_loss /= len(dataloader)
    tot_acc = accuracy_score(tot_labels, tot_preds)
    tot_ur = recall_score(tot_labels, tot_preds, average="macro")
    tot_cm = confusion_matrix(tot_labels, tot_preds)

    print("Epoch: {}, phase: {}, loss: {:.4f}, acc: {:.4f}, recall: {:.4f}, "
          "confusion matrix:\n{}".format(
            epoch_idx, phase, tot_loss, tot_acc, tot_ur, tot_cm))
    return tot_ur


def main(args):
    # Load data
    order_dict = get_dialog_order(args.iemocap_dir)
    emo_dict = get_label(args.iemocap_dir)
    features_dict = joblib.load(args.features_path)
    # Data loader
    val_prefix = None if args.val_prefix == "" else args.val_prefix
    test_prefix = None if args.test_prefix == "" else args.test_prefix
    df_train, df_val, df_test = generate_interaction_data(
        order_dict, emo_dict, val_prefix=val_prefix, test_prefix=test_prefix,
        mode=args.mode, emo=args.emo, num_processes=args.num_processes)

    train_loader = InteractionDataGenerator(
        df_train, args.batch_size, features_dict, sort_by_len=args.sort_by_len)
    val_loader = InteractionDataGenerator(
        df_val, args.batch_size, features_dict, sort_by_len=args.sort_by_len)
    test_loader = InteractionDataGenerator(
        df_test, args.batch_size, features_dict, sort_by_len=args.sort_by_len)

    # Initialize model
    features_dim = list(features_dict.values())[0].shape[-1]
    model = IAAN(
        save_dir=args.ckpt_dir, features_dim=features_dim,
        num_gru_units=args.num_gru_units, attention_size=args.attention_size,
        num_linear_units=args.num_linear_units, lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_keep_prob=args.keep_prob, num_classes=len(args.emo))

    # Train and evaluate
    best_ur = 0.0
    best_ur_idx = -1
    for epoch_idx in range(args.num_epochs):

        # Train
        train_loss = 0
        train_labels = list()
        train_preds = list()

        for batch in train_loader:
            loss, preds = model.train_one_batch(batch)
            train_loss += loss * len(batch[0])
            train_labels.append(batch[-1])
            train_preds.append(preds)

        # Compute accuracy, recall and confusion matrix
        train_labels = np.concatenate(train_labels)
        train_preds = np.concatenate(train_preds)

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_ur = recall_score(train_labels, train_preds, average="macro")
        train_cm = confusion_matrix(train_labels, train_preds)

        print("Epoch: {}, phase: {}, loss: {:.4f}, acc: {:.4f}, recall: "
              "{:.4f}, confusion matrix:\n{}".format(
                epoch_idx, "train", train_loss, train_acc, train_ur, train_cm))

        # Evaluate on validation data
        val_ur = evaluate(model, val_loader, epoch_idx, phase="val")
        if val_ur > best_ur:
            best_ur = val_ur
            best_ur_idx = epoch_idx

        # Save model
        model.save(epoch_idx)

    # Load the best model
    save_path = os.path.join(
        model.save_dir, "iaan_{}.ckpt".format(best_ur_idx))
    print("Loading the best model:", save_path)
    model.restore(best_ur_idx)
    # Evaluate on test set
    evaluate(model, test_loader, best_ur_idx, phase="test")


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'iemocap_dir', type=str,
        help='Directory to the "IEMOCAP_full_release" folder.')
    parser.add_argument(
        'features_path', type=str, help='Path to the pickled features file.')
    parser.add_argument(
        'ckpt_dir', type=str,
        help='Directory where the training checkpoints will be saved.')
    parser.add_argument(
        'num_epochs', type=int, help='Number of epochs.')
    parser.add_argument(
        'val_prefix', type=str,
        help='Scenes with this as prefix will be used as validation data, '
             'e.g., "Ses01F".')
    parser.add_argument(
        'test_prefix', type=str,
        help='Scenes with this as prefix will be used as test data, '
             'e.g., "Ses01M".')

    parser.add_argument(
        '--num_gru_units', type=int, default=128,
        help='Number of units in the GRU cells.')
    parser.add_argument(
        '--attention_size', type=int, default=16,
        help='Attention activation dimension.')
    parser.add_argument(
        '--num_linear_units', type=int, default=64,
        help='Number of linear units in the FCN layers.')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-3,
        help='Weight decay factor.')
    parser.add_argument(
        '--keep_prob', type=float, default=1.0,
        help='The probability of keeping connections in the drop out layers.')

    parser.add_argument(
        '--mode', type=str, default="context",
        help='Data loader mode. One of ["context", "random"]. If "context", '
             '"target" and "opposite" data will be chosen as the most recent '
             'ones. Otherwise, it will be chosen randomly.')
    parser.add_argument(
        '--emo', type=str, nargs="+", default=['ang', 'hap', 'neu', 'sad'],
        help='Emotions to be used. Note that the data preprocessing actually '
             'already convert excited to happy (in the `get_label` function). '
             'Passing multiple values is done by separating values by space.')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Data loader\'s batch size.')
    parser.add_argument(
        '--sort_by_len', default=False, action="store_true",
        help='Whether to sort the data by length or not.')
    parser.add_argument(
        '--num_processes', type=int,
        help='Number of threads in the thread pool.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

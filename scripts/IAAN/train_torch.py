import os
import sys
import json
import argparse
from collections.abc import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

from speech_utils.IAAN.torch.model import IAANTrainer
from speech_utils.IAAN.data_utils import (
    get_dialog_order, get_label, generate_interaction_data,
    InteractionDataGenerator)


def main(args):
    # Load config
    with open(args.config_path, "r") as fin:
        config = json.load(fin)
    # Load data
    order_dict = get_dialog_order(config["iemocap_dir"])
    emo_dict = get_label(config["iemocap_dir"])
    features_dict = joblib.load(config["features_path"])

    # Allow multiple configs for training
    val_prefix = config["val_prefix"]
    test_prefix = config["test_prefix"]
    if type(val_prefix) != type(test_prefix):
        raise TypeError(
            "val_prefix and test_prefix should have the same type. Got {} and "
            "{} instead.".format(type(val_prefix), type(test_prefix)))
    if isinstance(val_prefix, str):
        val_prefix = [None] if val_prefix == "" else [val_prefix]
        test_prefix = [None] if test_prefix == "" else [test_prefix]
    # If not str, val_prefix and test_prefix must be iterable of the same
    # length
    elif not isinstance(val_prefix, Iterable):
        raise TypeError("Expected val_prefix and test_prefix to be iterables, "
                        "got {} instead.".format(type(val_prefix)))
    elif len(val_prefix) != len(test_prefix):
        raise ValueError(
            "Expected val_prefix and test_prefix to have the same length, got "
            "{} and {} instead.".format(len(val_prefix), len(test_prefix)))

    # Loop over each config and train
    trainer = None
    results = dict()
    for val_pre, test_pre in zip(val_prefix, test_prefix):
        del trainer  # free up some memory
        print()
        print("=" * 50)
        print("Training with val_prefix={} and test_prefix={}".format(
            val_pre, test_pre))
        print("=" * 50)

        # Dataloader
        df_train, df_val, df_test = generate_interaction_data(
            order_dict, emo_dict, val_prefix=val_pre, test_prefix=test_pre,
            mode=config["mode"], emo=config["emo"],
            num_processes=config["num_processes"])

        train_loader = InteractionDataGenerator(
            df_train, config["batch_size"], features_dict,
            sort_by_len=config["sort_by_len"])
        val_loader = InteractionDataGenerator(
            df_val, config["batch_size"], features_dict,
            sort_by_len=config["sort_by_len"])
        test_loader = InteractionDataGenerator(
            df_test, config["batch_size"], features_dict,
            sort_by_len=config["sort_by_len"])

        # Calculate number of samples per class for training data
        uniques, samples_per_cls = np.unique(
            train_loader.label, return_counts=True)
        assert (uniques == range(len(uniques))).all()
        assert samples_per_cls.sum() == train_loader.num_samples
        config["trainer_kwargs"]["samples_per_cls"] = samples_per_cls
        config["trainer_kwargs"]["num_classes"] = len(samples_per_cls)

        # Initialize trainer
        features_dim = list(features_dict.values())[0].shape[-1]
        config["trainer_kwargs"]["features_dim"] = features_dim

        trainer = IAANTrainer(**config["trainer_kwargs"])

        # Train and evaluate
        best_ur = 0.0
        best_ur_idx = -1
        for epoch_idx in range(config["num_epochs"]):
            # Train
            train_loss, train_preds, train_labels = trainer.train_one_epoch(
                train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_ur = recall_score(train_labels, train_preds, average="macro")
            train_cm = confusion_matrix(train_labels, train_preds)

            print("Epoch: {}, phase: {}, loss: {:.4f}, acc: {:.4f}, recall: "
                  "{:.4f}, confusion matrix:\n{}".format(
                    epoch_idx, "train", train_loss, train_acc, train_ur,
                    train_cm))

            # Evaluate on validation data
            val_loss, val_preds, val_labels = trainer.test_one_epoch(
                val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_ur = recall_score(val_labels, val_preds, average="macro")
            val_cm = confusion_matrix(val_labels, val_preds)
            print("Epoch: {}, phase: {}, loss: {:.4f}, acc: {:.4f}, recall: "
                  "{:.4f}, confusion matrix:\n{}".format(
                    epoch_idx, "val", val_loss, val_acc, val_ur, val_cm))

            if val_ur > best_ur:
                best_ur = val_ur
                best_ur_idx = epoch_idx

            # Save model
            trainer.save(epoch_idx)

            # Early stopping
            if "early_stopping" in config:
                if epoch_idx - best_ur_idx >= config["early_stopping"]:
                    print("Model shows no improvement over {} steps. Early "
                          "stopping...".format(config["early_stopping"]))
                    break

        # Load the best model
        print("Loading the best model...")
        trainer.load(best_ur_idx)
        # Evaluate on test set
        test_loss, test_preds, test_labels = trainer.test_one_epoch(
            test_loader)
        test_acc = accuracy_score(test_labels, test_preds)
        test_ur = recall_score(test_labels, test_preds, average="macro")
        test_cm = confusion_matrix(test_labels, test_preds)
        print("Epoch: {}, phase: {}, loss: {:.4f}, acc: {:.4f}, recall: "
              "{:.4f}, confusion matrix:\n{}".format(
                best_ur_idx, "test", test_loss, test_acc, test_ur, test_cm))

        # Cache results
        key = "val_prefix={}, test_prefix={}".format(val_pre, test_pre)
        results[key] = {
            "accuracy": test_acc,
            "unweighted recall": test_ur,
        }

    df = pd.DataFrame(results).T
    print()
    print("Summary results on test sets:")
    print(df)
    print()
    print("accuracy: {:.4f} ± {:.4f}".format(
        df["accuracy"].mean(), df["accuracy"].std()))
    print("unweighted recall: {:.4f} ± {:.4f}".format(
        df["unweighted recall"].mean(), df["unweighted recall"].std()))


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'config_path', type=str,
        help='Path to the config file.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

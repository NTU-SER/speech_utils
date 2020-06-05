import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix


class TrainLoader(torch.utils.data.Dataset):
    """
    Holds data for a train dataset (e.g., holds training examples as well as
    training labels.)

    Parameters
    ----------
    data : ndarray
        Input data.
    target : ndarray
        Target labels.
    pre_process : fn
        A function to be applied to `data`.

    """
    def __init__(self, data, target, pre_process=None):
        super(TrainLoader).__init__()

        if pre_process is not None:
            data = pre_process(data)
        self.data = data
        self.target = target
        self.n_samples = data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def recall(self, predictions):
        """
        Calculate recall score given the predictions.

        Parameters
        ----------
        predictions : ndarray
            Model's predictions.

        Returns
        -------
        int
            Recall score (using `macro` method.)

        """
        return recall(self.target, predictions, average="macro")

    def accuracy(self, predictions):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        predictions : ndarray
            Model's predictions.

        Returns
        -------
        float
            Accuracy score.

        """
        acc = (self.target == predictions).sum() / self.n_samples
        return acc


class TestLoader(TrainLoader):
    """
    Holds data for a validation/test set.

    Parameters
    ----------
    data : ndarray
        Input data of shape `N x H x W x C`, where `N` is the number of
        examples (segments), `H` is image height, `W` is image width and `C`
        is the number of channels.
    actual_target : ndarray
        Actual target labels (labels for utterances) of shape `(N',)`, where
        `N'` is the number of utterances.
    seg_target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape `(N,)`.
    num_segs : ndarray
        Array of shape `(N',)` indicating how many segments each utterance
        contains.
    num_classes :
        Number of classes.
    pre_process : fn
        A function to be applied to `data`.

    """
    def __init__(self, data, actual_target, seg_target,
                 num_segs, num_classes=4, pre_process=None):
        super(TestLoader).__init__()

        if pre_process is not None:
            data = pre_process(data)
        self.data = data
        self.target = seg_target
        self.n_samples = data.shape[0]
        self.n_actual_samples = actual_target.shape[0]

        self.actual_target = actual_target
        self.num_segs = num_segs
        self.num_classes = num_classes

    def get_preds(self, seg_preds):
        """
        Get predictions for all utterances from their segments' prediction.
        This function will accumulate the predictions for each utterance by
        taking the maximum probability along the dimension 0 of all segments
        belonging to that particular utterance.
        """
        preds = np.empty(
            shape=(self.n_actual_samples, self.num_classes), dtype="float")

        end = 0
        for v in range(self.n_actual_samples):
            start = end
            end = start + self.num_segs[v]

            preds[v] = np.max(seg_preds[start:end], axis=0)

        preds = np.argmax(preds, axis=1)
        return preds

    def recall(self, utt_preds):
        """
        Calculate recall score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Recall score (using `macro` method.)

        """
        ur = recall(self.actual_target, utt_preds, average="macro")
        return ur

    def accuracy(self, utt_preds):
        """
        Calculate accuracy score given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        Returns
        -------
        float
            Accuracy score.

        """
        acc = (self.actual_target == utt_preds).sum() / self.n_actual_samples
        return acc

    def confusion_matrix(self, utt_preds):
        """Compute confusion matrix given the predictions.

        Parameters
        ----------
        utt_preds : ndarray
            Processed predictions.

        """
        conf = confusion_matrix(self.actual_target, utt_preds)
        # Make confusion matrix into data frame for readability
        conf = pd.DataFrame({"ang": conf[:, 0], "sad": conf[:, 1],
                             "hap": conf[:, 2], "neu": conf[:, 3]})
        conf = conf.to_string(index=False)
        return conf


class DatasetLoader:
    """
    Wrapper for both `TrainLoader` and `TestLoader`, which loads pre-processed
    speech features into `Dataset` objects.

    Parameters
    ----------
    data : tuple
        Data extracted using `speech_utils.ACRNN.data_utils.extract_mel`.
    num_classes : int
        Number of classes.
    pre_process : fn
        A function to be applied to `data`.

    """
    def __init__(self, data, num_classes, pre_process=None):
        self.num_classes = num_classes
        self.pre_process = pre_process
        # Unpack data
        self.train_data, self.train_labels = data[0:2]
        self.val_data, self.val_labels = data[2:4]
        self.val_segs_labels, self.val_segs = data[4:6]
        self.test_data, self.test_labels = data[6:8]
        self.test_segs_labels, self.test_segs = data[8:10]

    def get_train_dataset(self):
        return TrainLoader(
            self.train_data, self.train_labels, self.pre_process)

    def get_val_dataset(self):
        return TestLoader(
            self.val_data, self.val_labels,
            self.val_segs_labels, self.val_segs,
            num_classes=self.num_classes,
            pre_process=self.pre_process)

    def get_test_dataset(self):
        return TestLoader(
            self.test_data, self.test_labels,
            self.test_segs_labels, self.test_segs,
            num_classes=self.num_classes,
            pre_process=self.pre_process)

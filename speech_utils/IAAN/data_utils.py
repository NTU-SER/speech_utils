import os
import time
import math
import subprocess
from pathlib import Path
from multiprocessing.pool import Pool, ThreadPool

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn import preprocessing


def execute(wav_path, out_dir, smile_path, smile_conf):
    """Execute external openSMILE command to extract features from audio
    files"""
    _, wav_name = os.path.split(wav_path)
    name, _ = os.path.splitext(wav_name)
    save_path = os.path.join(out_dir, name + ".arff")
    # Execute commands
    command = "{} -C {} -I {} -O {}".format(
        smile_path, smile_conf, wav_path, save_path)

    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, close_fds=True)
    if process.stderr is not None:
        print("Error processing {}. message: {}".format(
            wav_path, process.stderr.read()))
        process.stderr.close()
    else:
        res = process.stdout.read().decode("utf-8")
        if res.split().count("successfully") != 2:
            print("Error processing {}".format(wav_path))
    process.stdout.close()


def extract(iemocap_dir, out_dir, smile_path, smile_conf,
            num_processes=32, step_size=32):
    """Extract features from input audio files using openSMILE and save
    extracted features to files.

    Parameters
    ----------
    iemocap_dir : str
        Directory to the "IEMOCAP_full_release" folder.
    out_dir : str
        Output directory.
    num_processes : int
        Number of threads in the thread pool.
    step_size : int
        Size of each step (like batch size).

    """
    wav_paths = list(Path(iemocap_dir).rglob("sentences/wav/**/*.wav"))
    num_steps = math.ceil(len(wav_paths) / step_size)
    pool = ThreadPool(processes=num_processes)

    for i in tqdm(range(num_steps)):
        start = i * step_size
        end = min(start + step_size, len(wav_paths))

        fn = lambda path: execute(path, out_dir, smile_path, smile_conf)
        pool.map(fn, wav_paths[start:end])


def get_feature_idxs():
    """Get feature indices for files extracted using `extract` function"""
    mfcc = np.arange(4, 16)
    mfcc_del = np.arange(30, 42)
    mfcc_del_del = np.arange(56, 68)

    loudness = np.array([3])
    loudness_del = np.array([29])
    loudness_del_del = np.array([55])

    f0 = np.array([26])
    f0_del = np.array([52])

    voice_prob = np.array([25])
    voice_prob_del = np.array([51])

    zcr = np.array([24])
    zcr_del = np.array([50])

    idxs = np.concatenate(
        [f0, f0_del, loudness, loudness_del, loudness_del_del, voice_prob,
         voice_prob_del, zcr, zcr_del, mfcc, mfcc_del, mfcc_del_del]
    )

    return idxs


def get_count_and_name(filepath):
    """Helper function to get the number of datapoints and file name without
    extension for a feature file extracted using `extract` function"""
    # First 86 lines is irrelevant
    count = len(open(filepath, "r").readlines()) - 86
    _, filename = os.path.split(filepath)
    name, _ = os.path.splitext(filename)
    return count, name


def get_feature(filepath):
    """Read features from a feature file extracted using `extract` function.

    Parameters
    ----------
    filepath : str
        Path to the feature file.

    Returns
    -------
    ndarray
        Array of shape (N, F), where N is sequence length and F is the number
        of features (45 by default).

    """
    feature_idxs = get_feature_idxs()
    with open(filepath, "r") as fin:
        for i in range(86):
            fin.readline()
        features = pd.read_csv(fin, header=None, usecols=feature_idxs)
    features = features[feature_idxs].to_numpy()
    return features


def get_feature_wrapper(i, filepaths, counts_cum, feats):
    """Helper function that wraps function `get_feature` to be used in
    multithreading"""
    feat = get_feature(filepaths[i])

    start = 0 if i == 0 else counts_cum[i - 1]
    end = start + len(feat)
    feats[start:end] = feat


def normalize(features_dir, num_processes=32):
    """Normalize features with respect to speakers.

    Parameters
    ----------
    features_dir : str
        Folder containing feature files extracted using `extract` function.
    num_processes : type
        Number of threads in the thread pool.

    Returns
    -------
    dict
        Dictionary of pairs "name:features", where "name" is the utterance file
        name without extension (e.g., Ses01F_impro01_F000) and features are
        respective features extracted and normalized.

    """
    speakers = {}
    pool = ThreadPool(processes=num_processes)

    # Read files
    for filename in os.listdir(features_dir):
        speaker = filename.split("_")[0]
        if speaker not in speakers:
            speakers[speaker] = list()

        filepath = os.path.join(features_dir, filename)
        speakers[speaker].append(filepath)

    # Loop over each speaker and do normalization
    all_feat = {}
    for speaker, filepaths in tqdm(speakers.items()):
        # Get sequence lengths and file names
        count_and_name = pool.map(get_count_and_name, filepaths)
        counts, names = tuple(zip(*count_and_name))
        counts_cum = np.cumsum(counts)

        # Get all features for this speaker
        feats = np.empty((counts_cum[-1], 45), dtype="float64")
        fn = lambda i: get_feature_wrapper(i, filepaths, counts_cum, feats)
        pool.map(fn, range(len(filepaths)))

        # Normalize
        feats_zscore = stats.zscore(feats)
        start = 0
        for name, count_cum in zip(names, counts_cum):
            end = count_cum
            all_feat[name] = feats_zscore[start:end]
            start = end
    return all_feat


def get_sequence(arr, step_size, overlap):
    seq = list()
    start = 0
    end = step_size
    cont = True

    while cont:
        if end >= len(arr):
            end = len(arr)
            cont = False
        seq.append(np.arange(start, end))
        start += step_size - overlap
        end = start + step_size
    return np.concatenate(seq)


def average_pool_single(arr, step_size=5, overlap=0, pad=True, max_len=2500):
    """Average pooling over the time axis of a single time series data.

    Parameters
    ----------
    arr : ndarray
        Array of shape [N, F], where N is the sequence length and F is the
        feature dimension.
    step_size : int
        Step size (width of pool).
    max_len : int
        Maximum sequence length.
        If `max_len=0` or negative, the array is kept as is.
        Otherwise, any data beyond this threshold is removed.
    overlap : int
        Overlapping length between two subsequences. This is set to 0 (no
        overlapping) in the original implementation.
    pad : bool
        Whether to pad zeros to the end of the sequence. This is set to True in
        the original implementation. However, it is fairly obvious that it is
        better set to False.

    Returns
    -------
    ndarray
        Array after average pooling.

    """
    # Truncate
    if max_len > 0 and len(arr) > max_len:
        arr = arr[:max_len]
    # Overlapping
    if overlap >= step_size:
        raise ValueError("Invalid overlap value: {}. It should be smaller "
                         "than `step_size` ({})".format(overlap, step_size))
    idxs = get_sequence(arr, step_size=step_size, overlap=overlap)
    arr = arr[idxs]
    # Pad
    excess = len(arr) % step_size
    if excess != 0 and pad:
        pad_len = step_size - excess
        arr = np.pad(arr, [(0, pad_len), (0, 0)])
    # Trim at the end
    excess = len(arr) % step_size
    split_point = len(arr) - excess
    last = arr[split_point:]
    arr = arr[:split_point]
    # Average pooling
    num_features = arr.shape[1]
    arr = arr.reshape(-1, step_size, num_features).mean(axis=1)
    if len(last) > 0:
        last = last.mean(axis=0)
        # Append the last cut
        arr = np.vstack([arr, last])
    return arr


def average_pool(features, num_processes=32, pool_size=32, step_size=5,
                 overlap=0, pad=True, max_len=2500):
    """Average pooling over the time axis of a single time series data.

    Parameters
    ----------
    features : ndarray
        Dictionary of pairs "name:features", where "name" is the utterance file
        name without extension (e.g., Ses01F_impro01_F000) and features are
        respective features extracted and normalized. This should be the
        results of executing the `extract_and_normalize.py`
    num_processes : int
        Number of threads in the thread pool.
    pool_size : int
        Thread pool size.
    step_size : int
        Step size (width of pool).
    max_len : int
        Maximum sequence length.
        If `max_len=0` or negative, the array is kept as is.
        Otherwise, any data beyond this threshold is removed.
    overlap : int
        Overlapping length between two subsequences. This is set to 0 (no
        overlapping) in the original implementation.
    pad : bool
        Whether to pad zeros to the end of the sequence. This is set to True in
        the original implementation. However, it is fairly obvious that it is
        better set to False.

    Returns
    -------
    ndarray
        Array after average pooling.

    """
    names = list(features.keys())
    feats = list(features.values())
    features_pooled = list()
    # Parameters for multithreading
    num_steps = math.ceil(len(features) / pool_size)
    pool = ThreadPool(processes=num_processes)
    fn = lambda arr: average_pool_single(arr, step_size, overlap, pad, max_len)
    # Loop over arrays and do average pooling
    for i in tqdm(range(num_steps)):
        start = i * pool_size
        end = min(start + pool_size, len(features))

        feat = feats[start:end]
        feats_pooled = pool.map(fn, feat)
        features_pooled.extend(feats_pooled)

    assert len(names) == len(features_pooled)
    return dict(zip(names, features_pooled))


def get_label(iemocap_dir):
    """For labels for every utterance under the form "utt_name:label".

    Parameters
    ----------
    iemocap_dir : str
        Path to the `IEMOCAP_full_release` directory.

    Returns
    -------
    dict
        Dictionary of pairs utt_name:label
        (e.g., 'Ses01F_impro01_F000': 'neu').

    """
    emo_dict = dict()

    def get_info(line):
        line = line.split()
        name = line[3]
        label = line[4]
        # Convert "exc" (excited) to "hap" (happy)
        label = "hap" if label == "exc" else label
        return name, label

    for filepath in Path(iemocap_dir).rglob("EmoEvaluation/*.txt"):
        with open(filepath, "r") as fin:
            lines = fin.readlines()

        lines = filter(lambda x: x[0] == "[", lines)  # filter lines with label
        emo_dict.update(dict(map(get_info, lines)))
    return emo_dict


def get_dialog_order(iemocap_dir):
    """Get order of speakers for every dialog. For session 2, the order is read
    from the label files *.lab (not from the transcription files *.txt as in
    other sessions). This differs from the original implementation that the
    order is sorted based on utterances' start time AND end time.

    Parameters
    ----------
    iemocap_dir : str
        Path to the `IEMOCAP_full_release` directory.

    Returns
    -------
    dict
        Dictionary of pairs "scene:utt_orders".

    """
    def filt(utt_name):
        return utt_name[:3] == "Ses" and "XX" not in utt_name

    order_dict = dict()
    # All sessions, except session 2
    for filepath in Path(iemocap_dir).rglob("dialog/transcriptions/*.txt"):
        _, filename = os.path.split(filepath)
        # Session 2 needs to be handled differently
        if int(filename[3:5]) == 2:
            continue
        scene, _ = os.path.splitext(filename)
        with open(filepath, "r") as fin:
            lines = fin.readlines()
        utt_orders = list(map(lambda x: x.split()[0], lines))
        # Filter unnecessary utterances
        utt_orders = list(filter(filt, utt_orders))
        order_dict[scene] = utt_orders
    # Session 2
    for filepath in Path(iemocap_dir).rglob(
            "Session2/dialog/lab/Ses02_F/*.lab"):
        _, filename = os.path.split(filepath)
        scene, _ = os.path.splitext(filename)

        # Read *.lab files as csv
        f_path = str(filepath)
        f_csv = pd.read_csv(
            f_path, delim_whitespace=True, header=None, skiprows=[0])
        m_path = f_path.replace("Ses02_F", "Ses02_M")
        m_csv = pd.read_csv(
            m_path, delim_whitespace=True, header=None, skiprows=[0])

        # Sort by start and end time
        tot_csv = pd.concat([f_csv, m_csv], ignore_index=True)
        tot_csv = tot_csv.sort_values([0, 1]).reset_index(drop=True)
        utt_orders = tot_csv[2].to_list()
        order_dict[scene] = utt_orders

    return order_dict


def generate_interaction_sample(index_words, emo_dict,
                                emo=['ang', 'hap', 'neu', 'sad']):
    """Generate interaction training samples of pairs "center - target -
    opposite", where "center" is the current utterance, "target" is the
    previous utterance of the current speaker, and "opposite" is the previous
    utterance of the interlocutor.

    Parameters
    ----------
    index_words : list
        Utterance order of the current scene.
    emo_dict : dict
        Dictionary of pairs "utt_name:label" generated by `get_label` function.
    emo : list
        List of emotions to use. Defaults to ['ang', 'hap', 'neu', 'sad'],
        which is used in the original implementation.

    Returns
    -------
    tuple
        Tuple of lists: center utterance names, target utterance names,
        opposite utterance names, center utterance labels, target utterance
        labels, opposite utterance labels, distance between center utterance
        and target utterance, distance between center utterance and opposite
        utterance.

    """
    # Utterance names
    centers, targets, opposites = list(), list(), list()
    # Utterance labels
    center_labels, target_labels, opposite_labels = list(), list(), list()
    # Distances
    target_dists, opposite_dists = list(), list()
    # Compatibility
    emo_dict[None] = None

    for i, center in enumerate(index_words):
        if emo_dict[center] not in emo:
            continue

        centers.append(center)
        center_labels.append(emo_dict[center])
        target = None
        target_dist = None
        opposite = None
        opposite_dist = None

        # Traverse up to previous 8 utterances
        for j in range(min(i, 8)):
            #  Stop if both are found
            if target is not None and opposite is not None:
                break

            dist = j + 1
            curr_word = index_words[i - dist]
            # If it is the same speaker
            if curr_word[-4] == center[-4]:
                if target is None:
                    target = curr_word
                    target_dist = dist
            elif opposite is None:
                opposite = curr_word
                opposite_dist = dist

        # Update
        targets.append(target)
        target_labels.append(emo_dict[target])
        target_dists.append(target_dist)

        opposites.append(opposite)
        opposite_labels.append(emo_dict[opposite])
        opposite_dists.append(opposite_dist)

    return (centers, targets, opposites, center_labels, target_labels,
            opposite_labels, target_dists, opposite_dists)


def generate_random_sample(index_words, emo_dict, keep_speakers=False,
                           emo=['ang', 'hap', 'neu', 'sad']):
    """Generate random training samples of pairs "center - target -
    opposite".

    Parameters
    ----------
    index_words : list
        Utterance order of the current scene.
    emo_dict : dict
        Dictionary of pairs "utt_name:label" generated by `get_label` function.
    keep_speakers : bool
        Whether to use the speaker type to sample.
            If True, "target" is the still the utterance of the current
            speaker, and "opposite" is still the utterance of the interlocutor.
            Otherwise, "target" and "opposite" are chosen completely randomly,
            i.e, "target" may now be the interlocutor's utterance, and vice
            versa. This is equivalent to the original implementation.
    emo : list
        List of emotions to use. Defaults to ['ang', 'hap', 'neu', 'sad'],
        which is used in the original implementation.

    Returns
    -------
    tuple
        Same as `generate_interaction_data`.

    """
    # Utterance names
    centers, targets, opposites = list(), list(), list()
    # Utterance labels
    center_labels, target_labels, opposite_labels = list(), list(), list()
    # Distances are always None (??)
    target_dists, opposite_dists = list(), list()

    for i, center in enumerate(index_words):
        if emo_dict[center] not in emo:
            continue

        centers.append(center)
        center_labels.append(emo_dict[center])

        if keep_speakers:
            # Get target and opposite indices
            target_idxs, opposite_idxs = list(), list()
            for j in range(len(index_words)):
                if j == i:
                    continue
                if index_words[i][-4] == index_words[j][-4]:
                    target_idxs.append(j)
                else:
                    opposite_idxs.append(j)
            # Sample
            target_i = np.random.choice(target_idxs)
            opposite_i = np.random.choice(opposite_idxs)

        # Sample completely randomly
        else:
            idxs = [j for j in range(len(index_words)) if j != i]
            target_i, opposite_i = np.random.choice(
                idxs, size=(2,), replace=False)

        targets.append(index_words[target_i])
        target_labels.append(emo_dict[index_words[target_i]])

        opposites.append(index_words[opposite_i])
        opposite_labels.append(emo_dict[index_words[opposite_i]])

        target_dists.append(None)
        opposite_dists.append(None)

    return (centers, targets, opposites, center_labels, target_labels,
            opposite_labels, target_dists, opposite_dists)


def generate_interaction_data(order_dict, emo_dict, val_prefix=None,
                              test_prefix=None, mode="context",
                              emo=['ang', 'hap', 'neu', 'sad'],
                              num_processes=32, **kwargs):
    """Generate train, validation and test data.

    Parameters
    ----------
    order_dict : dict
        Dictionary of pairs "scene:utt_orders" generated by the
        `get_dialog_order` function, where "utt_orders" is the order of the
        utterances in that particular scene.
    emo_dict : dict
        Dictionary of pairs "utt_name:label" generated by `get_label` function.
    val_prefix : str
        Scenes with this as prefix will be used as validation data, e.g.,
        "Ses01F".
    test_prefix ; str
        Scenes with this as prefix will be used as test data, e.g., "Ses01M".
    mode : str
        One of ["context", "random"]. Refer to `generate_interaction_sample`
        and `generate_random_sample`, respectively, for more details.
    emo : list
        List of emotions to use. Defaults to ['ang', 'hap', 'neu', 'sad'],
        which is used in the original implementation.
    num_processes : int
        Number of threads in the thread pool.
    **kwargs
        Keyword arguments to be passed to the sample generator (either
        `generate_interaction_sample` or `generate_random_sample`).

    Returns
    -------
    pd.DataFrame, pd.DataFrame, pd.DataFrame
        Dataframe of train, validation and test data, respectively.

    """
    if mode == "context":
        generator = generate_interaction_sample
    elif mode == "random":
        generator = generate_random_sample
    else:
        raise ValueError("Invalid mode value. Expected one of "
                         "['context', 'random'], got {} instead.".format(mode))

    train_scenes, val_scenes, test_scenes = list(), list(), list()
    train_utt_orders, val_utt_orders, test_utt_orders = list(), list(), list()
    for scene, utt_orders in order_dict.items():
        if val_prefix is not None and val_prefix in scene:
            val_scenes.append(scene)
            val_utt_orders.append(utt_orders)
        elif test_prefix is not None and test_prefix in scene:
            test_scenes.append(scene)
            test_utt_orders.append(utt_orders)
        else:
            train_scenes.append(scene)
            train_utt_orders.append(utt_orders)

    fn = lambda order: generator(order, emo_dict, emo=emo, **kwargs)

    pool = ThreadPool(processes=num_processes)

    train_data = list(zip(*pool.map(fn, train_utt_orders)))
    train_data = [sum(data, []) for data in train_data]

    val_data = list(zip(*pool.map(fn, val_utt_orders)))
    val_data = [sum(data, []) for data in val_data]

    test_data = list(zip(*pool.map(fn, test_utt_orders)))
    test_data = [sum(data, []) for data in test_data]

    # 8 is length of tuple returned by the generator
    assert len(train_data) in [0, 8]
    assert len(val_data) in [0, 8]
    assert len(test_data) in [0, 8]

    # Save results to a dataframe
    cols = ["center", "target", "opposite", "center_label", "target_label",
            "opposite_label", "target_dist", "opposite_dist"]
    df_train = pd.DataFrame(dict(zip(cols, train_data)))
    df_val = pd.DataFrame(dict(zip(cols, val_data)))
    df_test = pd.DataFrame(dict(zip(cols, test_data)))
    return df_train, df_val, df_test


class InteractionDataGenerator:
    """
    IEMOCAP interaction data generator, which returns a tuple of four arrays:
    center (the current utterances), target (previous utterance of the current
    speakers) and opposite (previous utterance of the interlocutors), as well
    as labels for the center utterances.
    """
    def __init__(self, df, batch_size, features_dict, sort_by_len=True,
                 original=False):
        """Initialize.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing data information.
        batch_size : int
            Batch size.
        features_dict : dict
            Dictionary of pairs "utt_name:features".
        sort_by_len : bool
            Whether to sort the data by length (maximum length of three:
            center, target and opposite across all samples). `sort_by_len=True`
            is equivalent to `mode="bucketing"` in the original implementation.
                If True, sort the data by length as described.
                Otherwise, shuffle the data randomly.
        original : bool
            Whether to truncate the data as in the original implementation. The
            data will be truncated to the maximum length of 3 arrays (center,
            target and opposite) of the LAST SAMPLE in the batch. This is a
            really strange behavior and could potentially be an unintentional
            mistake. This option is provided for the ONLY purpose of comparing
            with the original implementation. You don't want to set this to
            True in any other cases.

        """
        # Cache data
        self.df = df
        self.features_dict = features_dict
        self.sort_by_len = sort_by_len
        self.original = original

        # Info inferred from features
        self.dtype = list(features_dict.values())[0].dtype
        self.features_dim = list(features_dict.values())[0].shape[1]

        # Iterator
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(df) / batch_size)

        # Rearrange data
        self._rearrange()

        # One-hot encoding
        self.encoder = preprocessing.LabelEncoder()
        self.label = self.encoder.fit_transform(self.df["center_label"])

        # For compatibility
        self.features_dict[None] = np.zeros(shape=(0, self.features_dim))
        self.features_dict[np.NaN] = np.zeros(shape=(0, self.features_dim))

    def _rearrange(self):
        """
        If `self.sort_by_len=True`, sort by the maximum length of data
        across center, target and opposite features data.
        Otherwise, shuffle the dataframe randomly.
        """
        def transform(row):
            """Function to be applied to every row in the dataframe"""
            center_len = len(self.features_dict[row["center"]])
            target_len = opposite_len = 0
            if row["target"] not in [None, np.NaN]:
                target_len = len(self.features_dict[row["target"]])
            if row["opposite"] not in [None, np.NaN]:
                opposite_len = len(self.features_dict[row["opposite"]])
            return max(center_len, target_len, opposite_len)
        self.df["seq_len"] = self.df.apply(transform, axis=1)
        # Sort by len
        if self.sort_by_len:
            self.df = self.df.sort_values("seq_len")
            self.idxs = self.df.index.to_list()
            self.df = self.df.reset_index(drop=True)
        # Shuffle randomly
        else:
            self.idxs = np.random.permutation(len(self))
            self.df = self.df.iloc[self.idxs].reset_index(drop=True)

    def __iter__(self):
        """Initialize generator"""
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self))
        batch_size = end_idx - start_idx

        # Initialize
        max_len = max(self.df.iloc[start_idx:end_idx]["seq_len"])
        center_features = np.zeros(
            shape=(batch_size, max_len, self.features_dim), dtype=self.dtype)
        target_features = np.zeros(
            shape=(batch_size, max_len, self.features_dim), dtype=self.dtype)
        opposite_features = np.zeros(
            shape=(batch_size, max_len, self.features_dim), dtype=self.dtype)
        labels = np.zeros(shape=(batch_size,), dtype="int")

        # Read data
        df = self.df.loc[:, ["center", "target", "opposite",
                             "center_label", "seq_len"]]
        for i, j in enumerate(range(start_idx, end_idx)):
            center_f = self.features_dict[df.iloc[j, 0]]
            center_features[i, :len(center_f)] = center_f

            target_f = self.features_dict[df.iloc[j, 1]]
            target_features[i, :len(target_f)] = target_f

            opposite_f = self.features_dict[df.iloc[j, 2]]
            opposite_features[i, :len(opposite_f)] = opposite_f

            labels[i] = self.label[j]

        # In the original implementation, the features data is truncated to
        # the length of the LAST SAMPLE in the batch. This is a really strange
        # behavior and could be a unintentional mistake.
        if self.original:
            max_len = df.iloc[j]["seq_len"]
            center_features = center_features[:, :max_len]
            target_features = target_features[:, :max_len]
            opposite_features = opposite_features[:, :max_len]

        self.batch_idx += 1

        return center_features, target_features, opposite_features, labels

    def __len__(self):
        return len(self.df)

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

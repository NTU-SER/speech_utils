import os
import math
import pickle

import numpy as np
import python_speech_features as ps

from .. import read_wave_from_file

TRAIN_SESSIONS = ["Session1", "Session2", "Session3", "Session4"]
TEST_SESSIONS = ["Session5"]


def get_files(dataset_dir, train_sessions=TRAIN_SESSIONS,
              test_sessions=TEST_SESSIONS):
    """
    Adapted from https://github.com/xuanjihe/speech-emotion-recognition/blob/master/ExtractMel.py

    Get the lists of all the files needed for training, validation and test
    sets. By default, this function will parse all the improvised data in the
    first four sessions (1, 2, 3, 4) as training data, all the improvised female
    recordings in the last session as validation data, and all the improvised
    male recordings in the last session as test data.

    Parameters
    ----------
    dataset_dir : str
        Path to the `IEMOCAP_full_release` directory.
    train_sessions : list
        List of train sessions. By default, this is the list of the first four
        sessions.
    test_sessions: list
        List of sessions used for validation and test data.

    Returns
    -------
    tuple
        A tuple containing training, validation and test data, respectively,
        each with paths to *.wav files and path to its labels file.

    """
    sessions = train_sessions + test_sessions
    train_wav, val_wav, test_wav = list(), list(), list()
    train_lab, val_lab, test_lab = list(), list(), list()

    for session_name in os.listdir(dataset_dir):
        if session_name not in sessions:
            continue
        wav_dir = os.path.join(dataset_dir, session_name, "sentences/wav")
        eval_dir = os.path.join(dataset_dir, session_name, "dialog/EmoEvaluation")

        for speaker_name in os.listdir(wav_dir):
            # Only use improvised data, for example ".../wav/Ses01F_impro01"
            if speaker_name[7:12] != "impro":
                continue
            # Path to the directory containing all the *.wav files of the
            # current conversation
            speaker_dir = os.path.join(wav_dir, speaker_name)
            # Path to the labels of the current conversation
            label_path = os.path.join(eval_dir, speaker_name + ".txt")
            # Get a list of paths to all *.wav files
            wav_files = []
            for wav_name in os.listdir(speaker_dir):
                name, ext = os.path.splitext(wav_name)
                if ext != ".wav":
                    continue
                wav_files.append(os.path.join(speaker_dir, wav_name))
            # Training data
            if session_name in train_sessions:
                train_wav.append(wav_files)
                train_lab.append(label_path)
            else:
                # Female for validation data
                val_wav.append([path for path in wav_files if path[-8] == "F"])
                val_lab.append(label_path)
                # Male for test data
                test_wav.append([path for path in wav_files if path[-8] == "M"])
                test_lab.append(label_path)
    return train_wav, train_lab, val_wav, val_lab, test_wav, test_lab


def calc_zscore(dataset_dir, num_filters, emotions,
                sessions=TRAIN_SESSIONS, save_path=None):
    """
    Adapted from https://github.com/xuanjihe/speech-emotion-recognition/blob/master/zscore.py

    Read the entire IEMOCAP dataset and calculate z-score from the *.wav files.
    Remarks: Since the original implementation uses padding, the results are off
    slightly. In this implementation, the results are exact.

    Parameters
    -------
    dataset_dir: str
        Path to the `IEMOCAP_full_release` directory.
    num_filters: int
        Number of mel filters.
    emotions: list
        A list of emotions used for training and testing.
    sessions: list
        List of train sessions. Used to filter out unnecessary files.
    save_path: str
        Path to save the results as a pickle file.
        If None, the results will not be saved.

    Returns
    -------
    ans: tuple
        A tuple of (mean_mel_specs, std_mel_specs, mean_deltas,
                    std_deltas, mean_delta_deltas, std_delta_deltas).

    """
    mel_specs = list()
    deltas = list()
    delta_deltas = list()

    train_wav, train_lab, val_wav, val_lab, test_wav, test_lab = get_files(
        dataset_dir, train_sessions=sessions)

    for wav_files, label_path in zip(train_wav, train_lab):
        labels = dict() # to store labels of the current conversation
        # Read labels
        with open(label_path, "r") as fin:
            for line in fin:
                # If this line is sth like
                # [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                if line[0] == "[":
                    t = line.split()
                    # For e.g., {"Ses01F_impro01_F000": "neu", ...}
                    labels[t[3]] = t[4]
        # Read all *.wav files of the current conversation
        for wav_path in wav_files:
            # Get name
            _, wav_name = os.path.split(wav_path)
            wav_name, _ = os.path.splitext(wav_name)
            # Only keep allowed emotions
            emotion = labels[wav_name]
            if emotion not in emotions:
                continue
            # Read wave data
            wave_data, ticks, frame_rate = read_wave_from_file(wav_path)
            # Extract features
            mel_spec = ps.logfbank(wave_data, frame_rate, nfilt=num_filters)
            delta = ps.delta(mel_spec, 2)
            delta_delta = ps.delta(delta, 2)
            # Store
            mel_specs.append(mel_spec)
            deltas.append(delta)
            delta_deltas.append(delta_delta)

    mel_specs = np.vstack(mel_specs)
    deltas = np.vstack(deltas)
    delta_deltas = np.vstack(delta_deltas)
    # Calculate mean and std
    mean_mel_specs = mel_specs.mean(axis=0)
    mean_deltas = deltas.mean(axis=0)
    mean_delta_deltas = delta_deltas.mean(axis=0)
    std_mel_specs = mel_specs.std(axis=0)
    std_deltas = deltas.std(axis=0)
    std_delta_deltas = delta_deltas.std(axis=0)
    # To return
    ans = (mean_mel_specs, std_mel_specs, mean_deltas,
           std_deltas, mean_delta_deltas, std_delta_deltas)
    # Save results
    if save_path is not None:
        with open(save_path, "wb") as fout:
            pickle.dump(ans, fout)
    return ans


def _extract_data(emotion, mel_spec, mean_mel_specs, std_mel_specs,
                  delta, mean_deltas, std_deltas,
                  delta_delta, mean_delta_deltas, std_delta_deltas, eps=1e-5):
    """
    Helper function that extracts data from mel spectrograms, deltas and
    delta-deltas.

    Returns
    -------
    data: ndarray
        3-channel ndarray after normalization and padding.
    labels: list
        List of labels for each segment.
    num_segs: int
        Number of segments in this utterance.

    """
    time = mel_spec.shape[0]
    start, end = 0, 300
    num_segs = math.ceil(time / 300) # number of segments each with length of 300
    data_tot = []

    # Normalization
    mel_spec = (mel_spec - mean_mel_specs) / (std_mel_specs + eps)
    delta = (delta - mean_deltas) / (std_deltas + eps)
    delta_delta = (delta_delta - mean_delta_deltas) / (std_delta_deltas + eps)

    for i in range(num_segs):
        # The last segment
        if end > time:
            end = time
            start = max(0, end - 300)
        # Do padding
        mel_spec_pad = np.pad(
            mel_spec[start:end], ((0, 300 - (end - start)), (0, 0)), mode="constant")
        delta_pad = np.pad(
            delta[start:end], ((0, 300 - (end - start)), (0, 0)), mode="constant")
        delta_delta_pad = np.pad(
            delta_delta[start:end], ((0, 300 - (end - start)), (0, 0)), mode="constant")
        # Stack
        data = np.stack([mel_spec_pad, delta_pad, delta_delta_pad], axis=-1)
        data_tot.append(data)
        # Update variables
        start = end
        end = min(time, end + 300)

    data_tot = np.stack(data_tot)
    labels = [emotion] * num_segs
    return data_tot, labels, num_segs


def extract_data(wav_paths, lab_paths, metadata,
                 emot_map, num_filters, eps=1e-5):
    """
    Helper function that loop over the dataset and extracts data from it.

    Parameters
    -------
    metadata: tuple or list
        Values returned from function `calc_zscore`.
    emot_map: dict
        Mapping emotions with their corresponding integer label.

    Returns
    -------
    data_tot: ndarray
        Data, with the first dimension being the total number of segments.
    labels_tot: ndarray
        Array containing labels for each utterance.
    labels_segs_tot: ndarray
        Array containing labels for each segment.
    segs: ndarray
        Array containing the number of segments for each utterance.

    """
    assert len(wav_paths) == len(lab_paths)

    emotions = emot_map.keys()
    mean_mel_specs, std_mel_specs, mean_deltas, std_deltas, mean_delta_deltas, std_delta_deltas = metadata
    # labels_segs_tot stores labels for each segment, while segs store number
    # of segments per utterance
    data_tot, labels_tot, labels_segs_tot, segs = list(), list(), list(), list()

    for wav_files, label_path in zip(wav_paths, lab_paths):
        labels = dict() # to store labels of the current conversation
        # Read labels
        with open(label_path, "r") as fin:
            for line in fin:
                # If this line is sth like
                # [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                if line[0] == "[":
                    t = line.split()
                    # For e.g., {"Ses01F_impro01_F000": "neu", ...}
                    labels[t[3]] = t[4]
        # Read all *.wav files of the current conversation
        for wav_path in wav_files:
            # Get name
            _, wav_name = os.path.split(wav_path)
            wav_name, _ = os.path.splitext(wav_name)
            # Only keep allowed emotions
            emotion = labels[wav_name]
            if emotion not in emotions:
                continue
            emotion = emot_map[emotion]
            # Read wave data
            wave_data, ticks, frame_rate = read_wave_from_file(wav_path)
            # Extract features
            mel_spec = ps.logfbank(wave_data, frame_rate, nfilt=num_filters)
            delta = ps.delta(mel_spec, 2)
            delta_delta = ps.delta(delta, 2)
            # Process data
            data, labels_, num_segs = _extract_data(
                emotion, mel_spec, mean_mel_specs, std_mel_specs,
                delta, mean_deltas, std_deltas,
                delta_delta, mean_delta_deltas, std_delta_deltas, eps=eps)

            data_tot.append(data)
            labels_tot.append(emotion)
            labels_segs_tot.extend(labels_)
            segs.append(num_segs)
    # Post process
    data_tot = np.vstack(data_tot)
    labels_tot = np.asarray(labels_tot, dtype=np.int8)
    labels_segs_tot = np.asarray(labels_segs_tot, dtype=np.int8)
    segs = np.asarray(segs, dtype=np.int8)
    # Make sure everything works properly
    assert len(labels_tot) == len(segs)
    assert data_tot.shape[0] == labels_segs_tot.shape[0] == sum(segs)

    return data_tot, labels_tot, labels_segs_tot, segs


def extract_mel(dataset_dir, num_filters, emot_map, metadata,
                train_sessions=TRAIN_SESSIONS, test_sessions=TEST_SESSIONS,
                save_path=None, eps=1e-5):
    """
    Adapted from https://github.com/xuanjihe/speech-emotion-recognition/blob/master/ExtractMel.py

    Read the entire IEMOCAP dataset and extract data from it.

    Parameters
    -------
    dataset_dir: str
        Path to the `IEMOCAP_full_release` directory.
    num_filters: int
        Number of mel filters.
    emot_map: dict
        Mapping emotions with their corresponding integer label.
    metadata: tuple or list
        Values returned from function `calc_zscore`.
    train_sessions: list
        List of train sessions.
    test_sessions: list
        Liat of validation and test sessions.
    save_path: str
        Path to save the results as a pickle file.
        If None, the results will not be saved.
    eps: float
        To avoid dividing by 0 in normalization.

    Returns
    -------
    train_data: ndarray
    train_labels: ndarray
    val_data and test_data: ndarray
        Contain data for each segment.
    val_labels and test_labels: ndarray
        Contain labels for each utterance.
    val_segs_labels and test_segs_labels: ndarray
        Contain labels for each segment.
    val_segs and test_segs: ndarray
        Contain number of segments for each utterance.

    """
    # Paths
    train_wav, train_lab, val_wav, val_lab, test_wav, test_lab = get_files(
        dataset_dir, train_sessions, test_sessions)
    # Extract data
    train_data, _, train_labels, _ = extract_data(
        train_wav, train_lab, metadata, emot_map, num_filters)
    val_data, val_labels, val_segs_labels, val_segs = extract_data(
        val_wav, val_lab, metadata, emot_map, num_filters)
    test_data, test_labels, test_segs_labels, test_segs = extract_data(
        test_wav, test_lab, metadata, emot_map, num_filters)

    ans = (train_data, train_labels,
           val_data, val_labels, val_segs_labels, val_segs,
           test_data, test_labels, test_segs_labels, test_segs)
    # Save results
    if save_path is not None:
        with open(save_path, "wb") as fout:
            pickle.dump(ans, fout)
    return ans


"""
These below functions are for PyTorch version of 3D-ACRNN ARCNN
"""
